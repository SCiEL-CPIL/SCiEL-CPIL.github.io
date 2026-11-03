"""Logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
_START = time.time()
_DEADLINE = _START + 148.0  # safety margin before 150s kill


def _remaining():
    return _DEADLINE - time.time()


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _energy_fast(pts):
    """Fast energy using dot-product distance computation."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and tangent-space gradient using dot-product distances."""
    n = pts.shape[0]
    pts = _normalize(pts)
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))

    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)

    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)

    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts

    return energy, grad


def _riemannian_lbfgs(pts, deadline, maxiter=5000, ftol=1e-14, gtol=1e-10):
    """L-BFGS-B in tangent space: project gradient, optimize in flat coords, re-normalize."""
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)

    def obj_and_grad(flat):
        p = _normalize(flat.reshape(n, 3))
        e, g = _energy_grad(p)
        # Project gradient to tangent space (already done in _energy_grad)
        return e, g.reshape(-1)

    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            obj_and_grad, x0,
            method='L-BFGS-B', jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': ftol, 'gtol': gtol}
        )
        new_pts = _normalize(result.x.reshape(n, 3))
        e_new = _precise_energy(new_pts)
        if e_new < best_energy:
            best_pts = new_pts
            best_energy = e_new
    except Exception:
        pass

    return best_pts, best_energy


def _exp_map(pts, tangent):
    """Exponential map on the unit sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    theta = norms_safe
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    v_normalized = tangent / norms_safe
    new_pts = pts * cos_theta + v_normalized * sin_theta
    mask_small = norms < 1e-8
    if np.any(mask_small):
        new_pts[mask_small] = pts[mask_small] + tangent[mask_small]
    return _normalize(new_pts)


def _precise_energy(pts):
    """Compute energy using math.fsum for final scoring accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _simulated_annealing(pts, deadline, max_iter=10000, T_init=1.0, T_min=0.01, alpha=0.9997):
    """
    Simulated annealing on the sphere with adaptive step size.
    """
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)
    current_pts = best_pts.copy()
    current_energy = best_energy

    T = T_init
    rng = np.random.default_rng(42)
    step_scale = 0.15

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Adaptive step size based on temperature
        step_size = step_scale * (T / T_init + 0.05)
        tangent = rng.standard_normal((n, 3)) * step_size
        dot = np.sum(tangent * current_pts, axis=1, keepdims=True)
        tangent = tangent - dot * current_pts

        new_pts = _exp_map(current_pts, tangent)
        new_energy = _energy_fast(new_pts)

        # Metropolis acceptance
        delta_e = new_energy - current_energy
        accept = False
        if delta_e < 0:
            accept = True
        else:
            log_accept = -delta_e / T
            if log_accept > -700:  # avoid underflow
                accept = rng.random() < math.exp(log_accept)

        if accept:
            current_pts = new_pts
            current_energy = new_energy
            if new_energy < best_energy:
                best_pts = new_pts
                best_energy = new_energy
                step_scale = min(step_scale * 1.02, 0.5)
        else:
            step_scale = max(step_scale * 0.98, 0.02)

        T *= alpha
        if T < T_min:
            T = T_min

    return best_pts, best_energy


def _optimize_from_start(pts, deadline):
    """
    Optimize from a single starting configuration using multi-phase approach.
    Returns (best_pts, best_precise_energy).
    """
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)

    # Phase 1: Riemannian L-BFGS (coarse)
    try:
        p, e = _riemannian_lbfgs(best_pts, deadline, maxiter=2000, ftol=1e-10, gtol=1e-6)
        if e < best_energy:
            best_pts = p
            best_energy = e
    except Exception:
        pass

    if time.time() > deadline:
        return best_pts, best_energy

    # Phase 2: Simulated annealing to escape local minima
    if time.time() < deadline - 20:
        try:
            sa_pts, sa_e = _simulated_annealing(best_pts, deadline, max_iter=8000, T_init=0.8, T_min=0.02, alpha=0.9998)
            if sa_e < best_energy:
                best_pts = sa_pts
                best_energy = sa_e
        except Exception:
            pass

    if time.time() > deadline:
        return best_pts, best_energy

    # Phase 3: Riemannian L-BFGS (medium)
    try:
        p, e = _riemannian_lbfgs(best_pts, deadline, maxiter=3000, ftol=1e-12, gtol=1e-8)
        if e < best_energy:
            best_pts = p
            best_energy = e
    except Exception:
        pass

    if time.time() > deadline:
        return best_pts, best_energy

    # Phase 4: Riemannian L-BFGS (high-precision polish)
    try:
        p, e = _riemannian_lbfgs(best_pts, deadline, maxiter=5000, ftol=1e-14, gtol=1e-10)
        if e < best_energy:
            best_pts = p
            best_energy = e
    except Exception:
        pass

    return best_pts, best_energy


def _riemannian_polish(pts, deadline, max_iter=2000, lr_init=1e-3):
    """
    Riemannian gradient descent with backtracking line search on the sphere.
    """
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)
    
    lr = lr_init
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(best_pts)
        g_norm = np.linalg.norm(g)
        
        if g_norm < 1e-12:
            break
        
        alpha = lr
        for _ in range(30):
            new_pts = _exp_map(best_pts, -alpha * g)
            new_e = _energy_fast(new_pts)
            if new_e < e - 1e-4 * alpha * g_norm * g_norm:
                break
            alpha *= 0.5
        
        new_pts = _exp_map(best_pts, -alpha * g)
        new_e_precise = _precise_energy(new_pts)
        
        if new_e_precise < best_energy:
            best_pts = new_pts
            best_energy = new_e_precise
            lr = min(lr * 1.05, 0.05)
        else:
            lr *= 0.8
    
    return best_pts, best_energy


def _newton_polish(pts, deadline, max_iter=50):
    """
    Newton-like refinement using Hessian approximation via finite differences.
    """
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)
    
    eps = 1e-6
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(best_pts)
        g_flat = g.reshape(-1)
        g_norm = np.linalg.norm(g_flat)
        
        if g_norm < 1e-14:
            break
        
        if it < 10:
            hessian_flat = np.zeros_like(g_flat)
            dim = g_flat.shape[0]
            for i in range(0, dim, 3):
                for k in range(3):
                    idx = i + k
                    p_plus = best_pts.copy()
                    p_plus.flat[idx] += eps
                    p_plus = _normalize(p_plus)
                    _, g_plus = _energy_grad(p_plus)
                    
                    p_minus = best_pts.copy()
                    p_minus.flat[idx] -= eps
                    p_minus = _normalize(p_minus)
                    _, g_minus = _energy_grad(p_minus)
                    
                    hessian_flat[idx] = (g_plus.reshape(-1)[idx] - g_minus.reshape(-1)[idx]) / (2 * eps)
            
            h_diag = np.abs(hessian_flat) + 1e-8
        else:
            h_diag = np.abs(g_flat) + 1e-8
        
        step = -g_flat / h_diag
        step_norm = np.linalg.norm(step)
        
        if step_norm < 1e-14:
            break
        
        max_step = 1e-3
        if step_norm > max_step:
            step *= max_step / step_norm
        
        new_pts = best_pts + step.reshape(n, 3)
        new_pts = _normalize(new_pts)
        new_e = _precise_energy(new_pts)
        
        if new_e < best_energy:
            best_pts = new_pts
            best_energy = new_e
        else:
            new_pts = best_pts + 0.5 * step.reshape(n, 3)
            new_pts = _normalize(new_pts)
            new_e = _precise_energy(new_pts)
            if new_e < best_energy:
                best_pts = new_pts
                best_energy = new_e
            else:
                break
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    rng = np.random.default_rng(12345)

    # Generate diverse but fewer initializations (quality over quantity)
    initializations = []

    # 1. Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))

    # 2. A few random initializations
    for i in range(3):
        initializations.append(_random_sphere(N, rng))

    # 3. Perturbed Fibonacci (moderate perturbation)
    base = _fibonacci_spiral(N)
    for scale in [0.02, 0.08]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))

    # 4. Shifted golden spiral
    for offset in [0.2, 0.5]:
        i = np.arange(N) + 0.5 + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        initializations.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

    # Sort by quick energy to prioritize good starts
    scored_inits = []
    for init in initializations:
        e_quick = _energy_fast(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])

    best_pts = None
    best_energy = float('inf')

    for e_quick, init_pts in scored_inits:
        if time.time() > deadline - 10:
            break

        optimized_pts, e_precise = _optimize_from_start(init_pts, deadline)

        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _precise_energy(best_pts)

    # Final high-precision polish with remaining time
    if time.time() < deadline - 5:
        try:
            final_pts, final_e = _riemannian_lbfgs(best_pts, deadline, maxiter=8000, ftol=1e-15, gtol=1e-11)
            if final_e < best_energy:
                best_pts = final_pts
                best_energy = final_e
        except Exception:
            pass

    # Final normalization and validation
    best_pts = _normalize(best_pts)
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _normalize(_fibonacci_spiral(N))

    return best_pts
