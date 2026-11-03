"""Logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill


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
    # Ensure points are on the sphere for accurate distance computation
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

    # grad_i = -sum_j W_ij * x_i + sum_j W_ij * x_j
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)

    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts

    return energy, grad


def _exp_map(pts, tangent):
    """Exponential map on the unit sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    # For small angles, use first-order approximation
    mask_small = norms < 1e-8
    # General case: x' = x*cos(theta) + v*sin(theta)/|v|
    theta = norms_safe
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    v_normalized = tangent / norms_safe
    new_pts = pts * cos_theta + v_normalized * sin_theta
    # For very small steps, use linear approximation
    if np.any(mask_small):
        new_pts[mask_small] = pts[mask_small] + tangent[mask_small]
    return _normalize(new_pts)


def _riemannian_polish(pts, deadline, max_iter=2000, lr_init=1e-3):
    """
    Riemannian gradient descent with backtracking line search on the sphere.
    Uses exponential map for retraction. This helps escape shallow local minima
    that Euclidean L-BFGS-B might get stuck in.
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
        
        # Backtracking line search
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
            # Slightly increase learning rate if we made good progress
            lr = min(lr * 1.05, 0.05)
        else:
            lr *= 0.8
    
    return best_pts, best_energy


def _newton_polish(pts, deadline, max_iter=50):
    """
    Newton-like refinement using Hessian approximation via finite differences
    of the gradient. Applied as a final polish for very tight convergence.
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
        
        # Approximate Hessian in tangent space using central differences
        # Only do this for the first few iterations due to cost
        if it < 10:
            hessian_flat = np.zeros_like(g_flat)
            dim = g_flat.shape[0]
            for i in range(0, dim, 3):  # Every 3rd component (one per point)
                for k in range(3):
                    idx = i + k
                    # Forward difference
                    p_plus = best_pts.copy()
                    p_plus.flat[idx] += eps
                    p_plus = _normalize(p_plus)
                    _, g_plus = _energy_grad(p_plus)
                    
                    # Backward difference
                    p_minus = best_pts.copy()
                    p_minus.flat[idx] -= eps
                    p_minus = _normalize(p_minus)
                    _, g_minus = _energy_grad(p_minus)
                    
                    hessian_flat[idx] = (g_plus.reshape(-1)[idx] - g_minus.reshape(-1)[idx]) / (2 * eps)
            
            # Build diagonal Hessian approximation (cheap)
            h_diag = np.abs(hessian_flat) + 1e-8
        else:
            h_diag = np.abs(g_flat) + 1e-8
        
        # Newton step: -H^{-1} * g, using diagonal approximation
        step = -g_flat / h_diag
        step_norm = np.linalg.norm(step)
        
        if step_norm < 1e-14:
            break
        
        # Limit step size
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
            # Try half step
            new_pts = best_pts + 0.5 * step.reshape(n, 3)
            new_pts = _normalize(new_pts)
            new_e = _precise_energy(new_pts)
            if new_e < best_energy:
                best_pts = new_pts
                best_energy = new_e
            else:
                break
    
    return best_pts, best_energy


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


def _optimize_from_start(pts, deadline):
    """
    Optimize from a single starting configuration using multi-phase L-BFGS-B.
    Returns (best_pts, best_precise_energy).
    """
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)

    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_grad(p)
        return e, g.reshape(-1)

    # Phase 1: Coarse L-BFGS-B
    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            objective_and_grad, x0,
            method='L-BFGS-B', jac=True,
            options={'maxiter': 3000, 'maxfun': 15000, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_p = _precise_energy(new_pts)
            if e_p < best_energy:
                best_pts = new_pts
                best_energy = e_p
    except Exception:
        pass

    if time.time() > deadline:
        return best_pts, best_energy

    # Phase 2: Medium refinement
    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            objective_and_grad, x0,
            method='L-BFGS-B', jac=True,
            options={'maxiter': 5000, 'maxfun': 25000, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_p = _precise_energy(new_pts)
            if e_p < best_energy:
                best_pts = new_pts
                best_energy = e_p
    except Exception:
        pass

    if time.time() > deadline:
        return best_pts, best_energy

    # Phase 3: High-precision polish
    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            objective_and_grad, x0,
            method='L-BFGS-B', jac=True,
            options={'maxiter': 10000, 'maxfun': 50000, 'ftol': 1e-15, 'gtol': 1e-11}
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_p = _precise_energy(new_pts)
            if e_p < best_energy:
                best_pts = new_pts
                best_energy = e_p
    except Exception:
        pass

    return best_pts, best_energy


def _spectral_init(n, rng, l_max=8):
    """
    Generate a smooth initial configuration by sampling a random band-limited
    spherical function. This provides a structured, smooth starting point
    that avoids the combinatorial explosion of local minima.
    """
    # Generate random spherical harmonic coefficients up to l_max
    # We use a simplified approach: random smooth function on sphere
    # by combining random low-frequency spherical harmonics

    # Simple approach: random smooth perturbation of uniform distribution
    # Use a random smooth function defined by low-frequency components
    theta = rng.uniform(0, 2 * np.pi, n)
    cos_phi = 2.0 * rng.uniform(0, 1, n) - 1.0
    sin_phi = np.sqrt(1.0 - cos_phi ** 2)

    pts = np.column_stack([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        cos_phi
    ])

    # Add smooth low-frequency perturbation
    for l in range(1, l_max + 1):
        # Random amplitudes and phases for each m
        for m in range(-l, l + 1):
            amp = rng.normal(0, 0.01 / (l + 1))
            phase = rng.uniform(0, 2 * np.pi)
            # Simple approximation: modulate points
            # This is a heuristic smooth perturbation
            t = np.arctan2(pts[:, 1], pts[:, 0])
            c = pts[:, 2]
            # Use associated Legendre-like modulation (simplified)
            mod = amp * np.cos(m * t + phase) * (c ** abs(m))
            pts[:, 2] += mod * 0.1
            pts[:, 0] += amp * 0.05 * np.sin(m * t + phase) * sin_phi

    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    rng = np.random.default_rng(12345)

    # Generate diverse initializations
    initializations = []

    # 1. Fibonacci spiral (best structured start)
    initializations.append(_fibonacci_spiral(N))

    # 2. Spectral/smooth initializations
    for seed in range(5):
        r = np.random.default_rng(seed * 100 + 7)
        initializations.append(_spectral_init(N, r, l_max=6))

    # 3. Random initializations
    for i in range(10):
        initializations.append(_random_sphere(N, rng))

    # 4. Perturbed Fibonacci
    base = _fibonacci_spiral(N)
    for scale in [0.005, 0.01, 0.02, 0.05]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))

    # 5. Additional structured: shifted golden spiral
    for offset in [0.0, 0.15, 0.3, 0.5]:
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
        if time.time() > deadline:
            break

        optimized_pts, e_precise = _optimize_from_start(init_pts, deadline)

        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts

        # If we've found something very good, reduce remaining work
        if best_energy < -1083.37:
            # Keep optimizing but with tighter budget per remaining start
            pass

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)

    # Final high-precision polish on the best configuration
    if time.time() < deadline - 30:
        try:
            final_pts, final_e = _optimize_from_start(best_pts, deadline)
            if final_e < best_energy:
                best_pts = final_pts
                best_energy = final_e
        except Exception:
            pass
    
    # Riemannian manifold polish
    if time.time() < deadline - 15:
        try:
            riem_pts, riem_e = _riemannian_polish(best_pts, deadline, max_iter=1000, lr_init=5e-4)
            if riem_e < best_energy:
                best_pts = riem_pts
                best_energy = riem_e
        except Exception:
            pass
    
    # Newton-like final polish for tightest convergence
    if time.time() < deadline - 5:
        try:
            newton_pts, newton_e = _newton_polish(best_pts, deadline, max_iter=30)
            if newton_e < best_energy:
                best_pts = newton_pts
                best_energy = newton_e
        except Exception:
            pass

    # Final normalization and validation
    best_pts = _normalize(best_pts)
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _normalize(_fibonacci_spiral(N))

    return best_pts
