"""Logarithmic energy minimization for N points on the unit sphere.
Uses a hybrid approach: L-BFGS-B for initial convergence, followed by
a high-precision Damped Newton Method using the explicit Hessian for
final polishing. Includes multi-start and annealing for robustness."""

import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import scipy.linalg

N = 100
_DEADLINE = 145.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    grad = -raw_grad
    
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _compute_hessian(pts):
    """
    Compute the Euclidean Hessian of the energy for N points on the sphere.
    Returns a 3N x 3N matrix.
    """
    n = pts.shape[0]
    H = np.zeros((3 * n, 3 * n))
    
    # Compute pairwise distances and vectors
    # dist_sq[i, j] = |x_i - x_j|^2
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 0.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    # Precompute 1/d^2 and 2/d^4
    inv_d2 = 1.0 / np.maximum(dist_sq, 1e-30)
    inv_d4 = inv_d2 * inv_d2
    np.fill_diagonal(inv_d2, 0.0)
    np.fill_diagonal(inv_d4, 0.0)
    
    for i in range(n):
        for j in range(i + 1, n):
            d2 = dist_sq[i, j]
            if d2 < 1e-12:
                continue
            
            u = pts[i] - pts[j]
            u_u_T = np.outer(u, u)
            
            # Hessian block for pair (i, j)
            # H_ij = 2 u u^T / d^4 - I / d^2
            block = 2.0 * inv_d4[i, j] * u_u_T - inv_d2[i, j] * np.eye(3)
            
            # Fill the 3x3 blocks in the 3N x 3N matrix
            i_start = 3 * i
            j_start = 3 * j
            
            H[i_start:i_start+3, j_start:j_start+3] = block
            H[j_start:j_start+3, i_start:i_start+3] = block.T
            
            # Diagonal blocks
            # H_ii += I/d^2 - 2 u u^T / d^4
            # H_jj += I/d^2 - 2 u u^T / d^4
            diag_block = inv_d2[i, j] * np.eye(3) - 2.0 * inv_d4[i, j] * u_u_T
            H[i_start:i_start+3, i_start:i_start+3] += diag_block
            H[j_start:j_start+3, j_start:j_start+3] += diag_block
            
    return H


def _newton_step(pts, g_tangent, H_euclidean):
    """
    Compute a damped Newton step.
    1. Solve H_euclidean * delta = -g_euclidean (approximated by -g_tangent projected back?)
    Actually, we have g_tangent. The Euclidean gradient g_eucl is such that proj(g_eucl) = g_tangent.
    We can assume g_eucl = g_tangent + lambda * pts.
    To find the best Newton step, we solve H * delta = -g_eucl.
    However, we don't know lambda.
    Alternative: Solve the projected system.
    Let P = I - pts pts^T (block diagonal projection).
    We want delta in tangent space.
    Solve H_t * delta = -g_tangent, where H_t is the projected Hessian?
    
    Simpler robust approach:
    Solve H * delta = -g_tangent (treating g_tangent as the force).
    Then project delta to tangent space.
    """
    n = pts.shape[0]
    
    # Regularize Hessian if necessary (though it should be positive definite for log energy? 
    # Log energy is convex locally? No, it has saddle points. But near minimum it should be PD.)
    # Add small regularization to ensure invertibility
    reg = 1e-6 * np.eye(3 * n)
    
    try:
        # Solve for delta
        delta = scipy.linalg.solve(H_euclidean + reg, -g_tangent.reshape(-1), assume_a='sym')
    except Exception:
        # Fallback to gradient descent step if solve fails
        delta = -g_tangent.reshape(-1)
        
    delta = delta.reshape(n, 3)
    
    # Project delta to tangent space
    dot = np.sum(delta * pts, axis=1, keepdims=True)
    delta_tangent = delta - dot * pts
    
    return delta_tangent


def _newton_optimize(pts, deadline, max_iter=50, damp=0.5):
    """
    Run Damped Newton optimization.
    """
    current = pts.copy()
    best_pts = current.copy()
    best_energy = _energy(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        e, g = _energy_grad(current)
        
        # Check convergence
        if np.linalg.norm(g) < 1e-12:
            break
            
        H = _compute_hessian(current)
        
        # Newton step
        delta = _newton_step(current, g, H)
        
        # Line search
        alpha = damp
        improved = False
        for _ls in range(20):
            new_pts = current + alpha * delta
            new_pts = _project(new_pts)
            
            new_e = _energy(new_pts)
            
            if new_e < e:
                current = new_pts
                improved = True
                if new_e < best_energy:
                    best_energy = new_e
                    best_pts = current.copy()
                break
            else:
                alpha *= 0.5
                if alpha < 1e-6:
                    break
                    
        if not improved:
            # If no improvement, break or try smaller step
            if alpha < 1e-6:
                break
                
    return best_pts


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _tangent_to_sphere(pts, vec):
    """Project vec onto tangent space at each point in pts."""
    dot = np.sum(vec * pts, axis=1, keepdims=True)
    return vec - dot * pts


class _TimeoutException(Exception):
    pass


def _lbfgs_optimize(pts, deadline, max_iter=1000, ftol=1e-14, gtol=1e-12):
    """Run L-BFGS-B optimization from a starting point with tangent projection."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        if time.time() > deadline:
            raise _TimeoutException()
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)
    
    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
                'maxls': 20,
                'maxfun': max_iter * 10,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except _TimeoutException:
        return pts
    except Exception:
        return pts


def _anneal(pts, deadline, n_steps=100, base_lr=0.005, min_lr=0.0001, 
            base_pert=0.002, min_pert=0.0001, seed=42):
    """Gradient-flow annealing with adaptive perturbation."""
    current = pts.copy()
    n = current.shape[0]
    rng = np.random.default_rng(seed)
    
    best_in_anneal = current.copy()
    best_e = _energy(current)
    
    for step in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        
        frac = step / max(n_steps - 1, 1)
        lr = base_lr * (min_lr / base_lr) ** frac
        pert_scale = base_pert * (min_pert / base_pert) ** frac
        
        e, g = _energy_grad(current)
        perturbation = pert_scale * rng.standard_normal((n, 3))
        step_vec = -lr * (g + perturbation)
        
        step_vec = _tangent_to_sphere(current, step_vec)
        
        current = current + step_vec
        current = _project(current)
        
        e_now = _energy(current)
        if e_now < best_e:
            best_e = e_now
            best_in_anneal = current.copy()
    
    return best_in_anneal


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences
    for s in range(10):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(5):
        jitter_scale = 0.03 + 0.008 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random
    for s in range(10):
        initializations.append(_random_sphere(np.random.default_rng(200 + s)))
        
    best_pts = None
    best_energy = float('inf')
    
    # Score initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Phase 1: L-BFGS optimization
    top_candidates = scored_inits[:20]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 20.0:
            break
        
        optimized = _lbfgs_optimize(init_pts, deadline, max_iter=800, ftol=1e-14, gtol=1e-12)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Phase 2: Newton Polish
    if best_pts is not None and time.time() < deadline - 10.0:
        # Apply Newton polish to the best point
        polished = _newton_optimize(best_pts, deadline, max_iter=50)
        e_precise = _precise_energy(polished)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = polished.copy()
            
        # Apply Newton polish to a few other good candidates
        for e_q, init_p in top_candidates[1:5]:
            if time.time() > deadline - 5.0:
                break
            # Quick L-BFGS first
            quick_opt = _lbfgs_optimize(init_p, deadline, max_iter=400, ftol=1e-13, gtol=1e-11)
            # Then Newton
            newton_opt = _newton_optimize(quick_opt, deadline, max_iter=30)
            e_precise = _precise_energy(newton_opt)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = newton_opt.copy()
                
    # Phase 3: Final L-BFGS + Newton combo if time permits
    if best_pts is not None and time.time() < deadline - 5.0:
        # One more round of L-BFGS to ensure we are at a local min, then Newton
        final_lbfgs = _lbfgs_optimize(best_pts, deadline, max_iter=500, ftol=1e-15, gtol=1e-13)
        final_newton = _newton_optimize(final_lbfgs, deadline, max_iter=50)
        e_precise = _precise_energy(final_newton)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = final_newton.copy()

    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
