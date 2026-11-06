"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij) using math.fsum for precision."""
    n = len(pts)
    terms = []
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            d = np.linalg.norm(pi - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fast_energy(pts):
    """Fast vectorized energy for optimization loops. Returns (energy, min_dist_sq)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist_sq = np.min(dist_sq)
    if min_dist_sq < 1e-12:
        return float('inf'), min_dist_sq
    energy = -0.25 * np.sum(np.log(dist_sq))
    return float(energy), float(min_dist_sq)


def _fast_grad_tangent(pts):
    """Compute gradient of log energy projected to tangent space. Vectorized."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    mask = ~np.eye(n, dtype=bool)
    if np.any(dist_sq[mask] < 1e-12):
        return None
    
    inv_d2 = np.zeros_like(dist_sq)
    valid = dist_sq > 1e-24
    inv_d2[valid] = 1.0 / dist_sq[valid]
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_diff = diff * inv_d2[:, :, np.newaxis]
    grad_raw = -np.sum(weighted_diff, axis=1)
    
    dot_products = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad_tangent = grad_raw - dot_products * pts
    
    return grad_tangent


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=0):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_gradient_descent(pts, deadline, max_iters=8000):
    """Run vectorized gradient descent from initial points. Returns (pts, energy)."""
    pts = pts.copy()
    _, min_ds = _fast_energy(pts)
    if min_ds < 1e-12:
        pts += 0.01 * np.random.randn(*pts.shape)
        pts = _project_to_sphere(pts)
    
    lr = 0.08
    prev_energy = _fast_energy(pts)[0]
    min_lr = 1e-14
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        grad = _fast_grad_tangent(pts)
        if grad is None:
            pts += 0.01 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
            continue
        
        # Try a few step sizes if current one fails
        accepted = False
        for attempt in range(10):
            pts_new = pts - lr * grad
            pts_new = _project_to_sphere(pts_new)
            
            new_energy, _ = _fast_energy(pts_new)
            
            if new_energy < prev_energy:
                pts = pts_new
                prev_energy = new_energy
                # Adaptive learning rate: increase slightly if we're making good progress
                lr *= 1.002
                if lr > 0.15:
                    lr = 0.15
                accepted = True
                break
            else:
                lr *= 0.5
        
        if not accepted and lr < min_lr:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-13:
            break
    
    return pts, prev_energy


def _optimize_lbfgsb(pts, deadline, maxiter=3000):
    """Run L-BFGS-B optimization on flattened points. Returns (pts, energy)."""
    n = pts.shape[0]
    eval_count = [0]
    
    def _energy_and_grad_flat(x):
        eval_count[0] += 1
        if time.time() > deadline:
            return float('inf'), np.zeros_like(x)
        p = x.reshape(n, 3)
        # Project to sphere for valid energy computation
        p = _project_to_sphere(p)
        energy = _fast_energy(p)[0]
        grad_t = _fast_grad_tangent(p)
        if grad_t is None:
            return energy, np.zeros_like(x)
        return energy, grad_t.flatten()
    
    x0 = pts.flatten().copy()
    
    try:
        res = minimize(
            _energy_and_grad_flat,
            x0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 80, 'm': 50}
        )
        # Project result to sphere
        p = _project_to_sphere(res.x.reshape(n, 3))
        energy = _fast_energy(p)[0]
        return p, energy
    except Exception:
        return pts, _fast_energy(pts)[0]


def _polish(pts, deadline, max_iters=5000):
    """High-precision polish pass using gradient descent with very small steps."""
    pts = pts.copy()
    prev_energy = _fast_energy(pts)[0]
    lr = 0.02
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        grad = _fast_grad_tangent(pts)
        if grad is None:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        accepted = False
        for attempt in range(12):
            pts_new = pts - lr * grad
            pts_new = _project_to_sphere(pts_new)
            
            new_energy, _ = _fast_energy(pts_new)
            
            if new_energy < prev_energy:
                pts = pts_new
                prev_energy = new_energy
                # Gentle increase when making progress
                lr *= 1.001
                if lr > 0.08:
                    lr = 0.08
                accepted = True
                break
            else:
                lr *= 0.4
        
        if not accepted and lr < 1e-16:
            break
    
    return pts, prev_energy


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin of 2 seconds
    
    # Start with Fibonacci sphere
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy(best_pts)
    
    # Phase 1: Gradient descent from multiple initializations
    n_restarts = 16
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Different initializations
        if restart == 0:
            pts = _fibonacci_sphere(N, seed=0)
        elif restart == 1:
            pts = _fibonacci_sphere(N, seed=1)
            pts += 0.001 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
        elif restart == 2:
            pts = _fibonacci_sphere(N, seed=0)
            pts += 0.01 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
        elif restart == 3:
            pts = _fibonacci_sphere(N, seed=0)
            pts += 0.1 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
        elif restart == 4:
            pts = _fibonacci_sphere(N, seed=2)
            pts += 0.05 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
        elif restart == 5:
            pts = _fibonacci_sphere(N, seed=3)
            pts += 0.02 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
        elif restart == 6:
            pts = _fibonacci_sphere(N, seed=4)
            pts += 0.03 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
        elif restart == 7:
            pts = _fibonacci_sphere(N, seed=5)
            pts += 0.015 * np.random.randn(*pts.shape)
            pts = _project_to_sphere(pts)
        else:
            pts = _random_sphere(N, seed=restart * 42 + 7)
        
        # Optimize using gradient descent
        optimized_pts, opt_energy = _optimize_gradient_descent(pts, deadline - 30)
        
        # Check if this is better
        final_energy = _log_energy(optimized_pts)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = optimized_pts.copy()
    
    # Phase 2: L-BFGS-B from the best solution found so far
    if time.time() < deadline - 15:
        lbfgsb_pts, lbfgsb_energy = _optimize_lbfgsb(best_pts, deadline - 5, maxiter=2500)
        final_energy = _log_energy(lbfgsb_pts)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = lbfgsb_pts.copy()
    
    # Phase 3: Additional L-BFGS-B from Fibonacci sphere
    if time.time() < deadline - 10:
        fib_pts = _fibonacci_sphere(N)
        fib_pts, fib_energy = _optimize_lbfgsb(fib_pts, deadline - 5, maxiter=1500)
        final_energy = _log_energy(fib_pts)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = fib_pts.copy()
    
    # Phase 4: High-precision polish on the best solution
    if time.time() < deadline - 3:
        polished_pts, polished_energy = _polish(best_pts, deadline - 1, max_iters=5000)
        final_energy = _log_energy(polished_pts)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = polished_pts.copy()
    
    # Phase 5: Second polish pass for additional refinement
    if time.time() < deadline - 1:
        polished_pts2, polished_energy2 = _polish(best_pts, deadline - 0.5, max_iters=3000)
        final_energy2 = _log_energy(polished_pts2)
        if final_energy2 < best_energy:
            best_energy = final_energy2
            best_pts = polished_pts2.copy()
    
    # Final cleanup: ensure points are on unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    min_dist = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci sphere
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
