"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _log_energy_fsum(pts):
    """Compute logarithmic energy using math.fsum for precision."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(-math.log(d))
    return math.fsum(terms)


def _log_energy(pts):
    """Vectorized energy for optimization (fast but less precise)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    return -np.sum(np.log(d))


def _gradient(pts):
    """Gradient of E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij).
    
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    This is a repulsive force (points push each other apart).
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = xi - xj
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)  # avoid div by 0
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    # g_i = sum_{j!=i} W[i,j] * (xi - xj)
    g = np.sum(W[:, :, np.newaxis] * diff, axis=1)
    return g


def _project_tangent(pts, g):
    """Project gradient onto tangent space of sphere."""
    g_proj = g - np.sum(g * pts, axis=1, keepdims=True) * pts
    return g_proj


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _optimize_from_start(pts, deadline, max_iter=3000):
    """Run L-BFGS optimization from a starting configuration."""
    from scipy.optimize import minimize
    
    def obj(x):
        p = x.reshape(-1, 3)
        p = _normalize(p)
        return _log_energy(p)
    
    def grad(x):
        p = x.reshape(-1, 3)
        p = _normalize(p)
        g = _gradient(p)
        gp = _project_tangent(p, g)
        return gp.flatten()
    
    x0 = pts.flatten()
    
    # Phase 1: Coarse optimization
    result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
    
    if result.fun < float('inf'):
        p = _normalize(result.x.reshape(-1, 3))
        e = _log_energy(p)
        if e < float('inf'):
            return p, e
    
    return pts.copy(), _log_energy(pts)


def construct():
    deadline = time.time() + 140.0  # 10s safety margin before 150s limit
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = float('inf')
    
    # Multiple restarts with different initial configurations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            # Start with Fibonacci points
            pts = _fibonacci_points(N).astype(np.float64)
        elif restart == 1:
            # Start with uniform random points
            pts = rng.standard_normal((N, 3))
            pts = _normalize(pts)
        elif restart == 2:
            # Start with Fibonacci + perturbation
            pts = _fibonacci_points(N).astype(np.float64)
            pts = pts + rng.standard_normal((N, 3)) * 0.1
            pts = _normalize(pts)
        elif restart == 3:
            # Start with random points
            pts = rng.standard_normal((N, 3))
            pts = _normalize(pts)
        elif restart == 4:
            # Start with Fibonacci + larger perturbation
            pts = _fibonacci_points(N).astype(np.float64)
            pts = pts + rng.standard_normal((N, 3)) * 0.3
            pts = _normalize(pts)
        elif restart == 5:
            # Start with random + Fibonacci blend
            pts = 0.5 * _fibonacci_points(N).astype(np.float64) + 0.5 * rng.standard_normal((N, 3))
            pts = _normalize(pts)
        elif restart == 6:
            # Start with random
            pts = rng.standard_normal((N, 3))
            pts = _normalize(pts)
        else:
            # Start with Fibonacci + medium perturbation
            pts = _fibonacci_points(N).astype(np.float64)
            pts = pts + rng.standard_normal((N, 3)) * 0.2
            pts = _normalize(pts)
        
        # Optimize from this start
        try:
            opt_pts, opt_energy = _optimize_from_start(pts, deadline)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
        except Exception:
            continue
        
        # Check if we have time for a refinement phase
        if time.time() > deadline - 30:
            break
        
        # Refine the best solution found so far
        try:
            from scipy.optimize import minimize
            
            def obj(x):
                p = x.reshape(-1, 3)
                p = _normalize(p)
                return _log_energy(p)
            
            def grad(x):
                p = x.reshape(-1, 3)
                p = _normalize(p)
                g = _gradient(p)
                gp = _project_tangent(p, g)
                return gp.flatten()
            
            result = minimize(obj, best_pts.flatten(), jac=grad, method='L-BFGS-B',
                              options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
            
            if result.fun < best_energy:
                p = _normalize(result.x.reshape(-1, 3))
                e = _log_energy(p)
                if e < best_energy:
                    best_energy = e
                    best_pts = p.copy()
        except Exception:
            continue
    
    # Final validation and normalization
    if best_pts is not None:
        best_pts = _normalize(best_pts)
        # Verify no points are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < 1e-9:
                    # Perturb slightly
                    best_pts[j] += rng.standard_normal(3) * 1e-6
                    best_pts = _normalize(best_pts)
    else:
        # Fallback: return Fibonacci points
        best_pts = _fibonacci_points(N).astype(np.float64)
        best_pts = _normalize(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
