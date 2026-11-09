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


def _log_energy(pts):
    # Sum over i<j of log(1/|xi-xj|) = -Sum log(|xi-xj|)
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # log(1/d) = -log(d)
    return -np.sum(np.log(d))


def _gradient(pts):
    # Gradient of E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2  ... wait let me derive
    # E = -sum_{i<j} log(|xi-xj|)
    # dE/dxi = -sum_{j!=i} (xi-xj)/|xi-xj| * (1/|xi-xj|) = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    # So gradient (for minimization) is: g_i = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    # To minimize, we move in direction -g_i projected onto tangent
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = xi - xj
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)  # avoid div by 0
    # Weight: 1/dist_sq for each pair
    W = 1.0 / dist_sq  # W[i,j] = 1/|xi-xj|^2
    np.fill_diagonal(W, 0.0)
    # grad_i = -sum_j W[i,j] * (xi - xj) = -W @ diff[i]... 
    # Actually: g_i = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    # In matrix form: g = -(W * diff) summed over j axis
    # diff[i,j,k] = xi_k - xj_k
    # sum over j: for each i,k: sum_j W[i,j]*diff[i,j,k]
    g = -np.sum(W[:, :, np.newaxis] * diff, axis=1)
    return g


def _project_tangent(pts, g):
    # Project gradient onto tangent space of sphere
    # For each point, remove component along the point itself
    g_proj = g - np.sum(g * pts, axis=1, keepdims=True) * pts
    return g_proj


def construct():
    deadline = time.time() + 110.0  # 10s safety margin before 120s limit
    pts = _fibonacci_points(N).astype(np.float64)
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # L-BFGS-like optimization with line search
    # Use scipy if available for better optimization
    try:
        from scipy.optimize import minimize
        # Define objective and gradient for scipy (minimize E)
        def obj(x):
            p = x.reshape(-1, 3)
            p = p / np.linalg.norm(p, axis=1, keepdims=True)
            return _log_energy(p)
        
        def grad(x):
            p = x.reshape(-1, 3)
            p = p / np.linalg.norm(p, axis=1, keepdims=True)
            g = _gradient(p)
            gp = _project_tangent(p, g)
            return gp.flatten()
        
        x0 = pts.flatten()
        
        # Try L-BFGS-B with multiple iterations
        result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        
        if result.success or result.fun < best_energy:
            p = result.x.reshape(-1, 3)
            p = p / np.linalg.norm(p, axis=1, keepdims=True)
            e = _log_energy(p)
            if e < best_energy:
                best_energy = e
                best_pts = p.copy()
        
        # Check time
        if time.time() > deadline - 20:
            return best_pts
        
        # Try a second phase with tighter tolerance
        result2 = minimize(obj, best_pts.flatten(), jac=grad, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
        
        if result2.fun < best_energy:
            p = result2.x.reshape(-1, 3)
            p = p / np.linalg.norm(p, axis=1, keepdims=True)
            e = _log_energy(p)
            if e < best_energy:
                best_energy = e
                best_pts = p.copy()
        
    except Exception:
        # Fallback: manual gradient descent
        lr = 0.01
        for iteration in range(5000):
            if time.time() > deadline:
                break
            g = _gradient(pts)
            gp = _project_tangent(pts, g)
            # Adaptive step size with backtracking
            step = lr
            for _ in range(20):
                new_pts = pts - step * gp
                new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_e = _log_energy(new_pts)
                if new_e < best_energy:
                    pts = new_pts
                    best_energy = new_e
                    best_pts = pts.copy()
                    break
                step *= 0.5
            # Decay learning rate
            if iteration % 100 == 0:
                lr = max(0.0001, lr * 0.9)
    
    # Final normalization
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
# EVOLVE-BLOCK-END
