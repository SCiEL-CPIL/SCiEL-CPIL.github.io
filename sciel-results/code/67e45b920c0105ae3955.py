"""Optimized logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on the sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    """
    # pts shape (N, 3)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero (diagonal)
    np.fill_diagonal(dist, 1.0)
    
    # Only upper triangle for i<j
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    
    # Energy = -sum(log(d))
    energy = -np.sum(np.log(d))
    
    # Gradient: d/dx_i sum_{j!=i} log(1/|x_i - x_j|)
    # = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # But we need to be careful: the objective is sum_{i<j} log(1/|x_i-x_j|)
    # Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    # Wait, let me reconsider. 
    # E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    # dE/dx_i = -sum_{j != i} d/dx_i log(|x_i - x_j|)
    # d/dx_i log(|x_i - x_j|) = (x_i - x_j) / |x_i - x_j|^2
    # So dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    #             = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    
    # Compute gradient vectorized
    # For each pair (i,j), contribution to grad[i] is (x_j - x_i)/d^2, to grad[j] is (x_i - x_j)/d^2
    grad = np.zeros_like(pts)
    
    # diff[i,j] = x_i - x_j
    # For pair (i,j) with i<j:
    # grad[i] += (x_j - x_i) / d_ij^2 = -diff[i,j] / d_ij^2
    # grad[j] += (x_i - x_j) / d_ij^2 = diff[i,j] / d_ij^2
    
    inv_d2 = 1.0 / (dist[iu] ** 2)
    diff_iu = diff[iu]  # shape (num_pairs, 3)
    
    # Scatter into gradient
    i_idx, j_idx = iu
    np.add.at(grad, i_idx, -diff_iu * inv_d2[:, np.newaxis])
    np.add.at(grad, j_idx, diff_iu * inv_d2[:, np.newaxis])
    
    return energy, grad


def _retract(pts):
    """Project points back onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _optimize_lbfgs(pts0, maxiter=500, gtol=1e-10):
    """Run L-BFGS-B optimization with retraction."""
    def fun_and_grad(x):
        pts = x.reshape(N, 3)
        # Compute energy and gradient on the sphere
        e, g = _energy_and_grad(pts)
        # Project gradient onto tangent space: g_tan = g - (g.x)x
        g_proj = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_proj.flatten()
    
    x0 = pts0.flatten()
    result = minimize(fun_and_grad, x0, method='L-BFGS-B',
                      jac=True, options={'maxiter': maxiter, 'gtol': gtol,
                                         'ftol': 1e-12, 'maxfun': 10000})
    pts_opt = _retract(result.x.reshape(N, 3))
    return pts_opt, result.fun


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # Phase 1: L-BFGS from the initial guess
    pts1, e1 = _optimize_lbfgs(pts, maxiter=300, gtol=1e-10)
    if e1 < best_energy:
        best_energy = e1
        best_pts = pts1
    
    # Check time
    if time.time() > deadline - 30:
        return best_pts
    
    # Phase 2: Multiple random restarts with L-BFGS
    np.random.seed(42)
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate random initial points
        pts_rand = np.random.randn(N, 3)
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        
        pts_opt, e_opt = _optimize_lbfgs(pts_rand, maxiter=200, gtol=1e-9)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 10:
        pts_final, e_final = _optimize_lbfgs(best_pts, maxiter=500, gtol=1e-12)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final
    
    # Ensure all points are on the unit sphere and non-zero
    best_pts = _retract(best_pts)
    
    return best_pts
