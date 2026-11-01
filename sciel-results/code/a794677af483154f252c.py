import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _energy_and_grad(x_flat, mu=1000.0):
    """
    Compute energy and gradient for L-BFGS-B.
    x_flat: (N*3,) array
    """
    x = x_flat.reshape(N, 3)
    # Compute pairwise distances
    # dist^2 = |xi - xj|^2
    # We need sum over i<j log(1/d_ij)
    # grad_i = sum_{j!=i} (xj - xi) / |xi - xj|^2
    
    # Efficient calculation using broadcasting
    # diff[i, j] = xi - xj
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Mask diagonal to avoid division by zero
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/sqrt(dist_sq)) = -0.5 * sum_{i!=j} log(sqrt(dist_sq)) = -0.25 * sum_{i!=j} log(dist_sq)
    # Wait, log(1/d) = -log(d).
    # E = sum_{i<j} -log(|xi-xj|) = -0.5 * sum_{i!=j} log(|xi-xj|) = -0.25 * sum_{i!=j} log(dist_sq)
    
    # For gradient, we need the full sum.
    # grad_i = sum_{j!=i} (xj - xi) / dist_sq[i,j]
    # Note: diff[i,j] = xi - xj, so xj - xi = -diff[i,j]
    # grad_i = - sum_{j!=i} diff[i,j] / dist_sq[i,j]
    
    # Create a mask for off-diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Calculate gradient term
    # inv_dist_sq = 1 / dist_sq
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_log[i] = - sum_j diff[i,j] * inv_dist_sq[i,j]
    # diff[i,j] is (N, N, 3). inv_dist_sq is (N, N).
    # We need to multiply each vector by the scalar and sum over j.
    # grad_log = - np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    grad_log = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Penalty term
    # P = mu * sum (|xi|^2 - 1)^2
    # grad_P_i = 2 * mu * (|xi|^2 - 1) * xi
    norms_sq = np.sum(x**2, axis=1)
    grad_pen = 2.0 * mu * (norms_sq - 1.0)[:, np.newaxis] * x
    
    # Total gradient
    grad = grad_log + grad_pen
    
    # Energy value
    # E_log = -0.5 * sum_{i!=j} log(|xi-xj|)
    # We can compute this from dist_sq
    # log_dist = 0.5 * log(dist_sq)
    # E_log = -0.5 * sum_{i!=j} log_dist
    log_dist_sq = np.log(dist_sq)
    np.fill_diagonal(log_dist_sq, 0.0) # Doesn't matter as it's masked in sum if we are careful, but let's be safe
    # Actually, we just sum the off-diagonals
    E_log = -0.5 * np.sum(log_dist_sq[mask])
    
    E_pen = mu * np.sum((norms_sq - 1.0)**2)
    
    energy = E_log + E_pen
    
    return energy, grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x0_flat = x0.flatten()
    
    # Run L-BFGS-B
    # We use a callback to stop if time runs out
    state = {'x': x0_flat, 'energy': _energy_and_grad(x0_flat)[0]}
    
    def callback(xk):
        if time.time() > deadline:
            state['x'] = xk
            # We can't easily break minimize from callback, but we can set a flag
            # However, scipy's minimize doesn't support early exit via callback directly in a clean way for all versions.
            # Instead, we can just let it run, but we must ensure it finishes.
            # L-BFGS-B is fast. Let's just run it for a fixed number of iterations or until convergence.
            # Or, we can split the optimization into chunks.
            pass
    
    # Actually, a better way to handle time limits is to run minimize in a loop or just trust that L-BFGS-B converges quickly.
    # For N=100, L-BFGS-B usually converges in < 1 second.
    # Let's run it with a high maxiter and let it converge.
    
    result = minimize(
        _energy_and_grad, 
        x0_flat, 
        method='L-BFGS-B',
        jac=True, # _energy_and_grad returns (f, g)
        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    x_opt = result.x.reshape(N, 3)
    
    # Normalize to ensure exact unit sphere
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    return x_opt
