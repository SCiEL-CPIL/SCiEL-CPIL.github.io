import numpy as np
from math import fsum, log
from scipy.optimize import minimize

N = 150

def compute_energy_fsum(pts):
    """Energy with fsum for precision."""
    n = len(pts)
    iu = np.triu_indices(n, k=1)
    diff = pts[iu[0]] - pts[iu[1]]
    d = np.sqrt(np.sum(diff**2, axis=1))
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0/d)
    return fsum(terms.tolist())

def energy_wrapper(x):
    """Wrapper for scipy minimize."""
    pts = x.reshape(N, 3)
    # Project to sphere to handle any drift during optimization
    norms = np.linalg.norm(pts, axis=1)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms[:, np.newaxis]
    return compute_energy_fsum(pts)

def construct():
    np.random.seed(42)
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initialize Distance Matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(D, 1.0)
    
    # Compute initial energy
    iu = np.triu_indices(N, k=1)
    current_energy = fsum(np.log(1.0 / D[iu]).tolist())
    best_energy = current_energy
    best_pts = pts.copy()
    
    # Improved Simulated Annealing
    # We aim for ~60-90 seconds of SA, leaving time for polish.
    # Estimate iterations: 1 iter ~ 10-20 us? No, NumPy ops on N=150 are faster.
    # Let's try 500,000 iterations.
    n_iter = 500000
    T0 = 0.05
    Tf = 0.0005
    # Geometric cooling
    alpha = (Tf / T0) ** (1.0 / n_iter)
    T = T0
    
    # Pre-compute indices for "others" is not possible since k changes, 
    # but we can optimize the delta calculation.
    
    # To speed up SA, we will use a simpler move: 
    # Random rotation of a single point around a random axis.
    
    for it in range(n_iter):
        T = max(T * alpha, Tf)
        
        k = np.random.randint(N)
        
        # Generate new point via random rotation
        # Use a smaller step size as we cool down? 
        # Or keep it constant and let Metropolis handle it.
        # Let's use a step size that shrinks slightly with T.
        step_scale = 1.0 + 0.5 * (T / T0)
        angle = np.random.uniform(0.1, 0.5) * step_scale
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        
        # Rodrigues' rotation formula
        c = np.cos(angle)
        s = np.sin(angle)
        x, y, z = axis
        R = np.array([
            [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s,   c + y*y*(1-c),    y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s,   z*y*(1-c) + x*s,  c + z*z*(1-c)]
        ])
        
        old_pt = pts[k]
        new_pt = R @ old_pt
        new_pt /= np.linalg.norm(new_pt)
        
        # Compute delta energy efficiently
        # Only need distances from k to all j != k
        others_idx = np.arange(N)
        others_idx = others_idx[others_idx != k]
        
        d_old = D[k, others_idx]
        # Calculate new distances
        diff_new = pts[others_idx] - new_pt
        d_new = np.sqrt(np.sum(diff_new**2, axis=1))
        
        # Avoid log(0)
        d_old = np.maximum(d_old, 1e-15)
        d_new = np.maximum(d_new, 1e-15)
        
        # Delta E = sum(log(1/d_new)) - sum(log(1/d_old))
        # = -sum(log(d_new)) + sum(log(d_old))
        delta_E = np.sum(np.log(d_old)) - np.sum(np.log(d_new))
        
        # Metropolis criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            pts[k] = new_pt
            current_energy += delta_E
            
            # Update Distance Matrix
            D[k, others_idx] = d_new
            D[others_idx, k] = d_new
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        # Periodic recompute to avoid drift (less frequent)
        if it % 50000 == 0 and it > 0:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            D = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(D, 1.0)
            current_energy = fsum(np.log(1.0 / D[iu]).tolist())
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

    # Deterministic Polish: Vectorized Gradient Descent
    # E = sum_{i<j} log(1/||xi-xj||) = -sum_{i<j} log(||xi-xj||)
    # Gradient for point i:
    # dE/dxi = sum_{j != i} -d/dxi (log(1/||xi-xj||)) 
    # wait, E = sum log(1/d) = -sum log(d).
    # dE/dxi = - sum_{j != i} d/dxi log(||xi-xj||)
    # d/dxi log(||xi-xj||) = (xi - xj) / ||xi - xj||^2
    # So grad_i = - sum_{j != i} (xi - xj) / ||xi - xj||^2
    #             = sum_{j != i} (xj - xi) / ||xi - xj||^2
    
    pts = best_pts.copy()
    lr = 0.01
    min_lr = 1e-6
    max_iter = 500
    
    for p_iter in range(max_iter):
        # Vectorized Gradient Calculation
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dist_sq[i, j] = ||diff[i, j]||^2
        dist_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist_sq, 1.0) # Dummy value, we mask later
        
        # We want sum_{j != i} (xj - xi) / ||xi - xj||^2
        # Note: (xj - xi) = -diff[i, j]
        # So term is -diff[i, j] / dist_sq[i, j]
        # Weight matrix W where W[i, j] = 1 / dist_sq[i, j] for i != j, 0 for i == j
        W = 1.0 / dist_sq
        np.fill_diagonal(W, 0.0)
        
        # grad[i] = sum_j W[i, j] * (-diff[i, j])
        # grad = - sum_j W[i, j] * diff[i, j]
        # Using matrix multiplication:
        # grad = - (W @ diff) ??? 
        # Let's check dimensions.
        # W is (N, N), diff is (N, N, 3).
        # We want for each i, sum over j of W[i, j] * diff[i, j, k].
        # This is equivalent to einsum('ij,ijk->ik', W, diff)
        grad = np.einsum('ij,ijk->ik', W, diff)
        # Wait, the formula was:
        # grad_i = sum_{j != i} (xj - xi) / ||xi - xj||^2
        # (xj - xi) = - (xi - xj) = - diff[i, j]
        # So grad_i = sum_{j != i} -diff[i, j] * W[i, j]
        # So grad = - einsum('ij,ijk->ik', W, diff)
        
        grad = -grad
        
        # Project gradient to tangent space of sphere
        # The gradient on the sphere is the projection of the Euclidean gradient
        # onto the tangent plane: g_tan = g - (g . x) x
        dot_grad_x = np.sum(pts * grad, axis=1)
        grad_tangent = grad - dot_grad_x[:, np.newaxis] * pts
        
        # Update
        new_pts = pts - lr * grad_tangent
        
        # Normalize to sphere
        norms = np.linalg.norm(new_pts, axis=1)
        norms = np.maximum(norms, 1e-15)
        new_pts = new_pts / norms[:, np.newaxis]
        
        # Line search: check if energy decreased
        new_energy = compute_energy_fsum(new_pts)
        old_energy = compute_energy_fsum(pts)
        
        if new_energy < old_energy:
            pts = new_pts
            # Increase LR slightly if we are making good progress? Or keep constant.
            # Let's keep constant for stability, maybe decay slowly.
            lr = max(lr * 0.99, min_lr)
        else:
            lr *= 0.5
            if lr < min_lr:
                break

    final_energy = compute_energy_fsum(pts)
    if final_energy < best_energy:
        best_pts = pts
    
    return best_pts
