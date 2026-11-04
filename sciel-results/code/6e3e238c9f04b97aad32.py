import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_obj_grad(x, beta):
    """
    Objective and gradient for maximizing the minimum distance.
    x: (N*3,) array of centers.
    beta: inverse temperature for soft-min.
    """
    centers = x.reshape(N, 3)
    
    # 1. Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e-15) # Avoid division by zero
    
    v_pair = dists / 2.0
    # We only care about i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    v_pair_vals = v_pair[i_idx, j_idx]
    
    # 2. Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    w_vals = 0.5 - max_abs
    
    # Combine all values
    # v_pair_vals: (N*(N-1)/2,)
    # w_vals: (N,)
    
    # Compute exponentials
    # To avoid underflow, we can subtract the min, but soft-min is stable if we are careful.
    # However, for large beta, e^{-beta * v} can underflow for large v.
    # This is fine, it just means those terms don't contribute.
    
    e_pair = np.exp(-beta * v_pair_vals)
    e_wall = np.exp(-beta * w_vals)
    
    S = np.sum(e_pair) + np.sum(e_wall)
    
    if S == 0:
        # All distances are very large, gradient is 0
        return 0.0, np.zeros_like(x)
    
    p_pair = e_pair / S
    p_wall = e_wall / S
    
    # Objective value
    obj = (1.0 / beta) * np.log(S)
    
    # Gradient
    grad_centers = np.zeros_like(centers)
    
    # Pairwise contribution
    # grad_centers[i] += - sum_j p_pair_ij * 0.5 * (c_i - c_j) / dist_ij
    # grad_centers[j] += - sum_i p_pair_ij * 0.5 * (c_j - c_i) / dist_ij
    
    # We can compute the force for each pair and accumulate
    # vecs = (c_i - c_j) / dist_ij
    # force = p_pair * 0.5 * vecs
    # grad[i] -= force
    # grad[j] += force
    
    vecs = diff[i_idx, j_idx] / dists[i_idx, j_idx, np.newaxis]
    forces = (p_pair * 0.5)[:, np.newaxis] * vecs
    
    # Accumulate
    # np.add.at is slow, let's use a loop if N is small, or try to vectorize.
    # N=100, 4950 pairs. A loop in Python is ~5000 iterations, which is fast enough.
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        grad_centers[i] -= forces[k]
        grad_centers[j] += forces[k]
    
    # Wall contribution
    # grad_centers[i] += - p_wall[i] * grad(w_i)
    # grad(w_i) is -sign(c_ik) e_k where k is argmax |c_ik|
    
    # Find the axis with max abs value for each center
    abs_centers = np.abs(centers)
    max_axes = np.argmax(abs_centers, axis=1)
    
    # grad_wall[i] is a 3D vector
    # It is -sign(c_i, max_axis) in the max_axis component, 0 otherwise
    signs = np.sign(centers[np.arange(N), max_axes])
    
    # Create gradient vector for walls
    wall_grads = np.zeros_like(centers)
    wall_grads[np.arange(N), max_axes] = -signs
    
    # Accumulate
    grad_centers += (p_wall * 0.5)[:, np.newaxis] * wall_grads # Wait, w_i is distance to wall.
    # v_wall = 0.5 - max_abs.
    # d(v_wall)/d(c) = - d(max_abs)/d(c).
    # d(max_abs)/d(c_ik) = sign(c_ik) if k is argmax.
    # So d(v_wall)/d(c_ik) = -sign(c_ik).
    # My wall_grads calculation: -sign. This is correct.
    # The term in gradient sum is - p_wall * d(v_wall)/d(c).
    # So grad += - p_wall * (-sign * e_k) = p_wall * sign * e_k.
    # My code: grad_centers += p_wall * wall_grads.
    # wall_grads = -sign.
    # So grad += p_wall * (-sign).
    # This is WRONG.
    # Let's re-derive.
    # g = (1/beta) log(sum exp(-beta v)).
    # dg/dc = - sum p_v dv/dc.
    # For wall term v_w = 0.5 - max_abs.
    # dv_w/dc_ik = - sign(c_ik) (if k is argmax).
    # So contribution is - p_w * (- sign(c_ik) e_k) = p_w * sign(c_ik) e_k.
    # So grad_centers[i, max_axis] += p_wall[i] * sign(centers[i, max_axis]).
    
    # Let's fix the wall gradient accumulation
    grad_centers = np.zeros_like(centers) # Reset
    
    # Pairwise
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        grad_centers[i] -= forces[k]
        grad_centers[j] += forces[k]
        
    # Wall
    signs = np.sign(centers[np.arange(N), max_axes])
    # grad_wall_contrib = p_wall * signs
    # Add to the correct axis
    grad_centers[np.arange(N), max_axes] += p_wall * signs
    
    return obj, grad_centers.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Initialize with grid
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    init_centers = grid[:N].copy()
    
    # Add small noise to break symmetry
    init_centers += rng.normal(0, 0.01, size=(N, 3))
    init_centers = np.clip(init_centers, -0.49, 0.49)
    
    x0 = init_centers.flatten()
    
    # Annealing loop
    # Start with low beta, increase to high beta
    betas = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    
    current_x = x0
    best_r = _compute_radius(current_x.reshape(N, 3))
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        # Optimize with current beta
        try:
            res = minimize(
                lambda x: _soft_min_obj_grad(x, beta),
                current_x,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'disp': False}
            )
            new_x = res.x
            new_r = _compute_radius(new_x.reshape(N, 3))
            
            if new_r > best_r:
                best_r = new_r
                current_x = new_x
        except Exception:
            pass
            
    # Final polish with very high beta
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                lambda x: _soft_min_obj_grad(x, 5000),
                current_x,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'disp': False}
            )
            new_x = res.x
            new_r = _compute_radius(new_x.reshape(N, 3))
            if new_r > best_r:
                best_r = new_r
                current_x = new_x
        except Exception:
            pass
            
    final_centers = current_x.reshape(N, 3)
    final_centers = np.clip(final_centers, -0.5, 0.5)
    
    return final_centers
