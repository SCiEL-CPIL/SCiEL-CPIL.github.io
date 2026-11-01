"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(centers):
    """Calculate the score (radius) for a given set of centers."""
    if len(centers) != N:
        return 0.0
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Check finiteness
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Compute pairwise distances
    # d[i, j] is distance between i and j
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Min pairwise distance
    min_pair_dist = np.min(dists[np.triu_indices(N, k=1)])
    
    # Min distance to walls
    # Distance to each wall: 0.5 - |c|
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _potential(centers_flat):
    """
    Potential energy function to minimize.
    Minimizing this maximizes the minimum distance.
    """
    centers = centers_flat.reshape(-1, 3)
    # Pairwise repulsion
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # Avoid division by zero
    dists_sq[dists_sq < 1e-10] = 1e-10
    # Use a high power to emphasize close pairs
    # Sum over i < j
    iu = np.triu_indices(N, k=1)
    V_pair = np.sum(1.0 / (dists_sq[iu[0], iu[1]] ** 3)) # power 3 on squared dist -> 1/d^6
    
    # Wall repulsion
    # Distance to nearest wall for each point
    # w_i = min(0.5 - c_ix, 0.5 - c_iy, 0.5 - c_iz, 0.5 + c_ix, 0.5 + c_iy, 0.5 + c_iz)
    # Actually, since we bound to [-0.5, 0.5], 0.5 - |c| is the distance to the nearest wall face.
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dists = np.min(wall_dists, axis=1)
    min_wall_dists[min_wall_dists < 1e-10] = 1e-10
    V_wall = np.sum(1.0 / (min_wall_dists ** 6))
    
    return V_pair + V_wall

def _gradient_potential(centers_flat):
    """
    Analytical gradient of the potential.
    """
    centers = centers_flat.reshape(-1, 3)
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    # V_pair = sum_{i<j} d_ij^-6
    # dV/dc_i = sum_{j!=i} -6 * d_ij^-8 * (c_i - c_j) / d_ij ... wait
    # Let u = c_i - c_j. d = ||u||. V = d^-6.
    # dV/du = -6 d^-7 * (u/d) = -6 u d^-8.
    # So grad_i += sum_{j != i} -6 * (c_i - c_j) / ||c_i - c_j||^8
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    dists_sq[dists_sq < 1e-10] = 1e-10
    dists = np.sqrt(dists_sq)
    
    # coeff = -6 / dists^8
    coeff = -6.0 / (dists_sq ** 4)
    # Set diagonal to 0 to avoid self-interaction
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff_ij * diff_ij
    grad_pair = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    grad += grad_pair
    
    # Wall gradient
    # V_wall = sum_i (w_i)^-6 where w_i = min_k (0.5 - |c_ik|)
    # This is non-differentiable at the min. We approximate by taking the gradient of the min term.
    # For each point i, find the wall that is closest.
    # w_i = min(0.5 - c_ix, 0.5 + c_ix, 0.5 - c_iy, 0.5 + c_iy, 0.5 - c_iz, 0.5 + c_iz)
    # Let's compute distances to all 6 walls.
    # Walls: x=0.5, x=-0.5, y=0.5, y=-0.5, z=0.5, z=-0.5
    # d_x_pos = 0.5 - c_x
    # d_x_neg = 0.5 + c_x
    # ...
    
    d_x_pos = 0.5 - centers[:, 0]
    d_x_neg = 0.5 + centers[:, 0]
    d_y_pos = 0.5 - centers[:, 1]
    d_y_neg = 0.5 + centers[:, 1]
    d_z_pos = 0.5 - centers[:, 2]
    d_z_neg = 0.5 + centers[:, 2]
    
    # Clip to avoid 0
    d_x_pos = np.maximum(d_x_pos, 1e-10)
    d_x_neg = np.maximum(d_x_neg, 1e-10)
    d_y_pos = np.maximum(d_y_pos, 1e-10)
    d_y_neg = np.maximum(d_y_neg, 1e-10)
    d_z_pos = np.maximum(d_z_pos, 1e-10)
    d_z_neg = np.maximum(d_z_neg, 1e-10)
    
    # Find min for each point
    # We need to know WHICH wall is the min to compute the gradient sign.
    # If min is d_x_pos, grad_x += -6 * d_x_pos^-7 * (-1) = +6 * d_x_pos^-7 ?
    # V = w^-6. dV/dw = -6 w^-7.
    # w = 0.5 - c_x. dw/dc_x = -1.
    # dV/dc_x = -6 w^-7 * (-1) = 6 w^-7. (Pushes away from wall, i.e., towards -x? No, if c_x is near 0.5, w is small. Gradient should be negative (towards -x). Wait.
    # If c_x = 0.49, w=0.01. We want to decrease c_x to increase w. So grad should be negative.
    # Let's re-evaluate.
    # V = (0.5 - c_x)^-6.
    # dV/dc_x = -6 * (0.5 - c_x)^-7 * (-1) = 6 * (0.5 - c_x)^-7.
    # This is positive. So if we do gradient descent (c -= lr * grad), c_x will decrease. Correct.
    
    # However, we only take the gradient of the MINIMUM term.
    # For each point, determine which of the 6 distances is smallest.
    
    # Create a matrix of distances
    dists_matrix = np.stack([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg], axis=1)
    min_idx = np.argmin(dists_matrix, axis=1)
    min_dists = np.min(dists_matrix, axis=1)
    
    grad_wall = np.zeros_like(centers)
    
    # For each point, add gradient contribution from the closest wall
    for i in range(N):
        idx = min_idx[i]
        d = min_dists[i]
        # coeff = 6 * d^-7
        c = 6.0 / (d ** 7)
        
        if idx == 0: # x_pos wall (0.5 - c_x)
            grad_wall[i, 0] += c
        elif idx == 1: # x_neg wall (0.5 + c_x)
            # V = (0.5 + c_x)^-6. dV/dc_x = -6 (0.5+c_x)^-7.
            grad_wall[i, 0] -= c
        elif idx == 2: # y_pos
            grad_wall[i, 1] += c
        elif idx == 3: # y_neg
            grad_wall[i, 1] -= c
        elif idx == 4: # z_pos
            grad_wall[i, 2] += c
        elif idx == 5: # z_neg
            grad_wall[i, 2] -= c
            
    grad += grad_wall
    
    return grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    # Initializations
    # 1. Jittered Grid
    # 4x4x4 grid = 64 points. We need 50.
    # We can just take the first 50, or random 50.
    # Random 50 from a 4x4x4 grid is a good start.
    
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Generate initial config
        # Grid points in [-0.45, 0.45] to stay inside
        # Spacing 0.25
        ticks = np.array([-0.375, -0.125, 0.125, 0.375]) # Centered in 4 bins of 0.25?
        # Bin centers: -0.375, -0.125, 0.125, 0.375
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        
        # Shuffle and take 50
        idx = rng.permutation(len(grid))[:N]
        centers = grid[idx].copy()
        
        # Jitter
        centers += rng.normal(0, 0.01, centers.shape)
        
        # Clip to bounds
        centers = np.clip(centers, -0.49, 0.49)
        
        # Optimize
        x0 = centers.flatten()
        
        # Use L-BFGS-B
        # We can try different methods
        try:
            res = minimize(_potential, x0, method='L-BFGS-B', jac=_gradient_potential, 
                           bounds=[(-0.5, 0.5)] * (N * 3), options={'maxiter': 1000, 'ftol': 1e-10})
            
            opt_centers = res.x.reshape(-1, 3)
            # Clip again just in case
            opt_centers = np.clip(opt_centers, -0.5, 0.5)
            
            score = _score(opt_centers)
            
            if score > best_score:
                best_score = score
                best_centers = opt_centers.copy()
                
        except Exception as e:
            pass

    # If time remains, try a local refinement with Nelder-Mead on the best candidate?
    # Or just return.
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
