"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the common radius for given centers."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    wall_dists = np.minimum(0.5 - centers, centers + 0.5)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist) / 2.0

def _surrogate_objective_and_grad(x, beta=100.0):
    """
    Objective: Minimize -log(sum(exp(-beta * dist_i)))
    This approximates -min(dist_i) / beta for large beta.
    x is flattened centers of shape (N, 3) -> (3N,).
    """
    centers = x.reshape(-1, 3)
    n = centers.shape[0]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    # For each center, min distance to walls
    # walls are at -0.5, 0.5
    # d_w = min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5, 0.5 - z, z + 0.5)
    # We can compute these for each axis
    coords = centers # (N, 3)
    # dist to -0.5: coords + 0.5
    # dist to 0.5: 0.5 - coords
    d_walls = np.minimum(coords + 0.5, 0.5 - coords) # (N, 3)
    min_wall_dist_per_sphere = np.min(d_walls, axis=1) # (N,)
    
    # Combine all distances into a single vector for LSE
    # We need min of all pairwise dists and all wall dists.
    # LSE of min is -1/beta * log(sum(exp(-beta * d)))
    
    # To avoid overflow/underflow, we can just work with the log-sum-exp trick or just sum exps if beta is moderate.
    # With beta=100, exp(-100*d) can underflow for d>0.1. That's fine, it just becomes 0.
    # But if d is very small, exp(-100*d) is large.
    
    # Let's collect all distances
    # Pairwise: upper triangle
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    
    # Wall dists
    wall_dists_flat = min_wall_dist_per_sphere
    
    all_dists = np.concatenate([pair_dists, wall_dists_flat])
    
    # Compute LSE
    # obj = -1/beta * log(sum(exp(-beta * all_dists)))
    # grad = 1/beta * sum(exp(-beta * d) * (-beta * grad_d)) / sum(exp(-beta * d))
    #      = - sum(exp(-beta * d) * grad_d) / sum(exp(-beta * d))
    
    # To be numerically stable, let's find max d (min -beta*d)
    # Actually, we want to minimize the objective.
    # The term with the smallest distance will dominate the sum (largest exp).
    
    # Let's compute weights
    # If any distance is 0, exp(0)=1, others < 1.
    # If distances are small, exp is close to 1.
    
    # Simple implementation:
    # Weights w_i = exp(-beta * d_i)
    # To prevent underflow, we can subtract the max exponent (which is -beta * min_d)
    # max_exp = -beta * min(all_dists)
    # w_i = exp(-beta * d_i - max_exp)
    # sum_w = sum(w_i)
    # obj = -1/beta * (log(sum_w) + max_exp)
    #      = -1/beta * (log(sum_w) - beta * min_d)
    #      = -log(sum_w)/beta + min_d
    # Wait, the standard LSE approximation for min is:
    # min(x) ≈ -1/beta * log(sum(exp(-beta*x)))
    # Let's stick to the direct formula but handle stability.
    
    min_d = np.min(all_dists)
    # Shift to avoid underflow
    shifted = -beta * (all_dists - min_d)
    # Note: shifted <= 0. max is 0.
    weights = np.exp(shifted)
    sum_w = np.sum(weights)
    
    # Objective value
    obj = -1.0 / beta * (np.log(sum_w) - beta * min_d)
    # Actually, if we shift:
    # log(sum(exp(-beta*d))) = log(sum(exp(-beta*(d-min_d)) * exp(-beta*min_d)))
    #                         = log(sum(exp(-beta*(d-min_d)))) - beta*min_d
    # So -1/beta * log(...) = -1/beta * log(sum(exp(...))) + min_d
    # This is numerically stable.
    
    # Gradient
    # d(obj)/d(x) = 1/beta * sum( w_i * (-beta * d(d_i)/dx) ) / sum_w
    #             = - sum( w_i * d(d_i)/dx ) / sum_w
    #             = - sum( (w_i / sum_w) * d(d_i)/dx )
    
    grad = np.zeros_like(x)
    
    # Gradient of pairwise distances
    # d(||c_i - c_j||)/dc_i = (c_i - c_j) / ||c_i - c_j||
    # d(||c_i - c_j||)/dc_j = (c_j - c_i) / ||c_i - c_j||
    
    # We have pair_dists corresponding to iu indices.
    # weights_pair = weights[:len(pair_dists)]
    # weights_wall = weights[len(pair_dists):]
    
    w_pair = weights[:len(pair_dists)]
    w_wall = weights[len(pair_dists):]
    
    norm_w_pair = w_pair / sum_w
    norm_w_wall = w_wall / sum_w
    
    # Accumulate gradient for pairs
    # For each pair (i, j), contribution to grad[i] is - norm_w * (c_i - c_j) / d
    # Contribution to grad[j] is - norm_w * (c_j - c_i) / d
    
    # Vectorized update
    # diff is (N, N, 3). We only need upper triangle.
    # Let's use the full matrix and mask, or just loop if N is small? N=50, 1225 pairs. Loop is fine but vectorized is better.
    
    # Let's create a weighted difference matrix
    # G = sum_k w_k * (c_i - c_j) / d_k for pairs
    # This is tricky to vectorize directly into the gradient array without loops or advanced indexing.
    
    # Let's use advanced indexing
    i_idx, j_idx = iu
    # diff[i_idx, j_idx] is (M, 3) where M is num pairs
    pair_diffs = diff[i_idx, j_idx] # (M, 3)
    pair_dists_safe = np.where(pair_dists > 1e-12, pair_dists, 1e-12)
    
    # Unit vectors
    unit_vecs = pair_diffs / pair_dists_safe[:, np.newaxis] # (M, 3)
    
    # Weighted unit vectors
    # grad[i] -= norm_w * unit_vec
    # grad[j] += norm_w * unit_vec
    
    # Use np.add.at for accumulation
    # grad is flattened, so we need to map (i, axis) to flat index
    # i_flat = i_idx * 3 + axis
    
    # Let's reshape grad to (N, 3) for easier manipulation
    grad_3d = grad.reshape(-1, 3)
    
    # Accumulate for i
    # grad_3d[i_idx] -= norm_w_pair[:, np.newaxis] * unit_vecs
    np.add.at(grad_3d, i_idx, -norm_w_pair[:, np.newaxis] * unit_vecs)
    # Accumulate for j
    # grad_3d[j_idx] += norm_w_pair[:, np.newaxis] * unit_vecs
    np.add.at(grad_3d, j_idx, norm_w_pair[:, np.newaxis] * unit_vecs)
    
    # Gradient for walls
    # d(min(0.5-x, x+0.5, ...))/dx
    # If x < 0, min is x+0.5 (dist to -0.5), grad is 1
    # If x > 0, min is 0.5-x (dist to 0.5), grad is -1
    # If x == 0, non-differentiable, pick 0 or random?
    
    # d_walls was computed as min(coords+0.5, 0.5-coords)
    # Let's determine which wall was closer for each sphere and axis
    # d_neg = coords + 0.5
    # d_pos = 0.5 - coords
    # If d_neg < d_pos, grad is 1. Else -1.
    
    # min_wall_dist_per_sphere is min over axes.
    # We need to know which axis and which wall contributed to the min for each sphere.
    
    # Let's recompute to get indices
    # d_walls is (N, 3)
    # d_neg is (N, 3), d_pos is (N, 3)
    # d_walls = min(d_neg, d_pos)
    # min_axis = argmin(d_walls, axis=1)
    
    # For each sphere k:
    # axis = min_axis[k]
    # if d_neg[k, axis] < d_pos[k, axis]: grad[k, axis] += -1 * norm_w_wall[k]
    # else: grad[k, axis] += 1 * norm_w_wall[k]
    
    # Note: The objective is -sum(w * d). So grad = -sum(w * grad_d).
    # grad_d for wall dist:
    # d(x+0.5)/dx = 1
    # d(0.5-x)/dx = -1
    
    d_neg = centers + 0.5
    d_pos = 0.5 - centers
    
    # Determine which wall is closer
    # closer_is_neg = d_neg < d_pos
    # We need to find the global min per sphere
    # min_wall_dist_per_sphere is already computed.
    
    # Let's find the (sphere, axis, wall) that gives the min distance for each sphere
    # Actually, we just need the gradient of the min distance for each sphere.
    
    # For sphere k, min_d_k = min_{a, w} d_{k,a,w}
    # grad_k = - sum_{a,w} (w_{k,a,w} / sum_w_total) * grad_{d_{k,a,w}}
    # But our weights are assigned to the min distance per sphere?
    # No, in the LSE, we treated each sphere's min wall distance as a single variable in the global min.
    # So w_wall[k] is the weight for the min wall distance of sphere k.
    # The gradient of min_wall_dist(k) is the gradient of the specific wall distance that achieved the min.
    
    # So for each sphere k:
    # axis = argmin(d_walls[k])
    # if d_neg[k, axis] <= d_pos[k, axis]:
    #     grad_3d[k, axis] += -norm_w_wall[k] * 1  (because obj grad is -w * grad_d)
    # else:
    #     grad_3d[k, axis] += -norm_w_wall[k] * (-1) = +norm_w_wall[k]
    
    min_axes = np.argmin(d_walls, axis=1)
    
    # Create indices for advanced indexing
    sphere_indices = np.arange(N)
    
    # Check which wall
    is_neg = d_neg[sphere_indices, min_axes] <= d_pos[sphere_indices, min_axes]
    
    # Grad contribution: -w * grad_d
    # If neg: grad_d = 1 -> contrib = -w
    # If pos: grad_d = -1 -> contrib = +w
    
    # We add to grad_3d[sphere, axis]
    # We can do this with np.add.at
    
    # Create arrays for add.at
    # Indices: (sphere_indices, min_axes)
    # Values: 
    # val = np.where(is_neg, -norm_w_wall, norm_w_wall)
    
    vals = np.where(is_neg, -norm_w_wall, norm_w_wall)
    
    np.add.at(grad_3d, (sphere_indices, min_axes), vals)
    
    grad = grad_3d.flatten()
    
    return obj, grad

def _optimize_lbgfgs(centers, max_time=10.0):
    """
    Optimize centers using L-BFGS-B on the LSE surrogate.
    """
    start_time = time.time()
    x0 = centers.flatten()
    
    # Bounds: centers must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    # We want to maximize radius, which is minimizing the surrogate.
    # The surrogate approximates -radius.
    
    try:
        res = minimize(
            _surrogate_objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 200, 'disp': False}
        )
        if res.success or res.x is not None:
            new_centers = res.x.reshape(-1, 3)
            # Clamp just in case
            new_centers = np.clip(new_centers, -0.5, 0.5)
            return new_centers
    except Exception:
        pass
    
    return centers

def _grid_init():
    """
    Generate initial centers using a fine grid and greedy selection.
    """
    # Target radius is ~0.136. 
    # Grid spacing should be smaller than 2*r to allow flexibility, but large enough to have enough points.
    # Let's use a grid of 10x10x10 = 1000 points.
    # Spacing = 1/9 ~ 0.11.
    # 2*r ~ 0.27.
    # So points are close.
    
    # Actually, to get a good start, we want to pick 50 points from a grid such that they are as far apart as possible.
    # This is the greedy independent set approach.
    
    # Let's use a grid of size 8x8x8 = 512 points.
    # Spacing = 1/7 ~ 0.14.
    # This is close to the optimal radius.
    
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Greedy selection:
    # 1. Start with empty set.
    # 2. Pick the point that maximizes the minimum distance to already picked points.
    #    Initially, any point. To be symmetric, maybe center?
    #    Or random?
    #    Let's start with the center of the cube (0,0,0) if it's in the grid.
    #    In 8x8x8, 0 is not in the grid (ticks are 0.5/8, 1.5/8, ...).
    #    Closest to 0 are 0.5/8 = 0.0625.
    
    # Let's just pick a random point to start, or the one closest to center.
    dists_to_center = np.sum(grid**2, axis=1)
    start_idx = np.argmin(dists_to_center)
    
    selected = [grid[start_idx]]
    remaining = list(range(len(grid)))
    remaining.remove(start_idx)
    
    # Precompute pairwise distances? 512x512 is small.
    diff = grid[:, np.newaxis, :] - grid[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # For each remaining point, track min dist to selected
    # We can just recompute each time, N=50 is small.
    
    for _ in range(N - 1):
        if not remaining:
            break
        
        # Compute min distance from each remaining point to selected points
        # selected is list of indices
        sel_indices = np.array(selected)
        rem_indices = np.array(remaining)
        
        # Distances between remaining and selected
        # d[rem, sel]
        d_rem_sel = dists[np.ix_(rem_indices, sel_indices)]
        min_d_to_sel = np.min(d_rem_sel, axis=1)
        
        # Pick the one with max min_d_to_sel
        best_rem_idx = np.argmax(min_d_to_sel)
        best_global_idx = rem_indices[best_rem_idx]
        
        selected.append(best_global_idx)
        remaining.remove(best_global_idx)
        
    centers = np.array([grid[i] for i in selected])
    
    return centers

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # 1. Grid Init
    centers = _grid_init()
    
    # 2. Refine with L-BFGS-B
    # We have ~140s left.
    # L-BFGS-B is fast. We can do multiple restarts or just one long run.
    # Since the grid init is deterministic, one run might be enough, but random perturbations can help escape local minima.
    # Let's do a few random restarts from the grid init with small noise.
    
    best_centers = centers
    best_radius = _compute_radius(centers)
    
    np.random.seed(42)
    
    # Try a few variations
    for i in range(5):
        if time.time() - start_time > deadline - 10:
            break
            
        # Add small noise to grid init
        if i == 0:
            c = centers
        else:
            c = centers + np.random.randn(N, 3) * 0.05
            c = np.clip(c, -0.5, 0.5)
            
        # Optimize
        c_opt = _optimize_lbgfgs(c, max_time=20.0)
        
        r = _compute_radius(c_opt)
        if r > best_radius:
            best_radius = r
            best_centers = c_opt
            
    return best_centers
# EVOLVE-BLOCK-END
