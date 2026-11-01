import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_min_distance(centers):
    """
    Computes the minimum distance between any two circles and the boundary.
    centers: (N, 2) array of centers.
    Returns the radius r such that circles of radius r are disjoint and inside [-0.5, 0.5]^2.
    """
    # Pairwise distances
    # d[i, j] = ||c_i - c_j||
    # We need min_{i<j} d[i,j] / 2
    # And min_i min_k (0.5 - |c_ik|)
    
    # Efficient distance calculation
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    # np.fill_diagonal(dists, np.inf)
    # min_pair_dist = np.min(dists)
    
    # Vectorized boundary distance
    # dist_to_bound = np.min(np.abs(centers), axis=1)
    # min_bound_dist = np.min(dist_to_bound)
    
    # r = min(min_pair_dist / 2, min_bound_dist)
    
    # Let's implement this efficiently
    # Use broadcasting
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Pairwise squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Boundary distances
    # Distance to closest edge for each point
    # x in [-0.5, 0.5], y in [-0.5, 0.5]
    # dist_x = min(x - (-0.5), 0.5 - x) = min(x+0.5, 0.5-x)
    # Since x is in [-0.5, 0.5], x+0.5 >= 0 and 0.5-x >= 0.
    # Actually, if a point is outside, this might be negative?
    # The problem states "inside the square". The optimization should keep them inside.
    # For safety, we clip or handle it.
    # dist_to_left = x + 0.5
    # dist_to_right = 0.5 - x
    # dist_to_bottom = y + 0.5
    # dist_to_top = 0.5 - y
    # min_dist_to_bound = min(min(dist_to_left), min(dist_to_right), min(dist_to_bottom), min(dist_to_top))
    
    min_bound_dist = np.min([
        np.min(x + 0.5),
        np.min(0.5 - x),
        np.min(y + 0.5),
        np.min(0.5 - y)
    ])
    
    return min(min_pair_dist / 2.0, min_bound_dist)

def soft_min_surrogate(centers, beta):
    """
    Computes a smooth approximation of the negative minimum distance.
    f(x) = - (1/beta) * log(sum(exp(-beta * d_k)))
    where d_k are all pairwise distances and boundary distances.
    Minimizing this maximizes the soft-min of distances.
    """
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Pairwise distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    np.fill_diagonal(dist_sq, np.inf)
    dists = np.sqrt(dist_sq)
    
    # Boundary distances
    # We need to include boundary distances in the sum.
    # d_bound_i = min(x_i+0.5, 0.5-x_i, y_i+0.5, 0.5-y_i)
    # This is also non-smooth. 
    # To keep it simple and effective, we can just use pairwise distances in the soft-min
    # and add a penalty for boundary violations or include boundary terms.
    # Including boundary terms in the log-sum-exp is tricky because they are mins of linear functions.
    # However, for the final polish, we can just use the exact non-smooth objective 
    # with a numerical gradient method, or assume the points stay well inside the boundary
    # so the pairwise distances are the binding constraint.
    # Given N=50, the boundary is likely not the bottleneck if we start with a good packing.
    # The best known packing has r=0.0713, and the square side is 1.
    # If points are near boundary, we should handle it.
    
    # Let's add boundary terms to the distance list.
    # d_k for k in pairs: dist_ij
    # d_k for k in points: min dist to boundary
    
    # For smoothness, we can approximate the boundary min distance as well.
    # But let's stick to pairwise first, as it's the main constraint.
    # We will add a large penalty if a point goes outside.
    
    # Flatten pairwise distances
    pair_dists = dists[np.triu_indices_from(dists, k=1)]
    
    # Boundary distances (approximated as just the distance to the nearest edge)
    # To make it smooth, we can use a soft-min of the 4 distances to edges for each point.
    # d_b_i = softmin(x_i+0.5, 0.5-x_i, y_i+0.5, 0.5-y_i, beta)
    # Then sum exp(-beta * d_b_i)
    
    # Let's compute soft-min for boundary distances per point
    # d_left = x + 0.5
    # d_right = 0.5 - x
    # d_bottom = y + 0.5
    # d_top = 0.5 - y
    
    # This is getting complex. Let's simplify:
    # The objective is to maximize r.
    # r <= d_ij / 2 for all i<j
    # r <= dist_to_bound_i for all i
    
    # We can define the loss as sum of violations for a target r? No, we want to maximize r.
    
    # Alternative: Use the exact min distance as objective, but use a gradient-free method like Nelder-Mead?
    # Nelder-Mead is slow for high dimensions (100D).
    
    # Let's use the soft-min of pairwise distances only, and add a penalty for boundary.
    # L = -softmin(pair_dists, beta) + penalty(boundary)
    
    log_sum_exp_pair = np.log(np.sum(np.exp(-beta * pair_dists)))
    soft_min_pair = -log_sum_exp_pair / beta
    
    # Boundary penalty: if any point is outside, add a large cost.
    # Also, we want to maximize distance to boundary.
    # Let's include boundary distances in the soft-min as well.
    # We need a vector of all relevant distances: pair_dists and bound_dists.
    
    # Compute bound_dists for each point (min of 4 linear terms)
    # To make it smooth, we can just take the min, which is non-smooth.
    # But for the optimizer, we can provide a smooth approximation.
    # Let's just use the exact min for the boundary in the penalty term?
    
    # Actually, a simpler and very effective approach for circle packing:
    # Minimize the sum of inverse distances? No.
    
    # Let's try this:
    # Objective: -min(dists)
    # We will use the soft-min of ALL distances (pairwise and boundary).
    
    # Compute boundary distances (exact min for each point)
    d_left = x + 0.5
    d_right = 0.5 - x
    d_bottom = y + 0.5
    d_top = 0.5 - y
    
    # Soft-min of these 4 for each point
    # softmin_i = -1/beta * log(sum(exp(-beta * d_edge)))
    # This is smooth.
    
    exp_d_left = np.exp(-beta * d_left)
    exp_d_right = np.exp(-beta * d_right)
    exp_d_bottom = np.exp(-beta * d_bottom)
    exp_d_top = np.exp(-beta * d_top)
    
    sum_exp_bound = exp_d_left + exp_d_right + exp_d_bottom + exp_d_top
    soft_min_bound_per_point = -np.log(sum_exp_bound) / beta
    
    # Now we have two sets of "distances":
    # 1. Pairwise distances (N(N-1)/2 values)
    # 2. Soft-min boundary distances per point (N values)
    # We want the global soft-min of all these.
    
    # We can combine them by summing their exponentials.
    # sum_total = sum(exp(-beta * pair_dists)) + sum(exp(-beta * soft_min_bound_per_point))
    # Wait, soft_min_bound_per_point is already a scalar per point.
    # The "distance" represented by soft_min_bound_per_point is roughly the min boundary dist.
    # So we can treat it as a distance value.
    
    log_sum_exp_total = np.log(np.sum(np.exp(-beta * pair_dists)) + np.sum(np.exp(-beta * soft_min_bound_per_point)))
    global_soft_min = -log_sum_exp_total / beta
    
    # We want to MAXIMIZE global_soft_min.
    # So we minimize -global_soft_min.
    
    # Add a penalty for points going far outside (should not happen with soft-min, but just in case)
    # The soft-min boundary term handles this. If a point is outside, d_edge becomes negative.
    # exp(-beta * negative) = exp(positive) -> large sum -> log is large -> soft_min is negative.
    # So minimizing -soft_min will try to make soft_min less negative (i.e., increase it).
    
    return -global_soft_min

def hexagonal_init(N):
    """
    Generate a hexagonal packing initialization.
    """
    # Approximate hexagonal grid
    # Side length of hexagon ~ 2*r
    # We want to fit N circles.
    # Start with a grid and then optimize.
    
    # Let's create a dense hexagonal grid and pick the best N points?
    # Or just generate a grid of roughly sqrt(N) x sqrt(N) points in hex pattern.
    
    # Simpler: Start with a square grid, then let SA/Polish do the work.
    # But hex is better.
    
    # Let's generate a hexagonal lattice of points.
    # Spacing a = 2*r_est
    # r_est ~ 0.07
    # a ~ 0.14
    
    # Let's just use a square grid with jitter for simplicity and robustness, 
    # as the polish phase is strong.
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Initial guess
    centers = hexagonal_init(N)
    
    # Simulated Annealing
    # We will run SA for a while to find a good local minimum.
    
    def sa_step(centers, T):
        # Pick a random point
        i = np.random.randint(N)
        # Move it
        delta = np.random.normal(0, 0.01, size=2)
        new_centers = centers.copy()
        new_centers[i] += delta
        
        # Ensure it stays in bounds (optional, soft-min handles it, but helps convergence)
        new_centers[i, 0] = np.clip(new_centers[i, 0], -0.5, 0.5)
        new_centers[i, 1] = np.clip(new_centers[i, 1], -0.5, 0.5)
        
        old_r = compute_min_distance(centers)
        new_r = compute_min_distance(new_centers)
        
        delta_r = new_r - old_r
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            return new_centers, True
        else:
            return centers, False
    
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.99
    
    best_centers = centers
    best_r = compute_min_distance(centers)
    
    # SA Loop
    while time.time() < deadline - 20:  # Leave 20s for polish
        centers, _ = sa_step(centers, T)
        r = compute_min_distance(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart with higher T? Or just stop SA.
            break
            
    # Polish Phase
    # Use scipy minimize on the soft-min surrogate.
    # We will do a few iterations with increasing beta.
    
    current_centers = best_centers
    betas = [10, 50, 100, 500]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        # Objective function for minimize
        def obj(x):
            c = x.reshape(N, 2)
            return soft_min_surrogate(c, beta)
        
        # Bounds: [-0.5, 0.5] for all coordinates
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = minimize(obj, current_centers.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success:
                new_centers = res.x.reshape(N, 2)
                new_r = compute_min_distance(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers
                    current_centers = new_centers
        except Exception:
            pass
            
    # Final check
    final_r = compute_min_distance(best_centers)
    if final_r > best_r:
        best_r = final_r
        # No need to update best_centers if it's the same
    
    return best_centers

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_min_distance(centers):
    """
    Computes the minimum distance between any two circles and the boundary.
    centers: (N, 2) array of centers.
    Returns the radius r such that circles of radius r are disjoint and inside [-0.5, 0.5]^2.
    """
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Pairwise squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Boundary distances
    min_bound_dist = np.min([
        np.min(x + 0.5),
        np.min(0.5 - x),
        np.min(y + 0.5),
        np.min(0.5 - y)
    ])
    
    return min(min_pair_dist / 2.0, min_bound_dist)

def soft_min_surrogate(centers, beta):
    """
    Computes a smooth approximation of the negative minimum distance.
    """
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Pairwise distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    np.fill_diagonal(dist_sq, np.inf)
    dists = np.sqrt(dist_sq)
    
    # Flatten pairwise distances
    pair_dists = dists[np.triu_indices_from(dists, k=1)]
    
    # Boundary distances (soft-min of 4 edges for each point)
    d_left = x + 0.5
    d_right = 0.5 - x
    d_bottom = y + 0.5
    d_top = 0.5 - y
    
    exp_d_left = np.exp(-beta * d_left)
    exp_d_right = np.exp(-beta * d_right)
    exp_d_bottom = np.exp(-beta * d_bottom)
    exp_d_top = np.exp(-beta * d_top)
    
    sum_exp_bound = exp_d_left + exp_d_right + exp_d_bottom + exp_d_top
    soft_min_bound_per_point = -np.log(sum_exp_bound) / beta
    
    # Global soft-min
    log_sum_exp_total = np.log(np.sum(np.exp(-beta * pair_dists)) + np.sum(np.exp(-beta * soft_min_bound_per_point)))
    global_soft_min = -log_sum_exp_total / beta
    
    return -global_soft_min

def hexagonal_init(N):
    """
    Generate a hexagonal packing initialization.
    """
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Initial guess
    centers = hexagonal_init(N)
    
    # Simulated Annealing
    def sa_step(centers, T):
        i = np.random.randint(N)
        delta = np.random.normal(0, 0.01, size=2)
        new_centers = centers.copy()
        new_centers[i] += delta
        
        new_centers[i, 0] = np.clip(new_centers[i, 0], -0.5, 0.5)
        new_centers[i, 1] = np.clip(new_centers[i, 1], -0.5, 0.5)
        
        old_r = compute_min_distance(centers)
        new_r = compute_min_distance(new_centers)
        
        delta_r = new_r - old_r
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            return new_centers, True
        else:
            return centers, False
    
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.99
    
    best_centers = centers
    best_r = compute_min_distance(centers)
    
    # SA Loop
    while time.time() < deadline - 20:  # Leave 20s for polish
        centers, _ = sa_step(centers, T)
        r = compute_min_distance(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            break
            
    # Polish Phase
    current_centers = best_centers
    betas = [10, 50, 100, 500]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        def obj(x):
            c = x.reshape(N, 2)
            return soft_min_surrogate(c, beta)
        
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = minimize(obj, current_centers.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success:
                new_centers = res.x.reshape(N, 2)
                new_r = compute_min_distance(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers
                    current_centers = new_centers
        except Exception:
            pass
            
    return best_centers
# EVOLVE-BLOCK-END
