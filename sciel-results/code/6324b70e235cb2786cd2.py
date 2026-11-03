import numpy as np
from scipy.optimize import minimize
import time

N = 50
SIDE = 1.0
HALF = 0.5

def compute_radius(centers):
    """
    Compute the maximum radius for given centers.
    Centers are in [-0.5, 0.5]^2.
    """
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Boundary distance
    # min distance to edge
    dist_x = np.minimum(centers[:, 0] + HALF, HALF - centers[:, 0])
    dist_y = np.minimum(centers[:, 1] + HALF, HALF - centers[:, 1])
    boundary_min = np.min(np.minimum(dist_x, dist_y))
    
    # Pairwise distances
    # Use broadcasting for all pairs, then take min of upper triangle
    # N=50, 50x50 matrix is small
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, boundary_min)
    return r

def objective(x, sigma=0.02):
    """
    Smooth surrogate objective to minimize.
    We want to maximize the minimum distance.
    Minimizing sum of exp(-d^2/sigma^2) pushes points apart.
    """
    centers = x.reshape(N, 2)
    
    # Boundary penalty
    # dist to edge
    dx = np.minimum(centers[:, 0] + HALF, HALF - centers[:, 0])
    dy = np.minimum(centers[:, 1] + HALF, HALF - centers[:, 1])
    b = np.minimum(dx, dy)
    boundary_term = np.sum(np.exp(-b**2 / (sigma**2)))
    
    # Pairwise penalty
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    # Exclude diagonal
    dists_sq = dists_sq + np.eye(N) * 1e10
    # Only upper triangle to avoid double counting? Sum is symmetric, factor doesn't matter for minimization
    pair_term = np.sum(np.exp(-dists_sq / (sigma**2)))
    
    return pair_term + boundary_term

def gradient(x, sigma=0.02):
    """
    Gradient of the objective.
    """
    centers = x.reshape(N, 2)
    grad = np.zeros_like(centers)
    
    # Boundary gradient
    # d/dx exp(-b^2/s^2) = exp(...) * (-2b/s^2) * db/dx
    dx = np.minimum(centers[:, 0] + HALF, HALF - centers[:, 0])
    dy = np.minimum(centers[:, 1] + HALF, HALF - centers[:, 1])
    b = np.minimum(dx, dy)
    
    # db/dx
    # if x < 0, b = x + 0.5, db/dx = 1
    # if x > 0, b = 0.5 - x, db/dx = -1
    # if y dominates, db/dx = 0
    mask_x = (dx <= dy)
    db_dx = np.where(centers[:, 0] < 0, 1.0, -1.0)
    db_dx[~mask_x] = 0.0
    
    mask_y = (dy <= dx)
    db_dy = np.where(centers[:, 1] < 0, 1.0, -1.0)
    db_dy[~mask_y] = 0.0
    
    exp_b = np.exp(-b**2 / (sigma**2))
    grad[:, 0] += exp_b * (-2 * b / (sigma**2)) * db_dx
    grad[:, 1] += exp_b * (-2 * b / (sigma**2)) * db_dy
    
    # Pairwise gradient
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    # Exclude self
    np.fill_diagonal(dists_sq, 1e10)
    
    exp_d = np.exp(-dists_sq / (sigma**2))
    # d/dx_i exp(-d_ij^2/s^2) = exp(...) * (-2/s^2) * (x_i - x_j)
    # Sum over j
    # grad_i = sum_j exp_d_ij * (-2/s^2) * (x_i - x_j)
    
    factor = -2.0 / (sigma**2)
    # exp_d is (N, N)
    # diff is (N, N, 2)
    # We want sum over j for each i.
    # term = exp_d[:, :, np.newaxis] * diff
    # sum_j term[i, j, :] = sum_j exp_d[i, j] * (x_i - x_j)
    
    # Note: exp_d[i,i] is effectively 0 due to 1e10 in dists_sq
    # But we filled diagonal with 1e10, so exp is 0.
    
    weighted_diff = exp_d[:, :, np.newaxis] * diff
    grad_pair = np.sum(weighted_diff, axis=1)
    grad += grad_pair * factor
    
    return grad.flatten()

def optimize_config(centers, time_limit=10):
    """
    Optimize a configuration using L-BFGS-B.
    """
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-HALF, HALF)] * (2 * N)
    
    best_x = x0.copy()
    best_r = compute_radius(centers)
    
    # Try a few different sigmas
    sigmas = [0.05, 0.02, 0.01]
    
    for sigma in sigmas:
        if time.time() - start_time > time_limit:
            break
            
        res = minimize(
            objective, 
            x0, 
            jac=gradient, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50, 'maxfun': 1000, 'disp': False}
        )
        
        x_opt = res.x
        centers_opt = x_opt.reshape(N, 2)
        r_opt = compute_radius(centers_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_x = x_opt
        
        x0 = x_opt # Continue from previous result
    
    return best_x.reshape(N, 2), best_r

def generate_grid_config(m, n, N_points):
    """
    Generate a grid of m x n points, take first N_points.
    """
    xs = np.linspace(-HALF, HALF, m)
    ys = np.linspace(-HALF, HALF, n)
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid[:N_points]

def generate_hex_config(N_points):
    """
    Generate points from a hexagonal lattice clipped to the square.
    """
    # Hex lattice spacing
    # We want roughly N points.
    # Area per point in hex packing = sqrt(3)/2 * a^2
    # Total area = 1.
    # a = sqrt(2 / (N * sqrt(3)))
    a = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    points = []
    # Generate enough points to cover [-0.5, 0.5]^2
    # Range
    R = 1.0
    # Hex basis vectors
    v1 = np.array([a, 0])
    v2 = np.array([a/2, a*np.sqrt(3)/2])
    
    # Loop over lattice indices
    max_idx = int(R / a) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            p = i * v1 + j * v2
            if -HALF <= p[0] <= HALF and -HALF <= p[1] <= HALF:
                points.append(p)
    
    points = np.array(points)
    # If we have more than N, pick N random ones? Or nearest to grid?
    # Just take the first N after shuffling?
    # Better: take the ones that are most "central" or just random subset.
    # For initialization, any subset is fine, optimizer will fix.
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Pad with random points
        pad = np.random.uniform(-HALF, HALF, (N - len(points), 2))
        points = np.vstack([points, pad])
        
    return points

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    best_centers = None
    best_r = 0.0
    
    # Initial configurations
    configs = []
    
    # 1. 7x7 Grid + 1 random
    g = generate_grid_config(7, 7, 49)
    extra = np.random.uniform(-HALF, HALF, (1, 2))
    c1 = np.vstack([g, extra])
    configs.append(c1)
    
    # 2. 7x7 Grid + 1 in center of a hole
    g = generate_grid_config(7, 7, 49)
    # Hole at (0, 0) is occupied. Hole at (1/7, 1/7)?
    # Grid points are at (i-3)/7.
    # Hole centers are at (i-3+0.5)/7 = (i-2.5)/7.
    # e.g. (0.5-3)/7 = -2.5/7.
    hole = np.array([[-2.5/7, -2.5/7]])
    c2 = np.vstack([g, hole])
    configs.append(c2)
    
    # 3. Hexagonal
    c3 = generate_hex_config(N)
    configs.append(c3)
    
    # 4. Random
    c4 = np.random.uniform(-HALF, HALF, (N, 2))
    configs.append(c4)
    
    # 5. 6x8 Grid + 2 random
    g68 = generate_grid_config(6, 8, 48)
    extra2 = np.random.uniform(-HALF, HALF, (2, 2))
    c5 = np.vstack([g68, extra2])
    configs.append(c5)
    
    # 6. 5x10 Grid
    c6 = generate_grid_config(5, 10, 50)
    configs.append(c6)
    
    # Add some random perturbations to the best grid
    for i in range(5):
        c7 = generate_grid_config(7, 7, 49)
        extra = np.random.uniform(-HALF, HALF, (1, 2))
        c7 = np.vstack([c7, extra])
        c7 += np.random.normal(0, 0.01, c7.shape)
        configs.append(c7)
        
    # Optimize each configuration
    for i, config in enumerate(configs):
        if time.time() - start_time > deadline:
            break
            
        # Limit time per config
        remaining = deadline - (time.time() - start_time)
        alloc = min(10.0, remaining / max(1, len(configs) - i))
        
        opt_centers, r = optimize_config(config, time_limit=alloc)
        
        if r > best_r:
            best_r = r
            best_centers = opt_centers
            
        # If we have time, do a second pass on the best one
        # Or just continue with other configs
        
    # Final polish on the best found
    if best_centers is not None and time.time() - start_time < deadline - 10:
        # Use a very small sigma for high precision
        # Re-run optimize with small sigma
        best_centers, r = optimize_config(best_centers, time_limit=deadline - (time.time() - start_time))
        if r > best_r:
            best_r = r
    
    if best_centers is None:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

if __name__ == "__main__":
    c = construct()
    print(compute_radius(c))
