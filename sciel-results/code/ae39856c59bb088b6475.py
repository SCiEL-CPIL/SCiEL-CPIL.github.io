import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_score(centers):
    """
    Compute the actual maximum radius for a given set of centers.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Boundary distances
    max_abs = np.max(np.abs(centers), axis=1)
    boundary_dists = 0.5 - max_abs
    min_boundary = np.min(boundary_dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r

def compute_forces(centers, power=4.0, wall_power=4.0):
    """
    Compute repulsive forces for all particles.
    Force is proportional to -d^(power-1) for pairs, and similar for walls.
    We minimize potential V = sum 1/d^power.
    Gradient of V wrt pos_i is -power * sum_{j!=i} (pos_i - pos_j) / |pos_i - pos_j|^(power+1)
    Actually, let's stick to the force direction.
    F_i = sum_{j!=i} (pos_i - pos_j) / |pos_i - pos_j|^(power+1)
    Wall force: pushes inward if close to wall.
    """
    forces = np.zeros_like(centers)
    
    # Pairwise forces
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    mask = np.ones_like(dists, dtype=bool)
    np.fill_diagonal(mask, False)
    dists_safe = np.where(mask, dists, 1.0)
    
    # Force magnitude factor: 1 / d^(power+1)
    # We want force to be repulsive. 
    # Potential U = 1/d^p. Grad U = -p * r_vec / d^(p+1). 
    # Force = -Grad U = p * r_vec / d^(p+1).
    # Direction is from j to i (away from j).
    # diff[i,j] is c_i - c_j. So force on i from j is along (c_i - c_j).
    
    # Vectorized calculation
    # factor[i, j] = p / dists[i, j]^(p+1)
    p = power
    factors = np.zeros_like(dists)
    factors[mask] = p / (dists_safe[mask] ** (p + 1))
    
    # Force on i is sum over j of factors[i, j] * diff[i, j]
    forces = np.sum(factors[:, :, np.newaxis] * diff, axis=1)
    
    # Wall forces
    # Potential for wall: If x < -0.5 + r, push right. If x > 0.5 - r, push left.
    # Let's use a smooth potential for walls too.
    # U_wall = sum_k (1 / max(eps, 0.5 - |c_k|)^p) ? No, that's distance to boundary.
    # Let's use distance to the *limit* of the sphere center.
    # Actually, simpler: just push away from the boundary face.
    # d_left = c_k - (-0.5) = c_k + 0.5
    # d_right = 0.5 - c_k
    # If d_left < r_target (or some soft threshold), force is +x.
    # If d_right < r_target, force is -x.
    
    # To make it smooth and differentiable everywhere, we can use:
    # F_x = A * (1 / (c_x + 0.5)^(p+1) - 1 / (0.5 - c_x)^(p+1))
    # This assumes the sphere is near the walls. If it's far, the forces are small and cancel out roughly.
    # However, this potential goes to infinity at the walls, which is good.
    # Let's use this smooth wall force.
    
    eps = 1e-6
    d_left = centers + 0.5
    d_right = 0.5 - centers
    
    d_left_safe = np.maximum(d_left, eps)
    d_right_safe = np.maximum(d_right, eps)
    
    wall_force = (p / (d_left_safe ** (p + 1))) - (p / (d_right_safe ** (p + 1)))
    
    forces += wall_force
    
    return forces

def optimize_packing(initial_centers, time_budget, power=4.0, learning_rate=0.001, kick_interval=100, kick_strength=0.05):
    """
    Run gradient descent with repulsive forces.
    """
    centers = initial_centers.copy()
    best_centers = centers.copy()
    best_score = compute_score(best_centers)
    
    start_time = time.time()
    lr = learning_rate
    step = 0
    
    # Adaptive learning rate or fixed? Fixed with decay is safer.
    while time.time() - start_time < time_budget:
        forces = compute_forces(centers, power=power)
        
        # Gradient descent step: centers = centers - lr * forces
        # Since forces are repulsive (pointing away from neighbors), we want to move IN the direction of forces?
        # Wait. 
        # Potential U increases as spheres get closer. 
        # We want to MINIMIZE U. 
        # Grad U points in direction of increasing U. 
        # Force F = -Grad U points in direction of decreasing U.
        # My compute_forces returns F (repulsive).
        # So we should update: centers = centers + lr * forces.
        
        centers += lr * forces
        
        # Clamp to ensure they stay inside the cube (safety)
        # The wall forces should keep them in, but numerical errors might push them out.
        centers = np.clip(centers, -0.49, 0.49)
        
        # Periodic kick to escape local minima
        if step % kick_interval == 0 and step > 0:
            noise = np.random.normal(0, kick_strength, centers.shape)
            centers += noise
            centers = np.clip(centers, -0.49, 0.49)
            
            # Decay learning rate slightly to allow convergence after kicks
            lr *= 0.95
            
        # Evaluate score
        current_score = compute_score(centers)
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            
        step += 1
        
    return best_centers, best_score

def polish_gradient(centers, max_iter=2000):
    """
    Final polish using L-BFGS-B on a soft-min objective.
    """
    def objective(centers_flat):
        c = centers_flat.reshape(N, 3)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        max_abs = np.max(np.abs(c), axis=1)
        b = 0.5 - max_abs
        
        # Soft min of pairwise distances / 2 and boundary distances
        # We want to maximize r = min(min_dist/2, min_boundary)
        # Equivalent to minimizing -r.
        # Soft min: -eps * log(sum(exp(-x/eps)))
        
        # We have two sets of values: d_pairs/2 and b_walls
        # Combine them into one array
        min_dist_val = np.min(dists)
        # Actually, summing exp over all pairs is better for gradient
        eps = 0.005
        pair_terms = np.exp(-dists / (2.0 * eps))
        # Mask diagonal
        pair_terms[~np.eye(N, dtype=bool)] = pair_terms[~np.eye(N, dtype=bool)]
        np.fill_diagonal(pair_terms, 0)
        
        wall_terms = np.exp(-b / eps)
        
        sum_exp = np.sum(pair_terms) + np.sum(wall_terms)
        
        if sum_exp == 0:
            return 0.0
            
        soft_min = -eps * np.log(sum_exp)
        
        # We want to maximize soft_min, so return negative
        return -soft_min

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective,
            centers.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
        )
        return res.x.reshape(N, 3)
    except:
        return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    best_centers = None
    best_score = 0.0
    
    np.random.seed(42)
    
    # 1. Grid Initialization
    # 4x4x4 grid has 64 points. We need 50.
    # The best 50 points from a 4x4x4 grid likely exclude the corners or center?
    # Actually, for packing, we want to maximize min distance.
    # Let's just take the first 50 or random 50.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Random selection of 50 from 64
    idx = np.random.choice(64, 50, replace=False)
    init_grid = grid_points[idx].copy()
    
    # 2. Random Initialization
    init_random = np.random.uniform(-0.4, 0.4, (50, 3))
    
    # 3. Jittered Grid
    init_jitter = init_grid + np.random.normal(0, 0.02, (50, 3))
    init_jitter = np.clip(init_jitter, -0.45, 0.45)
    
    inits = [init_grid, init_random, init_jitter]
    
    # Allocate time for each initialization strategy
    # We have ~145s. Let's give 40s to each of the 3 main strategies.
    # Then use remaining time for polishing.
    
    time_per_strategy = 40.0
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10: # Keep 10s for polish
            break
            
        # Run optimization
        centers, score = optimize_packing(
            init, 
            time_budget=time_per_strategy, 
            power=4.0, 
            learning_rate=0.005, 
            kick_interval=50, 
            kick_strength=0.02
        )
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    # Final Polish
    if best_centers is not None and time.time() < deadline - 2:
        polished = polish_gradient(best_centers, max_iter=1000)
        score_polished = compute_score(polished)
        
        if score_polished > best_score:
            best_score = score_polished
            best_centers = polished
    elif best_centers is None:
        # Fallback
        best_centers = init_grid
        best_score = compute_score(best_centers)

    return best_centers

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    res = construct()
    end = time.time()
    score = compute_score(res)
    print(f"Time: {end-start:.2f}s, Score: {score}")
