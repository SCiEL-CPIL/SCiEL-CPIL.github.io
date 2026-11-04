"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.optimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def calc_radius(pts):
        # pts: (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        max_coord = np.max(np.abs(pts), axis=1)
        wall_dists = 0.5 - max_coord
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def objective(x, beta):
        pts = x.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        max_coord = np.max(np.abs(pts), axis=1)
        wall_dists = 0.5 - max_coord
        
        mask = dists < np.inf
        pair_dists = dists[mask] / 2.0
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        min_v = np.min(all_dists)
        exps = np.exp(-beta * (all_dists - min_v))
        soft_min = min_v - (1.0 / beta) * np.log(np.sum(exps))
        
        return -soft_min

    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def run_optimization(init_points):
        best_pts = init_points.copy()
        best_r = calc_radius(best_pts)
        
        beta_values = [5, 10, 20, 50, 100, 200, 500]
        current_pts = init_points.copy()
        
        for beta in beta_values:
            if time.time() > deadline:
                break
            try:
                res = scipy.optimize.minimize(
                    objective, 
                    current_pts.flatten(), 
                    args=(beta,), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                current_pts = res.x.reshape(-1, 2)
            except Exception:
                pass
            
            r = calc_radius(current_pts)
            if r > best_r:
                best_r = r
                best_pts = current_pts.copy()
                
        return best_pts, best_r

    # 1. Initial Guess: Hexagonal Lattice
    s = 0.15
    points = []
    range_x = int(1.2 / s) + 2
    range_y = int(1.2 / (s * np.sqrt(3)/2)) + 2
    
    for j in range(-range_y, range_y + 1):
        y = j * s * np.sqrt(3) / 2
        if y > 0.5 or y < -0.5:
            continue
        offset = (s / 2) if (j % 2 != 0) else 0
        for i in range(-range_x, range_x + 1):
            x = i * s + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < N:
        while len(points) < N:
            p = np.array([np.random.uniform(-0.49, 0.49), np.random.uniform(-0.49, 0.49)])
            points = np.vstack([points, p])
    elif len(points) > N:
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    
    np.random.shuffle(points)
    
    best_points, best_radius = run_optimization(points)
    
    # 2. Try a few random restarts if time permits
    restarts = 0
    while time.time() < deadline - 20:
        # Generate a new random-ish initial guess
        # Mix of hex and random
        if restarts % 2 == 0:
            # Perturbed hex
            p_hex = points.copy()
            p_hex += np.random.normal(0, 0.01, p_hex.shape)
            p_hex = np.clip(p_hex, -0.49, 0.49)
            init_pts = p_hex
        else:
            # Random
            init_pts = np.random.uniform(-0.49, 0.49, (N, 2))
            
        pts, r = run_optimization(init_pts)
        if r > best_radius:
            best_radius = r
            best_points = pts
            
        restarts += 1

    return best_points

# EVOLVE-BLOCK-END
