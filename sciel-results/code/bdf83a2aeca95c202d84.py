import numpy as np
import time
from scipy.optimize import minimize

N = 200
DEADLINE = 110.0

def _compute_radius(points):
    # points: (N, 2)
    # Returns radius r = min(dist_center, dist_wall) / 2
    
    # Center-center distances
    # Vectorized distance matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_center_dist = np.sqrt(min_dist_sq)
    
    # Wall distances
    # Square is [-0.5, 0.5]
    dx = 0.5 - np.abs(points[:, 0])
    dy = 0.5 - np.abs(points[:, 1])
    min_wall_dist = np.min(np.concatenate([dx, dy]))
    
    # Radius is half the minimum clear distance
    r = min(min_center_dist, min_wall_dist) / 2.0
    return r

def _generate_hex_grid(n):
    # Generate a hexagonal grid of points
    # Approximate aspect ratio of hex grid is 1 : sqrt(3)/2
    # We want to fill a square.
    
    # Estimate number of rows and cols
    # n ~ cols * rows * density
    # Density of hex is ~ 0.9
    # Let's just generate a grid that covers the square and take the first N
    # Spacing s
    # If we use s = 1/15, we get 15x15 = 225 points.
    # Hex spacing in x is s, in y is s*sqrt(3)/2.
    # Let's pick s such that we get roughly N points.
    
    # Try to fit N points in [-0.5, 0.5]
    # Area = 1.0. Area per point = 1/N.
    # Hex cell area = s^2 * sqrt(3)/2.
    # s = sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    # Number of cols and rows to cover [-0.5, 0.5]
    # Width = 1.0, Height = 1.0
    # cols = ceil(1.0 / s)
    # rows = ceil(1.0 / (s * sqrt(3)/2))
    
    cols = int(np.ceil(1.0 / s))
    rows = int(np.ceil(1.0 / (s * np.sqrt(3) / 2)))
    
    points = []
    for r in range(rows):
        for c in range(cols):
            x = -0.5 + (c + 0.5) * s
            y = -0.5 + (r + 0.5) * s * np.sqrt(3) / 2
            if r % 2 != 0:
                x += s / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Filter points inside the square
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have more than N, we need to select N.
    # To be fair, we can just take the first N, or randomly select.
    # Taking first N might bias towards top-left.
    # Let's shuffle and take N.
    if len(points) > N:
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        # If we don't have enough, pad with random points inside
        # This shouldn't happen with the s calculation, but just in case
        missing = N - len(points)
        random_pts = np.random.uniform(-0.5, 0.5, (missing, 2))
        points = np.vstack([points, random_pts])
        
    return points

def construct():
    start_time = time.time()
    
    # 1. Initialize
    init_points = _generate_hex_grid(N)
    # Clip to be strictly inside to avoid initial singularity
    init_points = np.clip(init_points, -0.499, 0.499)
    
    best_points = init_points
    best_score = _compute_radius(init_points)
    
    # Objective function for optimization
    # Minimize a soft-min of distances to maximize the minimum distance
    # f(x) = - log-sum-exp(-lambda * distances) / lambda
    # We want to maximize min(dist). So minimize -min(dist).
    # Smooth approximation: minimize (1/lambda) * log(sum(exp(-lambda * dist)))
    # This is the "soft-min" of the distances.
    
    lambda_val = 50.0 # High lambda makes it closer to hard min
    
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return np.inf
            
        pts = params.reshape(-1, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        
        # Wall distances
        dx = 0.5 - np.abs(pts[:, 0])
        dy = 0.5 - np.abs(pts[:, 1])
        wall_dists = np.concatenate([dx, dy])
        
        # Combine all distances
        # We need to avoid double counting pairs in the sum
        # But for log-sum-exp, we can just sum all pairs and divide by 2?
        # Or just take upper triangle.
        iu = np.triu_indices(N, k=1)
        pair_dists = dist[iu]
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Clip distances to avoid exp underflow/overflow issues in log-sum-exp
        # If dist is very small, exp(-lambda * dist) is large.
        # If dist is large, exp is small.
        # We want to minimize the soft-min.
        # soft_min = (1/lambda) * log(sum(exp(-lambda * d)))
        
        # Numerical stability: subtract max
        max_d = np.max(all_dists)
        # Actually, for soft-min, we want to emphasize small distances.
        # exp(-lambda * d) is largest for smallest d.
        # So we subtract the MAX of the exponents?
        # Exponents are -lambda * d.
        # Max exponent corresponds to Min distance.
        # Let's just compute directly with clipping.
        
        # Clip distances to [1e-6, 1.0] to prevent issues
        all_dists_clipped = np.clip(all_dists, 1e-6, 1.0)
        
        exp_terms = np.exp(-lambda_val * all_dists_clipped)
        log_sum = np.log(np.sum(exp_terms))
        
        # The objective to minimize is the soft-min value
        soft_min_val = log_sum / lambda_val
        
        return soft_min_val

    # Optimize
    # Use Powell as it doesn't require gradients and is robust
    try:
        res = minimize(objective, init_points.flatten(), method='Powell', 
                       options={'maxiter': 100, 'disp': False})
        
        if res.fun < objective(init_points.flatten()):
            candidate = res.x.reshape(-1, 2)
            if np.all(np.isfinite(candidate)):
                score = _compute_radius(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate
                    
    except Exception:
        pass

    # Second pass with higher lambda or different init if time permits
    if time.time() - start_time < DEADLINE - 10:
        # Increase lambda to sharpen the min
        lambda_val = 100.0
        
        def objective2(params):
            if time.time() - start_time > DEADLINE:
                return np.inf
            pts = params.reshape(-1, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            dx = 0.5 - np.abs(pts[:, 0])
            dy = 0.5 - np.abs(pts[:, 1])
            wall_dists = np.concatenate([dx, dy])
            iu = np.triu_indices(N, k=1)
            pair_dists = dist[iu]
            all_dists = np.concatenate([pair_dists, wall_dists])
            all_dists_clipped = np.clip(all_dists, 1e-6, 1.0)
            exp_terms = np.exp(-lambda_val * all_dists_clipped)
            log_sum = np.log(np.sum(exp_terms))
            return log_sum / lambda_val

        try:
            # Start from the best found so far
            res2 = minimize(objective2, best_points.flatten(), method='Powell', 
                           options={'maxiter': 50, 'disp': False})
            
            if res2.fun < objective2(best_points.flatten()):
                candidate2 = res2.x.reshape(-1, 2)
                if np.all(np.isfinite(candidate2)):
                    score2 = _compute_radius(candidate2)
                    if score2 > best_score:
                        best_score = score2
                        best_points = candidate2
                        
        except Exception:
            pass

    return best_points
