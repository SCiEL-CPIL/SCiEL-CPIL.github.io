import numpy as np
from scipy.optimize import minimize
import time

N = 21
START_TIME = time.time()
TIME_LIMIT = 115.0  # Leave 5s buffer

def construct():
    def time_remaining():
        return TIME_LIMIT - (time.time() - START_TIME)

    def check_validity(x, y, r, W, H):
        """
        Check if configuration is valid.
        Returns (is_valid, violation_value).
        violation_value is the max amount by which constraints are violated.
        """
        if W <= 0 or H <= 0:
            return False, 1e10
        if abs(2*(W+H) - 4) > 1e-4:
            return False, 1e10
        
        # Boundary checks
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        
        boundary_viol = max(0, -min_x, max_x - W, -min_y, max_y - H)
        
        # Overlap checks
        # Use numpy broadcasting for pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        rad_sum_sq = (r[:, None] + r[None, :])**2
        
        # We want dist_sq >= rad_sum_sq
        # violation is max(0, rad_sum_sq - dist_sq)
        # Exclude diagonal
        np.fill_diagonal(dist_sq, 1e10)
        np.fill_diagonal(rad_sum_sq, 0)
        
        overlap_viol = np.max(rad_sum_sq - dist_sq)
        if overlap_viol < 0:
            overlap_viol = 0
            
        total_viol = boundary_viol + overlap_viol
        return total_viol < 1e-6, total_viol

    def relax_and_expand(x, y, r, W, H, max_iters=500):
        """
        Iteratively resolve overlaps and expand radii.
        """
        # 1. Resolve overlaps by pushing circles apart
        for _ in range(max_iters):
            if time_remaining() < 5:
                break
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx**2 + dy**2
                    rad_sum = r[i] + r[j]
                    if dist_sq < rad_sum**2:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-10 else 1e-10
                        # Push apart
                        overlap = rad_sum - dist
                        if overlap > 0:
                            # Normalize direction
                            nx = dx / dist
                            ny = dy / dist
                            # Move each by half the overlap (or weighted by inverse radius if we cared, but equal is fine for start)
                            shift = 0.5 * overlap
                            x[i] += nx * shift
                            x[j] -= nx * shift
                            y[i] += ny * shift
                            y[j] -= ny * shift
                            moved = True
            if not moved:
                break
        
        # 2. Expand radii greedily
        step = 0.005
        for _ in range(100):
            if time_remaining() < 5:
                break
            improved = False
            for i in range(N):
                new_r = r[i] + step
                if x[i] - new_r < 0 or x[i] + new_r > W:
                    continue
                if y[i] - new_r < 0 or y[i] + new_r > H:
                    continue
                valid = True
                for j in range(N):
                    if i == j: 
                        continue
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    rad_sum = new_r + r[j]
                    if dist_sq < rad_sum**2:
                        valid = False
                        break
                if valid:
                    r[i] = new_r
                    improved = True
            if not improved:
                step *= 0.9
                if step < 1e-6:
                    break
        return x, y, r

    def local_search(x, y, r, W, H, iterations=500):
        """
        Simple local search: perturb positions, re-relax, re-expand.
        """
        best_score = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        temp = 0.05
        for _ in range(iterations):
            if time_remaining() < 5:
                break
            x_new = x.copy()
            y_new = y.copy()
            # Perturb
            noise = np.random.normal(0, temp, N)
            x_new += noise
            y_new += np.random.normal(0, temp, N)
            
            # Keep radii for now, try to relax and expand
            r_new = r.copy()
            x_new, y_new, r_new = relax_and_expand(x_new, y_new, r_new, W, H, max_iters=50)
            
            # Check validity
            is_valid, viol = check_validity(x_new, y_new, r_new, W, H)
            if is_valid:
                score = np.sum(r_new)
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_r = x_new.copy(), y_new.copy(), r_new.copy()
                    x, y, r = best_x, best_y, best_r
                    temp = max(0.01, temp * 0.95)
                else:
                    # Metropolis acceptance? Or just hill climb?
                    # Hill climb is safer for packing.
                    pass
            else:
                # If invalid, we can still accept if violation is small and score is high?
                # For simplicity, stick to valid solutions only.
                pass
        
        return best_x, best_y, best_r, best_score

    def sequential_init(W, H):
        """
        Initialize circles using sequential greedy insertion.
        This builds the packing layer by layer, respecting previous placements.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # 1. Place first circle: Max radius in center
        r[0] = min(W, H) / 2.0
        x[0] = W / 2.0
        y[0] = H / 2.0
        
        # Candidate points for subsequent circles
        # Use a grid + random points to find the best spot for each new circle
        n_grid = 15
        grid_x = np.linspace(0, W, n_grid)
        grid_y = np.linspace(0, H, n_grid)
        candidate_xs, candidate_ys = np.meshgrid(grid_x, grid_y)
        candidate_xs = candidate_xs.flatten()
        candidate_ys = candidate_ys.flatten()
        
        # Add random points for better coverage
        n_rand = 200
        candidate_xs = np.concatenate([candidate_xs, np.random.uniform(0, W, n_rand)])
        candidate_ys = np.concatenate([candidate_ys, np.random.uniform(0, H, n_rand)])
        
        for i in range(1, N):
            if time_remaining() < 5:
                # If out of time, place remaining circles with tiny radius at random spots to ensure valid output
                for k in range(i, N):
                    x[k] = np.random.uniform(0.01, W-0.01)
                    y[k] = np.random.uniform(0.01, H-0.01)
                    r[k] = 0.001
                break
            
            best_r = -1
            best_x = 0
            best_y = 0
            
            # For each candidate position, calculate max possible radius
            # Max radius is limited by:
            # 1. Distance to boundaries: min(cx, W-cx, cy, H-cy)
            # 2. Distance to existing circles: min_j (dist(c, c_j) - r_j)
            
            dist_to_bounds = np.minimum(
                np.minimum(candidate_xs, W - candidate_xs),
                np.minimum(candidate_ys, H - candidate_ys)
            )
            
            # Calculate distance to all existing circles
            # dist_sq = (cx - x_prev)^2 + (cy - y_prev)^2
            # We need dist >= r_prev + r_new => r_new <= dist - r_prev
            # So max_r_candidate = min_j (dist - r_prev)
            
            # Vectorized distance calculation
            # existing circles: 0 to i-1
            ex = x[:i]
            ey = y[:i]
            er = r[:i]
            
            # (n_cand, i) matrix of distances
            dx = candidate_xs[:, None] - ex[None, :]
            dy = candidate_ys[:, None] - ey[None, :]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Max feasible radius from circles is min_j (dists[:, j] - er[j])
            # dists - er[None, :]
            feasible_from_circles = np.min(dists - er[None, :], axis=1)
            
            # Total feasible radius is min(bound, circles)
            feasible_r = np.minimum(dist_to_bounds, feasible_from_circles)
            
            # Find the max
            idx = np.argmax(feasible_r)
            best_r = feasible_r[idx]
            
            # If best_r is very small or negative, it means we might be stuck or need to adjust
            # But sequential greedy usually finds *some* space unless it's very tight.
            if best_r < 1e-5:
                # Fallback: place at a random valid spot with tiny radius? 
                # Or just accept a very small radius.
                # To ensure validity, we can place it in the center of the largest empty spot.
                # For now, just use the best candidate found, even if small.
                pass
                
            x[i] = candidate_xs[idx]
            y[i] = candidate_ys[idx]
            r[i] = max(best_r, 1e-4) # Ensure non-zero
            
        return x, y, r

    best_global_score = 0
    best_global_x = None
    best_global_y = None
    best_global_r = None
    
    # Try various aspect ratios
    # Perimeter 2(W+H)=4 => W+H=2
    # W in [0.5, 1.5]
    W_values = np.linspace(0.6, 1.4, 9)
    
    np.random.seed(42)
    
    for W in W_values:
        H = 2.0 - W
        if H <= 0.1: continue
        
        # Initialize
        x, y, r = sequential_init(W, H)
        
        # Relax and expand
        x, y, r = relax_and_expand(x, y, r, W, H, max_iters=200)
        
        # Check validity
        is_valid, viol = check_validity(x, y, r, W, H)
        if not is_valid:
            # Try to fix with more relaxation
            x, y, r = relax_and_expand(x, y, r, W, H, max_iters=500)
            is_valid, viol = check_validity(x, y, r, W, H)
            if not is_valid:
                continue
        
        score = np.sum(r)
        
        # Local search
        x, y, r, score = local_search(x, y, r, W, H, iterations=100)
        
        if score > best_global_score:
            best_global_score = score
            best_global_x = x.copy()
            best_global_y = y.copy()
            best_global_r = r.copy()
            # print(f"New best: W={W:.2f}, H={H:.2f}, Score={score:.4f}")
            
        if time_remaining() < 10:
            break

    # If we didn't find anything valid (shouldn't happen), fallback
    if best_global_x is None:
        W, H = 1.0, 1.0
        x, y, r = sequential_init(W, H)
        x, y, r = relax_and_expand(x, y, r, W, H)
        # Force valid by shrinking if needed
        is_valid, viol = check_validity(x, y, r, W, H)
        if not is_valid:
             # Shrink radii until valid
             for _ in range(100):
                 r *= 0.95
                 is_valid, viol = check_validity(x, y, r, W, H)
                 if is_valid: break
        best_global_x = x
        best_global_y = y
        best_global_r = r

    result = np.zeros((N, 3))
    result[:, 0] = best_global_x
    result[:, 1] = best_global_y
    result[:, 2] = best_global_r
    
    return result
