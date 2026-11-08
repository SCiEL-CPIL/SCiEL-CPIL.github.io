"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Simulated Annealing / Stochastic Local Search approach
    # State: x, y, r for N circles
    
    def init_state():
        # Random initialization with some structure to help convergence
        # Grid-like start is often better for packing
        rows = 5
        cols = 6 # 5*6=30, we need 26. 
        # Let's just use a random jittered grid
        xs = np.linspace(0.1, 0.9, 6)
        ys = np.linspace(0.1, 0.9, 5)
        
        points = []
        for i in range(5):
            for j in range(6):
                if len(points) >= N:
                    break
                # Add jitter
                x = xs[j] + np.random.uniform(-0.05, 0.05)
                y = ys[i] + np.random.uniform(-0.05, 0.05)
                # Initial radius estimate
                r = 0.15
                points.append([x, y, r])
        
        # If we didn't get enough, add random
        while len(points) < N:
            points.append([np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8), 0.15])
            
        points = np.array(points[:N])
        return points

    def evaluate(points):
        """
        Returns (score, penalty).
        Score is sum of radii.
        Penalty is sum of squared overlaps and boundary violations.
        We want to maximize Score - Penalty.
        """
        x = points[:, 0]
        y = points[:, 1]
        r = points[:, 2]
        
        sum_r = np.sum(r)
        
        # Boundary penalties
        # x < r or x > 1-r
        bx = np.maximum(0, r - x) + np.maximum(0, x + r - 1)
        by = np.maximum(0, r - y) + np.maximum(0, y + r - 1)
        boundary_penalty = np.sum(bx**2) + np.sum(by**2)
        
        # Overlap penalties
        # Vectorized distance calculation
        # diff_x[i, j] = x[i] - x[j]
        diff_x = x[:, np.newaxis] - x[np.newaxis, :]
        diff_y = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        
        sum_r_pair = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Overlap amount: max(0, sum_r - dist)
        overlap = np.maximum(0, sum_r_pair - dist)
        # Zero out diagonal
        np.fill_diagonal(overlap, 0)
        
        overlap_penalty = np.sum(overlap**2)
        
        total_penalty = boundary_penalty + overlap_penalty
        # We want to maximize sum_r, so score = sum_r - penalty
        # But for SA, we usually minimize energy. Let's define Energy = Penalty - Sum_r * scaling?
        # Actually, let's just track (sum_r, penalty) and accept based on a combined metric.
        # Metric: sum_r - penalty * lambda
        # If lambda is large, it enforces validity. If small, it maximizes size.
        # We want valid solutions. So if penalty > 0, we prefer lower penalty.
        # If penalty == 0, we prefer higher sum_r.
        
        return sum_r, total_penalty

    def try_move(points, temp):
        # Pick 1-3 circles to move
        k = np.random.randint(1, 4)
        idxs = np.random.choice(N, k, replace=False)
        
        new_points = points.copy()
        
        for idx in idxs:
            x, y, r = new_points[idx]
            
            # Perturb position
            dx = np.random.uniform(-0.05, 0.05)
            dy = np.random.uniform(-0.05, 0.05)
            new_points[idx, 0] = x + dx
            new_points[idx, 1] = y + dy
            
            # Perturb radius
            dr = np.random.uniform(-0.02, 0.02)
            new_r = r + dr
            # Keep radius positive and reasonable
            new_points[idx, 2] = np.clip(new_r, 0.01, 0.5)
            
        return new_points

    def is_valid(points, tol=1e-5):
        x = points[:, 0]
        y = points[:, 1]
        r = points[:, 2]
        
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
            
        diff_x = x[:, np.newaxis] - x[np.newaxis, :]
        diff_y = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        sum_r = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Check if any dist < sum_r (excluding diagonal)
        # dist < sum_r - tol
        overlap = dist < (sum_r - tol)
        np.fill_diagonal(overlap, False)
        
        if np.any(overlap):
            return False
            
        return True

    # Main SA loop
    current_points = init_state()
    current_sum_r, current_penalty = evaluate(current_points)
    
    best_points = current_points.copy()
    best_sum_r = current_sum_r if current_penalty < 1e-6 else -np.inf
    
    T = 1.0
    T_min = 0.001
    cooling = 0.99
    
    max_iter = 20000
    no_improve_count = 0
    
    for i in range(max_iter):
        new_points = try_move(current_points, T)
        new_sum_r, new_penalty = evaluate(new_points)
        
        # Acceptance criterion
        # We want to maximize sum_r, but penalty must be 0.
        # Let's define a score: Score = sum_r - penalty * 10.0
        # This heavily penalizes overlaps.
        curr_score = current_sum_r - current_penalty * 10.0
        new_score = new_sum_r - new_penalty * 10.0
        
        delta_score = new_score - curr_score
        
        if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
            current_points = new_points
            current_sum_r, current_penalty = new_sum_r, new_penalty
            
            # Track best valid solution
            if current_penalty < 1e-6:
                if current_sum_r > best_sum_r:
                    best_sum_r = current_sum_r
                    best_points = current_points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Early stopping if stuck for too long
        if no_improve_count > 500:
            break
            
    # Final refinement: local search on best_points to polish
    # Use the simple local search from the seed but on the SA result
    # This combines global search (SA) with local refinement.
    
    def local_refine(points):
        points = points.copy()
        improved = True
        while improved:
            improved = False
            for idx in range(N):
                x, y, r = points[idx]
                # Try to increase radius
                new_r = r + 0.001
                if new_r > 0.5: continue
                
                valid = True
                # Check boundaries
                if new_r > x or new_r > 1-x or new_r > y or new_r > 1-y:
                    valid = False
                else:
                    # Check overlaps
                    for jdx in range(N):
                        if jdx == idx: continue
                        xj, yj, rj = points[jdx]
                        dist = np.sqrt((x-xj)**2 + (y-yj)**2)
                        if dist < rj + new_r - 1e-6:
                            valid = False
                            break
                if valid:
                    points[idx, 2] = new_r
                    improved = True
            
            # Try to move circles to allow larger radii? 
            # For simplicity, just radius increase loop.
            # If no radius can be increased, we might be stuck.
            # We can try random small moves again if stuck.
            if not improved:
                 # Try random perturbations to escape
                 for _ in range(100):
                     idx = np.random.randint(N)
                     x, y, r = points[idx]
                     nx = x + np.random.uniform(-0.01, 0.01)
                     ny = y + np.random.uniform(-0.01, 0.01)
                     if nx < r or nx > 1-r or ny < r or ny > 1-r: continue
                     valid = True
                     for jdx in range(N):
                         if jdx == idx: continue
                         xj, yj, rj = points[jdx]
                         dist = np.sqrt((nx-xj)**2 + (ny-yj)**2)
                         if dist < rj + r - 1e-6:
                             valid = False
                             break
                     if valid:
                         points[idx, 0] = nx
                         points[idx, 1] = ny
                         improved = True
                         break
        return points

    best_points = local_refine(best_points)
    
    # Final validation and clipping
    # If SA didn't find a valid one, try to fix the best one
    if not is_valid(best_points):
         # Fallback to seed logic or simple grid
         # But best_points should be close to valid
         # Let's try to shrink radii to make it valid
         for i in range(N):
             x, y, r = best_points[i]
             # Max radius allowed by boundaries
             r_bound = min(x, 1-x, y, 1-y)
             # Max radius allowed by neighbors
             r_neighbor = np.inf
             for j in range(N):
                 if i == j: continue
                 xj, yj, rj = best_points[j]
                 dist = np.sqrt((x-xj)**2 + (y-yj)**2)
                 r_neighbor = min(r_neighbor, dist - rj)
             
             best_points[i, 2] = max(0.01, min(r_bound, r_neighbor) - 1e-4)
             
         # If still invalid, use grid
         if not is_valid(best_points):
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    return best_points
# EVOLVE-BLOCK-END
