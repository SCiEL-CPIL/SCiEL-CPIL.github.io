"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 maximizing sum of radii.
    Returns array of shape (26, 3): rows (x, y, radius).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    best_result = None
    best_score = -np.inf

    # We'll use a gradient descent approach on a penalty function
    # Variables: positions (x, y) for each circle and radii r for each circle
    # Objective: maximize sum(r_i) subject to:
    #   1. Circles don't overlap: dist(i,j) >= r_i + r_j
    #   2. Circles inside unit square: r_i <= x_i, r_i <= y_i, r_i <= 1-x_i, r_i <= 1-y_i

    def make_initial_grid():
        """Create initial grid arrangement of 26 circles."""
        # 5x5 grid = 25 cells, need 26 circles
        # Use a slightly denser arrangement
        side = 5
        positions = []
        radii = []
        for i in range(side):
            for j in range(side):
                if len(positions) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    positions.append([x, y])
                    radii.append(0.5 / side - 0.01)
        
        # Add one more circle in a gap
        if len(positions) < N:
            positions.append([0.5, 0.5])
            radii.append(0.05)
        
        return np.array(positions), np.array(radii)

    def make_initial_random(seed_val):
        """Create random initial arrangement."""
        rng = np.random.RandomState(seed_val)
        positions = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = np.full(N, 0.08)
        return positions, radii

    def compute_penalty(positions, radii):
        """
        Compute penalty for constraint violations.
        Returns (score, penalty) where score = sum(radii) and penalty >= 0.
        """
        n = len(radii)
        score = np.sum(radii)
        penalty = 0.0
        
        # Check circle-circle overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        # Check boundary constraints
        for i in range(n):
            r = radii[i]
            x, y = positions[i]
            # Circle must be inside [0,1]^2
            if x < r:
                penalty += (r - x) ** 2
            if x > 1 - r:
                penalty += (x - (1 - r)) ** 2
            if y < r:
                penalty += (r - y) ** 2
            if y > 1 - r:
                penalty += (y - (1 - r)) ** 2
        
        # Also ensure radii are positive
        for i in range(n):
            if radii[i] < 0:
                penalty += radii[i] ** 2
        
        return score, penalty

    def gradient_descent(positions, radii, lr=0.001, max_iter=5000, tol=1e-8):
        """
        Gradient descent to maximize sum(radii) while satisfying constraints.
        Uses penalty method.
        """
        n = len(radii)
        penalty_weight = 100.0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute current score and penalty
            score, penalty = compute_penalty(positions, radii)
            
            # Compute gradients numerically
            grad_x = np.zeros_like(positions)
            grad_y = np.zeros_like(positions)
            grad_r = np.zeros_like(radii)
            
            eps = 1e-5
            base_obj = score - penalty_weight * penalty
            
            # Gradient w.r.t. positions
            for i in range(n):
                for d in range(2):
                    old_val = positions[i, d]
                    positions[i, d] = old_val + eps
                    s_plus, p_plus = compute_penalty(positions, radii)
                    obj_plus = s_plus - penalty_weight * p_plus
                    positions[i, d] = old_val - eps
                    s_minus, p_minus = compute_penalty(positions, radii)
                    obj_minus = s_minus - penalty_weight * p_minus
                    positions[i, d] = old_val
                    
                    if d == 0:
                        grad_x[i] = (obj_plus - obj_minus) / (2 * eps)
                    else:
                        grad_y[i] = (obj_plus - obj_minus) / (2 * eps)
            
            # Gradient w.r.t. radii
            for i in range(n):
                old_val = radii[i]
                radii[i] = old_val + eps
                s_plus, p_plus = compute_penalty(positions, radii)
                obj_plus = s_plus - penalty_weight * p_plus
                radii[i] = old_val - eps
                s_minus, p_minus = compute_penalty(positions, radii)
                obj_minus = s_minus - penalty_weight * p_minus
                radii[i] = old_val
                grad_r[i] = (obj_plus - obj_minus) / (2 * eps)
            
            # Update
            positions[:, 0] += lr * grad_x
            positions[:, 1] += lr * grad_y
            radii += lr * grad_r
            
            # Clamp radii to be positive
            radii = np.maximum(radii, 0.001)
            
            # Clamp positions to stay reasonable
            positions = np.clip(positions, 0.0, 1.0)
            
            # Check convergence
            grad_norm = np.sqrt(np.sum(grad_x**2) + np.sum(grad_y**2) + np.sum(grad_r**2))
            if grad_norm < tol:
                break
            
            # Adaptively increase penalty weight
            if it % 500 == 0 and it > 0:
                penalty_weight *= 1.5
        
        return positions, radii

    # Try multiple initializations
    initializations = []
    
    # Grid initialization
    pos_grid, rad_grid = make_initial_grid()
    initializations.append((pos_grid, rad_grid))
    
    # Random initializations with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        pos_rand, rad_rand = make_initial_random(seed)
        initializations.append((pos_rand, rad_rand))
    
    for init_idx, (pos, rad) in enumerate(initializations):
        if time.time() > deadline - 5:
            break
        
        # Run gradient descent
        pos_opt, rad_opt = gradient_descent(pos.copy(), rad.copy(), lr=0.002, max_iter=3000)
        
        # Evaluate
        score, penalty = compute_penalty(pos_opt, rad_opt)
        
        # Only accept if penalty is small
        if penalty < 1e-4:
            if score > best_score:
                best_score = score
                best_result = np.column_stack([pos_opt, rad_opt])
        
        # Also try with smaller learning rate for refinement
        if time.time() < deadline - 10:
            pos_opt2, rad_opt2 = gradient_descent(pos_opt.copy(), rad_opt2 if 'rad_opt2' in dir() else rad_opt.copy(), lr=0.0005, max_iter=2000)
            score2, penalty2 = compute_penalty(pos_opt2, rad_opt2)
            if penalty2 < 1e-4 and score2 > best_score:
                best_score = score2
                best_result = np.column_stack([pos_opt2, rad_opt2])
    
    # If no valid result found, fall back to grid
    if best_result is None:
        pos_grid, rad_grid = make_initial_grid()
        best_result = np.column_stack([pos_grid, rad_grid])
    
    # Final validation and cleanup
    # Ensure all circles are strictly inside the unit square
    for i in range(N):
        x, y, r = best_result[i]
        # Shrink radius slightly to ensure strict interior
        max_r = min(x, y, 1 - x, 1 - y)
        if r >= max_r:
            r = max_r * 0.999
        best_result[i] = [x, y, r]
    
    # Final check for overlaps and fix if needed
    for i in range(N):
        for j in range(i + 1, N):
            dx = best_result[i, 0] - best_result[j, 0]
            dy = best_result[i, 1] - best_result[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = best_result[i, 2] + best_result[j, 2]
            if dist < min_dist:
                # Shrink the smaller circle
                if best_result[i, 2] < best_result[j, 2]:
                    best_result[i, 2] = max(0.001, (dist - best_result[j, 2]) * 0.999)
                else:
                    best_result[j, 2] = max(0.001, (dist - best_result[i, 2]) * 0.999)
    
    return best_result
# EVOLVE-BLOCK-END
