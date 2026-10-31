import numpy as np
import time

N = 26

def construct():
    """
    Pack 26 circles in a unit square maximizing sum of radii.
    Uses Simulated Annealing with feasible moves.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Best solution found so far
    best_circles = None
    best_score = -np.inf

    # Number of restarts to attempt
    # We will dynamically decide based on time
    max_restarts = 100
    
    # SA Parameters
    T_start = 0.05
    T_end = 1e-6
    cooling_rate = 0.995
    
    # Move parameters
    pos_step = 0.1  # Step size for position moves
    rad_step = 0.01 # Step size for radius moves
    p_move = 0.7    # Probability of position move vs radius move

    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 10.0: # Leave 10s for final checks/return
            break
            
        # 1. Initialization
        # Use a jittered grid to ensure initial validity
        # 5x5 grid covers 25, we need 26. 
        # Let's use a 5x5 grid for 25 and one extra random spot far away? 
        # Or just 26 random points with tiny radius?
        # Tiny radius random points are always valid if N is small enough? 
        # No, 26 points in unit square with r=1e-4 is definitely valid.
        
        # Let's try a smarter init: 
        # Place circles in a 5x5 grid, but the 26th is placed in the center of the square 
        # and we start with very small radii.
        
        # Actually, a 5x5 grid has 25 cells. 
        # Let's just use 26 random positions and tiny radii.
        # This ensures validity.
        
        x = np.random.rand(N)
        y = np.random.rand(N)
        r = np.full(N, 1e-4)
        
        circles = np.column_stack((x, y, r))
        
        # Verify initial validity (should be true)
        if not is_valid(circles):
            continue

        current_score = np.sum(r)
        
        # 2. Simulated Annealing
        T = T_start
        no_improve_count = 0
        max_no_improve = 10000
        
        while T > T_end:
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Choose a random circle
            i = np.random.randint(N)
            
            # Choose move type
            if np.random.rand() < p_move:
                # Position move
                new_x = circles[i, 0] + np.random.uniform(-pos_step, pos_step)
                new_y = circles[i, 1] + np.random.uniform(-pos_step, pos_step)
                new_r = circles[i, 2]
                
                # Clamp to square
                new_x = np.clip(new_x, new_r, 1 - new_r)
                new_y = np.clip(new_y, new_r, 1 - new_r)
                
                # Check validity
                # Only need to check circle i against others
                if is_valid_move(circles, i, new_x, new_y, new_r):
                    # Calculate score change
                    delta_score = 0.0
                    # Always accept if valid? No, SA allows worse moves to escape local minima.
                    # But for packing, "validity" is a hard constraint.
                    # We only accept valid moves.
                    # To escape local minima, we need to allow moves that *temporarily* reduce score?
                    # In this formulation, we only propose valid moves.
                    # If a move is valid, it's a candidate.
                    # If delta_score > 0, accept.
                    # If delta_score < 0 (which is impossible for position move if r is same), 
                    # wait, position move doesn't change r. So delta_score is 0.
                    # So we should accept position moves if they are valid?
                    # Not necessarily. Moving might block future radius growth.
                    # However, usually we want to accept valid moves to explore space.
                    # Let's use a Metropolis criterion on the "potential" or just accept valid moves.
                    # If we only accept valid moves, we are doing Random Walk in feasible space.
                    # To maximize sum of radii, we need to bias towards larger radii.
                    
                    # Let's modify: 
                    # If move is valid, accept it with probability 1 if it doesn't hurt score?
                    # Position move: score unchanged. Accept with prob 1? 
                    # This might wander aimlessly.
                    # Better: Accept position moves with prob 1 to explore, 
                    # and rely on Radius moves to improve score.
                    
                    circles[i, 0] = new_x
                    circles[i, 1] = new_y
                    # Score unchanged
            else:
                # Radius move (increase or decrease)
                # Try to increase radius
                new_r = circles[i, 2] + np.random.uniform(0, rad_step)
                new_x = circles[i, 0]
                new_y = circles[i, 1]
                
                # Clamp radius to fit in square
                max_r = min(new_x, 1 - new_x, new_y, 1 - new_y)
                if new_r > max_r:
                    new_r = max_r
                
                if new_r > circles[i, 2]:
                    delta_score = new_r - circles[i, 2]
                    
                    if is_valid_move(circles, i, new_x, new_y, new_r):
                        # Accept if valid. 
                        # Since delta_score > 0, always accept.
                        circles[i, 2] = new_r
                        current_score += delta_score
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Try to decrease radius? 
                    # Decreasing radius allows neighbors to grow.
                    # This is a "make room" move.
                    new_r = circles[i, 2] - np.random.uniform(0, rad_step)
                    if new_r < 0: new_r = 1e-4
                    delta_score = new_r - circles[i, 2]
                    
                    if is_valid_move(circles, i, new_x, new_y, new_r):
                        # This move reduces score. Accept with probability exp(delta_score / T)
                        if np.random.rand() < np.exp(delta_score / T):
                            circles[i, 2] = new_r
                            current_score += delta_score
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1

            # Update temperature
            T *= cooling_rate
            
            # Update global best
            if current_score > best_score:
                best_score = current_score
                best_circles = circles.copy()
                
            # Early termination for this restart if stuck
            if no_improve_count > max_no_improve:
                break

    if best_circles is None:
        # Fallback: return a valid trivial solution
        # Grid of 26 tiny circles
        side = 5
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.01))
                    count += 1
        # Add remaining if any (should be none for 5x5=25, need 1 more)
        if count < N:
            cells.append((0.5, 0.5, 0.01))
        return np.array(cells)

    return best_circles

def is_valid(circles):
    """Check if entire configuration is valid."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False
        for j in range(i + 1, n):
            x2, y2, r2 = circles[j]
            dist_sq = (x - x2)**2 + (y - y2)**2
            req_dist = r + r2
            if dist_sq < req_dist**2 - 1e-12:
                return False
    return True

def is_valid_move(circles, i, new_x, new_y, new_r):
    """Check if moving circle i to (new_x, new_y, new_r) is valid."""
    # Boundary check
    if new_x < new_r or new_x > 1 - new_r:
        return False
    if new_y < new_r or new_y > 1 - new_r:
        return False
    
    # Overlap check with other circles
    # Vectorized check
    others = np.delete(circles, i, axis=0)
    # Calculate squared distances
    dx = others[:, 0] - new_x
    dy = others[:, 1] - new_y
    dist_sq = dx**2 + dy**2
    
    # Required squared distances
    req_dist_sq = (others[:, 2] + new_r)**2
    
    # Check if any distance is less than required
    # Add small epsilon for floating point
    if np.any(dist_sq < req_dist_sq - 1e-12):
        return False
        
    return True
