"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin for 120s limit
    
    # Initial state: Grid in a 1x1 square
    # 5x5 grid
    side = 5
    # Centers
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    for i in range(N):
        col = i % side
        row = i // side
        xs[i] = (col + 0.5) / side
        ys[i] = (row + 0.5) / side
        rs[i] = 0.5 / side - 0.001 # Slightly smaller to ensure validity initially
    
    W = 1.0
    H = 1.0
    
    def get_score():
        return np.sum(rs)
    
    def is_valid():
        # Check boundaries
        if np.any(xs - rs < 0) or np.any(xs + rs > W) or np.any(ys - rs < 0) or np.any(ys + rs > H):
            return False
        # Check overlaps
        # Vectorized distance calculation
        # dist[i,j] = sqrt((xi-xj)^2 + (yi-yj)^2)
        # We need dist[i,j] >= ri + rj
        # Using broadcasting
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        req = rs[:, None] + rs[None, :]
        # dist[i,i] is 0, req[i,i] is 2*ri. We ignore diagonal.
        # Check if any off-diagonal dist < req
        mask = ~np.eye(N, dtype=bool)
        if np.any(dist[mask] < req[mask]):
            return False
        return True
    
    best_score = get_score()
    best_state = (xs.copy(), ys.copy(), rs.copy(), W)
    
    current_score = best_score
    T = 0.01
    T_min = 0.0001
    cooling = 0.999
    
    while time.time() < deadline:
        # Randomly select a circle to perturb
        i = np.random.randint(N)
        
        # Perturbation
        # Move position
        dx = np.random.normal(0, 0.01)
        dy = np.random.normal(0, 0.01)
        # Change radius
        dr = np.random.normal(0, 0.005)
        
        old_xi = xs[i]
        old_yi = ys[i]
        old_ri = rs[i]
        
        xs[i] += dx
        ys[i] += dy
        rs[i] += dr
        
        # Ensure radius is positive
        if rs[i] <= 0:
            rs[i] = 0.001
        
        # Check validity
        if not is_valid():
            # Revert
            xs[i] = old_xi
            ys[i] = old_yi
            rs[i] = old_ri
            # Penalty for invalid move in SA logic? 
            # Simpler: just reject invalid moves.
            # To allow exploration, we could accept invalid moves with penalty,
            # but ensuring final validity is easier if we only accept valid ones.
            # However, pure hill climbing gets stuck.
            # Let's try to accept invalid moves with a penalty score.
            # Score = sum(r) - penalty
            # Penalty = sum of overlaps^2 + sum of boundary violations^2
            
            # Let's implement a penalty-based objective for the SA step
            pass # The above revert logic makes it pure hill climbing on valid states.
            # Let's switch to penalty-based objective to allow "tunneling" through overlaps.
            
    # The above loop is a bit inefficient if we just revert.
    # Let's rewrite the inner loop to use a penalty function.
    
    def objective():
        # Sum of radii
        score = np.sum(rs)
        
        # Boundary penalty
        pen = 0
        # x < 0 or x > W
        pen += np.sum(np.maximum(0, rs - xs)**2)
        pen += np.sum(np.maximum(0, xs + rs - W)**2)
        pen += np.sum(np.maximum(0, rs - ys)**2)
        pen += np.sum(np.maximum(0, ys + rs - H)**2)
        
        # Overlap penalty
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        req = rs[:, None] + rs[None, :]
        # Overlap amount = max(0, req - dist)
        overlap = np.maximum(0, req - dist)
        # Ignore diagonal
        overlap = overlap * (1 - np.eye(N))
        pen += np.sum(overlap**2)
        
        return score - 10 * pen
    
    # Restart the optimization with penalty-based SA
    xs = best_state[0].copy()
    ys = best_state[1].copy()
    rs = best_state[2].copy()
    W = best_state[3]
    H = 2.0 - W
    
    current_obj = objective()
    best_obj = current_obj
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_rs = rs.copy()
    best_W = W
    
    T = 0.05
    while time.time() < deadline:
        # Random move
        # 80% chance: move a circle
        # 20% chance: change aspect ratio
        if np.random.rand() < 0.8:
            i = np.random.randint(N)
            dx = np.random.normal(0, 0.02)
            dy = np.random.normal(0, 0.02)
            dr = np.random.normal(0, 0.01)
            
            old_xi = xs[i]
            old_yi = ys[i]
            old_ri = rs[i]
            
            xs[i] += dx
            ys[i] += dy
            rs[i] += dr
            if rs[i] < 0.001: rs[i] = 0.001
            
            new_obj = objective()
            
            delta = new_obj - current_obj
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current_obj = new_obj
                if current_obj > best_obj:
                    best_obj = current_obj
                    best_xs = xs.copy()
                    best_ys = ys.copy()
                    best_rs = rs.copy()
                    best_W = W
            else:
                xs[i] = old_xi
                ys[i] = old_yi
                rs[i] = old_ri
        else:
            # Change W
            dW = np.random.normal(0, 0.05)
            old_W = W
            W += dW
            # Clamp W to [0.5, 1.5] to keep H positive and reasonable
            if W < 0.5: W = 0.5
            if W > 1.5: W = 1.5
            H = 2.0 - W
            
            new_obj = objective()
            
            delta = new_obj - current_obj
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current_obj = new_obj
                if current_obj > best_obj:
                    best_obj = current_obj
                    best_xs = xs.copy()
                    best_ys = ys.copy()
                    best_rs = rs.copy()
                    best_W = W
            else:
                W = old_W
                H = 2.0 - W
                
        T *= 0.999
        if T < 0.001:
            T = 0.001
            
    # Final check: ensure the best state is valid
    # If not, we might need to project it or use the initial valid state
    # The penalty method might leave slight overlaps.
    # Let's try to "fix" the best state by slightly reducing radii if overlaps exist.
    
    xs = best_xs
    ys = best_ys
    rs = best_rs
    W = best_W
    H = 2.0 - W
    
    # Iterative correction: if overlap, reduce radii
    # Simple loop
    for _ in range(100):
        # Check overlaps
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        req = rs[:, None] + rs[None, :]
        overlap = np.maximum(0, req - dist)
        overlap = overlap * (1 - np.eye(N))
        
        if np.all(overlap == 0):
            break
            
        # Reduce radii proportional to overlap
        # For each pair i,j with overlap, reduce ri and rj
        # Sum of overlaps for each circle
        overlaps_per_circle = np.sum(overlap, axis=1)
        # Reduce radius by half the max overlap?
        # This is heuristic.
        # Simpler: just scale down all radii slightly if any overlap
        max_overlap = np.max(overlap)
        if max_overlap > 0:
            # Reduce all radii by max_overlap/2
            rs -= max_overlap / 2
            # Ensure positive
            rs[rs < 0.001] = 0.001
            
    # Check boundaries again
    # If out of bounds, shift or shrink
    # Shrink
    for _ in range(100):
        if np.any(xs - rs < 0) or np.any(xs + rs > W) or np.any(ys - rs < 0) or np.any(ys + rs > H):
            # Find max violation
            viol = 0
            viol = max(viol, np.max(0, rs - xs))
            viol = max(viol, np.max(0, xs + rs - W))
            viol = max(viol, np.max(0, rs - ys))
            viol = max(viol, np.max(0, ys + rs - H))
            if viol > 0:
                rs -= viol / 2
                rs[rs < 0.001] = 0.001
            else:
                break
        else:
            break
            
    # Final validity check
    if not is_valid():
        # Fallback to initial grid
        xs = best_state[0]
        ys = best_state[1]
        rs = best_state[2]
        W = 1.0
        H = 1.0

    # Construct output array
    # (x, y, r)
    out = np.column_stack((xs, ys, rs))
    return out

# EVOLVE-BLOCK-END
