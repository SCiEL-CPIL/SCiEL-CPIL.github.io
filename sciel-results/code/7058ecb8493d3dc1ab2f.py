"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    np.random.seed(42)
    random.seed(42)
    
    best_sum = -1
    best_result = None
    
    # Try a range of aspect ratios
    # Perimeter 4 => w + h = 2. Let w = 2 * ratio, h = 2 * (1-ratio) where ratio in [0,1]
    # Or simpler: w in [0.5, 1.5] (since w+h=2, min dim can be small but not too small for 21 circles)
    # Actually, optimal aspect ratio is likely close to 1.
    
    aspect_ratios = np.linspace(0.6, 1.4, 15) # w/h ratios
    
    for ar in aspect_ratios:
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Initial state: random positions inside rectangle, small radii
        # Ensure initial state is valid? No, SA handles penalties.
        # But starting with valid or near-valid is better.
        # Let's try a random non-overlapping initial config if possible, else random.
        
        # Strategy: Start with random positions, radii = 0.
        # Then grow radii while adjusting positions? 
        # Standard SA: x = [x1, y1, r1, ..., xN, yN, rN]
        
        max_iter = 5000
        temp = 1.0
        cooling = 0.9995
        
        # Initial vector
        # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        # To ensure validity, we need x in [r, w-r], y in [r, h-r]
        
        def random_state():
            xs = np.random.uniform(0, w, N)
            ys = np.random.uniform(0, h, N)
            rs = np.random.uniform(0, 0.05, N) # Small initial radii
            return np.concatenate([xs, ys, rs])
            
        def evaluate(state):
            xs = state[0:N]
            ys = state[N:2*N]
            rs = state[2*N:3*N]
            
            # Check boundaries
            # r_i <= x_i <= w - r_i
            # r_i <= y_i <= h - r_i
            boundary_violation = 0.0
            for i in range(N):
                if xs[i] < rs[i]:
                    boundary_violation += (rs[i] - xs[i]) ** 2
                if xs[i] > w - rs[i]:
                    boundary_violation += (xs[i] - (w - rs[i])) ** 2
                if ys[i] < rs[i]:
                    boundary_violation += (rs[i] - ys[i]) ** 2
                if ys[i] > h - rs[i]:
                    boundary_violation += (ys[i] - (h - rs[i])) ** 2
            
            # Check overlaps
            overlap_violation = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        overlap_violation += (min_dist - dist) ** 2
            
            total_violation = boundary_violation + overlap_violation
            objective = -np.sum(rs) + 10.0 * total_violation
            return objective, total_violation, np.sum(rs)

        def move(state):
            new_state = state.copy()
            # Pick a random circle
            i = np.random.randint(0, N)
            
            # Perturb position and radius
            # Scale of perturbation based on temperature?
            step_size = 0.05 * temp + 0.01
            
            # Move x, y
            new_state[i] += np.random.uniform(-step_size, step_size)
            new_state[N+i] += np.random.uniform(-step_size, step_size)
            
            # Move radius
            # Bias towards increasing radius?
            # Or symmetric?
            # Symmetric is safer for SA.
            new_state[2*N+i] += np.random.uniform(-step_size*0.5, step_size*0.5)
            
            # Ensure radius is positive
            if new_state[2*N+i] < 0:
                new_state[2*N+i] = 0
            
            # Clamp positions to [0, w] and [0, h] roughly to avoid huge penalties
            # But let the penalty handle it? Clamping might help.
            new_state[i] = np.clip(new_state[i], 0, w)
            new_state[N+i] = np.clip(new_state[N+i], 0, h)
            
            return new_state

        # Run SA
        current_state = random_state()
        current_obj, _, _ = evaluate(current_state)
        best_state = current_state.copy()
        best_obj = current_obj
        
        start_time = time.time()
        while temp > 0.01 and (time.time() - start_time) < 2.0: # 2s per aspect ratio
            next_state = move(current_state)
            next_obj, _, _ = evaluate(next_state)
            
            delta = next_obj - current_obj
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                current_state = next_state
                current_obj = next_obj
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_state = current_state.copy()
            
            temp *= cooling
        
        # Extract best result for this aspect ratio
        xs = best_state[0:N]
        ys = best_state[N:2*N]
        rs = best_state[2*N:3*N]
        
        # Validate strictly
        valid = True
        for i in range(N):
            if rs[i] <= 1e-9:
                valid = False
                break
            if xs[i] < rs[i] - 1e-7 or xs[i] > w - rs[i] + 1e-7:
                valid = False
                break
            if ys[i] < rs[i] - 1e-7 or ys[i] > h - rs[i] + 1e-7:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    if dist < rs[i] + rs[j] - 1e-7:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            s = np.sum(rs)
            if s > best_sum:
                best_sum = s
                best_result = np.column_stack([xs, ys, rs])
                
    if best_result is None:
        # Fallback: simple grid in square 1x1
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
