"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Initial hexagonal packing in a unit square (w=1, h=1, perimeter=4)
    # Approximate positions for 21 circles in a hex grid
    # Row 0: 5 circles
    # Row 1: 4 circles (offset)
    # Row 2: 5 circles
    # Row 3: 4 circles (offset)
    # Row 4: 3 circles
    # Total: 5+4+5+4+3 = 21
    
    # Let's generate a hex grid that fits in [0,1]x[0,1]
    # Radius approx 0.12
    # dx = 2*r, dy = r*sqrt(3)
    
    def generate_hex_initial():
        positions = []
        radii = []
        # Center the grid
        # 5 rows
        # y = 0.1, 0.3, 0.5, 0.7, 0.9
        # x depends on row
        # Let's just use a fixed hex-like pattern scaled to fit
        # r ~ 0.13
        # dx = 0.26, dy = 0.225
        
        # Simpler: use a grid and jitter, then let relaxation sort it out
        # But hex is better.
        
        # 5 rows of 5, 4, 5, 4, 3
        rows = [5, 4, 5, 4, 3]
        y_coords = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for i, count in enumerate(rows):
            y = y_coords[i]
            # Center horizontally
            # If count is odd, center is 0.5
            # If count is even, center is 0.5
            # Spacing 0.2
            start_x = 0.5 - (count - 1) * 0.1
            for j in range(count):
                x = start_x + j * 0.2
                # Add small jitter
                x += np.random.uniform(-0.01, 0.01)
                y += np.random.uniform(-0.01, 0.01)
                positions.append([x, y])
                radii.append(0.08) # Initial small radius
        
        return np.array(positions), np.array(radii)

    positions, radii = generate_hex_initial()
    
    # Optimization variables: x, y, r for each circle, and w (width)
    # h is determined by w: 2(w+h)=4 => h = 2-w
    # Constraints:
    # 1. w in [1, 2] (assuming w >= h for w.l.o.g., or just w in [0.5, 1.5]?)
    #    Actually, perimeter 4 means 2(w+h)=4 => h=2-w.
    #    If w=1, h=1. If w=2, h=0 (degenerate). If w=0.5, h=1.5.
    #    Let's search w in [0.5, 1.5].
    # 2. 0 <= x_i <= w
    # 3. 0 <= y_i <= h
    # 4. r_i > 0
    # 5. Distance between i, j >= r_i + r_j
    
    # Objective: Maximize sum(r_i)
    
    # We will use a simple gradient-based approach with penalty for overlaps.
    # Or simulated annealing.
    # Let's try a simple gradient ascent with penalty.
    
    # State: [x1, y1, r1, ..., x21, y21, r21, w]
    # Size: 21*3 + 1 = 64
    
    def compute_state(state):
        w = state[-1]
        h = 2.0 - w
        pos = state[:-1].reshape(N, 3)
        return w, h, pos
        
    def objective_and_penalty(state):
        w, h, pos = compute_state(state)
        sum_r = np.sum(pos[:, 2])
        
        # Penalties
        penalty = 0.0
        
        # Boundary violations
        # x < 0 or x > w
        # y < 0 or y > h
        # r < 0
        
        # Use soft penalties
        # Boundary: max(0, -x)^2 + max(0, x-w)^2
        for i in range(N):
            x, y, r = pos[i]
            if x < 0: penalty += (x)**2
            if x > w: penalty += (x-w)**2
            if y < 0: penalty += (y)**2
            if y > h: penalty += (y-h)**2
            if r < 0: penalty += (r)**2
        
        # Overlap violations
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = pos[i]
                xj, yj, rj = pos[j]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                min_dist = ri + rj
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        # We want to maximize sum_r - penalty
        return sum_r - penalty

    # Simple gradient ascent
    # Numerical gradient
    def gradient(state):
        eps = 1e-4
        grad = np.zeros_like(state)
        base_val = objective_and_penalty(state)
        for i in range(len(state)):
            state_p = state.copy()
            state_p[i] += eps
            val_p = objective_and_penalty(state_p)
            grad[i] = (val_p - base_val) / eps
        return grad

    # Simulated Annealing style random restarts / moves might be more robust
    # Let's do a mix: Random perturbations + Gradient steps
    
    best_state = np.zeros(64)
    best_state[:-1] = np.concatenate([positions, radii]).flatten()
    best_state[-1] = 1.0 # w=1
    best_val = objective_and_penalty(best_state)
    
    current_state = best_state.copy()
    current_val = best_val
    
    T = 0.1 # Temperature
    step_size = 0.05
    
    while time.time() < deadline:
        # Gradient step
        grad = gradient(current_state)
        
        # Random perturbation
        noise = np.random.normal(0, step_size, size=len(current_state))
        
        # Update
        # Move in direction of gradient + noise
        # We need to handle w specially to keep it in range
        new_state = current_state + 0.01 * grad + noise
        
        # Clamp w to [0.5, 1.5]
        new_state[-1] = np.clip(new_state[-1], 0.5, 1.5)
        
        # Clamp radii to be positive?
        # The penalty handles it, but let's keep them somewhat sane
        # new_state[2::3] = np.maximum(new_state[2::3], 0.01) # Maybe too aggressive
        
        new_val = objective_and_penalty(new_state)
        
        # Accept if better, or with probability exp((new_val - current_val)/T)
        if new_val > current_val:
            current_state = new_state
            current_val = new_val
            if current_val > best_val:
                best_state = current_state.copy()
                best_val = current_val
        else:
            prob = np.exp((new_val - current_val) / T)
            if np.random.rand() < prob:
                current_state = new_state
                current_val = new_val
                
        # Cool down
        T *= 0.999
        if T < 1e-5:
            T = 1e-5
            # Restart with new noise? Or just continue with low T
            
    # Final refinement: Local optimization on best_state
    # Simple gradient ascent for a few steps
    for _ in range(50):
        if time.time() > deadline - 5: break
        grad = gradient(best_state)
        best_state = best_state + 0.005 * grad
        best_state[-1] = np.clip(best_state[-1], 0.5, 1.5)
        # Check if valid
        # If invalid, we might need to project back, but the objective penalizes it.
        # We just want to maximize the raw value of the valid configuration.
        # The evaluator measures "raw value" which is sum of radii, but only if valid?
        # "Pack 21 circles with disjoint interiors... maximize sum of radii"
        # "evaluator measures the raw value"
        # Usually implies if invalid, score 0.
        # So we must ensure the final output is VALID.
        
    # Post-process: Ensure validity
    # If the best_state has overlaps, we need to shrink radii or move circles.
    # Let's check validity of best_state
    w, h, pos = compute_state(best_state)
    
    # Check overlaps
    valid = True
    for i in range(N):
        x, y, r = pos[i]
        if x < -1e-9 or x > w + 1e-9 or y < -1e-9 or y > h + 1e-9 or r < -1e-9:
            valid = False
            break
        for j in range(i+1, N):
            xi, yi, ri = pos[i]
            xj, yj, rj = pos[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - 1e-9:
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback: Shrink radii to fix overlaps
        # Or just use a simpler valid configuration
        # Let's try to fix by scaling radii
        # This is a heuristic
        for _ in range(100):
            # Check worst overlap
            worst_overlap = 0
            worst_pair = None
            for i in range(N):
                x, y, r = pos[i]
                if x < 0: 
                    # Move in
                    pos[i, 0] = max(0, x)
                if x > w:
                    pos[i, 0] = min(w, x)
                if y < 0:
                    pos[i, 1] = max(0, y)
                if y > h:
                    pos[i, 1] = min(h, y)
                if r < 0:
                    pos[i, 2] = 0.01
                for j in range(i+1, N):
                    xi, yi, ri = pos[i]
                    xj, yj, rj = pos[j]
                    dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    min_dist = ri + rj
                    overlap = min_dist - dist
                    if overlap > worst_overlap:
                        worst_overlap = overlap
                        worst_pair = (i, j)
                        
            if worst_overlap < 1e-9:
                break
                
            i, j = worst_pair
            # Shrink both radii by half the overlap
            shrink = worst_overlap / 2 + 1e-6
            pos[i, 2] = max(0.01, pos[i, 2] - shrink)
            pos[j, 2] = max(0.01, pos[j, 2] - shrink)
            # Move them apart?
            # Just shrinking is safer for validity
            
        # Re-check
        valid = True
        for i in range(N):
            x, y, r = pos[i]
            if x < -1e-9 or x > w + 1e-9 or y < -1e-9 or y > h + 1e-9 or r < -1e-9:
                valid = False
                break
            for j in range(i+1, N):
                xi, yi, ri = pos[i]
                xj, yj, rj = pos[j]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if dist < ri + rj - 1e-9:
                    valid = False
                    break
            if not valid: break
        
        if not valid:
            # Emergency fallback: Grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])
            
    # Construct output
    # Format: (n, 3) -> x, y, r
    # Ensure w, h are consistent with perimeter 4
    # The bounding box of the circles must be within a rectangle of perimeter <= 4.
    # Our w, h satisfy 2(w+h)=4.
    # The circles are inside [0, w] x [0, h].
    
    return pos

# EVOLVE-BLOCK-END
