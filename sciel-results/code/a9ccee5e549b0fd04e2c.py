"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Rectangle perimeter is 4, so 2*(w+h) = 4 => w+h = 2.
    # Let w = 2 * t, h = 2 * (1-t), where t in (0, 1).
    # To maximize sum of radii, we want to pack efficiently.
    # Initial guess: square-ish rectangle.
    # t = 0.5 => w=1, h=1. Perimeter = 4.
    
    # State: 
    # t: aspect ratio parameter (1 variable)
    # circles: (x, y, r) for N circles (3*N variables)
    
    # Total variables: 1 + 3*21 = 64
    
    def get_rect_dims(t):
        # t is in (0, 1)
        # Clip t to avoid degenerate rectangles
        t = np.clip(t, 0.1, 0.9)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def evaluate(state):
        """
        Returns a score (higher is better) and validity.
        Score = sum(radii) - penalty
        Penalty is large if constraints are violated.
        """
        t = state[0]
        circles = state[1:]
        
        w, h = get_rect_dims(t)
        
        sum_r = 0.0
        penalty = 0.0
        
        # Reshape circles
        # circles is flat array of x,y,r
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            
            if ri < 1e-6:
                ri = 1e-6
                penalty += 1.0 # small penalty for tiny radius? or just clamp
            
            sum_r += ri
            
            # Boundary constraints:
            # x - r >= 0, x + r <= w
            # y - r >= 0, y + r <= h
            if xi - ri < 0:
                penalty += (xi - ri) * 1000  # Quadratic or linear penalty? Linear is often faster to fix. Let's use squared for smoothness.
                # Actually, for SA, simple linear penalty on violation magnitude is robust.
                # violation = max(0, r - x), max(0, r - (w-x)), etc.
                penalty += (ri - xi) * 100
                penalty += (ri - (w - xi)) * 100
                penalty += (ri - yi) * 100
                penalty += (ri - (h - yi)) * 100
            
            # Wait, the above adds penalty for ALL sides if we just sum max(0, ...).
            # Let's be precise.
            # Penalty should be 0 if inside.
            # Violation if x < r => dist to left wall < r.
            # Violation if x > w - r => dist to right wall < r.
            
            # Let's restart penalty logic inside the loop for clarity and correctness.
            # I will rewrite the penalty calculation properly below.
        
        # Let's do a vectorized or clean loop for penalties
        penalty = 0.0
        sum_r = 0.0
        coords = []
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            if ri < 1e-5:
                ri = 1e-5
            sum_r += ri
            coords.append((xi, yi, ri))
            
            # Boundaries
            if xi < ri:
                penalty += (ri - xi) * 500
            if xi > w - ri:
                penalty += (xi - (w - ri)) * 500
            if yi < ri:
                penalty += (ri - yi) * 500
            if yi > h - ri:
                penalty += (yi - (h - ri)) * 500
        
        # Overlaps
        for i in range(N):
            xi, yi, ri = coords[i]
            for j in range(i+1, N):
                xj, yj, rj = coords[j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                dist = np.sqrt(dist_sq)
                min_dist = ri + rj
                if dist < min_dist:
                    # Penalty proportional to overlap depth
                    overlap = min_dist - dist
                    penalty += overlap * 1000
        
        score = sum_r - penalty
        return score, penalty

    def check_valid(state):
        """
        Check if the state is strictly valid for the final return.
        """
        t = state[0]
        circles = state[1:]
        w, h = get_rect_dims(t)
        
        coords = []
        for i in range(N):
            xi = circles[3*i]
            yi = circles[3*i+1]
            ri = circles[3*i+2]
            if ri < 1e-6:
                return False
            coords.append((xi, yi, ri))
            
            # Strict boundary check with small epsilon
            if xi - ri < -1e-6 or xi + ri > w + 1e-6 or yi - ri < -1e-6 or yi + ri > h + 1e-6:
                return False
        
        # Strict overlap check
        for i in range(N):
            xi, yi, ri = coords[i]
            for j in range(i+1, N):
                xj, yj, rj = coords[j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                if dist_sq < (ri + rj - 1e-6)**2:
                    return False
        return True

    # Initialization
    # Start with a grid-like initialization but randomize slightly
    # Square 1x1
    t_init = 0.5
    w, h = get_rect_dims(t_init)
    
    state = np.zeros(1 + 3*N)
    state[0] = t_init
    
    # Place circles in a grid pattern initially to get a decent starting point
    side = int(np.ceil(np.sqrt(N)))
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            # Grid center
            gx = (i + 0.5) / side * w
            gy = (j + 0.5) / side * h
            # Max radius in cell is 0.5/side * min(w,h) roughly, but let's start small
            r_init = 0.4 / side * min(w, h)
            
            state[1 + 3*idx] = gx + random.uniform(-0.01, 0.01)
            state[1 + 3*idx + 1] = gy + random.uniform(-0.01, 0.01)
            state[1 + 3*idx + 2] = r_init
            idx += 1
    
    best_state = state.copy()
    best_score, _ = evaluate(best_state)
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 1e-4
    alpha = 0.999
    max_iter = 200000  # Cap iterations, but loop until time
    
    T = T_start
    
    # Pre-compute some constants if needed? No, dynamic w/h.
    
    while time.time() < deadline:
        # Check time frequently
        if time.time() > deadline - 5:
            break
            
        # Generate neighbor
        new_state = best_state.copy()
        
        # Randomly perturb a variable
        var_idx = random.randint(0, len(new_state)-1)
        
        if var_idx == 0:
            # Perturb aspect ratio
            new_state[0] += random.gauss(0, 0.01)
            # Ensure t is valid
            if new_state[0] < 0.05 or new_state[0] > 0.95:
                continue
        else:
            # Perturb circle x, y, or r
            circle_idx = (var_idx - 1) // 3
            comp = (var_idx - 1) % 3
            
            if comp == 0: # x
                new_state[var_idx] += random.gauss(0, 0.05)
            elif comp == 1: # y
                new_state[var_idx] += random.gauss(0, 0.05)
            else: # r
                # Radius perturbation should be smaller
                new_state[var_idx] += random.gauss(0, 0.01)
                if new_state[var_idx] < 1e-4:
                    new_state[var_idx] = 1e-4
                # Cap radius to not exceed half min dimension
                w, h = get_rect_dims(new_state[0])
                max_r = 0.5 * min(w, h)
                if new_state[var_idx] > max_r:
                    new_state[var_idx] = max_r

        new_score, new_penalty = evaluate(new_state)
        
        # Acceptance criterion
        if new_score > best_score or random.random() < np.exp((new_score - best_score) / T):
            best_state = new_state
            best_score = new_score
            
            # If we found a valid solution, we can potentially tighten the "best" tracking
            # But we keep the highest score state as the candidate for annealing.
            # Note: The score includes penalties. A valid solution has penalty 0.
            # So maximizing score is equivalent to maximizing sum_r among valid solutions,
            # and moving towards valid solutions among invalid ones.
        
        T *= alpha
        if T < T_end:
            T = T_end
            
    # Final check and cleanup
    # We need to return a valid configuration.
    # If the best state found is invalid, we should try to repair it or return a known valid one.
    # The seed program returns a valid grid. We should keep a fallback.
    
    # Try to repair best_state if invalid
    if not check_valid(best_state):
        # Simple repair: shrink radii until valid?
        # Or just return the seed solution?
        # Let's try a quick local repair: reduce radii slightly.
        repair_state = best_state.copy()
        for _ in range(100):
            if check_valid(repair_state):
                break
            # Shrink all radii by 1%
            for i in range(N):
                r_idx = 1 + 3*i + 2
                repair_state[r_idx] *= 0.99
        
        if check_valid(repair_state):
            best_state = repair_state
        else:
            # Fallback to seed grid
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) >= N:
                        break
                    # In seed, w=1, h=1 was implicit? 
                    # Seed: side=5. x,y in (0,1). r=0.5/5 - eps = 0.1 - eps.
                    # Perimeter of bounding box of these circles:
                    # Min x: 0.1 - 0.1 = 0. Max x: 0.9 + 0.1 = 1.0. Width 1.
                    # Min y: 0.1 - 0.1 = 0. Max y: 0.9 + 0.1 = 1.0. Height 1.
                    # Perimeter = 4.
                    # So we need to map this to our w,h representation.
                    # Our representation: w+h=2. If w=1, h=1, t=0.5.
                    # The seed circles are in [0,1]x[0,1].
                    # Our evaluator expects coordinates in [0,w]x[0,h].
                    # So if we return t=0.5, w=1, h=1, the coordinates should be as is.
                    
                    gx = (i + 0.5) / side
                    gy = (j + 0.5) / side
                    r = 0.5 / side - 1e-3
                    cells.append([gx, gy, r])
            # Construct final array format: [t, x1, y1, r1, ...]
            final_state = np.zeros(1 + 3*N)
            final_state[0] = 0.5
            for i in range(N):
                final_state[1 + 3*i] = cells[i][0]
                final_state[1 + 3*i + 1] = cells[i][1]
                final_state[1 + 3*i + 2] = cells[i][2]
            best_state = final_state

    # Extract final result
    t_final = best_state[0]
    w_final, h_final = get_rect_dims(t_final)
    
    # The output format required is (N, 3) array of [x, y, r].
    # The problem statement says: "construct() -> array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4"
    # It does NOT ask to return t. It asks for the circles.
    # The evaluator will compute the bounding box of the returned circles.
    # So we just return the x, y, r coordinates.
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_state[1 + 3*i]
        result[i, 1] = best_state[1 + 3*i + 1]
        result[i, 2] = best_state[1 + 3*i + 2]
        
    return result

# EVOLVE-BLOCK-END
