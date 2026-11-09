# EVOLVE-BLOCK-START
import numpy as np
import time

def squares_in_square10() -> tuple:
    """
    Place 10 squares in the unit square to maximize the sum of their side lengths.
    Uses Simulated Annealing to find a high-quality packing.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s limit
    
    best_score = -np.inf
    best_params = None
    
    # Define the objective function to minimize (negative sum of sides + penalty)
    def objective(params):
        sides = params[:n]
        centers = params[n:].reshape(n, 2)
        
        # Penalty for violations
        penalty = 0.0
        tol = 1e-7
        
        # 1. Bounds penalty: squares must be inside [0, 1]^2
        # We use a quadratic penalty for boundary violations to make the landscape smoother
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            
            # Side length must be positive
            if s <= 0:
                penalty += 1000.0 * (1.0 - s)
                s = max(s, 1e-6) # Avoid division by zero or weird behavior in overlap calc if needed, though we use sides directly
                
            # X-bounds
            left = cx - s/2
            right = cx + s/2
            if left < -tol:
                penalty += 100.0 * (-tol - left)**2
            if right > 1 + tol:
                penalty += 100.0 * (right - (1 + tol))**2
            
            # Y-bounds
            bottom = cy - s/2
            top = cy + s/2
            if bottom < -tol:
                penalty += 100.0 * (-tol - bottom)**2
            if top > 1 + tol:
                penalty += 100.0 * (top - (1 + tol))**2
        
        # 2. Overlap penalty: squares must not overlap
        # Overlap condition: |dx| < (si+sj)/2 AND |dy| < (si+sj)/2
        # We penalize the "depth" of overlap in both dimensions if they are close
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                
                # If they are overlapping in both dimensions
                if dx < t and dy < t:
                    # The amount of overlap
                    overlap_x = t - dx
                    overlap_y = t - dy
                    # Penalty proportional to the area of overlap (roughly)
                    penalty += 100.0 * overlap_x * overlap_y
        
        return -np.sum(sides) + penalty

    def simulate_annealing(init_params, max_iters, t_start, t_end):
        current_params = init_params.copy()
        current_score = objective(current_params)
        best_params_local = current_params.copy()
        best_score_local = current_score
        
        # Cooling schedule: exponential
        alpha = (t_end / t_start) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Propose a move: perturb one or two random parameters
            # We perturb sides and centers
            num_perturb = np.random.randint(1, 3) # 1 or 2 parameters
            indices = np.random.choice(len(current_params), num_perturb, replace=False)
            
            new_params = current_params.copy()
            for idx in indices:
                # Step size depends on temperature
                step = np.random.uniform(-1, 1) * t_start * 0.1
                new_params[idx] += step
            
            new_score = objective(new_params)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / max(t_start * (alpha ** i), 1e-12)):
                current_params = new_params
                current_score = new_score
                
                if current_score < best_score_local:
                    best_params_local = current_params.copy()
                    best_score_local = current_score
                    
        return best_params_local, best_score_local

    # Initial configurations
    # 1. Grid 3x3 + 1 small
    def make_grid_3x3_plus():
        params = np.zeros(3 * n)
        side = 1.0 / 3.0
        for i in range(9):
            col = i % 3
            row = i // 3
            params[i] = side
            params[n + i] = (col + 0.5) * side
            params[n + n + i] = (row + 0.5) * side
        # 10th square: small in corner
        params[9] = 0.1
        params[n + 9] = 0.05
        params[n + n + 9] = 0.05
        return params

    # 2. Grid 2x5
    def make_grid_2x5():
        params = np.zeros(3 * n)
        side = 1.0 / 5.0
        for i in range(10):
            col = i % 5
            row = i // 5
            params[i] = side
            params[n + i] = (col + 0.5) * side
            params[n + n + i] = (row + 0.5) * side
        return params

    # 3. Random
    def make_random(seed):
        rng = np.random.default_rng(seed)
        params = np.zeros(3 * n)
        # Random sides between 0.1 and 0.3
        params[:n] = rng.uniform(0.1, 0.3, n)
        # Random centers
        params[n:n+2*n] = rng.uniform(0, 1, 2*n)
        return params

    # 4. "Jostled" grid: Start with 3x3 grid and randomize the 10th and perturb all
    def make_jostled_grid(seed):
        params = make_grid_3x3_plus()
        rng = np.random.default_rng(seed)
        # Perturb centers slightly
        for i in range(n):
            params[n+i] += rng.uniform(-0.05, 0.05)
            params[n+n+i] += rng.uniform(-0.05, 0.05)
        # Perturb sides slightly
        for i in range(n):
            params[i] *= rng.uniform(0.9, 1.1)
        return params

    inits = [
        make_grid_3x3_plus(),
        make_grid_2x5(),
        make_random(42),
        make_random(123),
        make_jostled_grid(7),
        make_jostled_grid(99)
    ]
    
    # Run SA for each initial configuration
    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Parameters for SA
        # We want to explore enough. 
        # t_start: large enough to allow big jumps initially
        # t_end: small enough to settle
        # max_iters: depends on time budget
        # Let's allocate roughly (deadline - now) / remaining_inits
        time_left = deadline - time.time()
        remaining = len(inits) - idx
        iters_per_init = int(5000 * (time_left / (remaining * 10.0))) # Heuristic
        iters_per_init = max(1000, min(50000, iters_per_init))
        
        try:
            local_best, local_score = simulate_annealing(init, iters_per_init, 1.0, 1e-4)
            if local_score < best_score:
                best_score = local_score
                best_params = local_best
        except Exception as e:
            pass

    # Fallback if SA failed or didn't find anything
    if best_params is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    sides = best_params[:n].astype(np.float64)
    centers = best_params[n:].reshape(n, 2).astype(np.float64)
    
    # Final validation and correction
    tol = 1e-7
    valid = True
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        valid = False
    if np.any(sides <= 0):
        valid = False
    
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            valid = False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            valid = False
            
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                valid = False

    if not valid:
        # Try to fix by shrinking
        sides *= 0.99
        centers = centers.copy() # Centers might need adjustment if shrinking helps, but usually shrinking is enough if they were close
        # Re-check validity quickly
        valid2 = True
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                valid2 = False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                valid2 = False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    valid2 = False
                    
        if valid2:
            sides = sides
            centers = centers
        else:
            # Fallback to grid
            k = int(np.ceil(np.sqrt(n)))
            side = 1.0 / k
            idx = np.arange(n)
            cols = idx % k
            rows = idx // k
            cx = (cols + 0.5) * side
            cy = (rows + 0.5) * side
            centers = np.stack([cx, cy], axis=1).astype(np.float64)
            sides = np.full(n, side, dtype=np.float64)

    return centers, sides

# EVOLVE-BLOCK-END
