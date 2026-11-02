"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

def _max_radius_at(x, y, w, h, circles):
    """
    Calculate the maximum radius a circle can have at (x, y) 
    given existing circles (list of (xi, yi, ri)) and rectangle (w, h).
    """
    # Distance to walls
    r_wall = min(x, w - x, y, h - y)
    if r_wall <= 0:
        return 0.0
        
    r_max = r_wall
    
    # Distance to existing circles
    for (xi, yi, ri) in circles:
        dist = np.sqrt((x - xi)**2 + (y - yi)**2)
        # The new circle must not overlap: dist >= r_new + ri
        # So r_new <= dist - ri
        r_candidate = dist - ri
        if r_candidate < r_max:
            r_max = r_candidate
            
    return max(0.0, r_max)

def _greedy_pack(w, h, n_circles, n_candidates=1000, seed=0):
    """
    Greedy packing algorithm.
    Places n_circles one by one, choosing the position that allows the largest radius.
    """
    rng = np.random.default_rng(seed)
    circles = []
    
    for _ in range(n_circles):
        best_r = -1.0
        best_pos = None
        
        # Generate random candidate positions
        # We can bias towards center or just uniform. Uniform is safer.
        candidates_x = rng.uniform(0, w, n_candidates)
        candidates_y = rng.uniform(0, h, n_candidates)
        
        # Also add some structured candidates (grid) to help initialization
        # But random is usually sufficient for N=21
        # Let's mix random and a coarse grid
        grid_size = 10
        gx = np.linspace(0, w, grid_size)
        gy = np.linspace(0, h, grid_size)
        grid_candidates_x = []
        grid_candidates_y = []
        for gx_i in gx:
            for gy_i in gy:
                grid_candidates_x.append(gx_i)
                grid_candidates_y.append(gy_i)
        
        # Combine
        all_x = np.concatenate([candidates_x, grid_candidates_x])
        all_y = np.concatenate([candidates_y, grid_candidates_y])
        
        # Evaluate all candidates
        # We can vectorize this partially, but N is small so loop is fine
        # However, _max_radius_at is O(k) where k is number of existing circles.
        # Total work is O(N * M * N) = O(N^2 * M). With N=21, M=1100, this is ~500k ops per restart. Very fast.
        
        for i in range(len(all_x)):
            x = all_x[i]
            y = all_y[i]
            r = _max_radius_at(x, y, w, h, circles)
            if r > best_r:
                best_r = r
                best_pos = (x, y)
                
        if best_pos is not None and best_r > 1e-6:
            circles.append((best_pos[0], best_pos[1], best_r))
        else:
            # Fallback: place a tiny circle if no space? 
            # This shouldn't happen if w,h are reasonable.
            # If it does, we might have failed to pack.
            break
            
    # Convert to numpy array
    if len(circles) < n_circles:
         # Fill remaining with 0 radius? Or fail?
         # The task requires N circles.
         # If we couldn't place enough, we should return a valid but low-score config or try harder.
         # For now, pad with small circles at random spots if possible, or just return what we have?
         # The evaluator expects shape (n, 3).
         # We must return N circles.
         # If we run out of space, we place tiny circles.
         while len(circles) < n_circles:
             x = rng.uniform(0, w)
             y = rng.uniform(0, h)
             r = _max_radius_at(x, y, w, h, circles)
             if r <= 0:
                 r = 1e-5
             circles.append((x, y, r))
             
    return np.array(circles)

def construct():
    best_score = -np.inf
    best_result = None
    
    start_time = time.time()
    time_limit = 110.0
    
    # Try different aspect ratios
    # w + h = 2 => h = 2 - w
    # w in [0.5, 1.5] is a reasonable range.
    # Square (1.0, 1.0) is often good.
    
    aspect_ratios = np.linspace(0.6, 1.4, 9)
    
    num_restarts_per_aspect = 5
    
    current_aspect_idx = 0
    current_restart_idx = 0
    
    while time.time() - start_time < time_limit:
        if current_aspect_idx >= len(aspect_ratios):
            break
            
        w = aspect_ratios[current_aspect_idx]
        h = 2.0 - w
        
        if current_restart_idx < num_restarts_per_aspect:
            seed = current_aspect_idx * 100 + current_restart_idx
            result = _greedy_pack(w, h, N, n_candidates=500, seed=seed)
            
            # Calculate score
            # Sum of radii
            score = np.sum(result[:, 2])
            
            # Check validity
            valid = True
            # Boundary check
            for i in range(N):
                x, y, r = result[i]
                if x < -1e-9 or x > w + 1e-9 or y < -1e-9 or y > h + 1e-9:
                    valid = False
                    break
                if r < -1e-9:
                    valid = False
                    break
                if r > min(x, w-x, y, h-y) + 1e-9:
                    valid = False
                    break
            
            if valid:
                # Overlap check
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((result[i,0]-result[j,0])**2 + (result[i,1]-result[j,1])**2)
                        if dist < result[i,2] + result[j,2] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                        
            if valid and score > best_score:
                best_score = score
                best_result = result
                
            current_restart_idx += 1
        else:
            current_aspect_idx += 1
            current_restart_idx = 0
            
    if best_result is None:
        # Fallback: Grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        w = 1.0
        h = 1.0
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * w / side
                y = (j + 0.5) * h / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        # Pad if necessary
        while len(cells) < N:
             cells.append([0.5, 0.5, 0.0])
        return np.array(cells)
        
    return best_result

# EVOLVE-BLOCK-END
