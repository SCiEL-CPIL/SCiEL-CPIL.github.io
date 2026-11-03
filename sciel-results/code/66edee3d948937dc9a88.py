"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    N = 21
    
    # Helper to check validity
    def is_valid(circles, w, h, eps=1e-9):
        # Check bounds
        if np.any(circles[:, 0] < circles[:, 2] - eps): return False
        if np.any(circles[:, 0] > w - circles[:, 2] + eps): return False
        if np.any(circles[:, 1] < circles[:, 2] - eps): return False
        if np.any(circles[:, 1] > h - circles[:, 2] + eps): return False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = circles[i, 2] + circles[j, 2]
                if dist_sq < min_dist*min_dist - eps:
                    return False
        return True

    def get_value(circles):
        return np.sum(circles[:, 2])

    def resolve_conflicts(circles, w, h):
        """
        Push circles apart to resolve overlaps and bound violations.
        This acts as a projection step for the hard constraints.
        Returns modified circles.
        """
        circles = circles.copy()
        # 1. Resolve bound violations
        for i in range(N):
            r = circles[i, 2]
            # x
            if circles[i, 0] < r:
                circles[i, 0] = r
            elif circles[i, 0] > w - r:
                circles[i, 0] = w - r
            # y
            if circles[i, 1] < r:
                circles[i, 1] = r
            elif circles[i, 1] > h - r:
                circles[i, 1] = h - r
        
        # 2. Resolve overlaps (simple iterative push)
        # We do a few iterations of pushing
        for _ in range(5):
            for i in range(N):
                for j in range(i + 1, N):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist_sq = dx*dx + dy*dy
                    min_dist = circles[i, 2] + circles[j, 2]
                    
                    if dist_sq < min_dist*min_dist:
                        dist = math.sqrt(dist_sq)
                        if dist < 1e-9:
                            # Random nudge if exactly overlapping
                            dx = 0.1
                            dy = 0.1
                            dist = math.sqrt(dx*dx + dy*dy)
                        
                        overlap = min_dist - dist
                        # Push apart along the vector
                        # Move each by half the overlap (weighted by radius if desired, but equal is simpler)
                        push_x = (dx / dist) * (overlap / 2.0)
                        push_y = (dy / dist) * (overlap / 2.0)
                        
                        circles[i, 0] += push_x
                        circles[i, 1] += push_y
                        circles[j, 0] -= push_x
                        circles[j, 1] -= push_y
                        
                        # Re-clip to bounds immediately to prevent flying out
                        for k in [i, j]:
                            r = circles[k, 2]
                            circles[k, 0] = np.clip(circles[k, 0], r, w - r)
                            circles[k, 1] = np.clip(circles[k, 1], r, h - r)
        return circles

    # State: circles (N, 3), w, h
    # w + h = 2. Let w be the variable, h = 2 - w.
    # w in [0.5, 1.5] is reasonable.
    
    best_circles = None
    best_w = None
    best_val = -np.inf

    # Initial Configuration: 3x7 grid
    def init_state(w_init):
        h_init = 2.0 - w_init
        # 3 rows, 7 cols
        rows, cols = 3, 7
        cell_w = w_init / cols
        cell_h = h_init / rows
        r_init = min(cell_w, cell_h) * 0.48 # Slightly smaller than max to allow movement
        
        circles = np.zeros((N, 3))
        idx = 0
        for i in range(rows):
            for j in range(cols):
                cx = (j + 0.5) * cell_w
                cy = (i + 0.5) * cell_h
                # Stagger middle row
                if i == 1:
                    cx += cell_w / 2.0
                    cx = np.clip(cx, r_init, w_init - r_init)
                
                circles[idx, 0] = cx
                circles[idx, 1] = cy
                circles[idx, 2] = r_init
                idx += 1
        
        # Resolve initial conflicts
        circles = resolve_conflicts(circles, w_init, h_init)
        return circles, w_init

    # Simulated Annealing
    # We will run SA for a fixed number of iterations or until time runs out.
    # Since we have 140s, we can do a lot of iterations.
    
    # Parameters
    T_start = 0.1
    T_end = 1e-6
    alpha = 0.9995 # Cooling rate
    
    # Try a few different starting aspect ratios to avoid bad local minima
    start_w_values = [1.2, 1.4, 1.0, 1.6]
    
    for w_start in start_w_values:
        if time.time() > deadline:
            break
            
        circles, w = init_state(w_start)
        h = 2.0 - w
        current_val = get_value(circles)
        
        # Ensure it's valid at start
        if not is_valid(circles, w, h):
             # If invalid, try to fix or skip
             circles = resolve_conflicts(circles, w, h)
             if not is_valid(circles, w, h):
                 continue
             current_val = get_value(circles)
        
        if current_val > best_val:
            best_val = current_val
            best_circles = circles.copy()
            best_w = w

        T = T_start
        rng = np.random.default_rng(42)
        
        # Number of iterations
        # Estimate: 1e6 iterations might take ~5-10s in Python. 
        # Let's do 500,000 iterations per restart if time permits.
        max_iters = 500000
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # 1. Propose move
            # With prob 0.1, change aspect ratio
            # Else, move a circle
            
            if rng.random() < 0.1:
                # Change w
                dw = rng.normal(0, 0.05)
                new_w = w + dw
                # Constrain w to [0.4, 1.6]
                if new_w < 0.4 or new_w > 1.6:
                    continue
                new_h = 2.0 - new_w
                
                # Project circles to new bounds
                new_circles = circles.copy()
                for i in range(N):
                    r = new_circles[i, 2]
                    new_circles[i, 0] = np.clip(new_circles[i, 0], r, new_w - r)
                    new_circles[i, 1] = np.clip(new_circles[i, 1], r, new_h - r)
                
                # Resolve overlaps in new shape
                new_circles = resolve_conflicts(new_circles, new_w, new_h)
                
                if not is_valid(new_circles, new_w, new_h):
                    continue
                    
                new_val = get_value(new_circles)
                delta_val = new_val - current_val
                
                if delta_val > 0 or rng.random() < math.exp(delta_val / T):
                    circles = new_circles
                    w = new_w
                    h = new_h
                    current_val = new_val
            else:
                # Move a circle
                i = rng.integers(0, N)
                new_circles = circles.copy()
                
                # Perturb position
                dx = rng.normal(0, 0.05)
                dy = rng.normal(0, 0.05)
                new_circles[i, 0] += dx
                new_circles[i, 1] += dy
                
                # Perturb radius
                dr = rng.normal(0, 0.01)
                new_r = new_circles[i, 2] + dr
                if new_r < 0.05:
                    new_r = 0.05
                new_circles[i, 2] = new_r
                
                # Resolve conflicts
                new_circles = resolve_conflicts(new_circles, w, h)
                
                if not is_valid(new_circles, w, h):
                    continue
                
                new_val = get_value(new_circles)
                delta_val = new_val - current_val
                
                if delta_val > 0 or rng.random() < math.exp(delta_val / T):
                    circles = new_circles
                    current_val = new_val
            
            # Update best
            if current_val > best_val:
                best_val = current_val
                best_circles = circles.copy()
                best_w = w
            
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end

    # Final cleanup: ensure the best result is strictly valid
    if best_circles is None:
        # Fallback: simple grid
        w = 1.4
        h = 0.6
        cells = []
        for i in range(3):
            for j in range(7):
                cx = (j + 0.5) * w / 7
                cy = (i + 0.5) * h / 3
                cells.append([cx, cy, 0.0999])
        best_circles = np.array(cells)
    else:
        # Final validation and slight shrink if needed to ensure strict validity
        # The evaluator might be strict. 
        # Our is_valid uses eps=1e-9.
        # Let's do a final resolve and check.
        w_final = best_w
        h_final = 2.0 - w_final
        final_circles = resolve_conflicts(best_circles, w_final, h_final)
        
        if is_valid(final_circles, w_final, h_final):
            best_circles = final_circles
        else:
            # If final resolve broke it (unlikely), fall back to a safe smaller grid
            w = 1.4
            h = 0.6
            cells = []
            for i in range(3):
                for j in range(7):
                    cx = (j + 0.5) * w / 7
                    cy = (i + 0.5) * h / 3
                    cells.append([cx, cy, 0.0999])
            best_circles = np.array(cells)

    return best_circles
# EVOLVE-BLOCK-END
