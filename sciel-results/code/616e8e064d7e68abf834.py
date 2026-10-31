"""Circle packing in perimeter-4 rectangle: Simulated Annealing + Local Expansion."""
import numpy as np
import time
import random

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _max_radius(cx, cy, r, W, H, i):
    """Max radius for circle i given others fixed."""
    r_wall = min(cx[i], W - cx[i], cy[i], H - cy[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = cx[i] - cx[j]
        dy = cy[i] - cy[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _expand_all(cx, cy, r, W, H, max_rounds=50):
    """Iteratively expand each circle to its max possible radius."""
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for i in range(N):
            r_new = _max_radius(cx, cy, r, W, H, i)
            if r_new > r[i] + 1e-10:
                r[i] = r_new
                improved = True
    return r


def _local_search(cx, cy, r, W, H, time_budget, seed=0):
    """
    Simulated annealing local search to improve sum of radii.
    Moves: perturb positions, or grow/shrink a radius.
    """
    rng = np.random.default_rng(seed)
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    current_val = float(np.sum(r))
    best_val = current_val
    best_state = (cx.copy(), cy.copy(), r.copy())
    
    T = 0.01
    T_min = 1e-6
    alpha = 0.9995
    start_time = time.time()
    
    n_iter = 0
    while T > T_min:
        if time.time() > time_budget:
            break
        n_iter += 1
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Perturb position of one circle
            i = rng.integers(N)
            dx = rng.normal(0, 0.02)
            dy = rng.normal(0, 0.02)
            new_x = cx[i] + dx
            new_y = cy[i] + dy
            # Clamp to walls
            new_x = np.clip(new_x, r[i], W - r[i])
            new_y = np.clip(new_y, r[i], H - r[i])
            
            # Check validity
            valid = True
            for j in range(N):
                if i == j:
                    continue
                dxj = new_x - cx[j]
                dyj = new_y - cy[j]
                if dxj * dxj + dyj * dyj < (r[i] + r[j]) ** 2 - 1e-9:
                    valid = False
                    break
            
            if valid:
                cx[i] = new_x
                cy[i] = new_y
                # Try to expand radius after move
                r_new = _max_radius(cx, cy, r, W, H, i)
                if r_new > r[i]:
                    r[i] = r_new
                new_val = float(np.sum(r))
                delta = new_val - current_val
                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_val = new_val
                    if current_val > best_val:
                        best_val = current_val
                        best_state = (cx.copy(), cy.copy(), r.copy())
                else:
                    # Revert
                    cx[i] = new_x - dx
                    cy[i] = new_y - dy
                    # Restore radius approximately (recompute)
                    r[i] = _max_radius(cx, cy, r, W, H, i)
                    current_val = float(np.sum(r))
            else:
                # Revert
                pass
                
        elif move_type < 0.8:
            # Try to grow a random circle's radius
            i = rng.integers(N)
            r_new = _max_radius(cx, cy, r, W, H, i)
            if r_new > r[i] + 1e-8:
                r[i] = r_new
                new_val = float(np.sum(r))
                delta = new_val - current_val
                if delta > 0:
                    current_val = new_val
                    if current_val > best_val:
                        best_val = current_val
                        best_state = (cx.copy(), cy.copy(), r.copy())
                else:
                    r[i] = _max_radius(cx, cy, r, W, H, i)
                    current_val = float(np.sum(r))
        else:
            # Perturb aspect ratio slightly (change W)
            dW = rng.normal(0, 0.01)
            new_W = W + dW
            if new_W < 0.3 or new_W > 1.7:
                continue
            new_H = 2.0 - new_W
            
            # Check if all circles fit in new rectangle
            valid = True
            for i in range(N):
                if cx[i] - r[i] < -1e-9 or cx[i] + r[i] > new_W + 1e-9:
                    valid = False
                    break
                if cy[i] - r[i] < -1e-9 or cy[i] + r[i] > new_H + 1e-9:
                    valid = False
                    break
            
            if valid:
                W = new_W
                H = new_H
                # Expand all radii
                r = _expand_all(cx, cy, r, W, H, max_rounds=10)
                new_val = float(np.sum(r))
                delta = new_val - current_val
                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_val = new_val
                    if current_val > best_val:
                        best_val = current_val
                        best_state = (cx.copy(), cy.copy(), r.copy(), W, H)
                    else:
                        pass
                else:
                    W = W - dW
                    H = 2.0 - W
                    r = _expand_all(cx, cy, r, W, H, max_rounds=10)
                    current_val = float(np.sum(r))
            else:
                pass
        
        T *= alpha
    
    # Return best state
    if len(best_state) == 4:
        return best_state[0], best_state[1], best_state[2], W
    else:
        return best_state[0], best_state[1], best_state[2], W


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_val = 0.0
    best_cx = None
    best_cy = None
    best_r = None
    best_W = 1.0
    best_H = 1.0
    
    # Try different initial aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67, 0.5, 2.0, 0.4, 1.5, 0.67]
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 5:
            break
        
        h0 = 2.0 - w0
        if h0 < 0.2:
            continue
        
        # Initialize with grid
        side = int(np.ceil(np.sqrt(N)))
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row, col = divmod(i, side)
            cx[i] = (col + 0.5) / side * w0
            cy[i] = (row + 0.5) / side * h0
            r[i] = min(w0 / side, h0 / side) * 0.45
        
        # Expand all radii
        r = _expand_all(cx, cy, r, w0, h0, max_rounds=100)
        
        val = float(np.sum(r))
        if val > best_val:
            best_val = val
            best_cx = cx.copy()
            best_cy = cy.copy()
            best_r = r.copy()
            best_W = w0
            best_H = h0
        
        # Local search
        if time.time() > deadline - 10:
            break
        
        time_budget = deadline - 5
        cx_ls, cy_ls, r_ls, W_ls = _local_search(cx, cy, r, w0, h0, time_budget, seed=int(w0 * 1000))
        
        if _is_valid(cx_ls, cy_ls, r_ls, W_ls, 2.0 - W_ls):
            val_ls = float(np.sum(r_ls))
            if val_ls > best_val:
                best_val = val_ls
                best_cx = cx_ls
                best_cy = cy_ls
                best_r = r_ls
                best_W = W_ls
                best_H = 2.0 - W_ls
    
    # Final expansion pass on best result
    if best_cx is not None:
        best_r = _expand_all(best_cx, best_cy, best_r, best_W, best_H, max_rounds=200)
        best_val = float(np.sum(best_r))
    
    if best_cx is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.column_stack((best_cx, best_cy, best_r))
