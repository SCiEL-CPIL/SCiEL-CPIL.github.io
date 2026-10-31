"""Circle packing in perimeter-4 rectangle: greedy sequential placement + aspect-ratio local search."""
import numpy as np
import time
import random
import math

N = 21
START = time.time()
DEADLINE = START + 135.0  # safety margin before 150s kill

def _time_ok():
    return time.time() < DEADLINE - 5

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)
    # clamp
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    # check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-8 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _max_radius_at(x, y, W, H, placed):
    """Compute the maximum radius for a circle at (x, y) given already-placed circles."""
    # Boundary constraint
    r_max = min(x, y, W - x, H - y)
    if r_max <= 0:
        return 0.0
    # Non-overlap with placed circles
    for (px, py, pr) in placed:
        d = math.hypot(x - px, y - py)
        if d > 0:
            r_max = min(r_max, d - pr)
        else:
            r_max = 0.0
    return max(r_max, 0.0)

def _find_best_spot(W, H, placed, grid_res=40):
    """
    Find the best (x, y) spot for the next circle by scanning a grid.
    Returns (x, y, r) or None if no valid spot.
    """
    best_r = -1
    best_x, best_y = 0.0, 0.0
    # Scan grid
    for i in range(grid_res + 1):
        for j in range(grid_res + 1):
            x = (i / grid_res) * W
            y = (j / grid_res) * H
            r = _max_radius_at(x, y, W, H, placed)
            if r > best_r:
                best_r = r
                best_x, best_y = x, y
    if best_r < 1e-6:
        return None
    return (best_x, best_y, best_r)

def _greedy_packing(W, H, order=None, grid_res=40):
    """
    Sequentially place N circles using greedy maximal-radius placement.
    order: list of indices indicating placement order (if None, use 0..N-1)
    Returns (x_array, sum_radii)
    """
    if order is None:
        order = list(range(N))
    placed = []
    cx_list = []
    cy_list = []
    r_list = []
    
    for idx in order:
        spot = _find_best_spot(W, H, placed, grid_res)
        if spot is None:
            # Fallback: place tiny circle at random corner
            x = random.uniform(0.01, 0.05)
            y = random.uniform(0.01, 0.05)
            r = 0.005
        else:
            x, y, r = spot
        placed.append((x, y, r))
        cx_list.append(x)
        cy_list.append(y)
        r_list.append(r)
    
    arr = np.array(list(zip(cx_list, cy_list, r_list)))
    s = _evaluate(arr, W, H)
    if s is None:
        s = 0.0
    return arr, s

def _local_aspect_search(best_arr, best_W, best_H, best_sum, n_trials=50):
    """
    Local search around the best aspect ratio by perturbing W and re-running greedy.
    Returns (best_arr, best_W, best_H, best_sum)
    """
    for _ in range(n_trials):
        if not _time_ok():
            break
        # Perturb W
        W_new = best_W + random.gauss(0, 0.05)
        W_new = max(0.3, min(1.7, W_new))  # keep H positive
        H_new = 2.0 - W_new
        if H_new < 0.1:
            continue
        
        # Use a random order to explore different packings
        order = list(range(N))
        random.shuffle(order)
        
        arr, s = _greedy_packing(W_new, H_new, order, grid_res=35)
        if s > best_sum:
            best_sum = s
            best_arr = arr
            best_W = W_new
            best_H = H_new
    
    return best_arr, best_W, best_H, best_sum

def _hill_climb_positions(arr, W, H, n_steps=200):
    """
    Simple hill-climbing: try small random moves on individual circle positions,
    recompute radii greedily, accept if sum improves.
    """
    cx = arr[:, 0].copy()
    cy = arr[:, 1].copy()
    r = arr[:, 2].copy()
    best_sum = _evaluate(arr, W, H) or 0.0
    best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
    
    for step in range(n_steps):
        if not _time_ok():
            break
        # Pick a random circle to move
        i = random.randint(0, N - 1)
        dx = random.gauss(0, 0.03)
        dy = random.gauss(0, 0.03)
        cx_new = cx.copy()
        cy_new = cy.copy()
        cx_new[i] = max(0, min(W, cx_new[i] + dx))
        cy_new[i] = max(0, min(H, cy_new[i] + dy))
        
        # Recompute all radii given the new positions
        r_new = np.zeros(N)
        for k in range(N):
            r_max = min(cx_new[k], cy_new[k], W - cx_new[k], H - cy_new[k])
            for j in range(N):
                if j != k:
                    d = math.hypot(cx_new[k] - cx_new[j], cy_new[k] - cy_new[j])
                    r_max = min(r_max, d - r[j])
            r_new[k] = max(r_max, 1e-6)
        
        # Check validity
        valid = True
        for a in range(N):
            for b in range(a + 1, N):
                d = math.hypot(cx_new[a] - cx_new[b], cy_new[a] - cy_new[b])
                if d + 1e-8 < r_new[a] + r_new[b]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            s = float(np.sum(r_new))
            if s > best_sum:
                best_sum = s
                best_cx, best_cy, best_r = cx_new.copy(), cy_new.copy(), r_new.copy()
                cx, cy, r = cx_new, cy_new, r_new
        else:
            # Reject, but still update positions for exploration
            cx, cy = cx_new, cy_new
            # Recompute r for current cx, cy
            for k in range(N):
                r_max = min(cx[k], cy[k], W - cx[k], H - cy[k])
                for j in range(N):
                    if j != k:
                        d = math.hypot(cx[k] - cx[j], cy[k] - cy[j])
                        r_max = min(r_max, d - r[j])
                r[k] = max(r_max, 1e-6)
    
    return np.column_stack([best_cx, best_cy, best_r]), best_sum

def construct():
    random.seed(42)
    np.random.seed(42)
    
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    # Phase 1: Greedy packing with diverse aspect ratios and orders
    Ws = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    for W in Ws:
        if not _time_ok():
            break
        H = 2.0 - W
        if H < 0.1:
            continue
        # Try a few different orders
        for trial in range(3):
            if not _time_ok():
                break
            order = list(range(N))
            random.shuffle(order)
            arr, s = _greedy_packing(W, H, order, grid_res=45)
            if s > best_sum:
                best_sum = s
                best_arr = arr
                best_W = W
                best_H = H
    
    # Phase 2: Local aspect ratio search around the best
    if best_arr is not None and _time_ok():
        best_arr, best_W, best_H, best_sum = _local_aspect_search(
            best_arr, best_W, best_H, best_sum, n_trials=80)
    
    # Phase 3: Hill-climbing on positions
    if best_arr is not None and _time_ok():
        best_arr, best_sum = _hill_climb_positions(best_arr, best_W, best_H, n_steps=300)
    
    # Phase 4: Another round of greedy + aspect search if time remains
    if _time_ok():
        # Try greedy again from current best aspect
        order = list(range(N))
        random.shuffle(order)
        arr, s = _greedy_packing(best_W, best_H, order, grid_res=50)
        if s > best_sum:
            best_sum = s
            best_arr = arr
        # More aspect search
        best_arr, best_W, best_H, best_sum = _local_aspect_search(
            best_arr, best_W, best_H, best_sum, n_trials=60)
        # More hill-climbing
        if _time_ok():
            best_arr, best_sum = _hill_climb_positions(best_arr, best_W, best_H, n_steps=200)
    
    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        arr, s = _greedy_packing(W, H, list(range(N)), grid_res=40)
        best_arr = arr
        best_W, best_H = W, H
        best_sum = s
    
    # Final clamping
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])
    
    # Final validation
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        # Fallback to grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return out
