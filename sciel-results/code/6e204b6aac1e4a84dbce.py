import numpy as np
import time

def compute_union_area_at_c(c, x_positions, y_positions, N=8):
    """
    Compute the area of the union of rectangles at height c.
    x_positions: array of shape (N,) for x-base positions (one per i)
    y_positions: array of shape (N,) for y-base positions (one per j)
    """
    i = np.arange(N)
    x_left = x_positions + c * i / N
    x_right = x_left + 1.0 / N
    j = np.arange(N)
    y_left = y_positions + c * j / N
    y_right = y_left + 1.0 / N
    
    x_edges = np.unique(np.concatenate([x_left, x_right]))
    y_edges = np.unique(np.concatenate([y_left, y_right]))
    
    area = 0.0
    for px in range(len(x_edges) - 1):
        x0 = x_edges[px]
        x1 = x_edges[px+1]
        if x1 <= x0:
            continue
        cover_i = np.where((x_left <= x0) & (x_right >= x1))[0]
        if len(cover_i) == 0:
            continue
        for py in range(len(y_edges) - 1):
            y0 = y_edges[py]
            y1 = y_edges[py+1]
            if y1 <= y0:
                continue
            cover_j = np.where((y_left <= y0) & (y_right >= y1))[0]
            if len(cover_j) == 0:
                continue
            area += (x1 - x0) * (y1 - y0)
            
    return area

def compute_volume(x_positions, y_positions, N=8, num_c=20):
    """
    Estimate the volume by integrating the area over c in [0,1].
    """
    c_vals = np.linspace(0, 1, num_c, endpoint=False) + 0.5 / num_c
    areas = [compute_union_area_at_c(c, x_positions, y_positions, N) for c in c_vals]
    return np.mean(areas)

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    max_time = 590.0  # Safety margin

    # Initial Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    x_positions = keich.copy()
    y_positions = keich.copy()
    
    best_vol = compute_volume(x_positions, y_positions)
    best_x = x_positions.copy()
    best_y = y_positions.copy()
    
    rng = np.random.default_rng(42)
    
    # Coordinate descent with adaptive steps
    def local_search(x_init, y_init, max_iters=2000):
        x = x_init.copy()
        y = y_init.copy()
        curr_vol = compute_volume(x, y)
        best_x_local = x.copy()
        best_y_local = y.copy()
        best_vol_local = curr_vol
        step = 0.05
        
        for it in range(max_iters):
            if time.time() - start_time > max_time - 10:
                break
            
            improved = False
            # Try moving each x coordinate
            for i in range(N):
                for sign in [-1, 1]:
                    x_new = x.copy()
                    x_new[i] += sign * step
                    v = compute_volume(x_new, y)
                    if v < curr_vol:
                        x = x_new
                        curr_vol = v
                        improved = True
                        if curr_vol < best_vol_local:
                            best_vol_local = curr_vol
                            best_x_local = x.copy()
                            best_y_local = y.copy()
                        break
            
            # Try moving each y coordinate
            for j in range(N):
                for sign in [-1, 1]:
                    y_new = y.copy()
                    y_new[j] += sign * step
                    v = compute_volume(x, y_new)
                    if v < curr_vol:
                        y = y_new
                        curr_vol = v
                        improved = True
                        if curr_vol < best_vol_local:
                            best_vol_local = curr_vol
                            best_x_local = x.copy()
                            best_y_local = y.copy()
                        break
            
            if not improved:
                # Try smaller step
                step *= 0.7
                if step < 1e-6:
                    break
            else:
                # Keep step or increase slightly
                step = min(step * 1.1, 0.1)
        
        return best_x_local, best_y_local, best_vol_local
    
    # Run local search from seed
    x_cand, y_cand, vol_cand = local_search(x_positions, y_positions)
    if vol_cand < best_vol:
        best_x = x_cand
        best_y = y_cand
        best_vol = vol_cand
    
    # Random restarts
    restarts = 0
    while time.time() - start_time < max_time - 10:
        x_init = rng.uniform(-0.5, 0.5, N)
        y_init = rng.uniform(-0.5, 0.5, N)
        
        x_cand, y_cand, vol_cand = local_search(x_init, y_init, max_iters=500)
        if vol_cand < best_vol:
            best_x = x_cand
            best_y = y_cand
            best_vol = vol_cand
        
        restarts += 1
        if restarts > 200:  # Safety
            break
    
    # Final polish with smaller steps
    x_positions = best_x.copy()
    y_positions = best_y.copy()
    final_vol = best_vol
    
    for _ in range(100):
        if time.time() - start_time > max_time - 5:
            break
        
        improved = False
        for i in range(N):
            for d in [-0.01, -0.005, 0.005, 0.01]:
                x_new = x_positions.copy()
                x_new[i] += d
                v = compute_volume(x_new, y_positions)
                if v < final_vol:
                    x_positions = x_new
                    final_vol = v
                    improved = True
                    break
        
        for j in range(N):
            for d in [-0.01, -0.005, 0.005, 0.01]:
                y_new = y_positions.copy()
                y_new[j] += d
                v = compute_volume(x_positions, y_new)
                if v < final_vol:
                    y_positions = y_new
                    final_vol = v
                    improved = True
                    break
        
        if not improved:
            break
        
        best_x = x_positions.copy()
        best_y = y_positions.copy()
        best_vol = final_vol
    
    x_positions = best_x
    y_positions = best_y
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_positions[i_flat]
    base_y = y_positions[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
