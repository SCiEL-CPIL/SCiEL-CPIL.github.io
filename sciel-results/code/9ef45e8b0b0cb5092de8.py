import numpy as np
import time
import sys
from scipy.optimize import minimize

def compute_2d_volume_fast(positions_2d):
    """
    Fast exact computation of the union volume of 64 sheared tubes.
    positions_2d: (64, 2) array of (x_k, y_k).
    Tube k at height c: [x_k + c*(k//8)/8, ... + 1/8] x [y_k + c*(k%8)/8, ... + 1/8]
    
    We find all c-breakpoints where the combinatorial structure of the union
    of axis-aligned rectangles changes, then integrate the area exactly.
    """
    N = 8
    base_x = positions_2d[:, 0]
    base_y = positions_2d[:, 1]
    
    # Precompute i and j for each tube
    i_arr = np.arange(64) // N
    j_arr = np.arange(64) % N
    
    # Find breakpoints in c where any two vertical or horizontal edges coincide
    breakpoints = [0.0, 1.0]
    
    # For efficiency, only check pairs with different i or j
    for k1 in range(64):
        i1 = i_arr[k1]
        j1 = j_arr[k1]
        for k2 in range(k1 + 1, 64):
            i2 = i_arr[k2]
            j2 = j_arr[k2]
            
            # Vertical edges (x-direction)
            di = (i1 - i2) / 8.0
            if abs(di) > 1e-12:
                # x_min(k1,c) = x_min(k2,c): base_x[k1] + c*i1/8 = base_x[k2] + c*i2/8
                c_val = (base_x[k2] - base_x[k1]) / di
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)
                # x_min(k1,c) = x_max(k2,c)
                c_val = (base_x[k2] + 1.0/8.0 - base_x[k1]) / di
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)
                # x_max(k1,c) = x_min(k2,c)
                c_val = (base_x[k2] - base_x[k1] - 1.0/8.0) / di
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)
            
            # Horizontal edges (y-direction)
            dj = (j1 - j2) / 8.0
            if abs(dj) > 1e-12:
                # y_min(k1,c) = y_min(k2,c)
                c_val = (base_y[k2] - base_y[k1]) / dj
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)
                # y_min(k1,c) = y_max(k2,c)
                c_val = (base_y[k2] + 1.0/8.0 - base_y[k1]) / dj
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)
                # y_max(k1,c) = y_min(k2,c)
                c_val = (base_y[k2] - base_y[k1] - 1.0/8.0) / dj
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)
    
    breakpoints = sorted(set(breakpoints))
    
    total_volume = 0.0
    
    for k in range(len(breakpoints) - 1):
        c_start = breakpoints[k]
        c_end = breakpoints[k + 1]
        c_mid = (c_start + c_end) / 2.0
        
        # Compute union area at c_mid using sweep line
        x_edges = []
        for kk in range(64):
            x_min = base_x[kk] + c_mid * i_arr[kk] / 8.0
            x_max = x_min + 1.0 / 8.0
            y_min = base_y[kk] + c_mid * j_arr[kk] / 8.0
            y_max = y_min + 1.0 / 8.0
            x_edges.append((x_min, 1, y_min, y_max))
            x_edges.append((x_max, -1, y_min, y_max))
        
        x_edges.sort(key=lambda e: e[0])
        
        # Sweep line over x
        area = 0.0
        cur_x = x_edges[0][0]
        active_y = []
        
        for x, delta, y_min, y_max in x_edges:
            dx = x - cur_x
            if dx > 0:
                if active_y:
                    active_y.sort()
                    merged_y = []
                    cy_min, cy_max = active_y[0]
                    for ym, yM in active_y[1:]:
                        if ym <= cy_max:
                            if yM > cy_max:
                                cy_max = yM
                        else:
                            merged_y.append((cy_min, cy_max))
                            cy_min, cy_max = ym, yM
                    merged_y.append((cy_min, cy_max))
                    y_len = sum(yM - ym for ym, yM in merged_y)
                else:
                    y_len = 0.0
                area += y_len * dx
            cur_x = x
            
            if delta == 1:
                active_y.append((y_min, y_max))
            else:
                # Remove matching interval
                for idx in range(len(active_y) - 1, -1, -1):
                    if active_y[idx][0] == y_min and active_y[idx][1] == y_max:
                        active_y.pop(idx)
                        break
        
        total_volume += area * (c_end - c_start)
    
    return total_volume


def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0  # Safety margin
    
    N = 8
    rng = np.random.default_rng(42)
    
    # Initial Keich-inspired 2D configuration
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich_1d = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich_1d[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Create initial 2D positions from Keich 1D
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich_1d[i_flat]
    base_y = keich_1d[j_flat]
    initial_pos = np.stack([base_x, base_y], axis=1)
    
    # Compute initial volume
    best_volume = compute_2d_volume_fast(initial_pos)
    best_positions = initial_pos.copy()
    
    # Multi-start optimization with Nelder-Mead and L-BFGS-B
    def objective_2d(flat_pos):
        """Objective function that takes flat array and returns volume."""
        pos_2d = flat_pos.reshape(64, 2)
        try:
            vol = compute_2d_volume_fast(pos_2d)
            if not np.isfinite(vol) or vol <= 0:
                return 1e10
            return vol
        except:
            return 1e10
    
    # Try multiple random restarts
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() - start_time > deadline - 60.0:
            break
        
        # Generate random initial position near the Keich configuration
        if restart == 0:
            current_pos = initial_pos.copy()
        else:
            current_pos = initial_pos + rng.normal(0, 0.3, (64, 2))
        
        # Nelder-Mead local optimization
        try:
            res_nm = minimize(objective_2d, current_pos.flatten(), method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                     'adaptive': True})
            if res_nm.success and res_nm.fun < best_volume:
                best_volume = res_nm.fun
                best_positions = res_nm.x.reshape(64, 2).copy()
        except:
            pass
        
        # L-BFGS-B local optimization from the best Nelder-Mead result
        if time.time() - start_time > deadline - 30.0:
            break
        
        try:
            start_lbfgs = best_positions.flatten()
            res_lbfgs = minimize(objective_2d, start_lbfgs, method='L-BFGS-B',
                                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6})
            if res_lbfgs.success and res_lbfgs.fun < best_volume:
                best_volume = res_lbfgs.fun
                best_positions = res_lbfgs.x.reshape(64, 2).copy()
        except:
            pass
        
        # Additional Nelder-Mead polish
        if time.time() - start_time > deadline - 15.0:
            break
        
        try:
            res_polish = minimize(objective_2d, best_positions.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                         'adaptive': True})
            if res_polish.success and res_polish.fun < best_volume:
                best_volume = res_polish.fun
                best_positions = res_polish.x.reshape(64, 2).copy()
        except:
            pass
    
    # Final validation
    if not np.all(np.isfinite(best_positions)) or best_volume <= 0:
        # Fallback to initial configuration
        best_positions = initial_pos
        best_volume = compute_2d_volume_fast(best_positions)
    
    return best_positions


if __name__ == "__main__":
    res = kakeya_needle_3d8()
    print(res.shape)
    print("Volume:", compute_2d_volume_fast(res))
