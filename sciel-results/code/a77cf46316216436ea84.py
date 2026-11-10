# EVOLVE-BLOCK-START
import numpy as np
import time


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = keich[i_flat].copy()
    base_y = keich[j_flat].copy()
    positions = np.stack([base_x, base_y], axis=1)
    
    start_time = time.time()
    deadline = start_time + 570.0

    i_k = np.arange(64) // N
    j_k = np.arange(64) % N

    def eval_volume_exact(positions, num_c=100):
        x_pos = positions[:, 0]
        y_pos = positions[:, 1]
        X0 = x_pos.min()
        X1 = (x_pos + (i_k + 1) / N).max()
        Y0 = y_pos.min()
        Y1 = (y_pos + (j_k + 1) / N).max()
        
        if (X1 - X0) * (Y1 - Y0) <= 1e-9:
            return float('inf')
        
        c_vals = np.linspace(0, 1, num_c)
        total_area = 0.0
        for c in c_vals:
            x_lo = x_pos + c * i_k / N
            x_hi = x_lo + 1.0 / N
            y_lo = y_pos + c * j_k / N
            y_hi = y_lo + 1.0 / N
            
            x_int = sorted(zip(x_lo, x_hi))
            x_merged = [list(x_int[0])]
            for lo, hi in x_int[1:]:
                if lo <= x_merged[-1][1]:
                    x_merged[-1][1] = max(x_merged[-1][1], hi)
                else:
                    x_merged.append([lo, hi])
            x_len = sum(hi - lo for lo, hi in x_merged)
            
            y_int = sorted(zip(y_lo, y_hi))
            y_merged = [list(y_int[0])]
            for lo, hi in y_int[1:]:
                if lo <= y_merged[-1][1]:
                    y_merged[-1][1] = max(y_merged[-1][1], hi)
                else:
                    y_merged.append([lo, hi])
            y_len = sum(hi - lo for lo, hi in y_merged)
            
            total_area += x_len * y_len
        
        return total_area / num_c

    current_vol = eval_volume_exact(positions, 100)
    best_positions = positions.copy()
    best_vol = current_vol
    
    rng = np.random.default_rng(123)
    step = 0.05
    no_improve = 0
    
    while time.time() < deadline:
        improved = False
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            k = rng.integers(0, 64)
            coord = rng.integers(0, 2)
            delta = rng.uniform(-step, step)
            
            trial = best_positions.copy()
            trial[k, coord] += delta
            vol = eval_volume_exact(trial, 50)
            
            if vol < best_vol - 1e-10:
                best_vol = vol
                best_positions = trial
                improved = True
                no_improve = 0
        if not improved:
            no_improve += 1
            step *= 0.8
            if step < 1e-4 or no_improve > 5:
                break
    
    return best_positions


# EVOLVE-BLOCK-END
