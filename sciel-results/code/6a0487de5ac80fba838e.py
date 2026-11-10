# EVOLVE-BLOCK-START
import numpy as np


def _exact_1d_union_integral(base_pos, N):
    """
    Compute the exact integral of the union measure over c in [0,1] for N intervals.
    Interval i: [base_pos[i] + c*i/N, base_pos[i] + 1/N + c*i/N]
    """
    n = len(base_pos)
    breakpoints = [0.0, 1.0]
    
    for i in range(n):
        for j in range(i + 1, n):
            di = i - j
            if di == 0:
                continue
            c1 = N * (base_pos[j] - base_pos[i]) / di
            c2 = N * (base_pos[j] + 1.0/N - base_pos[i]) / di
            c3 = N * (base_pos[j] - base_pos[i] - 1.0/N) / di
            
            for c_val in [c1, c2, c3]:
                if 0.0 < c_val < 1.0:
                    breakpoints.append(c_val)
    
    breakpoints = sorted(set(breakpoints))
    
    total_integral = 0.0
    for k in range(len(breakpoints) - 1):
        c_lo = breakpoints[k]
        c_hi = breakpoints[k + 1]
        c_mid = 0.5 * (c_lo + c_hi)
        
        lefts = base_pos + c_mid * np.arange(n) / N
        rights = lefts + 1.0 / N
        
        order = np.argsort(lefts)
        l_sorted = lefts[order]
        r_sorted = rights[order]
        
        merged_len = 0.0
        cl = l_sorted[0]
        cr = r_sorted[0]
        for m in range(1, n):
            if l_sorted[m] <= cr:
                cr = max(cr, r_sorted[m])
            else:
                merged_len += cr - cl
                cl = l_sorted[m]
                cr = r_sorted[m]
        merged_len += cr - cl
        
        total_integral += merged_len * (c_hi - c_lo)
    
    return total_integral

def _simulate_annealing_1d(N, n_iter=2000, T_init=0.01, T_end=1e-6, rng_seed=42):
    """
    Optimize 1-D base positions using Simulated Annealing.
    """
    rng = np.random.default_rng(rng_seed)
    
    # Initial guess: Randomized folding or simple random
    # Let's start with a random permutation of a folded grid to break symmetry
    base_fold = np.zeros(N)
    for i in range(N):
        base_fold[i] = (i % 2) * (1.0 / (2 * N))
    # Add some noise
    pos = base_fold + rng.uniform(-0.1, 0.1, N)
    pos -= pos.min()
    
    current_vol = _exact_1d_union_integral(pos, N)
    best_pos = pos.copy()
    best_vol = current_vol
    
    T = T_init
    cooling = (T_end / T_init) ** (1.0 / n_iter)
    
    for it in range(n_iter):
        # Perturb
        delta = rng.normal(0, T * 10, N)
        new_pos = pos + delta
        new_pos -= new_pos.min()
        
        new_vol = _exact_1d_union_integral(new_pos, N)
        
        # Acceptance criterion
        if new_vol < current_vol or rng.random() < np.exp(-(new_vol - current_vol) / T):
            pos = new_pos
            current_vol = new_vol
            if current_vol < best_vol:
                best_vol = current_vol
                best_pos = pos.copy()
        
        T *= cooling
        
    return best_pos, best_vol

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with N = 8.
    
    Uses Simulated Annealing to optimize 1D base positions independently for x and y.
    """
    N = 8
    
    # Run multiple restarts to find a better global minimum
    best_x = None
    best_vol_x = float('inf')
    best_y = None
    best_vol_y = float('inf')
    
    for seed in [1, 2, 3, 4, 5]:
        x_1d, vol_x = _simulate_annealing_1d(N, n_iter=1000, T_init=0.05, T_end=1e-6, rng_seed=seed)
        if vol_x < best_vol_x:
            best_vol_x = vol_x
            best_x = x_1d
            
        y_1d, vol_y = _simulate_annealing_1d(N, n_iter=1000, T_init=0.05, T_end=1e-6, rng_seed=seed + 10)
        if vol_y < best_vol_y:
            best_vol_y = vol_y
            best_y = y_1d
            
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
