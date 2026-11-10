import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 620.0

    # Keich binary construction for initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Scale and shift for better starting point
    X0 = keich * 0.5 + 0.25
    Y0 = keich * 0.5 + 0.25

    # Precompute indices for vectorization
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    def compute_1d_union_length(positions, c, N):
        """Compute total length of union of intervals [pos[i] + c*i/N, pos[i] + c*i/N + 1/N] for i=0..N-1"""
        starts = positions + c * np.arange(N) / N
        ends = starts + 1.0 / N
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        # Merge intervals
        total = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] <= cur_end:
                cur_end = max(cur_end, e_sorted[k])
            else:
                total += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
        total += cur_end - cur_start
        return total

    def exact_volume(X, Y, N):
        """
        Compute the exact integral of Lx(c)*Ly(c) dc using adaptive quadrature
        on the break points where the union structure changes.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        def get_breaks(f):
            breaks = {0.0, 1.0}
            invN = 1.0 / N
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    for offset in [0.0, invN, -invN]:
                        denom = (i - j)
                        if denom != 0:
                            c_val = (f[j] - f[i] + offset) * N / denom
                            if 0.0 < c_val < 1.0:
                                breaks.add(c_val)
            return sorted(breaks)
        
        x_breaks = get_breaks(X)
        y_breaks = get_breaks(Y)
        all_breaks = sorted(set(x_breaks + y_breaks))
        
        total_vol = 0.0
        for k in range(len(all_breaks) - 1):
            c_mid = (all_breaks[k] + all_breaks[k+1]) / 2.0
            Lx = compute_1d_union_length(X, c_mid, N)
            Ly = compute_1d_union_length(Y, c_mid, N)
            total_vol += Lx * Ly * (all_breaks[k+1] - all_breaks[k])
            
        return total_vol

    def objective(params):
        if time.time() > deadline - 5:
            return 1e9
        X = params[:N]
        Y = params[N:]
        return exact_volume(X, Y, N)

    # Precompute a lookup table for 1D union length on a fine grid of c
    # This allows fast evaluation of volume for given X, Y by interpolating
    # or using the precomputed break points
    c_grid = np.linspace(0, 1, 1000)
    
    def precompute_1d_union_table(positions, N):
        """Precompute union length at all c_grid points for given positions"""
        starts_all = positions[None, :] + c_grid[:, None] * np.arange(N) / N
        ends_all = starts_all + 1.0 / N
        
        # For each c, compute union length
        lengths = np.empty(len(c_grid))
        for k in range(len(c_grid)):
            s = starts_all[k]
            e = ends_all[k]
            order = np.argsort(s)
            s_sorted = s[order]
            e_sorted = e[order]
            total = 0.0
            cur_start = s_sorted[0]
            cur_end = e_sorted[0]
            for m in range(1, N):
                if s_sorted[m] <= cur_end:
                    cur_end = max(cur_end, e_sorted[m])
                else:
                    total += cur_end - cur_start
                    cur_start = s_sorted[m]
                    cur_end = e_sorted[m]
            total += cur_end - cur_start
            lengths[k] = total
        return lengths

    def fast_volume(X, Y, N, Lx_table, Ly_table):
        """Compute volume using precomputed lookup tables"""
        # Integrate Lx(c) * Ly(c) dc using trapezoidal rule on the grid
        return np.trapz(Lx_table * Ly_table, c_grid)

    initial_params = np.concatenate([X0, Y0])
    
    best_params = initial_params.copy()
    best_score = objective(initial_params)

    # Phase 1: Gradient-based optimization with smooth surrogate
    def soft_max_smoothed_volume(params, beta=50.0):
        X = params[:N]
        Y = params[N:]
        
        def smooth_1d_union(f, c, beta):
            starts = f + c * np.arange(N) / N
            ends = starts + 1.0 / N
            order = np.argsort(starts)
            s_sorted = starts[order]
            e_sorted = ends[order]
            total = 0.0
            for k in range(N):
                if k == 0:
                    right_boundary = e_sorted[0]
                else:
                    sub = e_sorted[:k+1]
                    m = np.max(sub)
                    log_sum = m + np.log(np.sum(np.exp((sub - m) * beta)) / beta)
                    right_boundary = log_sum / beta
                if k < N - 1:
                    gap = np.logaddexp(0, -beta * (right_boundary - s_sorted[k+1])) / beta
                    total += (right_boundary - s_sorted[k]) - gap
                else:
                    total += right_boundary - s_sorted[k]
            return total
        
        n_quad = 50
        c_vals = np.linspace(0, 1, n_quad)
        total_vol = 0.0
        for c in c_vals:
            Lx = smooth_1d_union(X, c, beta)
            Ly = smooth_1d_union(Y, c, beta)
            total_vol += Lx * Ly * (1.0 / n_quad)
        return total_vol

    def objective_smooth(params):
        if time.time() > deadline - 5:
            return 1e9
        return soft_max_smoothed_volume(params, beta=50.0)

    if time.time() < deadline - 120:
        try:
            res_smooth = minimize(objective_smooth, initial_params, method='L-BFGS-B',
                                  options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            if res_smooth.fun < 1e8:
                exact_val = objective(res_smooth.x)
                if exact_val < best_score:
                    best_score = exact_val
                    best_params = res_smooth.x.copy()
        except Exception:
            pass

    # Phase 2: Discrete combinatorial search with precomputed lookup tables
    # Use a fine grid for each coordinate and evaluate all combinations
    # Since N=8, we have 16 parameters. We'll use a reduced grid for speed.
    grid_size = 20  # 20 points per coordinate
    grid_range = (0.0, 1.0)
    
    # Generate grid points
    grid = np.linspace(grid_range[0], grid_range[1], grid_size)
    
    # Precompute union length tables for all grid points
    # For efficiency, we'll only do this for a subset of promising regions
    # First, let's try a coarser search
    coarse_grid_size = 10
    coarse_grid = np.linspace(grid_range[0], grid_range[1], coarse_grid_size)
    
    # Evaluate volume for all combinations in coarse grid
    # This is 10^16 combinations, too many. Instead, use random sampling.
    rng_search = np.random.default_rng(12345)
    
    best_discrete = best_params.copy()
    best_discrete_score = best_score
    
    # Random sampling phase
    num_samples = 5000
    for s in range(num_samples):
        if time.time() > deadline - 60:
            break
        
        # Sample from a distribution centered around current best
        center = best_params
        sigma = 0.1 if s < num_samples // 2 else 0.05
        sample = center + rng_search.normal(0, sigma, size=2*N)
        
        # Clip to valid range
        sample = np.clip(sample, 0.0, 1.0)
        
        score = objective(sample)
        if score < best_discrete_score:
            best_discrete_score = score
            best_discrete = sample.copy()
            best_score = score
            best_params = sample.copy()
    
    # Local refinement around best discrete solution
    for restart in range(4):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            start = best_params.copy()
        else:
            rng_noise = np.random.default_rng(restart * 100 + 42)
            noise = rng_noise.normal(0, 0.02, size=2*N)
            start = best_params + noise
        
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            res_final = minimize(objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'disp': False})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
            
    X_opt = best_params[:N]
    Y_opt = best_params[N:]
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
