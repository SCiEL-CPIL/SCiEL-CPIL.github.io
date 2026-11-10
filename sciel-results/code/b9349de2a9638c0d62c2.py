# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def _union_length(intervals):
    """
    Compute the length of the union of a list of intervals (start, end).
    intervals: list of (start, end) tuples.
    """
    if not intervals:
        return 0.0
    # Sort by start
    intervals = sorted(intervals, key=lambda x: x[0])
    total_len = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total_len += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total_len += current_end - current_start
    return total_len

def _compute_objective(params):
    """
    Compute the volume V = \int_0^1 L_x(c) L_y(c) dc.
    params: array of 16 values, first 8 are x_0..x_7, next 8 are y_0..y_7.
    """
    N = 8
    x = params[:8]
    y = params[8:16]
    
    # Find breakpoints for x
    break_x = [0.0, 1.0]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # x_i + c*i/N = x_j + c*j/N + 1/N
            denom = (i - j) / N
            if abs(denom) > 1e-12:
                c = (x[j] - x[i] + 1.0/N) / denom
                if 0.0 <= c <= 1.0:
                    break_x.append(c)
            # x_i + c*i/N + 1/N = x_j + c*j/N
            if abs(denom) > 1e-12:
                c = (x[j] - x[i]) / denom
                if 0.0 <= c <= 1.0:
                    break_x.append(c)
    
    # Find breakpoints for y
    break_y = [0.0, 1.0]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = (i - j) / N
            if abs(denom) > 1e-12:
                c = (y[j] - y[i] + 1.0/N) / denom
                if 0.0 <= c <= 1.0:
                    break_y.append(c)
            if abs(denom) > 1e-12:
                c = (y[j] - y[i]) / denom
                if 0.0 <= c <= 1.0:
                    break_y.append(c)
    
    # Combine and sort breakpoints
    all_breaks = sorted(set(break_x + break_y))
    # Ensure 0 and 1 are in the list
    if all_breaks[0] != 0.0:
        all_breaks.insert(0, 0.0)
    if all_breaks[-1] != 1.0:
        all_breaks.append(1.0)
    
    # Integrate piecewise
    V = 0.0
    for k in range(len(all_breaks) - 1):
        c0 = all_breaks[k]
        c1 = all_breaks[k+1]
        c_mid = (c0 + c1) / 2.0
        
        def get_lengths(c):
            ix = [(x[i] + c*i/N, x[i] + c*i/N + 1.0/N) for i in range(N)]
            iy = [(y[j] + c*j/N, y[j] + c*j/N + 1.0/N) for j in range(N)]
            return _union_length(ix), _union_length(iy)
        
        lx0, ly0 = get_lengths(c0)
        lxm, lym = get_lengths(c_mid)
        lx1, ly1 = get_lengths(c1)
        
        f0 = lx0 * ly0
        fm = lxm * lym
        f1 = lx1 * ly1
        
        # Integral of quadratic f(c) from c0 to c1
        # Using Simpson's rule which is exact for quadratics
        V += (c1 - c0) / 6.0 * (f0 + 4*fm + f1)
        
    return V

def _compute_objective_1d(params):
    """
    Compute the 1D cost: \int_0^1 L(c)^2 dc for a single dimension.
    params: array of 8 values.
    """
    N = 8
    v = params
    
    # Find breakpoints
    breaks = [0.0, 1.0]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = (i - j) / N
            if abs(denom) > 1e-12:
                c = (v[j] - v[i] + 1.0/N) / denom
                if 0.0 <= c <= 1.0:
                    breaks.append(c)
                c = (v[j] - v[i]) / denom
                if 0.0 <= c <= 1.0:
                    breaks.append(c)
    
    breaks = sorted(set(breaks))
    if breaks[0] != 0.0:
        breaks.insert(0, 0.0)
    if breaks[-1] != 1.0:
        breaks.append(1.0)
    
    cost = 0.0
    for k in range(len(breaks) - 1):
        c0 = breaks[k]
        c1 = breaks[k+1]
        c_mid = (c0 + c1) / 2.0
        
        def get_len(c):
            iv = [(v[i] + c*i/N, v[i] + c*i/N + 1.0/N) for i in range(N)]
            return _union_length(iv)
        
        l0 = get_len(c0)
        lm = get_len(c_mid)
        l1 = get_len(c1)
        
        # Integral of L(c)^2 using Simpson's rule
        cost += (c1 - c0) / 6.0 * (l0*l0 + 4*lm*lm + l1*l1)
        
    return cost

def _compute_objective_2d(params):
    """
    Compute the 2D volume as product of two 1D costs.
    params: array of 16 values, first 8 are x, next 8 are y.
    """
    x = params[:8]
    y = params[8:16]
    cost_x = _compute_objective_1d(x)
    cost_y = _compute_objective_1d(y)
    return cost_x * cost_y

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Initial params: x = keich, y = keich
    x0 = keich.copy()
    y0 = keich.copy()
    params0 = np.concatenate([x0, y0])
    
    # Optimization
    # We want to minimize V.
    # L-BFGS-B
    options = {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
    
    start_time = time.time()
    deadline = 600.0 - 15.0  # Safety margin
    
    best_params = params0
    best_obj = _compute_objective_2d(params0)
    
    # Try multiple restarts with different perturbations
    n_restarts = 10
    rng = np.random.default_rng(42)
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            params_curr = params0.copy()
        else:
            # Perturb the best solution found so far
            sigma = 0.05 / (r + 1)
            perturb = rng.normal(0, sigma, 16)
            params_curr = best_params + perturb
        
        try:
            res = scipy.optimize.minimize(_compute_objective_2d, params_curr, method='L-BFGS-B', options=options)
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        options_fine = {'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
        try:
            res = scipy.optimize.minimize(_compute_objective_2d, best_params, method='L-BFGS-B', options=options_fine)
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    x_opt = best_params[:8]
    y_opt = best_params[8:16]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
