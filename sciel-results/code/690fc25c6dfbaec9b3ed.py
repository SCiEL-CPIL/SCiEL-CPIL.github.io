import numpy as np
import time
from scipy.optimize import minimize

def _get_breakpoints(X, N=8):
    """
    Calculate all c in [0, 1] where the order of endpoints changes for the 1D intervals.
    """
    breakpoints = set([0.0, 1.0])
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = i - j
            # L_i = L_j
            c1 = N * (X[j] - X[i]) / denom
            if 0.0 <= c1 <= 1.0:
                breakpoints.add(c1)
            # L_i = R_j
            c2 = N * (X[j] + 1.0 / N - X[i]) / denom
            if 0.0 <= c2 <= 1.0:
                breakpoints.add(c2)
            # R_i = L_j
            c3 = N * (X[j] - X[i] - 1.0 / N) / denom
            if 0.0 <= c3 <= 1.0:
                breakpoints.add(c3)
            # R_i = R_j
            c4 = N * (X[j] + 1.0 / N - X[i] - 1.0 / N) / denom
            if 0.0 <= c4 <= 1.0:
                breakpoints.add(c4)
    return sorted(list(breakpoints))

def _union_length_1d(X, c, N=8):
    """
    Calculate the length of the union of intervals [X_i + i*c/N, X_i + i*c/N + 1/N]
    """
    intervals = []
    for i in range(N):
        start = X[i] + i * c / N
        end = start + 1.0 / N
        intervals.append((start, end))
    
    # Sort by start
    intervals.sort(key=lambda x: x[0])
    
    if not intervals:
        return 0.0
        
    total_length = 0.0
    current_start, current_end = intervals[0]
    
    for start, end in intervals[1:]:
        if start > current_end:
            total_length += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
            
    total_length += current_end - current_start
    return total_length

def _exact_volume(params, N=8):
    """
    Calculate exact union volume for 3D Kakeya set with separable X and Y.
    params: array of length 2*N. First N are X, next N are Y.
    """
    X = params[:N]
    Y = params[N:]
    
    # Get breakpoints for X and Y
    bp_x = _get_breakpoints(X, N)
    bp_y = _get_breakpoints(Y, N)
    
    # Combine and sort all breakpoints
    all_bps = sorted(set(bp_x) | set(bp_y))
    
    # Ensure 0 and 1 are included
    if all_bps[0] > 0.0:
        all_bps = [0.0] + all_bps
    if all_bps[-1] < 1.0:
        all_bps = all_bps + [1.0]
        
    volume = 0.0
    for k in range(len(all_bps) - 1):
        c_start = all_bps[k]
        c_end = all_bps[k+1]
        
        # Use midpoint for evaluation
        c_mid = (c_start + c_end) / 2.0
        
        len_x = _union_length_1d(X, c_mid, N)
        len_y = _union_length_1d(Y, c_mid, N)
        
        volume += (c_end - c_start) * len_x * len_y
        
    return volume

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    N = 8
    
    # Initial guesses
    # 1. Symmetric centered
    X0 = -np.arange(N) / (2 * N)
    Y0 = -np.arange(N) / (2 * N)
    init1 = np.concatenate([X0, Y0])
    
    # 2. Linear spread
    X1 = np.linspace(-0.5, 0.5, N)
    Y1 = np.linspace(-0.5, 0.5, N)
    init2 = np.concatenate([X1, Y1])
    
    # 3. Random
    rng = np.random.default_rng(42)
    init3 = rng.uniform(-0.5, 0.5, 2 * N)
    
    best_params = init1
    best_vol = _exact_volume(init1, N)
    
    inits = [init1, init2, init3]
    
    for init in inits:
        if time.time() > deadline - 30:
            break
            
        try:
            res = minimize(_exact_volume, init, args=(N,), method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Refine with tighter tolerances if time permits
    if time.time() < deadline - 10:
        try:
            res2 = minimize(_exact_volume, best_params, args=(N,), method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res2.fun < best_vol:
                best_params = res2.x
        except Exception:
            pass
            
    # Construct the 64x2 array
    X_opt = best_params[:N]
    Y_opt = best_params[N:]
    
    positions = np.zeros((N * N, 2))
    for i in range(N):
        for j in range(N):
            k = i * N + j
            positions[k, 0] = X_opt[i]
            positions[k, 1] = Y_opt[j]
            
    return positions
