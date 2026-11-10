import numpy as np
import time
from scipy.optimize import differential_evolution

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # Deadline guard
    start_time = time.time()
    timeout = 580  # Leave margin for overhead

    # Keich construction for initial guess
    def keich_positions(N):
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return keich

    u_init = keich_positions(N)
    v_init = keich_positions(N)
    
    # Variables: u_0..u_7, v_0..v_7 (16 variables)
    # Bounds: The positions can be anywhere. 
    # To keep the box [X0, X1] small, we should keep them close to each other.
    # The shear shifts them by at most 7/8.
    # Let's bound them in [-1, 1] relative to each other.
    # Actually, we can fix one variable to 0 to remove translation invariance?
    # The volume depends on relative positions.
    # Let's optimize relative to u_0=0 and v_0=0.
    # So we optimize u_1..u_7, v_1..v_7 (14 variables).
    
    def union_length(intervals):
        # intervals: list of (start, end)
        if len(intervals) == 0:
            return 0.0
        # Sort by start
        intervals = sorted(intervals, key=lambda x: x[0])
        total_len = 0.0
        current_start, current_end = intervals[0]
        for s, e in intervals[1:]:
            if s > current_end:
                total_len += current_end - current_start
                current_start, current_end = s, e
            else:
                current_end = max(current_end, e)
        total_len += current_end - current_start
        return total_len

    def compute_volume(u, v):
        # u, v are arrays of length 8
        # Sample c in [0, 1]
        # 100 samples should be enough for smooth integration
        c_samples = np.linspace(0, 1, 101)
        vol = 0.0
        for c in c_samples:
            x_intervals = []
            for i in range(N):
                x_start = u[i] + c * i / N
                x_end = x_start + 1.0 / N
                x_intervals.append((x_start, x_end))
            ax = union_length(x_intervals)
            
            y_intervals = []
            for j in range(N):
                y_start = v[j] + c * j / N
                y_end = y_start + 1.0 / N
                y_intervals.append((y_start, y_end))
            ay = union_length(y_intervals)
            
            vol += ax * ay
        return vol / len(c_samples)

    def objective(params):
        # params: 14 values
        u = np.zeros(N)
        v = np.zeros(N)
        u[1:] = params[0:7]
        v[1:] = params[7:14]
        return compute_volume(u, v)

    # Bounds for relative positions
    # The shear range is 7/8 = 0.875.
    # Keeping positions within [-1, 1] relative to 0 seems safe.
    bounds = [(-1.5, 1.5)] * 14

    best_result = None
    best_vol = float('inf')
    
    # Run Differential Evolution
    # maxiter and popsize can be tuned.
    # 14 variables, DE is fast.
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15,
            tol=1e-8,
            polish=True
        )
        if result.fun < best_vol:
            best_vol = result.fun
            best_result = result.x
    except Exception as e:
        pass

    # If time remains, try a local polish with Nelder-Mead
    if best_result is not None and time.time() - start_time < timeout:
        from scipy.optimize import minimize
        # Initialize with best result from DE
        # Add u0, v0 back? No, we fixed them to 0.
        # Let's just polish the 14 params.
        res = minimize(objective, best_result, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_vol:
            best_vol = res.fun
            best_result = res.x

    # Construct final positions
    if best_result is None:
        # Fallback to initial
        u_final = u_init
        v_final = v_init
    else:
        u_final = np.zeros(N)
        v_final = np.zeros(N)
        u_final[1:] = best_result[0:7]
        v_final[1:] = best_result[7:14]

    # Build the 64x2 array
    positions = np.zeros((64, 2))
    for k in range(64):
        i = k // 8
        j = k % 8
        positions[k, 0] = u_final[i]
        positions[k, 1] = v_final[j]
        
    return positions
