# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def _exact_1d_volume(x):
    """
    Compute the exact integral of the length of the union of sheared intervals
    [x_i + c*i/N, x_i + 1/N + c*i/N] for c in [0, 1].
    """
    N = 8
    x = np.asarray(x)
    # s_i = i / N
    s = np.arange(N) / N
    
    # Endpoints are of the form A(c) = x_i + s_i * c and B(c) = x_i + 1/N + s_i * c
    # We need to find critical points where A_i(c) == A_j(c), A_i(c) == B_j(c), etc.
    # Generally, any two endpoints P_i(c) and Q_j(c) coincide when:
    # x_i + offset_i + s_i * c == x_j + offset_j + s_j * c
    # (s_i - s_j) * c == x_j + offset_j - x_i - offset_i
    # c = (x_j + offset_j - x_i - offset_i) / (s_i - s_j)
    
    criticals = [0.0, 1.0]
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Pairs of endpoints:
            # 1. A_i vs A_j: offset_i=0, offset_j=0
            # 2. A_i vs B_j: offset_i=0, offset_j=1/N
            # 3. B_i vs A_j: offset_i=1/N, offset_j=0
            # 4. B_i vs B_j: offset_i=1/N, offset_j=1/N
            
            # We can just iterate over all 4 combinations of offsets for i and j
            for off_i in [0.0, 1.0/N]:
                for off_j in [0.0, 1.0/N]:
                    ds = s[i] - s[j]
                    if abs(ds) < 1e-12:
                        continue
                    c_val = (x[j] + off_j - (x[i] + off_i)) / ds
                    if 0.0 < c_val < 1.0:
                        criticals.append(c_val)
    
    # Remove duplicates and sort
    criticals = sorted(set([round(c, 15) for c in criticals]))
    
    total_vol = 0.0
    
    def union_length_at_c(c_val):
        # Compute the length of the union of intervals at height c_val
        starts = x + s * c_val
        ends = starts + 1.0/N
        # Merge intervals
        # Sort by start
        idx = np.argsort(starts)
        starts_sorted = starts[idx]
        ends_sorted = ends[idx]
        
        current_max_end = -np.inf
        length = 0.0
        for i in range(N):
            s_i = starts_sorted[i]
            e_i = ends_sorted[i]
            if s_i > current_max_end:
                length += e_i - s_i
                current_max_end = e_i
            else:
                if e_i > current_max_end:
                    length += e_i - current_max_end
                    current_max_end = e_i
        return length

    # Integrate over each segment
    for k in range(len(criticals) - 1):
        c_start = criticals[k]
        c_end = criticals[k + 1]
        if c_end - c_start < 1e-12:
            continue
        
        # The length function is linear in c within this segment.
        # We can just take the average of the length at the endpoints.
        L_start = union_length_at_c(c_start)
        L_end = union_length_at_c(c_end)
        
        total_vol += 0.5 * (L_start + L_end) * (c_end - c_start)
        
    return total_vol

def _objective(params):
    """
    Objective function: V = Vol_x * Vol_y
    """
    x = params[:8]
    y = params[8:16]
    vx = _exact_1d_volume(x)
    vy = _exact_1d_volume(y)
    return vx * vy

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimize the base positions to minimize the union volume.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initial guess: Keich construction from seed
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Symmetric initial guess
    x0 = keich
    y0 = keich
    params0 = np.concatenate([x0, y0])
    
    # Bounds for optimization
    # The relative positions are what matter. We can bound them to [-1, 1]
    bounds = [(-1.0, 1.0)] * 16
    
    best_params = params0
    best_score = _objective(params0)
    
    # Use Differential Evolution for global search
    # Maximize the number of iterations within the time limit
    # DE is stochastic, so we might run multiple runs or just one long one.
    # Given the smoothness, a single DE run with sufficient iterations should suffice.
    
    try:
        # differential_evolution is a global optimizer
        # maxiter is the max number of generations
        # popsize is the population size
        res = scipy.optimize.differential_evolution(
            _objective, 
            bounds, 
            maxiter=200, 
            popsize=10, 
            seed=42, 
            tol=1e-8,
            polish=True # Use L-BFGS-B to polish the result
        )
        
        if res.success or res.fun < best_score:
            best_params = res.x
            best_score = res.fun
            best_params = res.x
            
    except Exception:
        pass
    
    # If we have time, do a local refinement with L-BFGS-B
    if time.time() < deadline - 60:
        try:
            res_local = scipy.optimize.minimize(
                _objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res_local.fun < best_score:
                best_params = res_local.x
                best_score = res_local.fun
        except Exception:
            pass
    
    # Construct the output array
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
