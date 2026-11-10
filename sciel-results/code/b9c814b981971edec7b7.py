import numpy as np
import scipy.optimize as so
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    
    # Initial guess from Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    X0 = keich.copy()
    Y0 = keich.copy()
    
    # Build initial 128-dim parameter vector: [X0[0],...,X0[7], Y0[0],...,Y0[7]]
    # But we need 128 coords: for each (i,j) pair, we have (x_{ij}, y_{ij})
    # Actually, re-reading: base positions are (x_k, y_k) for k=0..63
    # x_k depends on i=k//8, y_k depends on j=k%8
    # So x_k = X[i] where i = k//8, y_k = Y[j] where j = k%8
    # This means only 8 unique x values and 8 unique y values!
    # So the parameter vector is just [X[0..7], Y[0..7]] = 16 params
    
    # Wait, let me re-read the task more carefully.
    # "Row k is tube (i, j) with i = k // 8 and j = k % 8"
    # "At height c in [0, 1] tube k is the closed square [x_k + c*i/8, x_k + c*i/8 + 1/8] x [y_k + c*j/8, y_k + c*j/8 + 1/8]"
    # So x_k and y_k are the base positions for tube k.
    # The current program uses X[i] for all tubes with the same i, and Y[j] for all tubes with the same j.
    # This is a structural assumption. Let's keep it since both current and inspiration use it.
    
    # Exact volume computation (kept from current program for final validation)
    def calc_union_length_vec(X, c):
        """Compute union length of intervals [X[i] + c*i/N, X[i] + c*i/N + delta] for i=0..N-1"""
        lefts = X + c * np.arange(N) / N
        rights = lefts + delta
        endpoints = np.empty(2 * N, dtype=np.float64)
        types = np.empty(2 * N, dtype=np.int8)
        endpoints[0::2] = lefts
        endpoints[1::2] = rights
        types[0::2] = 0
        types[1::2] = 1
        order = np.lexsort((types, endpoints))
        sorted_ends = endpoints[order]
        sorted_types = types[order]
        count = 0
        start_pos = 0.0
        total = 0.0
        for val, t in zip(sorted_ends, sorted_types):
            if t == 0:
                if count == 0:
                    start_pos = val
                count += 1
            else:
                count -= 1
                if count == 0:
                    total += val - start_pos
        return total

    def compute_volume(X, Y):
        A_x = np.empty(2 * N)
        B_x = np.empty(2 * N)
        for i in range(N):
            A_x[2*i] = X[i]
            B_x[2*i] = i / N
            A_x[2*i+1] = X[i] + delta
            B_x[2*i+1] = i / N
        
        A_y = np.empty(2 * N)
        B_y = np.empty(2 * N)
        for i in range(N):
            A_y[2*i] = Y[i]
            B_y[2*i] = i / N
            A_y[2*i+1] = Y[i] + delta
            B_y[2*i+1] = i / N
        
        crit_x = [0.0, 1.0]
        for i in range(2 * N):
            for j in range(i + 1, 2 * N):
                if B_x[i] != B_x[j]:
                    c = (A_x[j] - A_x[i]) / (B_x[i] - B_x[j])
                    if 0.0 < c < 1.0:
                        crit_x.append(c)
        
        crit_y = [0.0, 1.0]
        for i in range(2 * N):
            for j in range(i + 1, 2 * N):
                if B_y[i] != B_y[j]:
                    c = (A_y[j] - A_y[i]) / (B_y[i] - B_y[j])
                    if 0.0 < c < 1.0:
                        crit_y.append(c)
        
        all_crit = sorted(set(crit_x + crit_y))
        
        total_vol = 0.0
        for k in range(len(all_crit) - 1):
            c_start = all_crit[k]
            c_end = all_crit[k + 1]
            h = c_end - c_start
            if h < 1e-12:
                continue
            
            Lx_s = calc_union_length_vec(X, c_start)
            Lx_e = calc_union_length_vec(X, c_end)
            Ly_s = calc_union_length_vec(Y, c_start)
            Ly_e = calc_union_length_vec(Y, c_end)
            
            dx = Lx_e - Lx_s
            dy = Ly_e - Ly_s
            
            term = Lx_s * Ly_s
            term += 0.5 * (Lx_s * dy + Ly_s * dx)
            term += (1.0 / 3.0) * dx * dy
            
            total_vol += h * term
        
        return total_vol

    # Smooth sigmoid-based surrogate for gradient-based optimization
    def sigmoid(x, sharpness=100.0):
        """Smooth approximation of step function"""
        return 1.0 / (1.0 + np.exp(-sharpness * x))
    
    def smooth_tube_indicator(a, b, c, x_base, y_base, i, j, sharpness=100.0):
        """
        Smooth indicator that point (a, b, c) is inside tube (i, j) with base (x_base, y_base).
        Tube at height c: [x_base + c*i/N, x_base + c*i/N + delta] x [y_base + c*j/N, y_base + c*j/N + delta]
        """
        # x-range: [x_base + c*i/N, x_base + c*i/N + delta]
        x_lo = x_base + c * i / N
        x_hi = x_base + c * i / N + delta
        # y-range: [y_base + c*j/N, y_base + c*j/N + delta]
        y_lo = y_base + c * j / N
        y_hi = y_base + c * j / N + delta
        
        # Smooth indicators for being inside each interval
        # Inside x: sigmoid((a - x_lo)*s) * sigmoid((x_hi - a)*s)
        # This is 1 when a is well inside, 0 when well outside
        sx_in = sigmoid((a - x_lo) * sharpness) * sigmoid((x_hi - a) * sharpness)
        sy_in = sigmoid((b - y_lo) * sharpness) * sigmoid((y_hi - b) * sharpness)
        
        return sx_in * sy_in
    
    def smooth_union_volume(params, n_samples=20000, sharpness=100.0):
        """
        Estimate union volume using smooth sigmoid indicators.
        params: [X[0..7], Y[0..7]]
        """
        X = params[:N]
        Y = params[N:]
        
        # Compute bounding box
        i_vals = np.arange(N)
        j_vals = np.arange(N)
        
        # For each tube (i,j), base is (X[i], Y[j])
        # x-range at c=0: [X[i], X[i]+delta], at c=1: [X[i]+i/N, X[i]+i/N+delta]
        # So overall x-range for tube (i,j): [min(X[i], X[i]+i/N), max(X[i]+delta, X[i]+i/N+delta)]
        # = [X[i] + min(0, i/N), X[i] + delta + max(0, i/N)]
        # Since i >= 0, i/N >= 0, so: [X[i], X[i] + delta + i/N]
        
        x_mins = X  # X[i] + 0
        x_maxs = X + delta + i_vals / N  # X[i] + delta + i/N
        
        y_mins = Y
        y_maxs = Y + delta + j_vals / N
        
        X0 = np.min(x_mins)
        X1 = np.max(x_maxs)
        Y0 = np.min(y_mins)
        Y1 = np.max(y_maxs)
        
        box_area = (X1 - X0) * (Y1 - Y0)
        if box_area <= 1e-9:
            return 1e6  # penalty for degenerate box
        
        # Sample points
        rng = np.random.RandomState(42)
        a = rng.uniform(X0, X1, n_samples)
        b = rng.uniform(Y0, Y1, n_samples)
        c = rng.uniform(0, 1, n_samples)
        
        # Compute smooth union indicator
        # For each sample, compute 1 - prod(1 - tube_indicator) over all 64 tubes
        # This is expensive: 64 tubes * n_samples
        # Vectorize over samples for each tube
        
        union_ind = np.zeros(n_samples)
        for i in range(N):
            for j in range(N):
                ind = smooth_tube_indicator(a, b, c, X[i], Y[j], i, j, sharpness)
                union_ind = 1.0 - (1.0 - union_ind) * (1.0 - ind)
        
        volume = np.mean(union_ind) * box_area
        return volume
    
    def smooth_objective(params):
        """Smooth objective for L-BFGS-B"""
        return smooth_union_volume(params, n_samples=10000, sharpness=80.0)
    
    # Optimization strategy:
    # 1. Use smooth surrogate with L-BFGS-B for fast gradient-based optimization
    # 2. Polish with exact volume using Powell
    # 3. Multiple restarts
    
    p0 = np.concatenate([X0, Y0])
    
    start_time = time.time()
    deadline = start_time + 520.0
    
    best_p = p0.copy()
    best_v_exact = compute_volume(X0, Y0)
    best_v_smooth = smooth_objective(p0)
    
    # Phase 1: Smooth surrogate optimization with L-BFGS-B
    np.random.seed(42)
    n_restarts = 12
    for r in range(n_restarts):
        if time.time() > deadline - 120:
            break
        
        if r == 0:
            p_init = p0.copy()
        else:
            scale = 0.03 * (1.0 + 0.3 * np.sin(r * 0.5))
            p_init = p0 + np.random.normal(0, scale, 2 * N)
        
        try:
            res = so.minimize(smooth_objective, p_init, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_v_smooth:
                best_v_smooth = res.fun
                # Check exact volume
                X_try = res.x[:N]
                Y_try = res.x[N:]
                v_exact = compute_volume(X_try, Y_try)
                if v_exact < best_v_exact:
                    best_v_exact = v_exact
                    best_p = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Polish with exact volume using Powell
    if time.time() < deadline - 60:
        try:
            res2 = so.minimize(lambda p: compute_volume(p[:N], p[N:]), best_p, method='Powell',
                               options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-10})
            if res2.fun < best_v_exact:
                best_v_exact = res2.fun
                best_p = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final Nelder-Mead polish
    if time.time() < deadline - 20:
        try:
            res3 = so.minimize(lambda p: compute_volume(p[:N], p[N:]), best_p, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-11})
            if res3.fun < best_v_exact:
                best_v_exact = res3.fun
                best_p = res3.x.copy()
        except Exception:
            pass
    
    X_opt = best_p[:N]
    Y_opt = best_p[N:]
    
    # Build 64x2 position array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
