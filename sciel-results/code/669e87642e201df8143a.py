import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 620.0  # Safety margin before 630s limit

    # Precompute indices for the final output mapping
    # Row k is tube (i, j) with i = k // 8 and j = k % 8
    i_indices = np.repeat(np.arange(N), N)
    j_indices = np.tile(np.arange(N), N)

    def compute_1d_union_length(positions, c, N):
        """
        Compute the length of the union of intervals 
        [pos[i] + c*i/N, pos[i] + c*i/N + 1/N] for i=0..N-1.
        positions: array of shape (N,)
        c: scalar
        """
        starts = positions + c * np.arange(N) / N
        ends = starts + 1.0 / N
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        total = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] <= cur_end:
                if e_sorted[k] > cur_end:
                    cur_end = e_sorted[k]
            else:
                total += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
        total += cur_end - cur_start
        return total

    def get_break_points(positions, N):
        """
        Find the values of c in (0, 1) where the structure of the union changes.
        These occur when start_i(c) == start_j(c), start_i(c) == end_j(c), etc.
        start_i(c) = x_i + c*i/N
        end_i(c) = x_i + c*i/N + 1/N
        
        Equations:
        x_i + c*i/N = x_j + c*j/N  => c = N*(x_j - x_i)/(i-j)
        x_i + c*i/N = x_j + c*j/N + 1/N => c = N*(x_j - x_i + 1/N)/(i-j)
        x_i + c*i/N + 1/N = x_j + c*j/N => c = N*(x_j - x_i - 1/N)/(i-j)
        """
        breaks = set([0.0, 1.0])
        for i in range(N):
            for j in range(i + 1, N):
                denom = i - j
                if denom == 0: continue
                # Start-Start
                c1 = N * (positions[j] - positions[i]) / denom
                # Start-End
                c2 = N * (positions[j] - positions[i] + 1.0/N) / denom
                # End-Start
                c3 = N * (positions[j] - positions[i] - 1.0/N) / denom
                
                for c_val in [c1, c2, c3]:
                    if 0.0 < c_val < 1.0:
                        breaks.add(c_val)
        return sorted(breaks)

    def exact_volume(X, Y, N):
        """
        Compute the exact volume V = integral_0^1 Lx(c) Ly(c) dc.
        Uses the union of break points from X and Y to define integration intervals.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        x_breaks = get_break_points(X, N)
        y_breaks = get_break_points(Y, N)
        
        # Union of all break points
        all_breaks = sorted(set(x_breaks + y_breaks))
        
        total_vol = 0.0
        for k in range(len(all_breaks) - 1):
            c_mid = (all_breaks[k] + all_breaks[k+1]) / 2.0
            # Since the union length is piecewise linear, evaluating at midpoint
            # and multiplying by width is exact for the integral of the product?
            # No, Lx(c) is piecewise linear, Ly(c) is piecewise linear.
            # Their product is piecewise quadratic.
            # However, the "kinks" of the product occur at the kinks of Lx or Ly.
            # So on each interval [all_breaks[k], all_breaks[k+1]], Lx and Ly are linear.
            # Thus Lx*Ly is quadratic. We can integrate exactly using Simpson's rule
            # or just evaluate at endpoints and midpoint (Simpson's 1/3 rule is exact for quadratics).
            
            c0 = all_breaks[k]
            c1 = all_breaks[k+1]
            cm = (c0 + c1) / 2.0
            
            Lx0 = compute_1d_union_length(X, c0, N)
            Lx1 = compute_1d_union_length(X, c1, N)
            Lxm = compute_1d_union_length(X, cm, N)
            
            Ly0 = compute_1d_union_length(Y, c0, N)
            Ly1 = compute_1d_union_length(Y, c1, N)
            Lym = compute_1d_union_length(Y, cm, N)
            
            # Simpson's rule for integral of f(c) from c0 to c1
            # f(c) = Lx(c) Ly(c)
            # Integral approx = (c1-c0)/6 * (f(c0) + 4f(cm) + f(c1))
            f0 = Lx0 * Ly0
            fm = Lxm * Lym
            f1 = Lx1 * Ly1
            
            total_vol += (c1 - c0) / 6.0 * (f0 + 4.0 * fm + f1)
            
        return total_vol

    def objective(params):
        X = params[:N]
        Y = params[N:]
        return exact_volume(X, Y, N)

    # Initial Guesses
    # 1. Keich-like construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    X0 = keich * 0.5 + 0.25
    Y0 = keich * 0.5 + 0.25
    initial_params = np.concatenate([X0, Y0])
    
    best_params = initial_params.copy()
    best_score = objective(initial_params)
    
    # Multi-start Nelder-Mead
    rng = np.random.default_rng(12345)
    
    num_starts = 0
    max_starts = 50
    
    while time.time() < deadline - 10:
        # Generate a new starting point
        if num_starts == 0:
            start = initial_params.copy()
        else:
            # Mix of random uniform and perturbed best
            if num_starts % 2 == 0:
                # Random uniform in [0, 1]
                start = rng.uniform(0.0, 1.0, size=2*N)
            else:
                # Perturb current best
                sigma = 0.1
                start = best_params + rng.normal(0, sigma, size=2*N)
                start = np.clip(start, 0.0, 1.0)
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception:
            pass
        
        num_starts += 1
        if num_starts >= max_starts and time.time() > deadline - 60:
            break

    # Final Polish with tighter tolerances
    if time.time() < deadline - 5:
        try:
            res_final = minimize(objective, best_params, method='Nelder-Mead', 
                                 options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'disp': False})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass

    X_opt = best_params[:N]
    Y_opt = best_params[N:]
    
    base_x = X_opt[i_indices]
    base_y = Y_opt[j_indices]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
