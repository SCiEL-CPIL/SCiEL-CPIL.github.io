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
        # Midpoint to get the linear function values? No, we need the values at c0 and c1.
        # Actually, L(c) is linear on [c0, c1], so the integral of Lx(c)*Ly(c) is not simply 
        # the average of the products, because Lx*Ly is quadratic.
        # Wait, Lx(c) is linear, Ly(c) is linear, so Lx(c)*Ly(c) is quadratic.
        # We can integrate a quadratic exactly if we know its values at 3 points, or use 
        # the fact that it's a quadratic.
        # Let f(c) = Lx(c)*Ly(c). f is quadratic.
        # We can compute f(c0), f((c0+c1)/2), f(c1) and use Simpson's rule or exact quadratic integration.
        # Or simpler: since Lx and Ly are linear, we can write Lx(c) = ax*c + bx, Ly(c) = ay*c + by.
        # But it's easier to just evaluate at 3 points and use quadratic integration formula.
        c_mid = (c0 + c1) / 2.0
        
        # Compute Lx and Ly at c0, c_mid, c1
        # To avoid repeated sorting, we can just call _union_length
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
    options = {'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
    
    start_time = time.time()
    deadline = 600.0 - 10.0  # Safety margin
    
    best_params = params0
    best_obj = _compute_objective(params0)
    
    # Try a few restarts if time permits
    n_restarts = 5
    for r in range(n_restarts):
        if r > 0:
            # Perturb the initial guess
            np.random.seed(42 + r)
            perturb = np.random.uniform(-0.1, 0.1, 16)
            params0 = best_params + perturb
        else:
            params0 = keich.copy()
            params0 = np.concatenate([params0, params0])
            
        res = scipy.optimize.minimize(_compute_objective, params0, method='L-BFGS-B', options=options)
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
        
        if time.time() > deadline:
            break
            
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
