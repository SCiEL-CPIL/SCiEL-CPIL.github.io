import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def union_length_and_breakpoints(positions):
    """
    Computes the breakpoints of the union length L(c) and the linear coefficients
    a, b for L(c) = a*c + b in each interval.
    Returns:
        breakpoints: sorted list of c values in [0,1]
        coeffs: list of (a, b) for each interval [breakpoints[k], breakpoints[k+1]]
    """
    N = 8
    w = 1.0 / 8.0
    breakpoints = set([0.0, 1.0])
    
    # Find all c where any two interval endpoints meet
    # Intervals: [x_i + c*i/8, x_i + c*i/8 + w]
    # Endpoints: L_i(c) = x_i + c*i/8, R_i(c) = x_i + w + c*i/8
    # Pairs (i, j) with i != j
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = (i - j) / 8.0
            if abs(denom) < 1e-12:
                continue
            
            # L_i = L_j => x_i + c i/8 = x_j + c j/8 => c = (x_j - x_i) / (i/8 - j/8)
            c_val = (positions[j] - positions[i]) / denom
            if 0.0 < c_val < 1.0:
                breakpoints.add(c_val)
                
            # L_i = R_j => x_i + c i/8 = x_j + w + c j/8 => c = (x_j + w - x_i) / denom
            c_val = (positions[j] + w - positions[i]) / denom
            if 0.0 < c_val < 1.0:
                breakpoints.add(c_val)
                
            # R_i = L_j => x_i + w + c i/8 = x_j + c j/8 => c = (x_j - x_i - w) / denom
            c_val = (positions[j] - positions[i] - w) / denom
            if 0.0 < c_val < 1.0:
                breakpoints.add(c_val)
                
            # R_i = R_j => x_i + w + c i/8 = x_j + w + c j/8 => c = (x_j - x_i) / denom
            # Same as L_i = L_j, already covered
            
    breakpoints = sorted(breakpoints)
    
    coeffs = []
    for k in range(len(breakpoints) - 1):
        c_start = breakpoints[k]
        c_end = breakpoints[k + 1]
        c_mid = (c_start + c_end) / 2.0
        
        # Compute L(c_mid) and L(c_start) to get slope?
        # Better: L(c) is linear. We can compute L at c_start and c_end.
        # But at breakpoints, the merge structure might change, but L(c) is continuous.
        # So we can just evaluate L at c_start and c_end.
        
        def eval_L(c_val):
            ls = positions + (np.arange(N) / 8.0) * c_val
            rs = ls + w
            o = np.argsort(ls)
            sl = ls[o]
            sr = rs[o]
            
            # Merge intervals
            total_len = 0.0
            cur_l = sl[0]
            cur_r = sr[0]
            for idx in range(1, N):
                l = sl[idx]
                r = sr[idx]
                if l <= cur_r + 1e-12:
                    if r > cur_r:
                        cur_r = r
                else:
                    total_len += cur_r - cur_l
                    cur_l = l
                    cur_r = r
            total_len += cur_r - cur_l
            return total_len
        
        L_start = eval_L(c_start)
        L_end = eval_L(c_end)
        
        # L(c) = L_start + m * (c - c_start)
        # m = (L_end - L_start) / (c_end - c_start)
        dc = c_end - c_start
        if dc > 1e-15:
            m = (L_end - L_start) / dc
        else:
            m = 0.0
        b = L_start - m * c_start
        a = m
        coeffs.append((a, b))
        
    return breakpoints, coeffs

def compute_volume(p, q):
    """
    Computes the volume V = integral_0^1 L_p(c) L_q(c) dc
    """
    bp_p, coeffs_p = union_length_and_breakpoints(p)
    bp_q, coeffs_q = union_length_and_breakpoints(q)
    
    # Merge breakpoints
    all_bp = sorted(set(bp_p) | set(bp_q))
    
    # Map breakpoints to indices in coeffs_p and coeffs_q
    # We need to find which interval in p and q corresponds to each sub-interval in all_bp
    # Since coeffs are defined on [bp[k], bp[k+1]], we can just iterate over all_bp
    # and for each sub-interval, find the corresponding a,b from p and q.
    
    # To do this efficiently, we can pre-compute the interval index for each breakpoint
    # But since N is small (8), we can just do a linear search or binary search.
    
    total_vol = 0.0
    for k in range(len(all_bp) - 1):
        c_start = all_bp[k]
        c_end = all_bp[k + 1]
        c_mid = (c_start + c_end) / 2.0
        
        # Find the interval in p that contains c_mid
        # Since bp_p is sorted, we can use np.searchsorted
        idx_p = np.searchsorted(bp_p, c_mid, side='right') - 1
        if idx_p < 0: idx_p = 0
        if idx_p >= len(coeffs_p): idx_p = len(coeffs_p) - 1
        a_p, b_p = coeffs_p[idx_p]
        
        idx_q = np.searchsorted(bp_q, c_mid, side='right') - 1
        if idx_q < 0: idx_q = 0
        if idx_q >= len(coeffs_q): idx_q = len(coeffs_q) - 1
        a_q, b_q = coeffs_q[idx_q]
        
        # Integrate (a_p c + b_p) (a_q c + b_q) from c_start to c_end
        # = integral (a_p a_q c^2 + (a_p b_q + b_p a_q) c + b_p b_q) dc
        # = [a_p a_q c^3 / 3 + (a_p b_q + b_p a_q) c^2 / 2 + b_p b_q c]
        
        def F(c):
            return (a_p * a_q * c**3 / 3.0 + 
                    (a_p * b_q + b_p * a_q) * c**2 / 2.0 + 
                    b_p * b_q * c)
        
        total_vol += F(c_end) - F(c_start)
        
    return total_vol

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0
    
    N = 8
    
    # Initial guess: Keich configuration
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # Start with p = q = keich
    x0 = np.concatenate([keich, keich])
    
    # Bounds
    # Positions should be in a reasonable range.
    # The intervals are [x, x+1/8].
    # To cover [0,1] roughly, x should be in [-1, 1].
    # Let's use [-2, 2] to be safe.
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # Use Differential Evolution
    # tol, maxiter, popsize, etc.
    # We have 16 variables.
    # popsize=30 is good.
    
    def objective(x):
        p = x[:N]
        q = x[N:]
        return compute_volume(p, q)
        
    try:
        res = differential_evolution(objective, bounds, seed=12345, maxiter=100, popsize=30, tol=1e-8, polish=True)
        if time.time() - start_time > deadline:
            # Return best found so far? DE doesn't give intermediate easily.
            # But if it finishes, we use it.
            pass
    except Exception:
        # Fallback to L-BFGS-B on the same objective
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500})
        except Exception:
            res = None

    if res is not None and res.fun < 1.0: # Sanity check
        best_x = res.x
    else:
        best_x = x0
        
    p_best = best_x[:N]
    q_best = best_x[N:]
    
    # Construct 2D positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = p_best[i_flat]
    base_y = q_best[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

if __name__ == "__main__":
    res = kakeya_needle_3d8()
    print(res.shape)
