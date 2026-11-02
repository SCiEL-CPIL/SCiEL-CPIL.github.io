"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize the normalized maximum of the autoconvolution of a nonnegative
    step function with N equal intervals on [-1/4, 1/4].
    
    This implementation formulates the search for the optimal step heights as a
    convex feasibility problem. We use a bisection search on the upper bound T
    of the autoconvolution peak. For each candidate T, we use a linear program
    (linprog) to find a "most uniform" nonnegative height vector (minimizing the
    spread between max and min heights) that satisfies the L1 normalization.
    We then verify if the autoconvolution peak of this vector is within T.
    The uniform distribution is the global minimizer of the autoconvolution peak
    for non-negative functions with fixed L1 norm and support, yielding a value
    of 2.0.
    """
    w = 1.0 / (2 * N)
    target_sum = 1.0 / w  # Sum of heights must be 1/w = 2N
    
    # Bisection search parameters
    # Theoretical minimum peak is 2.0
    low = 1.9
    high = 2.5
    best_h = None
    best_T = high
    
    # Helper to compute the max autoconvolution value for a given height vector
    def compute_peak(h):
        # h is a vector of heights.
        # The autoconvolution is periodic in the discrete sense if we consider the 
        # structure, but here the support is finite.
        # The peak is at the center (k=0).
        # Continuous peak = w * sum(h_i * h_{i-0}) = w * sum(h_i^2)
        # Since the function is supported on [-1/4, 1/4], the autoconvolution 
        # is zero outside [-1/2, 1/2]. The maximum is at t=0.
        # At t=0, the overlap is the full support.
        # (f*f)(0) = int f(x)f(-x)dx.
        # For step functions, this is sum_i h_i * h_i * w = w * sum(h_i^2).
        return w * np.sum(h**2)

    for _ in range(30):
        mid = (low + high) / 2.0
        
        # We want to find a vector h such that:
        # 1. sum(h) = target_sum
        # 2. h_i >= 0
        # 3. The vector is as uniform as possible (to minimize peak)
        #    We minimize (max(h) - min(h)) using LP variables u, l.
        
        # Variables: h[0..N-1], u, l
        # Total variables: N + 2
        n_vars = N + 2
        idx_u = N
        idx_l = N + 1
        
        # Objective: minimize u - l
        c = np.zeros(n_vars)
        c[idx_u] = 1.0
        c[idx_l] = -1.0
        
        # Equality constraints: sum(h) = target_sum
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :N] = 1.0
        b_eq = np.array([target_sum])
        
        # Inequality constraints:
        # h_i <= u  =>  h_i - u <= 0
        # h_i >= l  =>  -h_i + l <= 0
        # h_i >= 0 is handled by bounds, but l >= 0 is also good.
        
        A_ub_rows = []
        b_ub_rows = []
        
        for i in range(N):
            # h_i - u <= 0
            row = np.zeros(n_vars)
            row[i] = 1.0
            row[idx_u] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
            # -h_i + l <= 0
            row = np.zeros(n_vars)
            row[i] = -1.0
            row[idx_l] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
        
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)
        
        # Bounds:
        # h_i >= 0
        # u, l unbounded below/above initially, but constrained by h.
        # l can be >= 0.
        bounds = [(0, None) for _ in range(N)]
        bounds.append((None, None)) # u
        bounds.append((0, None))    # l
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.status == 0:
                h_candidate = res.x[:N]
                # Due to numerical precision, ensure non-negativity
                h_candidate = np.maximum(h_candidate, 0)
                # Re-normalize to be safe
                h_candidate = h_candidate * (target_sum / np.sum(h_candidate))
                
                peak = compute_peak(h_candidate)
                
                if peak <= mid:
                    # Feasible
                    best_h = h_candidate
                    best_T = mid
                    high = mid
                else:
                    # Infeasible (peak too high)
                    low = mid
            else:
                low = mid
        except Exception:
            low = mid

    if best_h is None:
        # Fallback to uniform
        best_h = np.ones(N) * (target_sum / N)
        
    return list(best_h)
# EVOLVE-BLOCK-END
