# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
import sys

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    # Fallback to seed result
    def fallback():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Strategy: Linear Programming
    # We want to maximize a = -sum_{k=2}^K f(k) log(k)/k
    # Subject to:
    # 1. M(x) <= 1.0001 for all x in [1, 10K]
    #    M(x) = sum_{k=2}^K f(k) (floor(x/k) - x/k) + f(1)x
    #    With f(1) = -sum f(k)/k, M(x) = sum_{k=2}^K f(k) (floor(x/k) - x/k)
    #    Let C_{x,k} = x/k - floor(x/k) >= 0.
    #    Constraint: sum_{k=2}^K f(k) (-C_{x,k}) <= 1.0001
    #    => sum_{k=2}^K C_{x,k} f(k) >= -1.0001
    # 2. -10 <= f(k) <= 10 (clipping bounds)
    
    # To keep the LP solvable, we select a subset of x constraints.
    # The most restrictive constraints are likely those where sum C_{x,k} is large.
    # However, the direction of the constraint vector matters.
    # We will use a uniform subset of x values to approximate the full constraint set.
    
    K = 2000
    max_x = 10 * K
    
    # If time is tight, reduce K
    if time.time() > deadline - 10:
        return fallback()

    # Variables: f(2), ..., f(K)
    n_vars = K - 1
    
    # Objective: Maximize -sum w_k f_k  =>  Minimize sum w_k f_k
    # w_k = log(k) / k
    k_indices = np.arange(2, K + 1)
    w = np.log(k_indices) / k_indices.astype(np.float64)
    
    # Constraint Matrix A: rows for x in subset, cols for k in [2, K]
    # A_{x, k} = C_{x,k} = x/k - floor(x/k)
    # We need A @ f >= -1.0001
    # linprog standard form: A_ub @ x <= b_ub
    # So we use -A @ f <= 1.0001
    
    # Selecting x subset
    # Using all x in [1, 10K] is 20000 rows.
    # 20000 x 1999 matrix is ~40M floats (320MB).
    # This is borderline for memory and time in Python/SciPy.
    # Let's try to use a subset first, or use a sparse matrix if possible.
    # SciPy linprog supports sparse matrices.
    
    try:
        # Build sparse matrix A
        # Rows: x = 1 to max_x
        # Cols: k = 2 to K
        # A_{x,k} = x/k - floor(x/k)
        
        # We will use all x to be safe, but use sparse format.
        # If memory fails, we will catch and use a subset.
        
        # Precompute for each k, the values for all x
        # This is dense if we do it naively.
        # Let's build the sparse matrix by iterating k.
        
        row_indices = []
        col_indices = []
        data = []
        
        # To speed up, we can note that for a fixed k, x/k - floor(x/k) is periodic or has structure.
        # But for sparse matrix construction, we just need non-zeros.
        # C_{x,k} is non-zero for all x not divisible by k.
        # This is dense! C_{x,k} is rarely zero.
        # So sparse matrix will be almost dense.
        # 20000 * 2000 = 40 million entries.
        # Storing 40M floats is 320MB.
        # SciPy linprog with dense 40M matrix might be slow (minutes).
        
        # Alternative: Use a subset of x.
        # Which x are most critical?
        # The constraint is sum C_{x,k} f_k >= -1.0001.
        # If we violate it, M(x) > 1.0001.
        # M(x) is a step function.
        # Let's sample x uniformly.
        num_constraints = 1000
        x_vals = np.linspace(1, max_x, num_constraints, dtype=np.int64)
        
        # Build dense matrix for subset
        A = np.zeros((num_constraints, n_vars), dtype=np.float64)
        for i, x in enumerate(x_vals):
            # A[i, j] = x/k - floor(x/k) for k = 2+j
            ks = k_indices
            vals = x / ks.astype(np.float64)
            A[i, :] = vals - np.floor(vals)
        
        # Constraints: A @ f >= -1.0001
        # linprog: -A @ f <= 1.0001
        A_ub = -A
        b_ub = np.full(num_constraints, 1.0001)
        
        # Bounds: -10 <= f_k <= 10
        bounds = [(-10.0, 10.0) for _ in range(n_vars)]
        
        # Solve LP
        # Method 'highs' is fast and robust
        res = linprog(c=w, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            return fallback()
        
        f_vals = res.x
        k_domain = k_indices
        
        # Verify feasibility on ALL x in [1, 10K]
        # M(x) = sum f(k) (floor(x/k) - x/k)
        # We can compute M(x) for all x efficiently using the difference array method.
        
        D = np.zeros(max_x + 2, dtype=np.float64)
        for idx in range(n_vars):
            k = int(k_domain[idx])
            fk = f_vals[idx]
            if abs(fk) < 1e-9:
                continue
            j = 1
            while j * k <= max_x:
                lo = j * k
                hi = min((j + 1) * k - 1, max_x)
                # Contribution to M(x) is fk * floor(x/k)
                # In difference array for floor(x/k):
                # floor(x/k) increases by 1 at x = j*k
                # So we add fk to D[lo]
                # And subtract at D[hi+1]
                D[lo] += fk
                if hi + 1 <= max_x:
                    D[hi + 1] -= fk
                j += 1
        
        M_floor_part = np.cumsum(D)[:max_x + 1]
        
        # Calculate f(1)
        S = np.sum(f_vals / k_domain.astype(np.float64))
        f1 = -S
        
        # M(x) = M_floor_part(x) + f1 * x
        # Note: M_floor_part computed above is sum f(k) floor(x/k)
        # The formula derived was M(x) = sum f(k) (floor(x/k) - x/k)
        # So M(x) = M_floor_part(x) - S * x
        # Wait, f1 = -S.
        # M(x) = f1 * x + sum f(k) floor(x/k) = -S * x + M_floor_part(x)
        # This matches.
        
        x_range = np.arange(1, max_x + 1, dtype=np.float64)
        M_vals = M_floor_part[1:] - S * x_range
        
        if np.max(M_vals) > 1.0001:
            # Infeasible. Try to scale down or reduce K.
            # For now, return fallback or try a smaller K.
            # Let's try to salvage by scaling f_vals to satisfy constraint.
            # max M_vals = M_max.
            # We need M(x) <= 1.0001.
            # If M_max > 1.0001, we can scale f_vals by factor alpha < 1.
            # M(x) scales linearly with f_vals (since f1 also scales).
            # So new M(x) = alpha * M_old(x).
            # We need alpha * M_max <= 1.0001 => alpha <= 1.0001 / M_max.
            alpha = 1.0001 / np.max(M_vals)
            f_vals = f_vals * alpha
            S = S * alpha
            f1 = f1 * alpha
            
            # Re-verify
            M_vals_new = M_vals * alpha
            if np.max(M_vals_new) > 1.0001:
                return fallback()
        
        # Construct final domain and values
        domain = np.empty(n_vars + 1, dtype=np.int64)
        values = np.empty(n_vars + 1, dtype=np.float64)
        domain[0] = 1
        domain[1:] = k_domain
        values[0] = f1
        values[1:] = f_vals
        
        # Clip values to [-10, 10] as per evaluator (though LP bounds should ensure this)
        values = np.clip(values, -10.0, 10.0)
        
        return domain, values

    except Exception as e:
        # If LP fails or memory error, fallback
        return fallback()

# EVOLVE-BLOCK-END
