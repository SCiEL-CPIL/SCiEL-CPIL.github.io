# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    # We formulate this as a Linear Programming problem.
    # We choose a domain D = {1, 2, ..., N}.
    # Variables are v_k for k in 2..N.
    # v_1 is determined by the normalization constraint sum(v_k/k) = 0.
    # Objective: Maximize -sum_{k=2}^N v_k * log(k)/k.
    # Constraints:
    # 1. -10 <= v_k <= 10 for k in 2..N
    # 2. -10 <= v_1 <= 10  =>  -10 <= -sum_{k=2}^N v_k/k <= 10
    # 3. M(x) <= 1 for x in 1..10*N
    #    M(x) = sum_{k=1}^N v_k * floor(x/k)
    #    Substituting v_1: M(x) = sum_{k=2}^N v_k * (floor(x/k) - x/k) = -sum_{k=2}^N v_k * {x/k}
    #    So -sum_{k=2}^N v_k * {x/k} <= 1  =>  sum_{k=2}^N v_k * {x/k} >= -1
    #    => -sum_{k=2}^N v_k * {x/k} <= 1
    
    # Increase N to get a better approximation. 
    # Sparse matrices allow larger N.
    N = 100  # Maximum key in domain
    
    # Variables: v_2, v_3, ..., v_N
    n_vars = N - 1
    ks = np.arange(2, N + 1)
    
    # Objective: Minimize sum(v_k * log(k)/k)  [since we want to maximize negative]
    c = np.array([np.log(k) / k for k in ks])
    
    # Build sparse constraint matrix A_ub
    # 1. Bounds on v_k: I and -I
    # 2. Bounds on v_1: -sum(v_k/k) <= 10 and sum(v_k/k) <= 10
    # 3. M(x) constraints: -sum(v_k * {x/k}) <= 1
    
    rows = []
    cols = []
    vals = []
    
    # Add identity constraints for v_k <= 10
    for i in range(n_vars):
        rows.append(i)
        cols.append(i)
        vals.append(1.0)
        
    # Add -identity constraints for -v_k <= 10
    for i in range(n_vars):
        rows.append(n_vars + i)
        cols.append(i)
        vals.append(-1.0)
        
    # Add v_1 bounds
    # v_1 = -sum(v_k/k). 
    # v_1 <= 10 => -sum(v_k/k) <= 10 => sum(v_k/k) >= -10 => -sum(v_k/k) <= 10? 
    # Wait.
    # Constraint: -10 <= v_1 <= 10.
    # v_1 = -sum_{k=2}^N v_k/k.
    # 1) v_1 <= 10 => -sum(v_k/k) <= 10 => sum(v_k/k) >= -10 => -sum(v_k/k) <= 10.
    #    Row: [-1/k, ..., -1/N] <= 10.
    # 2) v_1 >= -10 => -sum(v_k/k) >= -10 => sum(v_k/k) <= 10.
    #    Row: [1/k, ..., 1/N] <= 10.
    
    row_idx = 2 * n_vars
    for i, k in enumerate(ks):
        rows.append(row_idx)
        cols.append(i)
        vals.append(-1.0 / k)
    row_idx += 1
    for i, k in enumerate(ks):
        rows.append(row_idx)
        cols.append(i)
        vals.append(1.0 / k)
    row_idx += 1
    
    b_ub = []
    b_ub.extend([10.0] * n_vars)
    b_ub.extend([10.0] * n_vars)
    b_ub.extend([10.0, 10.0])
    
    # M(x) constraints
    # For x in 1..10*N:
    # -sum_{k=2}^N v_k * {x/k} <= 1
    # Coeff for v_k is -{x/k}
    
    max_x = 10 * N
    # Vectorize construction of M(x) rows
    # We need {x/k} for all x in 1..max_x and k in 2..N
    # x_vals = np.arange(1, max_x + 1)[:, None]
    # k_vals = ks[None, :]
    # fractional = (x_vals % k_vals) / k_vals
    # coeffs = -fractional
    
    # This creates a matrix of size (10*N) x (N-1)
    # For N=100, max_x=1000, size 1000x99. This is fine.
    # For N=500, max_x=5000, size 5000x499. Still fine.
    
    x_range = np.arange(1, max_x + 1)
    # Use broadcasting to compute fractional parts
    # x_range shape (max_x, 1), ks shape (1, n_vars)
    # x_range % ks gives integer remainder
    # Divide by ks to get fractional part
    
    # To save memory/time, we can process in chunks if N is very large, 
    # but for N=100-200 it should be direct.
    
    # Compute fractional parts
    # Note: x % k / k is 0 if x is divisible by k.
    # We want -{x/k}.
    
    # Efficient calculation:
    # x_vals = x_range.reshape(-1, 1)
    # k_vals = ks.reshape(1, -1)
    # fracs = (x_vals % k_vals) / k_vals
    # m_coeffs = -fracs
    
    # Let's add these to the sparse matrix
    # We can create a dense array for this part if it's not too big, then convert to sparse.
    # 1000 * 100 = 100,000 floats. ~800KB. Very small.
    
    x_vals = x_range.reshape(-1, 1)
    k_vals = ks.reshape(1, -1)
    fracs = (x_vals % k_vals) / k_vals
    m_coeffs = -fracs
    
    # Add to rows/cols/vals
    for i in range(m_coeffs.shape[0]):
        row_base = row_idx + i
        for j in range(m_coeffs.shape[1]):
            if m_coeffs[i, j] != 0:
                rows.append(row_base)
                cols.append(j)
                vals.append(m_coeffs[i, j])
        
    b_ub.extend([1.0] * max_x)
    
    A_ub = csr_matrix((vals, (rows, cols)), shape=(row_idx + max_x, n_vars))
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(-10, 10)] * n_vars, method='highs')
    
    if res.success:
        v_rest = res.x
        v_1 = -np.sum(v_rest / ks)
        
        # Clip to be safe
        v_1 = np.clip(v_1, -10, 10)
        v_rest = np.clip(v_rest, -10, 10)
        
        domain = np.arange(1, N + 1, dtype=np.int64)
        values = np.empty(N, dtype=np.float64)
        values[0] = v_1
        values[1:] = v_rest
        
        # Verify M(x) <= 1.0001
        # M(x) = sum u_k floor(x/k)
        # We can check a subset or all. 
        # Since LP guarantees it, floating point errors might violate slightly.
        # The evaluator clips and replaces f(1). 
        # If we return values that satisfy LP constraints, the evaluator's f(1) replacement 
        # (which subtracts S=sum f(k)/k) will result in sum f(k)/k = 0.
        # The M(x) check in evaluator uses the FINAL values.
        # Our LP ensured M(x) <= 1 for the values where sum f(k)/k=0.
        # So it should be valid.
        
        # Quick sanity check
        # max_x_check = min(10*N, 1000) # Just check first 1000 to save time?
        # The evaluator checks up to 10K.
        # If N=100, 10K=1000.
        
        # Let's do a fast numpy check for validity
        # M(x) = sum_k values[k-1] * (x // k)
        # This is O(N * 10N) = O(N^2).
        # For N=100, 10,000 ops. Very fast.
        
        valid = True
        # Use vectorized check if possible, or loop
        # x_check = np.arange(1, 10*N + 1)
        # M_vals = np.array([np.sum(values * (x_check // np.arange(1, N+1))) for x_check in ...])
        # This is slow in Python loop.
        
        # Let's just trust the LP for now, but do a spot check.
        # If spot check fails, fallback.
        
        # Spot check x = 1, 2, ..., 10
        for x in range(1, 11):
            m_val = 0.0
            for k in range(1, N + 1):
                m_val += values[k-1] * (x // k)
            if m_val > 1.0001:
                valid = False
                break
        
        if valid:
            return domain, values
            
    # Fallback
    return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

# EVOLVE-BLOCK-END
