# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses linear programming to optimally distribute negative mass across the domain
    while satisfying M(x) <= 1.0001 for all x in [1, 10K].
    """
    start_time = time.time()
    DEADLINE = 610.0  # safety margin before 630s

    def build_lp(K, max_x):
        """
        Build the LP for domain {1, 2, ..., K}.
        Variables: y_i = f(i) for i=1..K (K variables)
        Constraints:
          1. sum_i f(i)/i = 0  (1 equality)
          2. M(x) = sum_i f(i)*floor(x/i) <= 1.0001 for x=1..max_x  (max_x inequalities)
        Objective: maximize a = -sum_i f(i)*log(i)/i
        """
        n = K
        # Objective: minimize -a = sum_i f(i)*log(i)/i
        # c_i = log(i)/i for i=1..K, with c_1 = 0 (log(1)=0)
        c = np.zeros(n, dtype=np.float64)
        if K >= 1:
            c[0] = 0.0  # log(1)/1 = 0
        if K >= 2:
            idx = np.arange(2, K + 1, dtype=np.float64)
            c[1:] = np.log(idx) / idx
        
        # Equality constraint: sum_i f(i)/i = 0
        A_eq = np.zeros((1, n), dtype=np.float64)
        A_eq[0, 0] = 1.0 / 1.0  # f(1)/1
        if K >= 2:
            idx = np.arange(2, K + 1, dtype=np.float64)
            A_eq[0, 1:] = 1.0 / idx
        b_eq = np.zeros(1, dtype=np.float64)
        
        # Inequality constraints: M(x) <= 1.0001 for x = 1..max_x
        # M(x) = sum_i f(i) * floor(x/i)
        # A_ub[i, j] = floor((i+1)/j) for row i (x=i+1), col j (f(j+1))
        # But this is max_x x K which could be huge
        # max_x = 10*K, so it's 10K^2 entries
        # For K=1000, that's 10^7 entries - manageable
        # For K=5000, that's 2.5*10^8 - might be too much
        # Let's use sparse representation
        
        from scipy.sparse import csr_matrix, csc_matrix
        
        # Build A_ub as sparse: for each x (row), A_ub[x, i] = floor(x/i)
        # We can build this by iterating over i and adding to rows
        rows = []
        cols = []
        data = []
        
        for i in range(1, K + 1):  # i is the divisor (1-indexed)
            col_idx = i - 1  # 0-indexed column
            # For each x, floor(x/i) is the value
            # x ranges from 1 to max_x
            # floor(x/i) = 0 for x < i
            # For x >= i: floor(x/i) = q where q*i <= x < (q+1)*i
            # We can add entries in blocks
            x_start = i
            if x_start > max_x:
                continue
            q = 1
            while q * i <= max_x:
                x_lo = q * i
                x_hi = min((q + 1) * i - 1, max_x)
                # For x in [x_lo, x_hi], floor(x/i) = q
                # Row indices: x_lo-1 to x_hi-1 (0-indexed)
                row_start = x_lo - 1
                row_end = x_hi - 1
                # Add (row_start..row_end, col_idx) with value q
                # Use slicing for efficiency
                n_rows = row_end - row_start + 1
                rows.extend(range(row_start, row_end + 1))
                cols.extend([col_idx] * n_rows)
                data.extend([float(q)] * n_rows)
                q += 1
        
        A_ub = csr_matrix((data, (rows, cols)), shape=(max_x, K), dtype=np.float64)
        b_ub = np.full(max_x, 1.0001, dtype=np.float64)
        
        # Bounds: f(i) in [-10, 10] (clipping is applied by evaluator, but we can bound)
        bounds = [(-10.0, 10.0)] * K
        
        return c, A_ub, b_ub, A_eq, b_eq, bounds

    def compute_score(domain, values):
        """Compute the objective a = -sum_k f(k)*log(k)/k"""
        return -np.sum(values * np.log(domain.astype(np.float64)) / domain.astype(np.float64))

    def check_feasibility(domain, values, K):
        """Check M(x) <= 1.0001 for all x in [1, 10K]."""
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        for k, f in zip(domain, values):
            if f == 0.0:
                continue
            kf = int(k)
            j = 1
            while j * kf <= max_x:
                lo = j * kf
                hi = min((j + 1) * kf, max_x + 1)
                M[lo:hi] += f
                j += 1
        return np.max(M[1:max_x + 1])

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try different K values with LP
    for K in [50, 100, 200, 500, 1000]:
        if time.time() > DEADLINE - 30:
            break
        
        max_x = 10 * K
        print(f"Trying K={K}, max_x={max_x}")
        
        try:
            c, A_ub, b_ub, A_eq, b_eq, bounds = build_lp(K, max_x)
            
            # Solve LP: minimize c^T y subject to A_ub y <= b_ub, A_eq y = b_eq
            result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs',
                options={'time_limit': 120}
            )
            
            if result.status == 0 and result.x is not None:
                f_vals = result.x
                # Clip to [-10, 10]
                f_vals = np.clip(f_vals, -10.0, 10.0)
                
                # Apply the f(1) adjustment: f(1) -= S where S = sum f(k)/k
                S = np.sum(f_vals / np.arange(1, K + 1, dtype=np.float64))
                f_vals[0] -= S
                
                # Re-clip after adjustment
                f_vals = np.clip(f_vals, -10.0, 10.0)
                
                domain = np.arange(1, K + 1, dtype=np.int64)
                
                # Check feasibility
                max_M = check_feasibility(domain, f_vals, K)
                
                if max_M <= 1.0001 + 1e-9:
                    a = compute_score(domain, f_vals)
                    print(f"  K={K}: feasible, max_M={max_M:.6f}, a={a:.6f}")
                    
                    if a > best_score:
                        best_score = a
                        best_domain = domain.copy()
                        best_values = f_vals.copy()
                else:
                    print(f"  K={K}: infeasible, max_M={max_M:.6f}")
            
            else:
                print(f"  K={K}: LP failed, status={result.status}")
                
        except Exception as e:
            print(f"  K={K}: error: {e}")
            continue
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: simple heuristic
    K = 100
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = np.zeros(K, dtype=np.float64)
    for k in range(2, K + 1):
        values[k - 1] = -0.01
    # Adjust f(1)
    S = np.sum(values / np.arange(1, K + 1, dtype=np.float64))
    values[0] = -S
    values = np.clip(values, -10.0, 10.0)
    
    return domain, values

# EVOLVE-BLOCK-END
