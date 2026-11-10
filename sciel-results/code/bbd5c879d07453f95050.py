import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    # Evaluator kills at 630s. Use 580s as soft deadline to allow for cleanup/return.
    deadline = 580.0
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try different K values.
    # Larger K allows for better optimization of the score.
    # K=100 is a baseline. K=1000-3000 should yield significant improvement.
    # We stop if time runs out.
    k_values = [100, 200, 500, 1000, 2000, 3000]
    
    for K in k_values:
        # Check time
        if time.time() > deadline - 20:
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        ks = np.arange(2, K + 1)
        # Objective: minimize sum( (log(k)/k) * f(k) )
        c = np.log(ks) / ks
        
        # Constraints:
        # 1. M(x) <= 1 for x in [1, 10K]
        #    M(x) = - sum_{k=2}^K f(k) {x/k}
        #    So: - sum_{k=2}^K f(k) {x/k} <= 1
        # 2. -10 <= f(k) <= 10
        # 3. -10 <= f(1) <= 10
        #    f(1) = -S, where S = sum_{k=2}^K f(k)/k
        #    -10 <= -S <= 10  =>  -10 <= S <= 10
        #    S <= 10 => sum (1/k) f(k) <= 10
        #    S >= -10 => sum (-1/k) f(k) <= 10
        
        A_rows = []
        b_vals = []
        
        max_x = 10 * K
        # Process constraints in chunks to manage peak memory if K is large
        # For K=3000, 10K=30000 rows, 2999 cols.
        # Full array is ~720MB. Chunking helps if we want to be safer, 
        # but vstacking is fine. Let's chunk to be safe on memory spikes.
        chunk_size = 2000
        
        for x_start in range(1, max_x + 1, chunk_size):
            x_end = min(x_start + chunk_size, max_x + 1)
            x_chunk = np.arange(x_start, x_end)
            # Compute {x/k} for all x in chunk and all k
            # x_chunk shape: (chunk_len,)
            # ks shape: (n_vars,)
            # Result shape: (chunk_len, n_vars)
            mods = x_chunk[:, None] % ks[None, :]
            row_block = - (mods / ks[None, :])
            A_rows.append(row_block)
            b_vals.extend([1.0] * len(x_chunk))
            
        # Add S constraints
        s_coeff = 1.0 / ks
        # S <= 10
        A_rows.append(s_coeff[None, :])
        b_vals.append(10.0)
        # -S <= 10
        A_rows.append(-s_coeff[None, :])
        b_vals.append(10.0)
        
        A_ub = np.vstack(A_rows)
        b_ub = np.array(b_vals)
        
        bounds = [(-10, 10)] * n_vars
        
        try:
            # Use HiGHS solver with a time limit to prevent hanging on a single K
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                          options={'time_limit': 40})
            
            if res.success:
                g = res.x
                # Construct f
                f = np.zeros(K + 1)
                f[2:K+1] = g
                S = np.sum(g / ks)
                f[1] = -S
                
                # Verify f(1) is within bounds
                if abs(f[1]) > 10.0:
                    # This should not happen if constraints are satisfied, 
                    # but if it does, the evaluator will clip it, potentially
                    # violating the sum condition. We skip this solution.
                    continue
                
                # Calculate score a
                # a = - sum_{k=2}^K f(k) log(k)/k
                a = -np.sum(f[2:K+1] * c)
                
                if a > best_score:
                    best_score = a
                    best_domain = np.arange(1, K + 1, dtype=np.int64)
                    best_values = f[1:K+1]
                    
        except Exception as e:
            # If an error occurs (e.g. memory), break the loop as larger K will likely fail too
            # or just continue to next K if it's a solver issue.
            # Memory errors usually indicate K is too big.
            if "Memory" in str(e) or "memory" in str(e):
                break
            continue

    if best_domain is None:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
        
    return best_domain, best_values
