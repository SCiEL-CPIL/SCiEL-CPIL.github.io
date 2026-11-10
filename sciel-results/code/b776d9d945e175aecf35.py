import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max] after clipping and f(1) adjustment."""
    # Clip values to [-10, 10]
    clipped = np.clip(values, -10.0, 10.0)
    
    # S = sum_k f(k)/k over the clipped values
    S = np.sum(clipped / domain.astype(np.float64))
    
    # f(1) is replaced by f(1) - S
    # If 1 is in domain, adjust it. If not, it's 0 - S = -S.
    # But for M(x) calculation, we use the full sum.
    # M(x) = sum_k f_adjusted(k) * floor(x/k)
    # Note: floor(x/1) = x.
    
    # Efficient check using the structure of M(x)
    # M(x) = sum_{j=1}^x sum_{k|j} f_adjusted(k)
    # Let h(j) = sum_{k|j} f_adjusted(k)
    # M(x) = cumsum(h[1..x])
    
    x_max = 10 * K_max
    h = np.zeros(x_max + 1, dtype=np.float64)
    
    for i in range(len(domain)):
        k = int(domain[i])
        f_val = clipped[i]
        if k == 1:
            f_val = f_val - S # Adjusted f(1)
        
        if k > x_max:
            continue
            
        # Add f_val to h[k], h[2k], h[3k], ...
        h[k::k] += f_val
        
    M_vals = np.cumsum(h[1:])
    return np.max(M_vals) <= tolerance

def _compute_a(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k."""
    clipped = np.clip(values, -10.0, 10.0)
    S = np.sum(clipped / domain.astype(np.float64))
    
    # Create adjusted array
    adjusted = clipped.copy()
    for i, k in enumerate(domain):
        if k == 1:
            adjusted[i] = clipped[i] - S
            break
    
    k_float = domain.astype(np.float64)
    return -np.sum(adjusted * np.log(k_float) / k_float)

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) log(k) / k subject to M(x) <= 1.
    """
    start_time = time.time()
    deadline = 580  # Leave 20s margin for the 600s limit (or 630s kill)
    
    # Fallback seed
    seed_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    seed_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    best_domain = seed_domain
    best_values = seed_values
    best_a = _compute_a(seed_domain, seed_values)
    
    # Try LP approach for small K using sparse matrix
    # K=1000 worked well in current program (1.0745)
    # Can we go higher?
    # LP size is (10K) x K.
    # Sparse matrix should allow larger K.
    # Let's try K=2000, 3000.
    
    K_candidates = [3000, 2000, 1500, 1000]
    
    for K in K_candidates:
        if time.time() - start_time > deadline - 30:
            break
            
        try:
            # Construct sparse constraint matrix A_ub
            # Variables: f(1), ..., f(K)
            # Constraints: M(x) <= 1 for x in 1..10K
            # M(x) = sum_{j=1}^K f(j) * floor(x/j)
            
            # Build sparse matrix
            # Row x (0-indexed) corresponds to constraint for x+1
            # Col j (0-indexed) corresponds to variable f(j+1)
            # Entry A[x, j] = floor((x+1)/(j+1))
            
            # We can build this by iterating over j (variables) and adding to rows
            # For each j, floor(x/j) is constant for ranges of x.
            # But simply iterating might be slow.
            
            # Alternative: Use the fact that floor(x/j) changes at multiples of j.
            # However, for simplicity and speed, let's try to build it efficiently.
            
            # Actually, for K=3000, 10K=30000.
            # 30000 rows, 3000 cols.
            # Number of non-zeros: Sum_{j=1}^K (10K/j) approx 10K ln K.
            # 30000 * ln(3000) ~ 30000 * 8 = 240,000.
            # This is very sparse!
            
            rows = []
            cols = []
            data = []
            
            for j in range(1, K + 1):
                # For each j, determine the values of floor(x/j) for x in 1..10K
                # floor(x/j) = q for x in [q*j, (q+1)*j - 1]
                max_q = (10 * K) // j
                
                for q in range(1, max_q + 1):
                    x_start = q * j
                    x_end = min((q + 1) * j - 1, 10 * K)
                    
                    # Rows are 0-indexed, so x-1
                    r_start = x_start - 1
                    r_end = x_end - 1
                    
                    # Add entries for rows r_start to r_end
                    # This is a block of identical values
                    # We can add them in a loop or use slice assignment if we were building dense,
                    # but for sparse we add individual entries or use scipy.sparse tools.
                    # Since the block size is j, and sum of j is not too bad?
                    # Wait, summing over q and then looping x is O(10K log K).
                    # 30000 * 8 = 240k operations. This is fast.
                    
                    for x_idx in range(r_start, r_end + 1):
                        rows.append(x_idx)
                        cols.append(j - 1)
                        data.append(q)
                        
            # This loop is O(10K log K). For K=3000, 10K=30000.
            # 30000 * ln(3000) ~ 240,000 iterations.
            # This is very fast in Python? Maybe a bit slow.
            # Let's optimize the inner loop.
            
            # Actually, let's just use the dense construction if K is small enough,
            # or a smarter sparse construction.
            
            # Let's try the sparse construction with list appending.
            
            # Optimization:
            # Instead of appending one by one, we can use numpy to generate indices.
            
            rows_arr = []
            cols_arr = []
            data_arr = []
            
            for j in range(1, K + 1):
                max_q = (10 * K) // j
                if max_q == 0:
                    continue
                
                # q ranges from 1 to max_q
                # For each q, x ranges from q*j to min((q+1)*j-1, 10K)
                
                # We can vectorize the generation of rows, cols, data for this j?
                # Or just loop q.
                
                for q in range(1, max_q + 1):
                    x_start = q * j
                    x_end = min((q + 1) * j - 1, 10 * K)
                    count = x_end - x_start + 1
                    
                    if count > 0:
                        # Rows: x_start-1 to x_end-1
                        # Cols: all j-1
                        # Data: all q
                        
                        rows_arr.extend(range(x_start - 1, x_end))
                        cols_arr.extend([j - 1] * count)
                        data_arr.extend([q] * count)
                        
            if not rows_arr:
                continue
                
            A_ub = csr_matrix((data_arr, (rows_arr, cols_arr)), shape=(10 * K, K))
            b_ub = np.ones(10 * K, dtype=np.float64)
            
            # Objective: Maximize -sum_{k=2}^K f(k) log(k)/k
            # f(1) term is 0.
            # Minimize sum_{k=2}^K f(k) log(k)/k
            c = np.zeros(K)
            k_vals = np.arange(1, K + 1)
            c[1:] = np.log(k_vals[1:]) / k_vals[1:]
            
            # Bounds -10 <= f <= 10
            bounds = [(-10.0, 10.0)] * K
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                f_vals = res.x
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = f_vals.astype(np.float64)
                
                # Verify feasibility with the exact evaluator logic (clipping + f(1) adj)
                if _check_feasibility(domain, values, K):
                    a_val = _compute_a(domain, values)
                    if a_val > best_a:
                        best_a = a_val
                        best_domain = domain
                        best_values = values
                        # print(f"K={K}, a={a_val}")
                        
        except Exception:
            continue
            
    return best_domain, best_values
