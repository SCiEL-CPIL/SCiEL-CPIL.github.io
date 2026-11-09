import numpy as np
import time
import sys
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def prime_number_theorem_partial_function():
    start_time = time.time()
    timeout = 500  # Safety margin
    
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0

    # Try a range of K values
    K_candidates = [50, 100, 200, 500, 1000]
    
    for K in K_candidates:
        if time.time() - start_time > timeout:
            break
            
        try:
            # Variables f_2 ... f_K
            n_vars = K - 1
            indices = np.arange(2, K + 1)
            
            # Objective: maximize -sum f(k) log(k)/k
            # linprog minimizes, so we minimize sum f(k) log(k)/k
            # c_k = log(k)/k
            c = np.log(indices) / indices
            
            # Constraints: sum {x/k} f_k >= -1.0001 for x in 1..10K
            # This is A @ f >= b
            # linprog handles A_ub @ x <= b_ub
            # So we use -A @ f <= 1.0001
            # A_{x,k} = {x/k}
            
            x_vals = np.arange(1, 10 * K + 1)
            # We need to construct A sparse
            # Rows: x, Cols: k (2..K)
            # Entry: {x/k}
            
            # To save memory and time, we can just iterate or vectorize carefully
            # But 10K rows and K cols for K=1000 is 10,000 x 999.
            # This is dense enough to be manageable, but sparse is better.
            
            # Let's build row by row or use broadcasting if K is small
            if K <= 200:
                # Dense might be faster for small K
                # x_vals shape (10K, 1), indices shape (1, K-1)
                frac_parts = (x_vals[:, None] / indices[None, :]) % 1
                A = -frac_parts  # We want -A @ f <= 1.0001
                b = np.full(10 * K, 1.0001)
            else:
                # Sparse construction
                rows = []
                cols = []
                data = []
                for x in x_vals:
                    # Compute {x/k} for k=2..K
                    # Vectorized:
                    vals = (x / indices) % 1
                    # Only store non-zeros? {x/k} is rarely 0.
                    # Just add all.
                    row_idx = np.full(K-1, len(rows)) # This is wrong, need unique row idx
                    # Actually, let's just build the matrix in chunks or use a different approach
                    pass
                
                # Simpler sparse approach:
                # For each k, it contributes to all x.
                # A is (10K, K-1)
                # Let's just use dense for K up to 500 if memory allows.
                # 10*500 = 5000 rows. 499 cols. 5000*499*8 bytes ~ 20MB. OK.
                frac_parts = (x_vals[:, None] / indices[None, :]) % 1
                A = -frac_parts
                b = np.full(10 * K, 1.0001)
            
            bounds = [(-10, 10)] * n_vars
            
            # Solve
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            
            if res.status == 0:
                f_vals = res.x
                # Check feasibility with original formulation just in case
                S = np.sum(f_vals / indices)
                # f(1) eval = -S
                # M(x) = -S*x + sum f(k) floor(x/k)
                # Max M(x)
                x_check = np.arange(1, 10*K + 1)
                M_vals = -S * x_check + np.dot(f_vals, np.floor(x_check[:, None] / indices[None, :]))
                if np.max(M_vals) <= 1.0001 + 1e-6:
                    # Calculate score
                    score = -np.sum(f_vals * np.log(indices) / indices)
                    if score > best_score:
                        best_score = score
                        # Construct domain and values
                        # domain: 1, 2, ..., K
                        domain = np.arange(1, K + 1, dtype=np.int64)
                        values = np.zeros(K, dtype=np.float64)
                        values[0] = 0.0 # f(1) orig, evaluator adjusts
                        values[1:] = f_vals
                        best_domain = domain
                        best_values = values
                        
        except Exception as e:
            print(f"Error for K={K}: {e}", file=sys.stderr)
            continue
            
    return best_domain, best_values
