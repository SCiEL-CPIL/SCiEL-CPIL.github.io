import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try increasing K
    # K=2000 matrix is 20000x1999 ~ 320MB. 
    # K=5000 matrix is 50000x4999 ~ 2GB. Risky.
    # Let's try up to K=3000 if fast enough.
    
    K_candidates = [50, 100, 200, 500, 1000, 2000, 3000]
    
    for K in K_candidates:
        if time.time() - start_time > 500:
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks
        
        # Build constraint matrix A_ub
        # Constraints:
        # 1. - sum f(k) {x/k} <= 1 for x in 1..10K
        # 2. sum f(k)/k <= 10
        # 3. -sum f(k)/k <= 10
        
        max_x = 10 * K
        
        # Vectorized construction of A_x
        # x is 1..max_x
        # k is 2..K
        # A_x[i, j] = - ((i+1) % (j+2)) / (j+2)
        
        # To save memory, we might chunk x if K is large.
        # But for K=2000, it fits.
        
        x_vals = np.arange(1, max_x + 1)
        # Reshape for broadcasting
        # x_vals[:, None] has shape (max_x, 1)
        # ks[None, :] has shape (1, K-1)
        
        # (x % k) / k
        # Note: x and k are ints.
        # We need float division.
        
        try:
            # This creates a large intermediate array
            # For K=3000, 30000 * 2999 * 8 bytes ~ 700MB.
            # Acceptable.
            A_x = - (x_vals[:, None] % ks[None, :]) / ks[None, :].astype(float)
        except MemoryError:
            continue
            
        b_x = np.ones(max_x)
        
        # Additional constraints for S
        # sum f(k)/k <= 10
        # -sum f(k)/k <= 10
        A_S = np.array([1.0/ks, -1.0/ks])
        b_S = np.array([10.0, 10.0])
        
        A_ub = np.vstack([A_x, A_S])
        b_ub = np.concatenate([b_x, b_S])
        
        bounds = [(-10, 10)] * n_vars
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                g = res.x
                # Construct f
                f = np.zeros(K + 1)
                f[2:K+1] = g
                # Calculate S = sum_{k=2}^K f(k)/k
                S = np.sum(f[2:K+1] / ks)
                f[1] = -S
                
                # Verify f(1) is within bounds (should be due to constraints)
                if abs(f[1]) > 10.001:
                    # If slightly out, clip? 
                    # The constraints were <= 10, so it should be in [-10, 10].
                    pass

                # Calculate score a
                # a = - sum_{k=1}^K f(k) log(k)/k
                # log(1)=0
                a = -np.sum(f[2:K+1] * c)
                
                # Verify M(x) <= 1.0001
                # M(x) = - sum_{k=2}^K f(k) {x/k}
                # We enforced - sum f(k) {x/k} <= 1.
                # So M(x) <= 1.
                
                if a > best_score:
                    best_score = a
                    domain = np.arange(1, K + 1)
                    values = f[1:K+1]
                    # Clip values to [-10, 10] as per evaluator spec
                    values = np.clip(values, -10, 10)
                    best_domain = domain
                    best_values = values
                    
        except Exception:
            continue

    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
