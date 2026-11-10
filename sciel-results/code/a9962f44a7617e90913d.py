import numpy as np
import time
import math
import scipy.optimize as opt

def _verify_solution(domain, values, K):
    """Verify that the solution satisfies all constraints efficiently."""
    if len(domain) != len(values):
        return False
    if len(domain) < 1:
        return False
    if np.any(domain <= 0) or np.any(domain > K):
        return False
    if not np.all(np.isfinite(values)):
        return False
    
    # Clip to [-10, 10]
    clipped = np.clip(values, -10, 10)
    
    # Compute S = sum f(k)/k
    S = np.sum(clipped / domain.astype(np.float64))
    
    # Adjust f(1)
    has_1 = 1 in domain
    if has_1:
        f1_idx = np.where(domain == 1)[0]
        f1_val = clipped[f1_idx[0]] - S
        if f1_val < -10 or f1_val > 10:
            return False
    else:
        if -S < -10 or -S > 10:
            return False
    
    # Build full f array up to K
    f = np.zeros(K + 1)
    for i, k in enumerate(domain):
        f[k] = clipped[i]
    
    # Apply adjustment
    if has_1:
        f[1] = f[1] - S
    else:
        f[1] = -S
    
    # Check M(x) for all x in [1, 10K]
    # M(x) = sum_{k=1}^K f[k] * floor(x/k)
    # We only need to check critical points: multiples of k in [1, K]
    # But to be safe, we check all x in [1, 10K] if K is small, 
    # or a subset if K is large. 
    # Given the 600s limit, we should be careful.
    # For verification, we can use the fact that M(x) is constant between multiples.
    # So we only need to check x that are multiples of any k in [1, K].
    
    max_x = 10 * K
    if max_x <= 10000:
        # Check all x
        M = np.zeros(max_x + 1)
        for k in range(1, K + 1):
            if f[k] != 0:
                M += f[k] * np.floor(np.arange(max_x + 1) / k)
        if np.any(M[1:] > 1.0001):
            return False
    else:
        # Check only multiples
        crit_x = set()
        for k in range(1, K + 1):
            if f[k] != 0:
                for m in range(1, max_x // k + 1):
                    crit_x.add(m * k)
        if not crit_x:
            return True
        for x in crit_x:
            mx = 0.0
            for k in range(1, K + 1):
                if f[k] != 0:
                    mx += f[k] * (x // k)
            if mx > 1.0001:
                return False
    return True

def _compute_metric(domain, values, K):
    """Compute the metric a = -sum f(k)*log(k)/k"""
    clipped = np.clip(values, -10, 10)
    S = np.sum(clipped / domain.astype(np.float64))
    
    f = np.zeros(K + 1)
    for i, k in enumerate(domain):
        f[k] = clipped[i]
    
    has_1 = 1 in domain
    if has_1:
        f[1] = f[1] - S
    else:
        f[1] = -S
    
    a = 0.0
    for k in range(2, K + 1):
        if f[k] != 0:
            a -= f[k] * math.log(k) / k
    return a

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580.0
    
    best_domain = None
    best_values = None
    best_metric = -np.inf
    
    # Try different K values
    K_candidates = [10, 20, 50, 100, 200, 500, 1000, 2000]
    
    for K in K_candidates:
        if time.time() > deadline - 10:
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        k_vals = np.arange(2, K + 1, dtype=np.float64)
        
        # Objective: maximize sum (log k / k) * y_k, where y_k = -f(k)
        # Minimize -sum (log k / k) * y_k
        c = -np.log(k_vals) / k_vals
        
        # Constraints:
        # 1. y_k <= 10 (since f(k) >= -10)
        # 2. -y_k <= 0 (since f(k) <= 10 => y_k >= 0) -> y_k >= 0
        # 3. sum_{k=2}^K y_k * {x/k} <= 1 for all x in [1, 10K]
        
        # Determine critical x values
        max_x = 10 * K
        # Use multiples of small k's to reduce constraint count
        # If K is small, use all multiples. If K is large, use multiples of k <= 100
        limit_k_for_crit = min(K, 100)
        crit_x_set = set()
        for k in range(2, limit_k_for_crit + 1):
            for m in range(1, max_x // k + 1):
                crit_x_set.add(m * k)
        
        # Also add x=10K and some random points if needed?
        # The multiples of small k's should cover the "peaks" of the fractional parts.
        # Let's also add a few random points to be safe
        if len(crit_x_set) < 5000:
            # Add random points
            random_x = np.random.choice(range(1, max_x + 1), size=min(1000, max_x), replace=False)
            crit_x_set.update(random_x.tolist())
        
        x_vals = np.array(sorted(crit_x_set), dtype=np.float64)
        n_constraints = len(x_vals)
        
        # Build A matrix for inequality constraints: A_ub @ y <= b_ub
        # Row i corresponds to x_i: sum_k y_k * {x_i/k} <= 1
        # {x/k} = x/k - floor(x/k)
        A_frac = (x_vals[:, None] / k_vals[None, :]) - np.floor(x_vals[:, None] / k_vals[None, :])
        
        # Bounds for y: [0, 10]
        bounds = [(0, 10)] * n_vars
        
        # Add constraint for f(1) bound?
        # f(1) = sum y_k / k. We need -10 <= f(1) <= 10.
        # Since y_k >= 0, f(1) >= 0. So we need sum y_k/k <= 10.
        # This is another linear constraint.
        A_f1 = np.array([1.0 / k_vals])
        b_f1 = 10.0
        
        # Combine constraints
        # A_total has rows from A_frac and A_f1
        # b_total has 1s and 10
        A_total = np.vstack([A_frac, A_f1[None, :]])
        b_total = np.concatenate([np.ones(n_constraints), [b_f1]])
        
        try:
            res = opt.linprog(c, A_ub=A_total, b_ub=b_total, bounds=bounds, method='highs', 
                              options={'time_limit': min(30, deadline - time.time() - 5)})
            
            if res.success and res.x is not None:
                y = res.x
                # y_k = -f(k) for k >= 2
                # f(k) = -y_k
                # f(1) = sum y_k / k
                f1 = np.sum(y / k_vals)
                
                if f1 > 10 or f1 < -10:
                    continue
                
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = np.zeros(K, dtype=np.float64)
                values[0] = f1
                values[1:] = -y
                
                # Verify
                if _verify_solution(domain, values, K):
                    metric = _compute_metric(domain, values, K)
                    if metric > best_metric:
                        best_metric = metric
                        best_domain = domain.copy()
                        best_values = values.copy()
                        
        except Exception as e:
            pass
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values
