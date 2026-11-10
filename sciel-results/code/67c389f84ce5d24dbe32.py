# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    DEADLINE = 610.0  # safety margin before 630s

    def check_feasibility_fast(domain, values, K):
        """Check M(x) <= 1.0001 for all x in [1, 10K] using efficient range updates."""
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        
        for k, f in zip(domain, values):
            if f == 0.0:
                continue
            # For each j, add f to M[j*k : (j+1)*k]
            j = 1
            while j * k <= max_x:
                lo = j * k
                hi = min((j + 1) * k, max_x + 1)
                M[lo:hi] += f
                j += 1
        
        return np.max(M[1:max_x + 1])

    def compute_score(domain, values):
        """Compute the objective a = -sum_k f(k)*log(k)/k"""
        return -np.sum(values * np.log(domain) / domain)

    def compute_max_g(K, p):
        """Compute max_x h(x) where h(x) = x*sum g(k)/k - sum g(k)*floor(x/k), g(k)=1/k^p"""
        ks = np.arange(2, K + 1, dtype=np.float64)
        g_vals = 1.0 / ks**p
        sum_g_over_k = np.sum(g_vals / ks)
        
        max_x = 10 * K
        # Compute sum_{k=2}^K g(k)*floor(x/k) for all x efficiently
        # For each k, g(k)*floor(x/k) is constant on intervals [j*k, (j+1)*k - 1]
        floor_sum = np.zeros(max_x + 1, dtype=np.float64)
        
        for k_int in range(2, K + 1):
            g_k = 1.0 / (k_int**p)
            for j in range(1, max_x // k_int + 1):
                lo = j * k_int
                hi = min((j + 1) * k_int, max_x + 1)
                floor_sum[lo:hi] += g_k
        
        # h(x) = x * sum_g_over_k - floor_sum[x]
        h = np.arange(1, max_x + 1, dtype=np.float64) * sum_g_over_k - floor_sum[1:max_x + 1]
        return np.max(h)

    def compute_a_coeff(K, p):
        """Compute sum g(k)*log(k)/k where g(k)=1/k^p"""
        ks = np.arange(2, K + 1, dtype=np.float64)
        g_vals = 1.0 / ks**p
        return np.sum(g_vals * np.log(ks) / ks)

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try different power laws f(k) = -c/k^p for various p and K
    p_values = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    K_values = [50, 100, 500, 1000, 5000, 10000]

    for p in p_values:
        if time.time() > DEADLINE - 60:
            break
        
        for K in K_values:
            if time.time() > DEADLINE - 60:
                break
            
            # Compute max h(x)
            max_h = compute_max_g(K, p)
            
            if max_h <= 0:
                continue
            
            # c_max = 1.0001 / max_h
            c = 1.0001 / max_h
            
            # a = c * sum g(k)*log(k)/k
            a_coeff = compute_a_coeff(K, p)
            a_val = c * a_coeff
            
            if a_val > best_score:
                # Build domain and values
                ks_int = np.arange(2, K + 1, dtype=np.int64)
                domain = np.concatenate([[1], ks_int])
                values = np.empty(K, dtype=np.float64)
                
                # f(1) = c * sum g(k)/k
                ks_float = ks_int.astype(np.float64)
                g_vals = 1.0 / ks_float**p
                sum_g_over_k = np.sum(g_vals / ks_float)
                values[0] = c * sum_g_over_k  # f(1)
                values[1:] = -c * g_vals  # f(k) = -c/k^p
                
                # Verify feasibility
                max_val = check_feasibility_fast(domain, values, K)
                
                if max_val <= 1.0001:
                    best_score = a_val
                    best_domain = domain.copy()
                    best_values = values.copy()
                else:
                    # If infeasible, try slightly smaller c
                    c = 0.999 * c
                    values[0] = c * sum_g_over_k
                    values[1:] = -c * g_vals
                    max_val = check_feasibility_fast(domain, values, K)
                    if max_val <= 1.0001:
                        a_val = c * a_coeff
                        if a_val > best_score:
                            best_score = a_val
                            best_domain = domain.copy()
                            best_values = values.copy()
        
        if time.time() > DEADLINE - 60:
            break
    
    # Also try the original strategy with p=1 for larger K
    for K in [20000, 50000, 100000, 200000]:
        if time.time() > DEADLINE - 30:
            break
        
        ks = np.arange(2, K + 1, dtype=np.float64)
        sum_inv_sq = np.sum(1.0 / ks**2)
        
        # Compute max g(x) efficiently using the same method
        max_h = compute_max_g(K, 1.0)
        
        if max_h <= 0:
            continue
        
        c = 1.0001 / max_h
        a_coeff = np.sum(np.log(ks) / ks**2)
        a_val = c * a_coeff
        
        if a_val > best_score:
            ks_int = np.arange(2, K + 1, dtype=np.int64)
            domain = np.concatenate([[1], ks_int])
            values = np.empty(K, dtype=np.float64)
            values[0] = c * sum_inv_sq
            values[1:] = -c / ks
            
            max_val = check_feasibility_fast(domain, values, K)
            if max_val <= 1.0001:
                best_score = a_val
                best_domain = domain.copy()
                best_values = values.copy()
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: use the original seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
