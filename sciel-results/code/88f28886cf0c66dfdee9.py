# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin

    K_max = 200000

    def check_feasibility(K_val, f_arr, c_val=1.0):
        """
        Check if f(k) = c_val * f_arr(k) for k in [1, K_val] is feasible.
        Feasibility means M(x) = sum_{k=1}^{K_val} f(k) * floor(x/k) <= 1.0001 for all x in [1, 10*K_val].
        """
        max_x = 10 * K_val
        g = np.zeros(max_x + 1, dtype=np.float64)
        
        # g(j) = sum_{d|j} f(d) for d <= K_val
        # We compute g efficiently by iterating over d and adding f(d) to multiples of d.
        # f(d) = c_val * f_arr(d)
        # We only need to consider d where f_arr(d) != 0 to save time, but f_arr is dense here (mu).
        # Actually, for mu, about 60% are non-zero.
        
        # Optimization: Only iterate if time allows, but this is O(K log K) ~ 200k * 12 ~ 2.4M ops, very fast.
        for d in range(1, K_val + 1):
            val = c_val * f_arr[d]
            if val != 0:
                g[d:max_x+1:d] += val
        
        M = np.cumsum(g[1:max_x+1])
        max_M = np.max(M)
        return max_M <= 1.0001, max_M

    # 1. Compute Mobius function mu(k) for k <= K_max
    mu = np.ones(K_max + 1, dtype=np.int64)
    primes = []
    is_composite = np.zeros(K_max + 1, dtype=bool)
    
    mu[1] = 1
    for i in range(2, K_max + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # 2. Determine the maximum feasible scaling factor c for the full domain [1, K_max]
    # M(x) = c * sum mu(k) floor(x/k)
    # Let T(x) = sum mu(k) floor(x/k). We need max_{x in [1, 10K]} c * T(x) <= 1.0001.
    # So c <= 1.0001 / max(T(x)).
    # If max(T(x)) <= 0, c can be arbitrarily large (limited by clipping to 10).
    # If max(T(x)) > 0, c_max = 1.0001 / max(T(x)).
    
    # Check time
    if time.time() > deadline - 10:
        # Fallback to small baseline
        return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    # Compute max T(x) for K = K_max
    feasible, max_T = check_feasibility(K_max, mu, 1.0)
    
    if feasible:
        # If max_T <= 1.0001, we can scale up.
        if max_T > 0:
            c_max = 1.0001 / max_T
        else:
            c_max = 10.0 # Limited by clipping
    else:
        # If infeasible with c=1, we must scale down or reduce K.
        if max_T > 0:
            c_max = 1.0001 / max_T
        else:
            # This shouldn't happen if max_T is the max, but if max_T is negative, 
            # it means M(x) is always negative, so c can be large. 
            # Wait, check_feasibility returns max_M. If max_M > 1.0001, it's infeasible.
            # If max_M <= 1.0001, it's feasible.
            # If max_M is negative, c_max is limited by clipping.
            c_max = 10.0

    c_max = min(c_max, 10.0)
    c_max = max(c_max, 0.0)
    
    # We want to maximize a = -sum f(k) log(k) / k.
    # f(k) = c * mu(k).
    # a = -c * sum mu(k) log(k) / k.
    # Let S = sum mu(k) log(k) / k.
    # We want to maximize -c * S.
    # If S < 0, we want c as large as possible.
    # If S > 0, we want c as small as possible (0).
    
    # For K large, S approximates -1 (since sum mu(k) log(k)/k ~ -1).
    # So -S ~ 1. Thus a ~ c.
    # So we want c as large as possible.
    
    c = c_max

    # Verify feasibility with the chosen c
    feasible_c, max_M_c = check_feasibility(K_max, mu, c)
    
    if feasible_c:
        domain = np.arange(1, K_max + 1, dtype=np.int64)
        values = (c * mu[1:K_max+1]).astype(np.float64)
        return domain, values
    else:
        # If c_max calculation was wrong or numerical issues, fallback to c=1 if feasible
        if max_T <= 1.0001:
            domain = np.arange(1, K_max + 1, dtype=np.int64)
            values = mu[1:K_max+1].astype(np.float64)
            return domain, values
        else:
            # Try reducing K to find a feasible solution with c=1 or higher
            # This is expensive, so we try a few values
            for K_try in [100000, 50000, 20000, 10000]:
                if time.time() > deadline - 5:
                    break
                feasible_t, max_T_t = check_feasibility(K_try, mu, 1.0)
                if feasible_t:
                    if max_T_t > 0:
                        c_try = min(1.0001 / max_T_t, 10.0)
                    else:
                        c_try = 10.0
                    
                    feasible_cf, _ = check_feasibility(K_try, mu, c_try)
                    if feasible_cf:
                        domain = np.arange(1, K_try + 1, dtype=np.int64)
                        values = (c_try * mu[1:K_try+1]).astype(np.float64)
                        return domain, values
                    else:
                        domain = np.arange(1, K_try + 1, dtype=np.int64)
                        values = mu[1:K_try+1].astype(np.float64)
                        return domain, values
            
            # Fallback to small baseline
            return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

# EVOLVE-BLOCK-END
