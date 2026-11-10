# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin
    
    # Maximum K to consider
    K_max = 50000
    
    # Compute Mobius function up to K_max
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(K_max + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    # Candidate K values to try, from large to small
    # We want the largest feasible K to maximize the objective
    candidate_Ks = [50000, 20000, 10000, 5000, 2000, 1000, 500, 100, 30]
    
    best_domain = None
    best_values = None
    
    for K in candidate_Ks:
        if K > K_max:
            continue
            
        # Check time
        if time.time() > deadline - 10:
            break
            
        # Feasibility check for this K
        # M(x) = sum_{k=1}^K mu(k) * floor(x/k)
        # We need M(x) <= 1 for all x in [1, 10*K]
        
        max_x = 10 * K
        # Use difference array to compute M(x) efficiently
        # diff[i] stores the change in M(x) at x=i
        diff = np.zeros(max_x + 2, dtype=np.float64)
        
        mu_k = mu[1:K+1]
        ks = np.arange(1, K+1)
        
        # For each k, add mu(k) * floor(x/k)
        # floor(x/k) = j for x in [j*k, (j+1)*k - 1]
        # We iterate k and update diff
        # This loop is O(K * log(10K))
        for k in range(1, K + 1):
            if mu[k] == 0:
                continue
            m_k = mu[k]
            # j goes from 1 to floor(max_x / k)
            max_j = max_x // k
            # We can vectorize the updates for a fixed k?
            # start_idx = j*k
            # end_idx = (j+1)*k - 1
            # diff[start_idx] += m_k * j
            # diff[end_idx + 1] -= m_k * j
            
            # Vectorized approach for a single k
            js = np.arange(1, max_j + 1)
            starts = js * k
            ends = (js + 1) * k - 1
            # Clip ends to max_x
            ends = np.minimum(ends, max_x)
            
            # We need to be careful with array bounds
            # starts are in [k, max_x]
            # ends are in [k, max_x]
            
            # Use np.add.at for accumulation if there are overlaps?
            # No, for a fixed k, the intervals [j*k, (j+1)*k-1] are disjoint.
            # So we can just do:
            diff[starts] += m_k * js
            diff[ends + 1] -= m_k * js
            
        # Prefix sum to get M(x)
        M = np.cumsum(diff)
        
        # Check constraint: M(x) <= 1 for x in [1, max_x]
        # Note: M(0) is 0, we care about x >= 1
        if np.max(M[1:max_x+1]) <= 1.0:
            # Feasible
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = mu[1:K + 1].astype(np.float64)
            
            # Calculate objective to ensure it's better? 
            # The task says "score is proportional to a". 
            # Larger K generally means a closer to 1 (which is the max).
            # So the first feasible (largest) K is best.
            best_domain = domain
            best_values = values
            break
            
    if best_domain is None:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values
        
    return domain, values

# EVOLVE-BLOCK-END
