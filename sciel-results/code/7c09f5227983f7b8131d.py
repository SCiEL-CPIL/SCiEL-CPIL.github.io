# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s limit
    
    # 1. Compute Mobius function up to MAX_K
    MAX_K = 200000
    mu = np.ones(MAX_K + 1, dtype=np.int64)
    primes = []
    is_prime = np.ones(MAX_K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, MAX_K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, MAX_K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 > MAX_K:
                continue
            for j in range(i2, MAX_K + 1, i2):
                mu[j] = 0
    
    # Precompute log(k)/k for objective
    k_vals = np.arange(1, MAX_K + 1)
    log_k_over_k = np.log(k_vals) / k_vals
    
    # We will evaluate candidate K values.
    # We want large K to minimize the tail error term (ln K)/K * M(K).
    # But we must ensure feasibility.
    # Let's try a set of K values, descending.
    candidate_Ks = [200000, 100000, 50000, 20000, 10000, 5000, 1000, 500]
    
    best_score = -np.inf
    best_K = 1
    best_domain = np.array([1], dtype=np.int64)
    best_values = np.array([0.0], dtype=np.float64)
    
    # Helper to check feasibility and compute score for a given K
    def evaluate_K(K):
        if K > MAX_K:
            return -np.inf, None
        
        # Compute M(x) for x in [1, 10K]
        # M(x) = sum_{k=1}^K mu(k) floor(x/k)
        # We use the additive method:
        # M(x) = sum_{n=1}^x S(n), where S(n) = sum_{d|n, d<=K} mu(d)
        # But simpler:
        # Initialize M array to 0
        # For each k in 1..K:
        #   For each multiple x of k up to 10K:
        #     M[x] += mu[k]
        
        limit = 10 * K
        M = np.zeros(limit + 1, dtype=np.float64)
        
        # Vectorized update is hard due to variable lengths, but we can use slicing
        # Or loop over k. K is at most 200000.
        # Looping 200000 times in Python is slow.
        # Let's use NumPy operations where possible.
        # M[k::k] += mu[k]
        
        for k in range(1, K + 1):
            m_val = mu[k]
            if m_val != 0:
                M[k:limit+1:k] += m_val
        
        # Check constraint: max(M[1:limit+1]) <= 1.0001
        max_M = np.max(M[1:limit+1])
        if max_M > 1.0001:
            return -np.inf, None
        
        # Compute objective: a = -sum_{k=1}^K mu(k) * log(k) / k
        # Note: The evaluator clips values to [-10, 10]. mu is in {-1, 0, 1}, so no clipping effect.
        # The evaluator also adjusts f(1) to make sum f(k)/k = 0.
        # Let's calculate the raw sum first.
        raw_sum = np.sum(mu[1:K+1] * log_k_over_k[1:K+1])
        
        # The evaluator does:
        # 1. Clip values. (No effect)
        # 2. S = sum_k f(k)/k
        # 3. f(1) = f(1) - S
        # 4. Check M(x) <= 1.0001.
        #    Wait, does the adjustment to f(1) affect M(x)?
        #    M(x) = sum f(k) floor(x/k).
        #    If we change f(1) by -S, M(x) changes by -S * floor(x/1) = -S * x.
        #    So M_new(x) = M_old(x) - S * x.
        #    We need M_new(x) <= 1.0001 for all x.
        #    This changes the feasibility condition!
        
        # Let's re-evaluate feasibility with the adjusted f(1).
        S = raw_sum
        # f_new(1) = mu(1) - S = 1 - S
        # f_new(k) = mu(k) for k > 1
        
        # M_new(x) = (1-S)*floor(x/1) + sum_{k=2}^K mu(k) floor(x/k)
        # M_old(x) = 1*floor(x/1) + sum_{k=2}^K mu(k) floor(x/k)
        # So M_new(x) = M_old(x) - S*x
        
        # We need max_{x in [1, 10K]} (M_old(x) - S*x) <= 1.0001
        # x is integer from 1 to 10K.
        x_range = np.arange(1, limit + 1)
        M_new = M[1:limit+1] - S * x_range
        
        max_M_new = np.max(M_new)
        if max_M_new > 1.0001:
            return -np.inf, None
        
        # Also check if any M_new(x) is very negative? No, only upper bound constraint.
        
        # Calculate final objective a
        # a = -sum f_new(k) log(k)/k
        # f_new(1) = 1 - S
        # f_new(k) = mu(k) for k > 1
        # sum f_new(k) log(k)/k = (1-S)*log(1)/1 + sum_{k=2}^K mu(k) log(k)/k
        # log(1) = 0.
        # So sum = sum_{k=2}^K mu(k) log(k)/k
        # Note raw_sum = mu(1)log(1)/1 + sum_{k=2}^K ... = 0 + sum_{k=2}^K ...
        # So sum f_new(k) log(k)/k = raw_sum
        # Wait.
        # S = sum_{k=1}^K f_old(k)/k.
        # f_new(1) = f_old(1) - S.
        # sum f_new(k)/k = sum f_old(k)/k - S*1/1 = S - S = 0. Correct.
        
        # Objective a = -sum f_new(k) log(k)/k
        # a = - [ (1-S)*0 + sum_{k=2}^K mu(k) log(k)/k ]
        # a = - sum_{k=2}^K mu(k) log(k)/k
        
        # We know sum_{k=1}^K mu(k) log(k)/k = raw_sum.
        # mu(1)log(1)/1 = 0.
        # So sum_{k=2}^K mu(k) log(k)/k = raw_sum.
        # Thus a = -raw_sum.
        
        score = -raw_sum
        return score, K

    # Evaluate candidates
    # We should also consider that smaller K might be feasible if larger ones aren't.
    # And we want the highest score.
    # Score approx 1 - M(K)lnK/K.
    
    for K in candidate_Ks:
        if time.time() > deadline - 60:
            break
        score, _ = evaluate_K(K)
        if score > best_score:
            best_score = score
            best_K = K
            best_domain = np.arange(1, K + 1, dtype=np.int64)
            best_values = mu[1:K+1].astype(np.float64)
    
    # If no candidate worked, fall back to a very small K (e.g., K=1)
    if best_K == 1:
        # Check K=1
        # f(1)=1. S=1. f_new(1)=0.
        # M(x) = 0. Feasible.
        # a = 0.
        # But we can do better?
        # Try K=2
        # mu=[1,-1]. S = 1 - 0.5 = 0.5.
        # f_new(1) = 0.5. f_new(2) = -1.
        # M(x) = 0.5*floor(x) - 1*floor(x/2).
        # x=1: 0.5.
        # x=2: 1 - 1 = 0.
        # x=3: 1.5 - 1 = 0.5.
        # x=4: 2 - 2 = 0.
        # Max M = 0.5 <= 1.0001. Feasible.
        # a = - [ 0.5*0 + (-1)*ln2/2 ] = ln2/2 ~ 0.346.
        
        # Let's just run the loop for small Ks if large ones failed.
        small_Ks = [100, 50, 10, 5, 2, 1]
        for K in small_Ks:
            if time.time() > deadline - 10:
                break
            score, _ = evaluate_K(K)
            if score > best_score:
                best_score = score
                best_K = K
                best_domain = np.arange(1, K + 1, dtype=np.int64)
                best_values = mu[1:K+1].astype(np.float64)
                
    return best_domain, best_values

# EVOLVE-BLOCK-END
