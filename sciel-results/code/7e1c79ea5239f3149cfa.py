# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy: Use the exact Mobius function mu(k) for k <= K.
    The identity sum_{k=1}^x mu(k) floor(x/k) = 1 holds for all x >= 1 in the infinite case.
    For a finite truncation at K, M(x) = sum_{k<=K} mu(k) floor(x/k) approximates 1.
    Specifically, for x <= K, sum_{k=1}^x mu(k) floor(x/k) = 1 exactly (since floor(x/k)=0 for k>x).
    Wait, the identity is sum_{d|n} mu(d) = 1 if n=1 else 0.
    Then sum_{k=1}^x mu(k) floor(x/k) = sum_{k=1}^x mu(k) sum_{j=1}^{floor(x/k)} 1 = sum_{j=1}^x sum_{k|j, k<=x} mu(k) = sum_{j=1}^x (1 if j=1 else 0) = 1.
    So for any x, if we include all k such that k|x and k<=K, the sum is 1 if we have all divisors of j for j<=x?
    No, the identity sum_{k=1}^x mu(k) floor(x/k) = 1 holds for ALL x, provided we sum up to k=x.
    If we truncate at K, then for x <= K, we sum k=1..x (since floor(x/k)=0 for k>x), so M(x)=1 exactly.
    For x > K, M(x) = sum_{k=1}^K mu(k) floor(x/k).
    The constraint is M(x) <= 1.0001 for all x in [1, 10K].
    For x <= K, M(x) = 1.
    For x > K, M(x) can deviate.
    The error term is related to the Mertens function.
    M(x) = 1 + sum_{j=K+1}^x (sum_{d|j, d<=K} mu(d) - sum_{d|j} mu(d)) ? No.
    Let's just compute M(x) for the chosen K and check the max.
    
    The objective is a = -sum_{k in domain} f(k) log(k) / k.
    If we use f(k) = mu(k) for all k <= K, then a = -sum_{k=1}^K mu(k) log(k) / k.
    This sum converges to 1.
    The current seed has a specific small set. Let's try a larger K.
    
    Constraint: K <= 200000.
    Computation cost: m * 10K. If m ~ K, cost ~ 10 K^2.
    For K=200000, 10 * (2*10^5)^2 = 4*10^11 operations. This is too slow for Python/NumPy naive loops, but vectorized NumPy might handle it?
    Actually, the evaluator computes M(x) for every integer x in [1, 10K].
    M(x) = sum_{k in domain} f(k) floor(x/k).
    This is a Dirichlet convolution with the floor function.
    We can compute this efficiently using the fact that floor(x/k) is constant for ranges of x.
    Or simply: M = np.zeros(10*K + 1)
    for k, v in zip(domain, values):
        M[k::k] += v
    This is O(K log K) roughly? No, sum_{k=1}^K K/k = K log K.
    For K=200000, K log K ~ 200000 * 12 ~ 2.4 million operations. This is very fast.
    
    So we can use K = 200000.
    We need to check if max(M[1:10K+1]) <= 1.0001.
    If not, we might need to reduce K or adjust values.
    However, the Mertens function M(x) = sum_{k<=x} mu(k) is small.
    The error in the identity for x > K is roughly bounded by M(x) or similar.
    Actually, sum_{k=1}^K mu(k) floor(x/k) = sum_{j=1}^x (sum_{d|j, d<=K} mu(d)).
    Let S(j) = sum_{d|j} mu(d) = 1 if j=1 else 0.
    Let T(j) = sum_{d|j, d<=K} mu(d).
    Then M(x) = sum_{j=1}^x T(j).
    T(j) = S(j) - sum_{d|j, d>K} mu(d).
    So M(x) = 1 - sum_{j=1}^x sum_{d|j, d>K} mu(d).
    The term sum_{d|j, d>K} mu(d) is non-zero only if j has a divisor d > K.
    For x <= 10K, the largest divisor d can be x.
    This deviation is expected to be small.
    
    Let's implement:
    1. Compute mu(k) for k=1..K.
    2. Set domain = k where mu(k) != 0.
    3. Set values = mu(k) for those k.
    4. Compute M(x) for x=1..10K using the vectorized update method.
    5. Check max(M) <= 1.0001.
    6. If feasible, return.
    7. If not, maybe reduce K? Or clip values?
       The task says "values are clipped to [-10, 10]". mu(k) is in {-1, 0, 1}, so no clipping needed.
       "f(1) is replaced by f(1) - S".
       S = sum f(k)/k. For mu, sum mu(k)/k converges to 0.
       For finite K, S_K = sum_{k=1}^K mu(k)/k.
       The evaluator does this automatically.
       
    We just need to ensure M(x) <= 1.0001 BEFORE the f(1) adjustment?
    "M(x) = sum_k f(k) * floor(x/k) is computed ... if any M(x) > 1.0001 ... infeasible".
    The text says: "Then f(1) is replaced by ... so that afterwards sum f(k)/k = 0. M(x) ... is computed ...".
    This implies M(x) is computed on the ADJUSTED values.
    So we must simulate the adjustment in our feasibility check.
    
    Steps:
    1. Generate mu for k=1..K.
    2. domain = [k for k in 1..K if mu[k] != 0]
    3. values = [mu[k] for k in domain]
    4. Clip values to [-10, 10] (no-op for mu).
    5. Compute S = sum(values / domain).
    6. Find index of 1 in domain. If present, values[idx] -= S.
       If 1 not in domain, f(1) is effectively 0? "f(1) = 0 before this if 1 is not in the domain".
       Since mu(1)=1, 1 is always in domain for K>=1.
    7. Compute M(x) for x=1..10K with adjusted values.
    8. Check max(M) <= 1.0001.
    
    If K=200000 is too risky, we can start with a smaller K and increase.
    But K=200000 should be fine. The Mertens function M(n) is O(sqrt(n)) typically.
    The error term in the Dirichlet hyperbola method suggests small deviations.
    
    Let's try K=200000.
    
    One detail: "distinct positive integers". mu(k) is 0 for non-square-free.
    So domain size m is approx 6/pi^2 * K ~ 0.6 * 200000 = 120,000.
    Memory is fine.
    
    Let's write the code.
    """
    K = 200000
    
    # Compute Mobius function up to K
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    # Sieve of Eratosthenes to find primes and compute mu
    # mu[1] = 1
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 > K:
                continue
            for j in range(i2, K + 1, i2):
                mu[j] = 0
    
    # Build domain and values
    mask = mu != 0
    domain = np.where(mask)[0]
    # domain starts at 1 because mu[1]=1
    values = mu[domain].astype(np.float64)
    
    # Clip values (no-op but required by spec logic)
    values = np.clip(values, -10.0, 10.0)
    
    # Compute S = sum f(k)/k
    S = np.sum(values / domain)
    
    # Adjust f(1)
    # 1 is always in domain (mu[1]=1)
    idx_1 = np.where(domain == 1)[0][0]
    values[idx_1] -= S
    
    # Now compute M(x) for x in [1, 10K]
    max_x = 10 * K
    M = np.zeros(max_x + 1, dtype=np.float64)
    
    # Efficient computation of M(x) = sum_{k in domain} f(k) floor(x/k)
    # M[x] = sum_{k} f[k] * (x // k)
    # We can iterate over k and add f[k] to M[k], M[2k], ...
    # This is O(K log K)
    for k, v in zip(domain, values):
        if v == 0:
            continue
        M[k::k] += v
        
    # Check feasibility
    # M[0] is unused, we care about x in [1, 10K]
    if np.max(M[1:]) > 1.0001:
        # If infeasible, we might need to reduce K or tweak.
        # For now, let's assume K=200000 is feasible.
        # If not, we could try a smaller K.
        # Let's try to find the largest feasible K?
        # Or just return a smaller safe K like 10000?
        # The seed was very small.
        # Let's try K=10000 first to be safe?
        # No, the score is proportional to a. Larger K -> a closer to 1.
        # Let's check if K=200000 works. If not, fallback.
        
        # Let's try to reduce K until feasible.
        # But recomputing mu is expensive.
        # We can just use the precomputed mu and slice.
        
        # Let's check the max value.
        max_val = np.max(M[1:])
        # print(f"K={K}, max M={max_val}")
        
        # If infeasible, try smaller K
        # We'll do a binary search or linear reduction?
        # Actually, the error grows with K?
        # Let's just try K=100000, 50000, etc.
        # But we already computed mu for 200000.
        
        feasible_K = K
        while feasible_K > 100:
            # Slice domain and values to feasible_K
            mask_k = domain <= feasible_K
            dom_k = domain[mask_k]
            val_k = values[mask_k]
            
            # Recalculate S and adjust f(1) for this subset
            S_k = np.sum(val_k / dom_k)
            idx_1_k = np.where(dom_k == 1)[0][0]
            val_k[idx_1_k] -= S_k
            
            # Compute M for this subset
            max_x_k = 10 * feasible_K
            M_k = np.zeros(max_x_k + 1, dtype=np.float64)
            for k, v in zip(dom_k, val_k):
                if v == 0:
                    continue
                M_k[k::k] += v
            
            if np.max(M_k[1:]) <= 1.0001:
                return dom_k, val_k
            feasible_K //= 2
            
        # If all failed, return the smallest one (should be feasible)
        mask_k = domain <= feasible_K
        dom_k = domain[mask_k]
        val_k = values[mask_k]
        S_k = np.sum(val_k / dom_k)
        idx_1_k = np.where(dom_k == 1)[0][0]
        val_k[idx_1_k] -= S_k
        return dom_k, val_k

    return domain, values

# EVOLVE-BLOCK-END
