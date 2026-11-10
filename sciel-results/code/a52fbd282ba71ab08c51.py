# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    # Fallback to seed result
    def fallback():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Strategy: Use the Möbius function mu(k) for k in [2, K] and include k=1
    # with f(1) = -S where S = sum_{k>=2} mu(k)/k.
    # This ensures sum_{k in domain} f(k)/k = 0, which makes M(x) = 1 for all x >= 1.
    # The constraint is M(x) <= 1.0001, so M(x)=1 is feasible.
    # The objective is a = -sum f(k) log(k) / k.
    # Since f(1) log(1) / 1 = 0, the objective is independent of f(1).
    # a = -sum_{k=2}^K mu(k) log(k) / k.
    # To maximize a, we want to include as many k as possible, especially those
    # with mu(k) = -1 (since -(-1)log(k)/k = +log(k)/k > 0) and minimize contribution
    # from mu(k) = 1 (since -(1)log(k)/k = -log(k)/k < 0).
    # However, the "inherent" PNT construction suggests using all k where mu(k) != 0.
    # Let's try K=200000.

    K = 200000
    
    if time.time() > deadline:
        return fallback()

    # Compute Möbius function up to K using linear sieve
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # Domain: k where mu[k] != 0, k >= 2
    mask = (mu != 0)
    mask[0] = False
    mask[1] = False
    domain_k_geq_2 = np.where(mask)[0]
    values_k_geq_2 = mu[domain_k_geq_2].astype(np.float64)
    m = len(domain_k_geq_2)

    if m < 1:
        return fallback()

    # Compute S = sum f(k)/k over k >= 2
    # Note: domain_k_geq_2 is int64, values_k_geq_2 is float64
    S = np.sum(values_k_geq_2 / domain_k_geq_2.astype(np.float64))

    # f(1) = -S
    f1 = -S

    # Construct full domain and values
    # domain: [1] + domain_k_geq_2
    # values: [f1] + values_k_geq_2
    domain = np.empty(m + 1, dtype=np.int64)
    values = np.empty(m + 1, dtype=np.float64)
    domain[0] = 1
    domain[1:] = domain_k_geq_2
    values[0] = f1
    values[1:] = values_k_geq_2

    # Verification step (optional but good for debugging/safety)
    # Check if M(x) <= 1.0001 for all x in [1, 10K]
    # M(x) = sum_{k in domain} f(k) floor(x/k)
    # We know analytically that sum f(k)/k = 0 implies M(x) = 1 for x >= 1?
    # Let's re-verify the logic.
    # M(x) = sum_{k=1}^K f(k) floor(x/k)
    # If f(k) = mu(k) for k>1 and f(1) = -sum_{k>1} mu(k)/k, does M(x) = 1?
    # Actually, the standard identity is sum_{d|n} mu(d) = 1 if n=1 else 0.
    # sum_{k=1}^x mu(k) floor(x/k) = 1.
    # Here our f(k) is slightly different.
    # Let's just do a quick spot check or trust the analytic derivation if time permits.
    # Given the 600s limit and K=200000, checking M(x) for x in [1, 2000000] might be slow
    # if done naively. But we can do it efficiently.
    
    # Efficient check of M(x) for x in [1, 10K]
    max_x = 10 * K
    # M(x) = f1 * x + sum_{k=2}^K f(k) floor(x/k)
    # We can compute this for a subset of x to verify, or all x if fast enough.
    # The loop in the previous program took a while. Let's try a vectorized approach for a few points
    # or just trust the math if we are confident.
    # The math:
    # sum_{k=1}^K f(k) floor(x/k)
    # = f1 * x + sum_{k=2}^K mu(k) floor(x/k)
    # We know sum_{k=1}^K mu(k) floor(x/k) = 1 for x >= 1 (if K >= x, which is true for x <= 10K? No, K=200000, max_x=2000000. So for x > K, the sum is not exactly 1).
    # Wait, the identity sum_{d|n} mu(d) = 1_{n=1} leads to sum_{k=1}^x mu(k) floor(x/k) = 1.
    # This sum is over k=1 to x.
    # Our sum is over k=1 to K.
    # If x <= K, then sum_{k=1}^K mu(k) floor(x/k) = sum_{k=1}^x mu(k) floor(x/k) = 1.
    # If x > K, the sum is partial.
    # However, f(1) is NOT mu(1)=1. f(1) = -S.
    # So M(x) = -S * x + sum_{k=2}^K mu(k) floor(x/k).
    # This is NOT necessarily 1.
    
    # Let's re-read the task carefully.
    # "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0."
    # "M(x) = sum_k f(k) * floor(x/k) is computed... if any M(x) > 1.0001 ... infeasible"
    
    # If sum f(k)/k = 0, does that imply M(x) is small?
    # M(x) = sum f(k) floor(x/k).
    # This is a Dirichlet convolution type sum.
    # If f(k) = mu(k), sum mu(k) floor(x/k) = 1 for x <= K.
    # But we modified f(1).
    # Let f(k) = mu(k) for k > 1.
    # S = sum_{k=2}^K mu(k)/k.
    # f(1) = -S.
    # M(x) = -S * x + sum_{k=2}^K mu(k) floor(x/k).
    # For x=1: M(1) = -S * 1 + sum_{k=2}^K mu(k) * 1 (since floor(1/k)=0 for k>1) = -S.
    # S is approx -1/2 (since sum mu(k)/k approx 0? No, sum mu(k)/k = 0 is the limit, but partial sums fluctuate).
    # Actually, sum_{k=1}^K mu(k)/k is close to 0.
    # If M(1) = -S, and S is small, M(1) is small.
    # But for large x, say x=K:
    # M(K) = -S * K + sum_{k=2}^K mu(k) floor(K/k).
    # We know sum_{k=1}^K mu(k) floor(K/k) = 1.
    # So sum_{k=2}^K mu(k) floor(K/k) = 1 - mu(1)floor(K/1) = 1 - 1 = 0.
    # So M(K) = -S * K + 0 = -S * K.
    # If S is not 0, M(K) can be large.
    # S = sum_{k=2}^K mu(k)/k.
    # For K=200000, S is likely small but multiplied by K=200000, it could be > 1.
    # Example: if S = 0.001, M(K) = -200. This would be infeasible.
    
    # So the simple mu(k) construction with f(1)=-S is NOT guaranteed to be feasible for large K.
    # The previous program excluded k=1 from the domain initially, then f(1) was set to -S.
    # Wait, the previous program:
    # "Domain: k where mu[k] != 0, k >= 2 (exclude k=1 to avoid large f1*x term)"
    # "f(1) = -S (since 1 not in domain, initial f(1)=0, then replaced by 0-S)"
    # It then checked M(x).
    # And it tried decreasing K until feasible.
    # This implies that for large K, the mu(k) construction is infeasible.
    
    # So I cannot just use K=200000 with mu(k). I need to ensure feasibility.
    # The previous program found a feasible K (probably smaller).
    # To improve, I should try to find the largest feasible K efficiently, or use a different function.
    
    # Alternative: Use f(k) = 0 for all k? No, score 0.
    # Use f(k) = mu(k) but clip values? No, clipping is done by evaluator.
    
    # Let's stick to the previous strategy but optimize the search.
    # The previous program tried K in [100000, 50000, ...].
    # It computed mu for each K. This is wasteful.
    # Compute mu for K=200000 once.
    # Then for each candidate K, check feasibility using the precomputed mu.
    
    # Let's do that.
    
    K_max = 200000
    # Compute mu up to K_max
    mu = np.ones(K_max + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(K_max + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
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

    # Candidate K values
    candidate_Ks = [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000]
    
    for K in candidate_Ks:
        if time.time() > deadline:
            return fallback()
            
        # Domain: k in [2, K] where mu[k] != 0
        mask = (mu[2:K+1] != 0)
        domain_k_geq_2 = np.where(mask)[0] + 2
        values_k_geq_2 = mu[domain_k_geq_2].astype(np.float64)
        m = len(domain_k_geq_2)
        
        if m < 1:
            continue
            
        S = np.sum(values_k_geq_2 / domain_k_geq_2.astype(np.float64))
        f1 = -S
        
        # Check feasibility: M(x) <= 1.0001 for all x in [1, 10K]
        # M(x) = f1 * x + sum_{k=2}^K f(k) floor(x/k)
        # We can compute this for all x in [1, 10K] efficiently?
        # 10K can be 2,000,000.
        # Direct loop over x and k is too slow.
        # Use the difference array method as in the previous program.
        
        max_x = 10 * K
        D = np.zeros(max_x + 2, dtype=np.float64)
        
        # Add contributions from k >= 2
        # For each k, f(k) is added to floor(x/k)
        # floor(x/k) = j for x in [j*k, (j+1)*k - 1]
        # So f(k) is added to D[j*k] and subtracted from D[(j+1)*k]
        
        # Vectorize this part?
        # The loop over m (number of non-zero mu) is ~ 6/pi^2 * K ~ 0.6 K.
        # For K=200000, m ~ 120000.
        # Inner loop j goes up to max_x/k = 10K/k.
        # Total operations sum_{k in domain} 10K/k ~ 10K * sum_{k in domain} 1/k ~ 10K * log(K) * density.
        # This might be too slow for K=200000 in Python loop.
        
        # Optimization: Only check a subset of x? No, must be all.
        # Can we use numpy to add the intervals?
        # For a fixed k, the intervals are [k, 2k-1], [2k, 3k-1], ...
        # We can create an array of indices for the starts and ends.
        
        # Let's try to speed up the check.
        # If K is large, the check might time out.
        # The previous program succeeded, so it must have been fast enough for the K it found.
        # It likely found a smaller K.
        
        # Let's try to do the check faster.
        # Instead of looping over all k, we can loop over j?
        # No, the structure is sparse.
        
        # Let's just run the check and see. If it's too slow, we reduce K.
        
        # Optimization: Use a larger step for D updates?
        # No, we need exact values.
        
        # Let's try to vectorize the inner loop for a specific k.
        # j = 1, 2, ..., max_x // k
        # starts = j * k
        # ends = (j+1) * k
        # D[starts] += fk
        # D[ends] -= fk
        
        # This is hard to vectorize across different k because lengths vary.
        
        # Alternative: Approximate feasibility?
        # M(x) is roughly 1 for x <= K if f(k)=mu(k).
        # The issue is the f1*x term.
        # M(x) = -S*x + 1 (approx for x<=K).
        # We need -S*x <= 0.0001.
        # Since S is sum mu(k)/k, it oscillates around 0.
        # For K=200000, S might be small.
        # If S < 0, then -S*x is positive and grows.
        # If S > 0, then -S*x is negative, which is good for upper bound.
        
        # Let's check the sign of S for various K.
        # If S > 0, the f1*x term is negative, helping feasibility.
        # If S < 0, the f1*x term is positive, hurting feasibility.
        
        # We can check S quickly.
        if S < 0:
            # f1 > 0. M(x) will have a positive linear component.
            # Likely infeasible for large x unless the mu part cancels it.
            # The mu part is bounded (oscillates). The linear part grows.
            # So if S < 0, it's likely infeasible for large K.
            # We can skip the expensive M(x) check and try smaller K.
            continue
            
        # If S >= 0, f1 <= 0. The linear term is negative.
        # M(x) = f1*x + (mu part).
        # The mu part is roughly 1 (for x<=K) or bounded.
        # So M(x) <= 1 - |f1|x + bound.
        # This is likely <= 1.0001.
        
        # So we only need to check if S >= 0.
        # If S >= 0, we assume it's feasible (with high confidence) and return.
        # To be safe, we can do a quick check for a few x values.
        
        # Quick check:
        # M(1) = f1
        # M(K) = f1*K + sum_{k=2}^K mu(k) floor(K/k)
        # sum_{k=2}^K mu(k) floor(K/k) = 1 - mu(1)K = 1 - K? No.
        # sum_{d|n} mu(d) = 1 if n=1 else 0.
        # sum_{k=1}^K mu(k) floor(K/k) = 1.
        # So sum_{k=2}^K mu(k) floor(K/k) = 1 - mu(1)*K = 1 - K.
        # Wait, mu(1)=1. floor(K/1)=K.
        # So sum_{k=2}^K ... = 1 - K.
        # M(K) = f1*K + (1 - K) = -S*K + 1 - K = 1 - K(1+S).
        # If S >= 0, then 1+S >= 1, so M(K) <= 1 - K <= 0.
        # This is definitely <= 1.0001.
        
        # What about x > K?
        # For x > K, floor(x/k) is larger.
        # But f1 is negative.
        # The mu part grows?
        # sum_{k=2}^K mu(k) floor(x/k).
        # For x = 10K, floor(10K/k) is large.
        # But the sum is still bounded by the properties of mu?
        # Actually, sum_{k=1}^K mu(k) floor(x/k) is not necessarily 1 for x > K.
        # It is related to the Mertens function.
        
        # Given the risk, let's do the full check for the first feasible S>=0 case.
        # But to save time, we can just check M(x) for x in [1, 10K] using the fast method.
        
        # Let's try the fast method.
        D = np.zeros(max_x + 2, dtype=np.float64)
        
        # Loop over k in domain
        # To speed up, we can process k in batches?
        # No, let's just run it. If it's too slow, we'll know.
        
        # Optimization: Use numba? Not available.
        # Use numpy broadcasting?
        
        # Let's try to vectorize the update for all k at once?
        # No, different k have different lengths.
        
        # Let's just run the loop.
        # If K=200000, m~120000.
        # Average 1/k ~ log(K)/K.
        # Total ops ~ 120000 * 10 * log(200000) ~ 120000 * 10 * 12 ~ 14 million.
        # Python loop: 14M iterations might take 1-2 seconds.
        # This is acceptable.
        
        for idx in range(m):
            k = int(domain_k_geq_2[idx])
            fk = values_k_geq_2[idx]
            j = 1
            while j * k <= max_x:
                lo = j * k
                hi = min((j + 1) * k - 1, max_x)
                D[lo] += fk
                if hi + 1 <= max_x:
                    D[hi + 1] -= fk
                j += 1
        
        M_k_geq_2 = np.cumsum(D)[:max_x + 1]
        M = M_k_geq_2 + f1 * np.arange(max_x + 1, dtype=np.float64)
        M[0] = 0.0
        
        if np.max(M[1:]) > 1.0001:
            continue
            
        # Feasible!
        # Construct final domain and values
        domain = np.empty(m + 1, dtype=np.int64)
        values = np.empty(m + 1, dtype=np.float64)
        domain[0] = 1
        domain[1:] = domain_k_geq_2
        values[0] = f1
        values[1:] = values_k_geq_2
        
        return domain, values

    return fallback()


# EVOLVE-BLOCK-END
