# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580

    # Fallback to seed result
    def fallback():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # The analytic strategy uses the Möbius function.
    # For f(k) = mu(k) for k >= 2 and f(1) = -sum_{k>=2} mu(k)/k,
    # M(x) = sum_{k|n} mu(k) = 0 for n > 1 and M(1) = f(1) = -sum_{k>=2} mu(k)/k.
    # Wait, the definition is M(x) = sum_{k in D} f(k) floor(x/k).
    # If f(k) = mu(k), then sum_{k|x} mu(k) = 0 for x > 1.
    # However, M(x) sums over ALL k in domain, not just divisors.
    # M(x) = sum_{k=1}^K mu(k) floor(x/k).
    # It is a known identity that sum_{k=1}^x mu(k) floor(x/k) = 1 for x >= 1.
    # Specifically, sum_{d|n} mu(d) = 1 if n=1 else 0.
    # sum_{k=1}^x mu(k) floor(x/k) = sum_{k=1}^x mu(k) sum_{j: kj <= x} 1 = sum_{n=1}^x sum_{k|n} mu(k) = 1 + 0 + ... + 0 = 1.
    # So if we set f(k) = mu(k) for all k in 1..K, then M(x) = 1 for all x in 1..K.
    # What about x > K?
    # The problem states M(x) is computed for x in [1, 10K].
    # For x > K, floor(x/k) is not just counting divisors in the simple way if the domain is finite.
    # Let's re-read carefully: "M(x) = sum_k f(k) * floor(x/k) is computed for every integer x in [1, 10K]".
    # The domain is finite, say D = {1, ..., K}.
    # If we choose f(k) = mu(k) for k=1..K, then for x <= K, M(x) = 1.
    # For x > K, M(x) = sum_{k=1}^K mu(k) floor(x/k).
    # Does this stay <= 1.0001?
    # The sum sum_{k=1}^infty mu(k) floor(x/k) = 1 for any x.
    # The tail sum sum_{k=K+1}^infty mu(k) floor(x/k) is small.
    # Specifically, sum_{k=1}^K mu(k) floor(x/k) = 1 - sum_{k=K+1}^infty mu(k) floor(x/k).
    # The error term is bounded by O(x/K * something) or related to Mertens function.
    # Actually, sum_{k=1}^K mu(k) floor(x/k) = 1 + O(x/K) is not quite right.
    # Let's check the magnitude.
    # If K is large enough, the error should be small.
    # However, we must also satisfy the constraint: sum_k f(k)/k = 0 after clipping and f(1) adjustment.
    # The problem says: "f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the clipped values ... so that afterwards sum_k f(k)/k = 0."
    # This adjustment changes f(1).
    # If we start with f(k) = mu(k), then f(1) = 1.
    # Clipping: mu(k) is in {-1, 0, 1}, so no clipping needed.
    # S = sum_{k=1}^K mu(k)/k.
    # New f(1) = 1 - S.
    # New M(x) = f(1)floor(x/1) + sum_{k=2}^K mu(k)floor(x/k) = (1-S)x + sum_{k=2}^K mu(k)floor(x/k).
    # We know sum_{k=1}^K mu(k)floor(x/k) = 1 (for x <= K) approx.
    # sum_{k=2}^K mu(k)floor(x/k) = 1 - 1*floor(x/1) = 1 - x (for x <= K, assuming identity holds exactly for finite sum? No, identity holds for infinite sum).
    # For finite sum, sum_{k=1}^K mu(k)floor(x/k) = 1 + Error(x, K).
    # So sum_{k=2}^K mu(k)floor(x/k) = 1 + Error(x, K) - x.
    # New M(x) = (1-S)x + 1 + Error(x, K) - x = 1 + Error(x, K) - S*x.
    # We need New M(x) <= 1.0001 for all x in [1, 10K].
    # This requires 1 + Error(x, K) - S*x <= 1.0001 => Error(x, K) - S*x <= 0.0001.
    # S = sum_{k=1}^K mu(k)/k. As K -> inf, S -> 0 (since sum mu(k)/k = 1/zeta(1) = 0).
    # For finite K, S is small.
    # The term -S*x could become large negative, which is good for the upper bound <= 1.0001.
    # But wait, if S is positive, -S*x is negative. If S is negative, -S*x is positive.
    # S = 1 - 1/2 + 0 - 1/3 + ... 
    # Mertens function M(K) = sum mu(k). S is related to M(K)/K roughly? No.
    # Let's estimate S for K=200000.
    # sum_{k=1}^K mu(k)/k is very close to 0.
    # If S is small, the term -S*x is small.
    # The main risk is Error(x, K).
    # Error(x, K) = sum_{k=K+1}^infty mu(k) floor(x/k).
    # For x <= K, floor(x/k) = 0 for k > x. So Error is 0 for x <= K.
    # So for x in [1, K], New M(x) = 1 - S*x.
    # We need 1 - S*x <= 1.0001. This is always true if S >= -0.0001/x.
    # Since S is small (approx 0), this is likely safe.
    # What about x in [K+1, 10K]?
    # For x > K, floor(x/k) is non-zero for k up to x.
    # But our sum only goes to K.
    # M(x) = sum_{k=1}^K mu(k) floor(x/k).
    # Identity: sum_{k=1}^x mu(k) floor(x/k) = 1.
    # So sum_{k=1}^K mu(k) floor(x/k) = 1 - sum_{k=K+1}^x mu(k) floor(x/k).
    # Let T = sum_{k=K+1}^x mu(k) floor(x/k).
    # For x <= 10K, the range K+1..x has length at most 9K.
    # floor(x/k) for k in (K, 2K] is 1.
    # For k in (2K, 3K] is 1 if x >= 3K, else 1 or 0?
    # If x <= 2K, floor(x/k) = 1 for k in (x/2, x].
    # This error term T can be significant.
    # However, we also have the -S*x term from the f(1) adjustment.
    # New M(x) = 1 - T(x) - S*x.
    # We need 1 - T(x) - S*x <= 1.0001 => -T(x) - S*x <= 0.0001.
    # If T(x) is positive, this helps. If T(x) is negative, it hurts.
    # T(x) = sum_{k=K+1}^x mu(k) floor(x/k).
    # This is a partial sum of the Möbius function weighted by floor(x/k).
    # Given the randomness of mu(k), T(x) might fluctuate.
    # To be safe, we can scale f(k) by a factor alpha < 1.
    # Let f(k) = alpha * mu(k).
    # Clipping: alpha * mu(k) is in [-alpha, alpha]. If alpha <= 10, no clipping.
    # S' = sum alpha * mu(k)/k = alpha * S.
    # f(1) adjusted: f(1) = alpha * 1 - alpha * S = alpha(1-S).
    # M(x) = alpha * sum_{k=1}^K mu(k) floor(x/k) + (alpha(1-S)) * x - alpha * 1 * x ?
    # No, M(x) = sum_{k=1}^K f(k) floor(x/k).
    # f(1) is part of the sum.
    # M(x) = f(1) x + sum_{k=2}^K alpha mu(k) floor(x/k).
    # Sum_{k=2}^K alpha mu(k) floor(x/k) = alpha (Sum_{k=1}^K mu(k) floor(x/k) - 1*x).
    # Let S_K(x) = Sum_{k=1}^K mu(k) floor(x/k).
    # M(x) = alpha(1-S)x + alpha(S_K(x) - x) = alpha(S_K(x) - Sx).
    # We know S_K(x) = 1 - T(x) for x > K, and 1 for x <= K.
    # For x <= K: M(x) = alpha(1 - Sx).
    # For x > K: M(x) = alpha(1 - T(x) - Sx).
    # We need M(x) <= 1.0001.
    # Also we want to maximize a = -sum f(k) log(k)/k.
    # a = -sum_{k=1}^K alpha mu(k) log(k)/k (approx, f(1) term is small or handled).
    # Actually, the metric is a = -sum_k f(k) log(k)/k.
    # If we scale by alpha, a scales by alpha.
    # So we want alpha as large as possible while maintaining feasibility.
    # The constraint is alpha * max_x (S_K(x) - Sx) <= 1.0001.
    # Also M(x) can be negative, which is fine (no lower bound constraint other than feasibility check M(x) <= 1.0001).
    # Wait, is there a lower bound? "if any M(x) > 1.0001 the function is infeasible".
    # So we only care about the upper bound.
    # We need alpha <= 1.0001 / max_x (S_K(x) - Sx).
    # Note that S_K(x) - Sx can be negative, in which case it doesn't constrain alpha from above (since alpha > 0).
    # We only care about the maximum positive value of (S_K(x) - Sx).
    # Let C = max_{x in [1, 10K]} (S_K(x) - Sx).
    # If C > 0, alpha = 1.0001 / C.
    # If C <= 0, we can choose alpha = 10 (max allowed by clipping, though mu is +/-1 so alpha=1 is natural, but we can go up to 10 if we want? No, f(k) is clipped to [-10, 10]. If alpha=10, f(k)=10*mu(k) is in [-10, 10]. So alpha can be up to 10).
    # But wait, if C <= 0, M(x) is always <= 0, so any alpha works?
    # Let's check x=1. S_K(1) = mu(1)*1 = 1. S = sum mu(k)/k.
    # Term = 1 - S.
    # S is small. So Term is approx 1.
    # So C is at least approx 1.
    # Thus alpha will be approx 1.
    # This suggests the analytic solution with mu(k) gives a score proportional to sum mu(k) log(k)/k.
    # sum_{k=1}^K mu(k) log(k)/k is related to the derivative of zeta?
    # sum mu(k)/k^s = 1/zeta(s).
    # d/ds (1/zeta(s)) = -sum mu(k) log(k)/k^s.
    # At s=1, this diverges? No, 1/zeta(s) has a simple zero at s=1? No, zeta(s) has a pole, so 1/zeta(s) has a zero.
    # Actually, sum mu(k)/k = 0.
    # The sum sum mu(k) log(k)/k is conditionally convergent?
    # Let's just compute it numerically for K=200000.
    
    # Strategy:
    # 1. Compute mu(k) for k=1..K.
    # 2. Compute S = sum mu(k)/k.
    # 3. Compute C = max_{x in 1..10K} (S_K(x) - S*x).
    # 4. Set alpha = min(10.0, 1.0001 / C) if C > 0 else 10.0.
    # 5. Set f(k) = alpha * mu(k).
    # 6. Return domain 1..K and values f.
    
    # To speed up, we can sample x or use a smaller K if time is tight, but 200000 is fast enough for sieve.
    # Computing S_K(x) for all x in 1..10K takes O(K log K) or O(K sqrt K)?
    # S_K(x) = sum_{k=1}^K mu(k) floor(x/k).
    # We can compute this for all x efficiently using a difference array or similar?
    # floor(x/k) is constant for ranges of x.
    # For a fixed k, floor(x/k) = j for x in [j*k, (j+1)*k - 1].
    # So we can add mu(k) to a difference array for each segment.
    # Total operations sum_{k=1}^K (10K/k) = 10K log K.
    # For K=200000, 10*200000*12 ~ 2.4M operations. Very fast.
    
    K = 200000
    if time.time() > deadline - 10:
        return fallback()
        
    # Sieve for Mobius
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
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
                
    # mu[1] = 1
    mu[1] = 1
    
    # Compute S = sum_{k=1}^K mu(k)/k
    k_arr = np.arange(1, K + 1, dtype=np.float64)
    S = np.sum(mu / k_arr)
    
    # Compute S_K(x) for x in 1..10K
    max_x = 10 * K
    S_K = np.zeros(max_x + 1, dtype=np.float64)
    
    # Use difference array technique
    # diff[x] += val, diff[x_end+1] -= val
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for k in range(1, K + 1):
        mk = mu[k]
        if mk == 0:
            continue
        # floor(x/k) takes value j for x in [j*k, min((j+1)*k-1, max_x)]
        j = 1
        while j * k <= max_x:
            lo = j * k
            hi = min((j + 1) * k - 1, max_x)
            diff[lo] += mk
            diff[hi + 1] -= mk
            j += 1
            
    # Prefix sum to get S_K(x)
    np.cumsum(diff, out=diff)
    S_K = diff[:max_x+1]
    
    # Compute C = max (S_K(x) - S*x)
    x_arr = np.arange(1, max_x + 1, dtype=np.float64)
    terms = S_K - S * x_arr
    C = np.max(terms)
    
    if C > 0:
        alpha = 1.0001 / C
    else:
        alpha = 10.0
        
    # Cap alpha at 10 to respect clipping bounds (mu is +/-1, so alpha*mu is +/-alpha)
    if alpha > 10.0:
        alpha = 10.0
        
    # Construct values
    values = alpha * mu.astype(np.float64)
    domain = np.arange(1, K + 1, dtype=np.int64)
    
    # Verify time
    if time.time() > deadline - 5:
        return fallback()
        
    return domain, values

# EVOLVE-BLOCK-END
