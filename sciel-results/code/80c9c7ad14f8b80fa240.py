import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) * log(k) / k subject to
    sum_k f(k)/k = 0 and M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K].
    """
    start_time = time.time()
    deadline = 600.0  # Leave safety margin
    
    # Maximum K to consider
    K_max = 100000
    
    # 1. Sieve for Mobius function up to K_max
    mu = np.ones(K_max + 1, dtype=np.int8)
    is_prime = np.ones(K_max + 1, dtype=bool)
    primes = []
    
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K_max + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            for j in range(i*i, K_max + 1, i*i):
                mu[j] = 0
    mu[1] = 1
    
    # Precompute log(k)/k and mu(k)/k for efficient score calculation
    k_vals = np.arange(1, K_max + 1, dtype=np.float64)
    log_k = np.log(k_vals)
    inv_k = 1.0 / k_vals
    
    # Q_K = sum_{k=1}^K mu(k) * log(k) / k  (note: log(1)=0 so k=1 contributes 0)
    weighted_mu_log = mu.astype(np.float64) * log_k * inv_k
    Q_cumulative = np.cumsum(weighted_mu_log)
    
    # H_K = sum_{k=2}^K mu(k)/k  (Note: mu(1)=1, so sum_{k=1}^K mu(k)/k = 1 + H_K)
    # Actually, the constraint is sum_{k=1}^K f(k)/k = 0.
    # f(k) = c * mu(k) for k >= 2, and f(1) is chosen to satisfy the constraint.
    # Let f(1) = v_1.
    # v_1/1 + c * sum_{k=2}^K mu(k)/k = 0
    # v_1 = -c * sum_{k=2}^K mu(k)/k
    # Let S_K = sum_{k=2}^K mu(k)/k.
    # Then f(1) = -c * S_K.
    
    weighted_mu = mu.astype(np.float64) * inv_k
    # Sum of mu(k)/k for k=2..K
    # cumsum[0] is mu[1]/1 = 1.
    # We want sum for k=2..K, which is cumsum[K-1] - cumsum[0] (if 0-indexed array of size K)
    # Let's just compute cumulative sum of mu(k)/k.
    S_cumulative = np.cumsum(weighted_mu)
    # S_cumulative[K-1] = sum_{k=1}^K mu(k)/k
    # So S_K = S_cumulative[K-1] - 1.0 (since mu(1)/1 = 1)
    
    best_score = -np.inf
    best_K = 1
    best_c = 1.0
    best_S_K = 0.0
    
    # Candidate K values
    # We want to find K where the feasibility constraint allows a large c,
    # and Q_K is sufficiently negative.
    # M(x) = sum_{k=1}^K f(k) floor(x/k)
    # = f(1)*x + sum_{k=2}^K c*mu(k) floor(x/k)
    # = -c*S_K*x + c*sum_{k=2}^K mu(k) floor(x/k)
    # = c * [ sum_{k=2}^K mu(k) floor(x/k) - S_K * x ]
    
    # Let G_K(x) = sum_{k=2}^K mu(k) floor(x/k) - S_K * x
    # Then M(x) = c * G_K(x)
    # Constraint: c * G_K(x) <= 1.0001 for all x in [1, 10K]
    # If max G_K(x) > 0, then c <= 1.0001 / max G_K(x)
    # Score a = -sum f(k) log(k)/k = -[f(1)*log(1)/1 + c*sum_{k=2}^K mu(k) log(k)/k]
    # = -[0 + c * Q_K]  (since log(1)=0)
    # Note: Q_K defined above as sum_{k=1}^K mu(k) log(k)/k, which is same as sum_{k=2}^K since log(1)=0.
    # So a = -c * Q_K
    
    # We want to maximize a.
    # a = -c * Q_K.
    # If Q_K > 0, we want c as small as possible (but c must be positive? No, c can be negative?
    # Wait, if c is negative, M(x) = c * G_K(x).
    # If G_K(x) has both positive and negative values, we need c*G_K(x) <= 1.0001.
    # If c > 0, we need c * max(G_K) <= 1.0001.
    # If c < 0, we need c * min(G_K) <= 1.0001 (since c is negative, min is negative, product is positive).
    # Also, we usually want c > 0 to make sense with "scaling".
    # If Q_K < 0, then -Q_K > 0. To maximize a, we want c as large as possible.
    # If Q_K > 0, then -Q_K < 0. To maximize a (make it less negative), we want c as small as possible.
    # But c is determined by the constraint.
    
    # Let's assume we pick c to saturate the constraint: c = 1.0001 / C_max where C_max = max_{x} G_K(x).
    # This assumes C_max > 0. If C_max <= 0, then G_K(x) <= 0 for all x, so c can be arbitrarily large?
    # No, if G_K(x) <= 0, then c * G_K(x) <= 0 <= 1.0001 for any c > 0.
    # But we also have the clipping constraint on values: f(k) in [-10, 10].
    # f(1) = -c * S_K. f(k) = c * mu(k).
    # So |c * mu(k)| <= 10 => c <= 10 (since |mu(k)| <= 1).
    # And |c * S_K| <= 10.
    # So c is bounded by 10.
    
    # Let's iterate over K.
    K_candidates = []
    # Include a wide range of K
    # Small K for quick check
    for k in range(10, 100, 2):
        K_candidates.append(k)
    # Medium K
    for k in range(100, 1000, 10):
        K_candidates.append(k)
    # Large K
    for k in range(1000, K_max + 1, 50):
        K_candidates.append(k)
    
    for K in K_candidates:
        if time.time() > deadline:
            break
        
        # Compute S_K = sum_{k=2}^K mu(k)/k
        S_K = S_cumulative[K-1] - 1.0
        
        # Compute Q_K = sum_{k=1}^K mu(k) log(k)/k
        Q_K = Q_cumulative[K-1]
        
        # Compute G_K(x) for x in [1, 10K]
        # G_K(x) = sum_{k=2}^K mu(k) floor(x/k) - S_K * x
        # We can compute H_K(x) = sum_{k=2}^K mu(k) floor(x/k) using difference array.
        # Let A[x] = sum_{d|x, d>=2} mu(d).
        # Then H_K(x) = sum_{j=1}^x A[j].
        # Wait, sum_{k=2}^K mu(k) floor(x/k) = sum_{k=2}^K mu(k) sum_{j=1}^{floor(x/k)} 1
        # = sum_{j=1}^x sum_{k=2}^K mu(k) [k|j]
        # = sum_{j=1}^x (sum_{d|j} mu(d) - mu(1))   (since sum_{d|j} mu(d) includes d=1)
        # = sum_{j=1}^x (1_{j=1} - 1)
        # = sum_{j=1}^x (-1) + 1 (if x>=1) ?
        # Let's re-evaluate.
        # sum_{d|j} mu(d) is 1 if j=1, 0 otherwise.
        # So sum_{k=2}^K mu(k) [k|j] = (sum_{d|j, d<=K} mu(d)) - mu(1)[1|j]
        # If K >= j, sum_{d|j, d<=K} mu(d) = 1_{j=1}.
        # So for j=1: sum = 1 - 1 = 0.
        # For j>1: sum = 0 - 1 = -1 (if K >= j).
        # This assumes K >= x (since j goes up to x).
        # If K < x, then for j > K, the sum is not necessarily 0 or 1.
        # But in our case, we compute for x in [1, 10K]. So x can be up to 10K.
        # So K is not always >= x.
        
        # Let's use the direct computation of A[x] = sum_{d|x, d<=K, d>=2} mu(d).
        # A = np.zeros(10*K + 1)
        # for k in range(2, K+1):
        #     if mu[k] != 0:
        #         A[k::k] += mu[k]
        # H = np.cumsum(A)
        # G = H - S_K * np.arange(1, 10*K+1)
        
        N = 10 * K
        A = np.zeros(N + 1, dtype=np.float64)
        # Only iterate over non-zero mu
        non_zero_k = np.where(mu[2:K+1] != 0)[0] + 2
        for k in non_zero_k:
            A[k::k] += mu[k]
        
        H = np.cumsum(A)
        # H[0] = 0. H[x] for x>=1 is sum_{j=1}^x A[j] = sum_{k=2}^K mu(k) floor(x/k).
        
        x_vals = np.arange(1, N + 1, dtype=np.float64)
        G = H[1:] - S_K * x_vals
        
        C_max = np.max(G)
        
        # Determine c
        if C_max > 0:
            c_bound = 1.0001 / C_max
            c = min(10.0, c_bound)
        else:
            c = 10.0
            
        # Check if c * S_K clips f(1)
        f1 = -c * S_K
        if abs(f1) > 10.0:
            # Reduce c
            c = 10.0 / abs(S_K)
            # Re-check M constraint
            # M(x) = c * G(x)
            # max M = c * C_max
            if c * C_max > 1.0001:
                # Infeasible with this K?
                # Try to find a c that satisfies both?
                # If C_max > 0 and S_K != 0, we have two constraints.
                # c <= 1.0001 / C_max and c <= 10 / |S_K|
                # So c = min(1.0001 / C_max, 10 / |S_K|)
                c = min(1.0001 / C_max, 10.0 / abs(S_K))
        
        # Calculate score
        score = -c * Q_K
        
        if score > best_score:
            best_score = score
            best_K = K
            best_c = c
            best_S_K = S_K
    
    # Construct result
    K = best_K
    c = best_c
    S_K = best_S_K
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = c * mu[1:K + 1].astype(np.float64)
    
    # Adjust f(1)
    # f(1) = -c * S_K
    values[0] = -c * S_K
    
    # Clip values to [-10, 10] just in case (though we tried to avoid it)
    values = np.clip(values, -10.0, 10.0)
    
    return domain, values

# EVOLVE-BLOCK-END
