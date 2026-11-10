import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) * log(k) / k subject to
    sum_k f(k)/k = 0 and M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K].
    
    The optimal strategy is to use f(k) = c * mu(k) for k in [1, K], where c is chosen
    to saturate the feasibility constraint.
    """
    start_time = time.time()
    deadline = 550.0  # Leave 50s margin
    
    # 1. Sieve for Mobius function up to K_max
    K_max = 50000
    mu = np.ones(K_max + 1, dtype=np.int8)
    is_prime = np.ones(K_max + 1, dtype=bool)
    primes = []
    
    # Standard sieve for mu
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K_max + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            for j in range(i*i, K_max + 1, i*i):
                mu[j] = 0
    # mu[1] is 1 by definition, which is already set.
    # Wait, the standard sieve loop above handles mu correctly?
    # Let's use a more robust sieve for mu.
    
    mu = np.ones(K_max + 1, dtype=np.int8)
    mu[1] = 1
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
    # This sieve is correct.

    # Precompute log(k)/k for k=2..K_max
    # We need Q_K = sum_{k=2}^K mu(k) * log(k) / k
    k_vals = np.arange(1, K_max + 1, dtype=np.float64)
    log_k = np.log(k_vals)
    # Handle k=1 log(1)=0
    inv_k = 1.0 / k_vals
    weighted_mu = mu.astype(np.float64) * log_k * inv_k
    # weighted_mu[0] is 0 (index 0 unused), weighted_mu[1] is 0.
    Q_cumulative = np.cumsum(weighted_mu)
    # Q_cumulative[K] = sum_{k=1}^K ... which is sum_{k=2}^K ...

    best_score = 0.0
    best_K = 1
    best_c = 1.0
    best_M_max = 1.0

    # We will sample K values
    # Start small, go large.
    # The score a ~ -Q_K / M_max.
    # -Q_K grows roughly like log(K) or similar.
    # M_max oscillates.
    
    # Check a range of K values
    # To save time, we don't check every K.
    # We check K = 100, 200, ..., 1000, 2000, ..., 50000
    
    K_candidates = []
    # Small K
    for k in range(50, 500, 10):
        K_candidates.append(k)
    # Medium K
    for k in range(500, 2000, 20):
        K_candidates.append(k)
    # Large K
    for k in range(2000, K_max + 1, 50):
        K_candidates.append(k)
    
    for K in K_candidates:
        if time.time() > deadline:
            break
            
        Q_K = Q_cumulative[K]
        if Q_K >= 0:
            # If Q_K is positive, -Q_K is negative, score is negative.
            # We want to maximize, so we prefer Q_K < 0.
            # However, if all Q_K are positive, we pick the least negative?
            # Actually, if Q_K > 0, then a < 0. Seed score is 1.0 (a~1).
            # So we must find K where Q_K < 0.
            # Let's just calculate and compare.
            pass
            
        # Compute M_max(K) = max_{x in [1, 10K]} sum_{k=1}^K mu(k) floor(x/k)
        N = 10 * K
        M = np.zeros(N + 1, dtype=np.float64)
        
        # Vectorized update
        # M[x] += mu[k] for all x that are multiples of k
        # This is O(N log K)
        
        # We only need to add for k where mu[k] != 0
        non_zero_k = np.where(mu[1:K+1] != 0)[0] + 1
        mu_vals = mu[non_zero_k].astype(np.float64)
        
        # To speed up, we can batch process?
        # Loop is over K non-zero mu values.
        # For each k, M[k::k] += mu[k]
        # This is fast in numpy.
        
        for idx in non_zero_k:
             M_val = mu[idx]
             if M_val != 0:
                 M[idx:K+1:idx] += M_val  # Wait, floor(x/k) is 1, 2, 3...
                 # M(x) = sum mu(k) floor(x/k).
                 # My previous logic:
                 # M(x) = sum_{k=1}^K mu(k) floor(x/k)
                 # If I do M[k::k] += mu[k], I am adding mu[k] to M(k), M(2k), ...
                 # This corresponds to sum_{k|x} mu(k), which is 1 if x=1 else 0.
                 # This is NOT M(x).
                 
                 # M(x) = sum_{k=1}^K mu(k) floor(x/k)
                 # floor(x/k) = 1 for x in [k, 2k-1], 2 for [2k, 3k-1], etc.
                 # So for a fixed k, mu(k) is added to M(x) for x in [k, 10K] with value floor(x/k).
                 # This is harder to vectorize simply.
                 
                 # Alternative:
                 # M(x) = sum_{j=1}^{floor(x/K)} ... no.
                 # Let's stick to the definition.
                 # For each k, we need to add mu(k) * floor(x/k) to M(x) for x=1..10K.
                 # floor(x/k) takes values 1, 2, ..., floor(10K/k).
                 # Range for value v: x in [v*k, (v+1)*k - 1].
                 
                 # This is too slow in a Python loop over K if done naively for each x.
                 # But we can do it for each k.
                 # For a given k, the sequence floor(x/k) for x=1..N is:
                 # 0 (x<k), 1 (k<=x<2k), 2 (2k<=x<3k), ...
                 # We can construct the array for this k and add it.
                 # But constructing it is O(N/k).
                 # Total O(N log K).
                 
                 # Let's try a different approach.
                 # M(x) is the Dirichlet convolution of mu and 1?
                 # sum_{d|x} mu(d) = 1 if x=1 else 0.
                 # M(x) = sum_{k=1}^K mu(k) floor(x/k)
                 # = sum_{k=1}^K mu(k) sum_{j=1}^{floor(x/k)} 1
                 # = sum_{j=1}^{x} sum_{k=1}^{\lfloor x/j \rfloor} mu(k)
                 # = sum_{j=1}^{x} M_mertens(\lfloor x/j \rfloor)
                 # This doesn't seem to simplify computation for fixed K.
                 
                 pass # Skip this inner loop for now, need better method

        # Let's use the O(N log K) method properly.
        # M = np.zeros(N+1)
        # for k in range(1, K+1):
        #    if mu[k] != 0:
        #       # Add mu[k] * floor(x/k) to M
        #       # This is equivalent to:
        #       # For x in [k, 2k-1], add mu[k]
        #       # For x in [2k, 3k-1], add 2*mu[k]
        #       # ...
        #       # We can do this by adding mu[k] to M[k::k], 2*mu[k] to M[2k::k]? No.
        #       # M(x) = sum_{k|x} ... no.
        
        # Correct vectorization:
        # M[x] = sum_{k=1}^K mu[k] * (x // k)
        # This is a linear combination of step functions.
        # We can compute it by:
        # M = np.zeros(N+1)
        # for k in range(1, K+1):
        #    if mu[k] != 0:
        #       # We need to add mu[k] * floor(x/k) to M
        #       # This can be done by:
        #       # M[k:] += mu[k]  # adds 1 for all x>=k
        #       # M[2k:] += mu[k] # adds 1 for all x>=2k
        #       # M[3k:] += mu[k] # adds 1 for all x>=3k
        #       # So M[x] gets mu[k] added for each j such that j*k <= x.
        #       # i.e., j <= x/k. So it adds mu[k] * floor(x/k).
        #       # This is O(N/k) operations per k.
        #       for j in range(1, N//k + 1):
        #           M[j*k:] += mu[k]
        #       # This inner loop is O(N/k). Total O(N log K).
        #       # But it's a Python loop inside NumPy? No, M[j*k:] is a slice.
        #       # The loop is over j. Number of j's is N/k.
        #       # So for k=1, N loops. k=2, N/2 loops.
        #       # Total loops sum(N/k) = N log N.
        #       # For N=500,000, 500,000 * 13 = 6.5 million slice additions.
        #       # This is feasible in NumPy if done carefully, but Python loop overhead might be high.
        
        # Let's try a simpler approach for smaller K or a heuristic.
        # Given time constraints, maybe we should just check a few K values with a simpler check?
        # Or accept that the seed is 1.0 and just return a valid Mobius function for K=100?
        # The seed has score 1.0.
        # If I return mu(k) for k=1..100, c=1.
        # M(1)=1. M(2)=1-1=0. M(3)=1-1-1=-1.
        # M_max is likely 1.
        # Q_100 = sum mu(k) log k / k.
        # If Q_100 is negative, score = -Q_100.
        # -Q_100 is approx 1?
        
        # Let's just return the seed solution if we can't find a better one quickly.
        # But the prompt asks for a stronger algorithm.
        
        # Let's try to compute M_max for K=100, 200, 500, 1000, 5000.
        
        if K <= 5000: # Only do expensive check for small K
            N = 10 * K
            M = np.zeros(N + 1, dtype=np.float64)
            
            # Efficient addition
            # For each k, we add mu[k] to M[k::k], 2*mu[k] to M[2k::k]... no.
            # We add mu[k] to M[j*k:] for j=1..N//k.
            
            # To avoid Python loop over j, we can note:
            # M[x] = sum_{k=1}^K mu[k] floor(x/k)
            # This is the same as:
            # M = np.zeros(N+1)
            # for k in range(1, K+1):
            #    if mu[k] != 0:
            #       # Use a difference array?
            #       # floor(x/k) is a step function.
            #       # Let D be difference array of M.
            #       # M[x] - M[x-1] = sum_{k|x} mu[k].
            #       # This is 1 if x=1 else 0.
            #       # So M[x] = sum_{j=1}^x sum_{k|j} mu[k] = x * 1 (if x=1) ... no.
            #       # sum_{k|j} mu[k] is 1 if j=1 else 0.
            #       # So M[x] = 1 for x>=1? No.
            #       # M(x) = sum_{k=1}^K mu(k) floor(x/k).
            #       # If K=x, M(x) = 1.
            #       # If K < x, M(x) = 1 - sum_{k=K+1}^x mu(k) floor(x/k).
            
            # The difference array trick:
            # Let A[x] = sum_{k|x, k<=K} mu(k).
            # Then M[x] = M[x-1] + A[x].
            # We can compute A[x] for all x in O(N log K) using sieve-like addition.
            # A = np.zeros(N+1)
            # for k in range(1, K+1):
            #    if mu[k] != 0:
            #        A[k::k] += mu[k]
            # This is O(N log K).
            # Then M = np.cumsum(A)
            
            A = np.zeros(N + 1, dtype=np.float64)
            non_zero_k = np.where(mu[1:K+1] != 0)[0] + 1
            for k in non_zero_k:
                A[k::k] += mu[k]
            
            M_vals = np.cumsum(A)
            # M_vals[0] is 0. M_vals[1] is A[1]=mu[1]=1.
            # We need max M(x) for x in [1, N].
            M_max_val = np.max(M_vals[1:])
            
            if M_max_val > 0:
                c = 1.0001 / M_max_val
                # Check clipping
                if c > 10:
                    c = 10.0
                # Calculate score
                # a = -c * Q_K
                # Note: Q_K is sum_{k=2}^K mu(k) log k / k
                score = -c * Q_K
                
                if score > best_score:
                    best_score = score
                    best_K = K
                    best_c = c
                    best_M_max = M_max_val
        
    # If no better K found, return seed-like or just K=1
    if best_K == 1 and best_score <= 0:
         # Fallback to a small domain
         K = 5
         domain = np.arange(1, K+1, dtype=np.int64)
         # Use mu values
         vals = mu[1:K+1].astype(np.float64)
         # Adjust c
         N = 10*K
         A = np.zeros(N+1, dtype=np.float64)
         for k in range(1, K+1):
             if mu[k] != 0:
                 A[k::k] += mu[k]
         M_vals = np.cumsum(A)
         M_max_val = np.max(M_vals[1:])
         if M_max_val > 0:
             c = 1.0001 / M_max_val
         else:
             c = 1.0
         vals = c * vals
         return domain, vals

    domain = np.arange(1, best_K + 1, dtype=np.int64)
    values = best_c * mu[1:best_K + 1].astype(np.float64)
    
    return domain, values

# EVOLVE-BLOCK-END
