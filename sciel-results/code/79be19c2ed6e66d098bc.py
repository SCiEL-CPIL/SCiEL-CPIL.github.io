"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def _sigma_small(n):
    """Compute sigma(n) for n up to ~10^7 efficiently using precomputed SPF."""
    # We'll use a global precomputed SPF array if available
    pass

def construct():
    start_time = time.time()
    DEADLINE = 135  # safety margin before 150s kill
    
    # Strategy: brute-force search for k*sigma(k) collisions.
    # For each k, compute M = k * sigma(k). Group by M.
    # Find M with the most distinct k values (k <= sqrt(M)).
    # Note: k <= sqrt(M) is automatically satisfied when k*sigma(k)=M and sigma(k)>=k+1 (for k>1),
    # but we must verify k <= isqrt(M) for each candidate.
    
    # Precompute sigma for all k up to some limit N.
    # M = k * sigma(k). For k up to N, M can be up to N * sigma(N).
    # We want M <= 10^18.
    
    # Let's estimate N. sigma(k) is roughly O(k log log k).
    # So M ~ k^2 log log k. For M <= 10^18, k^2 ~ 10^18 / log log k ~ 10^18 / 4 ~ 2.5*10^17.
    # k ~ sqrt(2.5*10^17) ~ 5*10^8. That's too large for brute force.
    
    # However, we know from the knowledge that brute force up to k=5e6 finds M with >3 solutions.
    # Let's try to go higher if possible, but 5e6 is a good start.
    # Actually, let's think about the constraint k <= sqrt(M).
    # M = k * sigma(k). So k <= sqrt(k * sigma(k)) => k^2 <= k * sigma(k) => k <= sigma(k).
    # This is always true for k >= 1 (since sigma(k) >= k+1 for k>1, and sigma(1)=1, k=1: 1<=1 ok).
    # So the constraint k <= sqrt(M) is automatically satisfied for all k where M=k*sigma(k).
    # Wait, let's double check.
    # k <= sqrt(M)  <=>  k^2 <= M  <=>  k^2 <= k * sigma(k)  <=>  k <= sigma(k).
    # Yes, this is always true. So we just need distinct k such that k*sigma(k) = M.
    
    # Let's precompute sigma(k) for k up to N.
    # We'll use a sieve-like approach.
    # sigma[k] = sum of divisors of k.
    # Initialize sigma array with 0s.
    # For i from 1 to N:
    #   for j from i to N, step i:
    #     sigma[j] += i
    
    # This is O(N log N). For N=5e6, this is about 5e6 * 15 = 7.5e7 operations.
    # In Python, this might be slow. Let's use NumPy if possible, or optimize the loop.
    
    # Let's try N = 2_000_000 first to see if it fits in time, then increase.
    # Actually, the knowledge says "Brute force search for k*sigma(k) collisions up to k=5e+06 finds M with >3 solutions, beating seed".
    # So let's aim for N = 5_000_000.
    
    # To speed up sigma computation in Python, we can use a list and nested loops.
    # Or we can use a more efficient method.
    
    # Let's use a dictionary to store M -> list of k.
    # But M can be up to 10^18, so we can't use a list. Dictionary is fine.
    
    # Let's estimate time for N=5e6.
    # The inner loop runs N/i times for each i. Total iterations ~ N * H_N ~ 5e6 * 15 = 7.5e7.
    # In Python, 7.5e7 iterations might take 10-20 seconds. That's acceptable.
    
    N = 5_000_000
    
    # Precompute sigma using a sieve
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Now compute M = k * sigma(k) for each k, and group by M.
    # We only care about M <= 10^18.
    M_limit = 10**18
    
    # Use a dictionary: M -> list of k
    groups = {}
    
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= M_limit:
            if M not in groups:
                groups[M] = []
            groups[M].append(k)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        # Return best found so far (which is empty or partial, but let's handle gracefully)
        # Actually, if we haven't finished, we should return the seed result to be safe.
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Find M with the most k values
    best_M = None
    best_k_list = []
    
    for M, k_list in groups.items():
        if len(k_list) > len(best_k_list):
            best_M = M
            best_k_list = k_list
    
    # If we found something better than the seed, return it.
    # Seed has 3 solutions.
    if best_M is not None and len(best_k_list) >= 3:
        # Ensure k_list is sorted and distinct (it should be by construction)
        best_k_list = sorted(set(best_k_list))
        # Verify the constraint k <= isqrt(M) just in case (though it's always true)
        filtered_k = [k for k in best_k_list if k <= math.isqrt(best_M)]
        if len(filtered_k) >= 3:
            return {'M': best_M, 'k': filtered_k}
    
    # If nothing better, return seed
    return {'M': 333312, 'k': [336, 372, 434]}
# EVOLVE-BLOCK-END
