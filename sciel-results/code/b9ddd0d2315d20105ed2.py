import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer for safety
    
    # We want to find M with many k such that k * sigma(k) = M.
    # Strategy: Iterate k, compute M = k * sigma(k), store in dict M -> list of k.
    # k must be <= sqrt(M). Since M = k * sigma(k) >= k * (1+k) = k^2 + k, 
    # sqrt(M) >= sqrt(k^2+k) > k. So any k we find satisfies k < sqrt(M).
    # Actually, the constraint is k <= isqrt(M).
    # M = k * sigma(k). 
    # isqrt(k * sigma(k)) >= k  <=>  k * sigma(k) >= k^2  <=>  sigma(k) >= k.
    # sigma(k) >= 1 + k > k. So the condition k <= isqrt(M) is always satisfied for k >= 1.
    
    # We need to maximize the number of distinct k in the list for a single M.
    # We can iterate k up to some limit K_max.
    # M <= 10^18.
    # k * sigma(k) <= 10^18.
    # sigma(k) is roughly k * log(log(k)) for large k, but can be much larger for highly composite.
    # Worst case sigma(k) ~ k * e^gamma * log(log(k)) is too high? No, sigma(k)/k is bounded by exp(gamma) log log k + ...
    # For k ~ 10^9, sigma(k) ~ 10^10. M ~ 10^19. So k should be less than 10^9.
    # Let's estimate max k. If sigma(k) ~ 2k (very low, e.g. prime), M ~ 2k^2.
    # 2k^2 <= 10^18 => k <= sqrt(5*10^17) ~ 7*10^8.
    # If sigma(k) is large, k can be smaller.
    # So we can iterate k up to ~ 10^8 or 10^9.
    
    # 10^8 iterations in Python might be slow (~5-10s for simple ops).
    # 10^9 is too slow (~50-100s).
    # We need to be efficient.
    
    # Optimization:
    # We can precompute sigma(k) for all k up to K_max using a sieve-like method.
    # Sieve for sigma:
    # Initialize sigma[1..K_max] = 1.
    # For i from 1 to K_max:
    #   for j from i to K_max step i:
    #     sigma[j] += i
    # This is O(N log N). For N=10^7, N log N ~ 10^7 * 16 ~ 1.6 * 10^8 ops.
    # In Python, 1.6 * 10^8 simple additions might take ~10-20s.
    # For N=10^6, it's very fast.
    
    # Let's try N = 10^7 first. If time allows, we can go higher?
    # Or use a faster method?
    # Python loops are slow. Using NumPy or array module might help, but standard library is preferred for portability unless specified.
    # The prompt says "Python standard library, NumPy and SciPy are available."
    # Let's use a standard list for sigma, but the sieve loop is the bottleneck.
    
    # Alternative: Factorization based sigma calculation for each k?
    # Too slow for 10^8 k's.
    
    # Let's use the sieve approach with optimized inner loop.
    # Or, restrict K_max to 2*10^6 or 5*10^6 to be safe, and hope for collisions there.
    # Collisions are rare. We need many k's to have a chance.
    # The "inspiration" mentioned searching up to 10^7 found M with >3 preimages.
    # So 10^7 is a good target.
    
    # Let's implement the sieve for sigma up to N.
    # To speed up the sieve in Python, we can use a list of ints.
    # The inner loop `for j in range(i, N+1, i): sigma[j] += i` is the cost.
    # Total iterations ~ N * H_N ~ N * ln N.
    # For N=10^7, ln(10^7) ~ 16.1. Total ~ 1.6 * 10^8.
    # In Python, a tight loop of 10^8 iterations takes ~5-10 seconds.
    # This is feasible within 110s.
    
    N = 10_000_000  # 10^7
    
    # Check time before starting heavy work
    if time.time() > deadline - 30.0:
        # Fallback to small search or known solution
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Initialize sigma array
    # Using a list of zeros/ones
    sigma = [1] * (N + 1)
    
    # Sieve
    # Optimization: local variable references
    sig = sigma
    for i in range(2, N + 1):
        # If i is prime, we could optimize, but we don't know primes yet without extra work.
        # Just run the loop.
        for j in range(i, N + 1, i):
            sig[j] += i
            
    if time.time() > deadline - 10.0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Now, bucket M values
    # M = k * sigma[k]
    # We only care if M <= 10^18
    # And we want to count distinct k for each M.
    
    buckets = defaultdict(list)
    limit_M = 10**18
    
    # Iterate k
    # We can break early if k * sigma[k] > limit_M and sigma[k] is increasing?
    # sigma[k] is not monotonic, but generally grows.
    # However, k * sigma[k] is strictly increasing for k >= 1?
    # Let's check: f(k) = k * sigma(k).
    # f(k+1) > f(k)?
    # sigma(k+1) >= 1 + (k+1) = k+2.
    # f(k+1) = (k+1) * sigma(k+1) >= (k+1)(k+2) = k^2 + 3k + 2.
    # f(k) = k * sigma(k). Max sigma(k) is unbounded relative to k? No, sigma(k) < e^gamma k log log k.
    # For small k, it's monotonic?
    # k=1: 1*1=1
    # k=2: 2*3=6
    # k=3: 3*4=12
    # k=4: 4*7=28
    # k=5: 5*6=30
    # k=6: 6*12=72
    # It seems monotonic.
    # If it is monotonic, we can break when M > 10^18.
    
    for k in range(1, N + 1):
        M = k * sig[k]
        if M > limit_M:
            # Since f(k) is likely monotonic increasing, we can break.
            # But to be safe, if it's not strictly monotonic, we might miss some?
            # Actually, if f(k) is not monotonic, a later k could have smaller M.
            # But sigma(k) >= k+1, so f(k) >= k(k+1) which is monotonic.
            # So f(k) is monotonic increasing.
            break
            
        # Store k in bucket M
        # To save memory, we can just count first, then re-run to collect?
        # Or store list. N=10^7, distinct M's might be up to 10^7.
        # Storing lists of ints is fine.
        buckets[M].append(k)
        
        if time.time() > deadline - 5.0:
            # Time out, process current buckets
            break
            
    # Find best M
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Iterate through buckets
    # Only consider M with count > best_count
    # We can sort by count or just max.
    # max is O(N_buckets).
    
    # Optimization: we can track the max during the bucketing phase?
    # No, because we need the list of k's.
    # So we must iterate buckets at the end.
    
    for M, ks in buckets.items():
        cnt = len(ks)
        if cnt > best_count:
            best_count = cnt
            best_M = M
            best_ks = ks
            
    # Ensure types are correct
    # M is int, k is list of ints.
    # best_M is from dict key (int).
    # best_ks is list of ints.
    
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
