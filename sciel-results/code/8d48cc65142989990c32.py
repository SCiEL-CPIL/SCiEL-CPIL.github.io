"""Rewritten program: Dijkstra-like generation by ascending LPF."""
import time
import math
import heapq

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Structural change: Instead of generating B-smooth numbers for increasing B and
    scanning the full range, we use a Dijkstra-like priority queue to generate
    integers in ascending order of their largest prime factor. We maintain a
    sliding window of recently-generated numbers and check for consecutive pairs.
    """
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Precompute primes up to sqrt(N_MAX) for factorization
    limit = int(math.isqrt(N_MAX)) + 2
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # We need primes up to N_MAX for the LPF computation in the priority queue
    # Actually, for the Dijkstra approach, we need primes up to N_MAX
    # But that's too many. Instead, we'll compute LPF on the fly.
    
    # For the priority queue approach:
    # Each entry is (lpf, value, max_prime_used)
    # We start with all primes (lpf = prime, value = prime)
    # We can multiply by any prime <= lpf to get new entries
    
    # But we need to be careful: the LPF of a product is the max of the LPFs
    # of the factors. So if we have a number with LPF = p, and we multiply by
    # a prime q <= p, the new number has LPF = p.
    # If we multiply by a prime q > p, the new number has LPF = q.
    
    # Strategy: 
    # 1. Generate numbers in ascending order of LPF using a priority queue
    # 2. For each number in [N_MIN, N_MAX], record its LPF
    # 3. Check consecutive pairs: L(n) = max(lpf(n), lpf(n+1))
    # 4. The minimum L(n) over n in [N_MIN, N_MAX-1] is our answer
    
    # However, generating ALL numbers up to N_MAX in LPF order might be expensive.
    # Let's think about the expected LPF values.
    # For n ~ 10000, typical LPF is around 100-1000.
    # For n ~ 1000000, typical LPF is around 1000-10000.
    
    # Alternative: Use the fact that we only need to find the minimum L(n).
    # We can binary search on the answer B: is there an n in [N_MIN, N_MAX-1]
    # such that both n and n+1 have all prime factors <= B?
    # But this is essentially what the original approach does.
    
    # Let me try a hybrid: use the priority queue to generate numbers in LPF order,
    # but stop early once we've found a good candidate.
    
    # Actually, let me reconsider. The key insight is:
    # L(n) = max(lpf(n), lpf(n+1))
    # We want to minimize this.
    # 
    # A number with small LPF is "smooth". Consecutive smooth numbers are rare.
    # The minimum L(n) is likely small (maybe < 100 or < 200).
    #
    # Let me try a direct approach: for each candidate B (in increasing order),
    # check if there exist consecutive B-smooth numbers in [N_MIN, N_MAX].
    # But do it efficiently using a hash set of B-smooth numbers.
    
    # Precompute: for each B, generate B-smooth numbers in [N_MIN, N_MAX]
    # and check for consecutive pairs.
    
    # To make this faster, let's use a different generation method:
    # For a given set of primes P, generate all products of primes in P
    # that fall in [N_MIN, N_MAX].
    
    # Let's precompute primes up to 200 (should be more than enough)
    primes_up_to_200 = [p for p in primes if p <= 200]
    
    # For efficiency, let's use a different strategy:
    # 1. Compute lpf for all numbers in [N_MIN, N_MAX] using a modified sieve
    # 2. Find the minimum max(lpf(n), lpf(n+1))
    
    # Sieve approach for LPF:
    # lpf[i] = largest prime factor of i
    # Initialize lpf[i] = 1 for all i
    # For each prime p, for each multiple of p, set lpf[multiple] = max(lpf[multiple], p)
    
    # This is O(N log log N) which should be fast for N = 1000000.
    
    lpf = [1] * (N_MAX + 2)
    for p in primes:
        if p > N_MAX:
            break
        for multiple in range(p, N_MAX + 2, p):
            lpf[multiple] = p  # Since we go in increasing order, the last prime that divides is the largest
    
    # Now find the best n
    best_n = N_MIN
    best_l = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    for n in range(N_MIN, N_MAX):
        l = max(lpf[n], lpf[n + 1])
        if l < best_l:
            best_l = l
            best_n = n
            # Early termination: if best_l == 2, we can't do better
            if best_l <= 2:
                break
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
