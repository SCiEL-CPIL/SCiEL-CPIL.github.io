"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    # Strategy: Maximize tuple size by selecting a set of integers that are "rough"
    # with respect to small primes. Specifically, we exclude multiples of primes p <= k.
    # This ensures that for each p <= k, the residue class 0 mod p is missed.
    # We find the largest k such that the count of such integers is >= k.
    
    primes = _primes(N)
    
    # is_rough[x] is True if x is not divisible by any prime <= current_k
    # We start with k=1 (no primes <= 1), so all numbers are "rough"
    is_rough = [True] * (N + 1)
    count = N + 1  # 0 to N inclusive
    
    best_k = 0
    best_set = []
    
    # We iterate through primes. 
    # The condition is p <= k.
    # If we include prime p in the "excluded" set, it means we are enforcing the constraint for p.
    # This requires k >= p.
    # So we can only "activate" the exclusion for prime p if our current target size k is at least p.
    
    # We want to find max k such that count_rough(k) >= k.
    # count_rough(k) depends on all primes p <= k.
    
    # Let's iterate k from 1 upwards.
    # Maintain the set of active primes (p <= k).
    active_prime_idx = 0
    current_k = 0
    
    # We need to be careful: k is the size of the resulting set.
    # But we are using k as the threshold for primes.
    # Let's call the threshold T.
    # We want max T such that |{x in [0,N] : no prime p <= T divides x}| >= T.
    
    # Iterate T from 1 to N.
    # This is O(N * N/p) which is too slow if done naively for each T.
    # Instead, we incrementally add primes to the exclusion set.
    
    # Start with T=1. No primes excluded. Count = N+1.
    # If N+1 >= 1, T=1 is valid.
    
    # We will iterate T from 1 to N.
    # When T increases, we check if T is a prime. If so, we exclude its multiples.
    
    # Optimization: We only need to check T up to some limit.
    # The count drops as 1/ln(T).
    # So T is at most ~ N / ln(N).
    
    max_T = int(N / (0.5 * (N ** 0.5))) + 100 # Rough upper bound, just to be safe.
    # Actually, let's just go up to N, but break early if count < T and count is decreasing fast.
    
    # is_rough is modified in place.
    # We need to restore state? No, we just want the best T.
    # But if we go past the best T, we can't go back.
    # So we must find the best T in one pass.
    
    # Since count is non-increasing with T, and T is increasing,
    # the condition count >= T will hold for a range of T, then fail.
    # We want the largest T in that range.
    
    # Initialize
    is_rough = [True] * (N + 1)
    count = N + 1
    prime_idx = 0
    primes_list = _primes(N)
    
    best_T = 0
    best_count = 0
    best_rough_set = []
    
    # We iterate T from 1 to N.
    # If T is a prime, we update is_rough.
    # Then we check if count >= T.
    
    # To avoid O(N^2), we only update when T is a prime.
    # Between primes, count is constant, T increases.
    # So if count >= T for some T, it might fail for T+1.
    
    # Let's iterate through primes.
    # Between prime p_i and p_{i+1}, T can range from p_i to p_{i+1}-1.
    # The count is constant in this interval.
    # The max T in this interval is p_{i+1}-1.
    # We check if count >= p_{i+1}-1.
    
    current_count = N + 1
    prev_prime = 1
    
    # Handle T=1 to first prime
    # First prime is 2.
    # Interval [1, 1]. T=1. Count=N+1. N+1 >= 1. Best T=1.
    
    # We'll iterate primes p.
    # Before processing p, we are at T = p-1.
    # Count is current_count.
    # Check if current_count >= p-1.
    # If yes, best_T = p-1.
    
    # Then process p: exclude multiples of p.
    # current_count decreases.
    # prev_prime = p.
    
    # After loop, check up to N.
    
    best_T = 0
    best_set = []
    
    # We need to store the set for the best T.
    # Storing the set for every T is expensive.
    # We can store the list of excluded numbers?
    # Or just reconstruct at the end for the best T.
    
    # Let's just find the best T first.
    
    p_idx = 0
    current_count = N + 1
    last_checked_T = 0
    
    # We need to know the count for any T.
    # Count(T) is the number of integers in [0, N] not divisible by any prime <= T.
    
    # We iterate T from 1 to N.
    # This is 5000 steps.
    # In each step, if T is prime, we update count.
    # Updating count for prime p:
    #   We need to count how many "rough" numbers are multiples of p.
    #   This requires checking is_rough[multiple].
    #   Multiples of p: p, 2p, ...
    #   Number of multiples is N/p.
    #   Sum of N/p for p <= N is N log log N.
    #   For N=5000, this is very fast.
    
    is_rough = [True] * (N + 1)
    count = N + 1
    
    # Map prime to index for fast lookup?
    # Just use a set or boolean array for primes.
    is_prime = [False] * (N + 1)
    for p in _primes(N):
        is_prime[p] = True
        
    best_T = 0
    best_count_val = 0
    
    # We iterate T from 1 to N.
    # If T is prime, update count.
    # Check if count >= T.
    
    # To save memory/time, we don't store the set for every T.
    # We just find the best T.
    
    for T in range(1, N + 1):
        if is_prime[T]:
            # Exclude multiples of T
            # Only count those that are currently "rough"
            removed = 0
            for multiple in range(T, N + 1, T):
                if is_rough[multiple]:
                    is_rough[multiple] = False
                    removed += 1
            count -= removed
        
        if count >= T:
            best_T = T
            best_count_val = count
        else:
            # Since count is non-increasing and T is increasing,
            # once count < T, it will stay < T+1?
            # count(T+1) <= count(T) < T < T+1.
            # Yes, it will never become valid again.
            break
            
    # Now reconstruct the set for best_T
    # We need the state of is_rough after processing all primes <= best_T.
    # The loop above broke early. 
    # If best_T is the last valid T, we have the correct is_rough state?
    # Yes, because we update is_rough as we go.
    # If we broke at T=100, is_rough reflects primes <= 99 (if 100 is not prime) or 100 (if prime).
    # Wait, if we break at T where count < T, we have already processed T (if prime).
    # But best_T is the PREVIOUS T where count >= T.
    # So we need the state of is_rough BEFORE processing the prime that caused the drop?
    # Or just before T became invalid?
    
    # Example:
    # T=455: count=460. Valid. best_T=455.
    # T=456: count=460. 460 >= 456. Valid. best_T=456.
    # ...
    # T=460: count=456. 456 < 460. Invalid. Break.
    # best_T=459.
    # The is_rough state at T=460 includes exclusions for primes <= 460.
    # But for T=459, we should only exclude primes <= 459.
    # If 460 is not prime, the state is the same as T=459.
    # If 460 was prime, we excluded its multiples, but T=459 didn't require it.
    # So we might have over-excluded.
    
    # To be safe, we should reconstruct is_rough for best_T from scratch.
    # N=5000 is small.
    
    is_rough_final = [True] * (N + 1)
    for p in _primes(best_T):
        for multiple in range(p, N + 1, p):
            is_rough_final[multiple] = False
            
    result = [i for i in range(N + 1) if is_rough_final[i]]
    
    # Verify size
    # if len(result) < best_T:
    #    # This shouldn't happen if logic is correct
    #    pass
    
    return result
# EVOLVE-BLOCK-END
