import math
import time
import random
from itertools import combinations

def _score_primes(prime_list):
    """
    Compute tau_perp for a squarefree number defined by prime_list.
    Returns (score, product).
    """
    n = 1
    for p in prime_list:
        n *= p
        if n > 10**18:
            return -1, n
            
    k = len(prime_list)
    if k == 0:
        return 0, 1
        
    # Generate divisors as (value, bitmask)
    # Bitmask i is set if prime_list[i] divides the divisor
    divs = [(1, 0)]
    for i, p in enumerate(prime_list):
        bit = 1 << i
        # Current length
        l = len(divs)
        # Extend with new divisors
        # Optimization: list comprehension might be faster
        # new_divs = [(v * p, m | bit) for v, m in divs[:l]]
        # divs.extend(new_divs)
        
        # Manual loop for clarity and potential micro-opts
        for j in range(l):
            v, m = divs[j]
            divs.append((v * p, m | bit))
            
    # Sort by value
    divs.sort(key=lambda x: x[0])
    
    # Count coprime adjacent pairs
    count = 0
    for i in range(len(divs) - 1):
        if (divs[i][1] & divs[i + 1][1]) == 0:
            count += 1
            
    return count, n

def _generate_primes(limit):
    """Sieve of Eratosthenes."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer
    
    # Generate a pool of primes
    # We need primes up to roughly 500-1000 to have enough candidates for swaps
    all_primes = _generate_primes(1000)
    
    best_score = -1
    best_n = 2
    best_primes = []
    
    def update_best(primes, score, n):
        nonlocal best_score, best_n, best_primes
        if score > best_score:
            best_score = score
            best_n = n
            best_primes = list(primes)
            
    # Phase 1: Initial candidates
    # Try primorials of first k primes
    max_k = 0
    prod = 1
    for i, p in enumerate(all_primes):
        prod *= p
        if prod <= 10**18:
            max_k = i + 1
        else:
            break
            
    # Evaluate primorials up to max_k
    for k in range(2, max_k + 1):
        primes_k = all_primes[:k]
        s, n = _score_primes(primes_k)
        update_best(primes_k, s, n)
        
    # Phase 2: Local Search
    # We will perform local search from the best candidate found so far
    # and from a few other promising starting points
    
    def local_search(initial_primes, time_limit):
        nonlocal best_score, best_n, best_primes
        current_primes = list(initial_primes)
        current_score, current_n = _score_primes(current_primes)
        if current_n > 10**18:
            return
            
        update_best(current_primes, current_score, current_n)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_swap_score = current_score
            best_swap_primes = current_primes
            
            # Try swapping each prime with a larger prime
            # To reduce search space, we only consider primes that are not already in the set
            # and that keep the product <= 10^18
            
            current_set = set(current_primes)
            
            # Sort current primes to maintain order? Not strictly necessary for set logic, 
            # but _score_primes expects a list. The order in the list determines the bitmask bits.
            # However, the set of divisors is the same regardless of order.
            # The sorted order of divisors depends on the values, not the index in prime_list.
            # So we can keep the list sorted for consistency.
            
            for i in range(len(current_primes)):
                p_old = current_primes[i]
                # Try replacing p_old with p_new
                # p_new should be > p_old to avoid duplicates and move towards "larger" primes
                # which might spread divisors more? Or smaller?
                # Actually, we should try both directions or just a wide range.
                # Let's try primes from the pool that are not in current_set.
                
                # Optimization: Only try a subset of candidates to keep it fast
                # We can iterate through all_primes
                for p_new in all_primes:
                    if p_new in current_set:
                        continue
                    # Check product bound
                    # new_n = current_n // p_old * p_new
                    # If current_n // p_old * p_new > 10^18, skip
                    # Since p_new can be large, this check is cheap.
                    
                    # Heuristic: If p_new is very small and p_old is large, it might be a "downhill" move
                    # but could lead to a different local optimum.
                    # Let's just try all valid swaps.
                    
                    new_n = current_n // p_old * p_new
                    if new_n > 10**18:
                        continue
                        
                    new_primes = current_primes.copy()
                    new_primes[i] = p_new
                    # Sort to keep bitmask indices consistent? 
                    # Actually, the bitmask indices just need to be unique.
                    # But sorting the primes ensures the same set always produces the same bitmask mapping?
                    # No, the bitmask mapping depends on the position in the list.
                    # If we change the list, the bitmasks change.
                    # But the *set* of divisors is the same.
                    # The *sorted order* of divisors is the same.
                    # The *coprimality* is the same.
                    # So the score is invariant to the permutation of the input list.
                    # We can keep the list unsorted or sorted, it doesn't matter for correctness.
                    # However, for `in` checks on `current_set`, we use the set.
                    
                    s, n = _score_primes(new_primes)
                    if s > best_swap_score:
                        best_swap_score = s
                        best_swap_primes = new_primes
                        improved = True
                        
            if improved:
                current_primes = best_swap_primes
                current_score = best_swap_score
                current_n = 1
                for p in current_primes:
                    current_n *= p
                update_best(current_primes, current_score, current_n)

    # Run local search from the best primorial
    if time.time() < deadline - 10:
        local_search(all_primes[:max_k], deadline - 5)
        
    # Run local search from other starting points
    # e.g., first max_k-1 primes + a larger prime
    if time.time() < deadline - 10:
        for offset in range(1, 5):
            if time.time() > deadline - 5:
                break
            if max_k - 1 + offset < len(all_primes):
                start_primes = all_primes[:max_k-1] + [all_primes[max_k-1+offset]]
                # Ensure product <= 10^18
                prod = 1
                for p in start_primes:
                    prod *= p
                if prod <= 10**18:
                    local_search(start_primes, deadline - 5)
                    
    return best_n
