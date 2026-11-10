import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin

    # 1. Generate a list of small primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # We need primes up to a reasonably large number to allow swaps.
    # If we remove small primes, we can add larger ones.
    # Max prime if we keep 14 smallest: ~77.
    # Max prime if we remove 2,3,5: ~2500.
    # Max prime if we remove 2,3,5,7: ~10^18 / (1.3e16 / 210) = ~16000.
    # Let's generate primes up to 50000 to be safe and have plenty of candidates.
    primes = sieve(50000)
    prime_set_global = set(primes)

    # 2. Define tau_perp calculation optimized for speed
    def tau_perp(n, prime_factors):
        # prime_factors is a list of distinct primes
        # Generate divisors in sorted order
        # For square-free n with k distinct primes, there are 2^k divisors.
        # We can generate them in sorted order by starting with [1] and
        # for each prime, merging the existing list with existing*prime.
        
        # To speed up, we can use a list of (divisor, mask)
        # But for k=15, 32768 elements is small.
        
        divs = [1]
        for p in prime_factors:
            # divs is sorted. We need to merge divs with [d*p for d in divs].
            # Both are sorted. Use two-pointer merge.
            merged = []
            i = 0
            j = 0
            len_d = len(divs)
            while i < len_d or j < len_d:
                if j >= len_d:
                    merged.append(divs[i])
                    i += 1
                elif i >= len_d:
                    merged.append(divs[j] * p)
                    j += 1
                else:
                    a = divs[i]
                    b = divs[j] * p
                    if a <= b:
                        merged.append(a)
                        i += 1
                    else:
                        merged.append(b)
                        j += 1
            divs = merged
        
        # Count adjacent coprime pairs
        # We need to know the prime factors of each divisor.
        # Instead of recalculating, we can track masks during generation.
        # Let's redo with masks for correctness and speed (bitwise AND is fast).
        
        pairs = [(1, 0)]  # (divisor, bitmask)
        for idx, p in enumerate(prime_factors):
            bit = 1 << idx
            # pairs is sorted by divisor. We merge with [(d*p, m|bit) for (d,m) in pairs]
            merged = []
            i = 0
            j = 0
            len_p = len(pairs)
            while i < len_p or j < len_p:
                if j >= len_p:
                    merged.append(pairs[i])
                    i += 1
                elif i >= len_p:
                    d, m = pairs[j]
                    merged.append((d * p, m | bit))
                    j += 1
                else:
                    a_d, a_m = pairs[i]
                    b_d, b_m = (pairs[j][0] * p, pairs[j][1] | bit)
                    if a_d <= b_d:
                        merged.append((a_d, a_m))
                        i += 1
                    else:
                        merged.append((b_d, b_m))
                        j += 1
            pairs = merged
        
        # Count adjacent coprime pairs
        score = 0
        for i in range(len(pairs) - 1):
            if (pairs[i][1] & pairs[i+1][1]) == 0:
                score += 1
        return score

    # 3. Initialize with the "greedy" baseline: product of the smallest primes
    # This maximizes the number of distinct prime factors (k) within 10^18.
    prod = 1
    max_k = 0
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1
        
    # Start with the first max_k primes
    initial_primes = primes[:max_k]
    current_primes = list(initial_primes)
    current_n = 1
    for p in current_primes:
        current_n *= p
        
    best_n = current_n
    current_score = tau_perp(current_n, current_primes)
    best_score = current_score
    
    # 4. Local Search (Hill Climbing)
    # Strategy: Try to swap each prime in the current set with a prime not in the set.
    # If the product remains <= 10^18 and the score improves, accept the move.
    # Repeat until no improvement is found or time runs out.
    
    # Precompute the set of primes for O(1) lookup
    current_prime_set = set(current_primes)
    
    improved = True
    no_improve_count = 0
    
    # Candidate primes: all primes up to 50000
    # We will iterate through them.
    
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Shuffle the order of checking primes to avoid getting stuck in a specific local minimum
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5.0:
                return best_n
                
            p_old = current_primes[idx]
            
            # Calculate the maximum value for the new prime
            # new_n = (current_n / p_old) * p_new <= 10^18
            # p_new <= 10^18 * p_old / current_n
            max_p_new = (10**18 * p_old) // current_n
            
            # Find candidate primes.
            # We only consider primes up to max_p_new.
            # And not in current_prime_set.
            
            found_swap = False
            # To speed up, we can pre-filter candidates.
            # But max_p_new changes.
            # Let's iterate through the global primes list.
            for p_new in primes:
                if p_new > max_p_new:
                    break
                if p_new in current_prime_set:
                    continue
                    
                # Potential swap
                new_n = (current_n // p_old) * p_new
                
                # Evaluate score
                # To save time, we can do a quick check if the new prime is "too similar" to old?
                # No, just evaluate.
                s_new = tau_perp(new_n, [p if p != p_old else p_new for p in current_primes])
                
                if s_new > current_score:
                    # Accept swap
                    current_score = s_new
                    current_n = new_n
                    current_primes[idx] = p_new
                    current_prime_set.remove(p_old)
                    current_prime_set.add(p_new)
                    
                    best_score = current_score
                    best_n = current_n
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                # Restart the index loop with the new state
                break
        
        if not improved:
            no_improve_count += 1
            # If we've had several rounds without improvement, try a random restart
            # by perturbing the current solution slightly
            if no_improve_count >= 5 and time.time() < deadline - 20.0:
                # Try a random swap to escape local minimum
                if len(current_primes) > 1:
                    idx1 = random.randint(0, len(current_primes) - 1)
                    p_old = current_primes[idx1]
                    max_p_new = (10**18 * p_old) // current_n
                    # Find a random candidate
                    candidates = [p for p in primes if p not in current_prime_set and p <= max_p_new]
                    if candidates:
                        p_new = random.choice(candidates)
                        new_n = (current_n // p_old) * p_new
                        s_new = tau_perp(new_n, [p if p != p_old else p_new for p in current_primes])
                        # Accept even if it doesn't improve, to escape local minimum
                        current_n = new_n
                        current_primes[idx1] = p_new
                        current_prime_set.remove(p_old)
                        current_prime_set.add(p_new)
                        current_score = s_new
                        no_improve_count = 0
                        # Reset improved to continue searching from this new point
                        # But we don't want to count this as a "best" improvement unless it is
                        if s_new > best_score:
                            best_score = s_new
                            best_n = current_n

    return best_n
