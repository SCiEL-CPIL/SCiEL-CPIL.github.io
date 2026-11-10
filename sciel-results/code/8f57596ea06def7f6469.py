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
    # The product of the first 15 primes is ~6e17. 
    # We can swap larger primes with smaller ones to fit more primes or rearrange.
    # Let's generate primes up to 10000 to have plenty of candidates.
    primes = sieve(10000)
    
    # 2. Define tau_perp calculation optimized for speed
    def tau_perp(n):
        # Factorize n
        temp = n
        factors = []
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                factors.append(d)
                while temp % d == 0:
                    temp //= d
            else:
                d += 1
        if temp > 1:
            factors.append(temp)
            
        # Generate divisors
        divs = [1]
        for p in factors:
            new_divs = []
            for d in divs:
                new_divs.append(d * p)
            divs.extend(new_divs)
            
        divs.sort()
        
        # Count adjacent coprime pairs
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # 3. Initialize with the "greedy" baseline: product of the smallest primes
    # This maximizes the number of distinct prime factors (k) within 10^18.
    # Product of first 15 primes: 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47
    # Let's calculate max k.
    prod = 1
    max_k = 0
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1
        
    # Start with the first max_k primes
    initial_primes = primes[:max_k]
    best_n = 1
    for p in initial_primes:
        best_n *= p
        
    best_score = tau_perp(best_n)
    
    # 4. Local Search (Hill Climbing)
    # Strategy: Try to swap each prime in the current set with a larger prime not in the set.
    # If the product remains <= 10^18 and the score improves, accept the move.
    # Repeat until no improvement is found or time runs out.
    
    # To make this efficient, we track the current set of primes as a sorted list.
    current_primes = list(initial_primes)
    current_n = best_n
    current_score = best_score
    
    # Precompute the set of primes for O(1) lookup
    current_prime_set = set(current_primes)
    
    # We will iterate through the primes in the current set.
    # For each prime p_i, we try to replace it with a prime q_j > p_i 
    # such that (current_n / p_i) * q_j <= 10^18.
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Shuffle the order of checking primes to avoid getting stuck in a specific local minimum
        # or to explore different parts of the space.
        # Actually, systematic iteration is often better for deterministic local search,
        # but randomizing the order of attempts can help escape shallow local minima.
        # Let's use a mix: iterate through indices, but maybe randomize which prime to swap first.
        
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
            
            # Find candidate primes to swap in.
            # We should look for primes greater than p_old to maintain some ordering logic,
            # but technically any prime not in the set works.
            # However, swapping a small prime for a large one reduces the "density" of small factors,
            # which might hurt the score. Swapping a large prime for a smaller one (not in set)
            # is impossible if we started with the smallest k primes.
            # So we are effectively trying to replace primes with LARGER primes.
            # This seems counter-intuitive for maximizing divisors, but the *arrangement*
            # of divisors depends on the specific values.
            
            # Let's search for candidates in the primes list.
            # To be efficient, we only check primes that are not already in the set.
            # We'll scan primes from the beginning, but skip those in current_prime_set.
            # We only consider p_new such that p_new <= max_p_new.
            
            # Optimization: If max_p_new is small, we only check small primes.
            # Since we started with the smallest primes, most small primes are already in the set.
            # So we will mostly be looking at primes just outside the current range.
            
            # Find the index in the global primes list where we can start searching.
            # We want p_new not in current_prime_set.
            
            # A simple loop over primes is fast enough if we break early.
            # primes is sorted.
            found_swap = False
            for p_new in primes:
                if p_new in current_prime_set:
                    continue
                if p_new > max_p_new:
                    break
                
                # Potential swap
                # new_n = current_n // p_old * p_new
                new_n = (current_n // p_old) * p_new
                
                # Evaluate score
                s_new = tau_perp(new_n)
                
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
                    break # Restart the inner loop or continue? 
                         # Breaking allows us to re-evaluate the new state immediately in the next outer iteration 
                         # or continue with other indices. 
                         # Since the set changed, other indices' validity (max_p_new) changed slightly, 
                         # but the structure is similar. 
                         # Let's break the inner loop to restart the index iteration with the new state.
                         # This is "First Improvement" strategy.
            
            if found_swap:
                # Restart the index loop with the new state
                break

    return best_n
