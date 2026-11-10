import time
import math

def construct():
    start_time = time.time()
    LIMIT = 10**18
    DEADLINE = start_time + 145.0  # Safety margin

    # List of first ~60 primes for more search space
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283]
    
    def tau_perp_score(prime_list):
        """
        Calculate tau_perp for a square-free number given its list of prime factors.
        """
        k = len(prime_list)
        # Generate divisors using bitmasks
        divs = []
        for mask in range(1 << k):
            val = 1
            for i in range(k):
                if mask & (1 << i):
                    val *= prime_list[i]
            divs.append(val)
        
        divs.sort()
        
        score = 0
        prev = divs[0]
        for curr in divs[1:]:
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    def product_of_set(pset):
        prod = 1
        for p in pset:
            prod *= p
        return prod

    best_n = 0
    best_score = -1
    best_set = None

    # Strategy: Try multiple candidate sets and use local search to improve each
    
    # Candidate 1: First 15 primes
    std_set = primes[:15]
    n_std = product_of_set(std_set)
    if n_std <= LIMIT:
        score_std = tau_perp_score(std_set)
        best_score = score_std
        best_n = n_std
        best_set = std_set[:]

    # Try some alternative initial sets
    # Set 2: Replace some small primes with larger ones to create more "spread"
    # The idea: if divisors are more spread out, adjacent pairs might be more likely to be coprime
    
    # Let's try a few heuristic initial sets
    initial_sets = [
        primes[:15],  # Already tried
    ]
    
    # Generate some alternative sets by swapping
    for swap_idx in range(1, 14):
        for new_prime_idx in range(15, min(30, len(primes))):
            alt_set = primes[:15][:]
            alt_set[swap_idx] = primes[new_prime_idx]
            alt_set.sort()
            alt_prod = product_of_set(alt_set)
            if alt_prod <= LIMIT:
                initial_sets.append(alt_set)
    
    # Try each initial set with local search
    for init_set in initial_sets:
        if time.time() > DEADLINE - 30:
            break
            
        current_set = init_set[:]
        current_score = tau_perp_score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_n = product_of_set(current_set)
            best_set = current_set[:]
        
        # Local search: try swapping each prime with a larger one
        improved = True
        while improved and time.time() < DEADLINE - 15:
            improved = False
            current_score = tau_perp_score(current_set)
            current_prod = product_of_set(current_set)
            
            # Try swapping each prime in current_set with a larger prime
            for idx in range(len(current_set)):
                p_old = current_set[idx]
                # Find candidate replacement primes > p_old
                for p_new in primes:
                    if p_new <= p_old:
                        continue
                    if p_new in current_set:
                        continue
                    
                    new_prod = current_prod // p_old * p_new
                    if new_prod > LIMIT:
                        break # Since primes are sorted, larger p_new will also exceed limit
                    
                    new_set = current_set[:]
                    new_set[idx] = p_new
                    new_set.sort()
                    
                    score_new = tau_perp_score(new_set)
                    if score_new > best_score:
                        best_score = score_new
                        best_n = new_prod
                        best_set = new_set[:]
                        current_set = new_set
                        improved = True
                        break
                if improved:
                    break
            
            # If no improvement in a full pass, try a random perturbation
            if not improved and time.time() < DEADLINE - 20:
                import random
                random.seed(int(time.time() * 1000) % 10000)
                # Try a random swap
                idx1 = random.randint(0, len(current_set) - 1)
                p_old = current_set[idx1]
                candidates = [p for p in primes if p > p_old and p not in current_set]
                if candidates:
                    # Limit candidates to those that keep product <= LIMIT
                    current_prod = product_of_set(current_set)
                    valid_candidates = [p for p in candidates if current_prod // p_old * p <= LIMIT]
                    if valid_candidates:
                        p_new = random.choice(valid_candidates)
                        new_set = current_set[:]
                        new_set[idx1] = p_new
                        new_set.sort()
                        score_new = tau_perp_score(new_set)
                        if score_new > best_score:
                            best_score = score_new
                            best_n = product_of_set(new_set)
                            best_set = new_set[:]
                            current_set = new_set
                            improved = True

    if best_n == 0:
        # Fallback to first 15 primes
        best_n = 1
        for p in primes[:15]:
            best_n *= p
            
    assert 2 <= best_n <= LIMIT
    return best_n
