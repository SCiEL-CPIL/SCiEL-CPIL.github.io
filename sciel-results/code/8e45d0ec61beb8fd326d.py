import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin

    # 1. Generate primes up to 20000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(20000)
    prime_index = {p: i for i, p in enumerate(primes)}

    # 2. Fast tau_perp for square-free numbers using bitmask DP
    # This function is the core of the new strategy.
    # It calculates the number of adjacent coprime pairs in the sorted divisor list.
    # For square-free n with k primes, there are 2^k divisors.
    # The divisors are subsets of the prime set.
    # We sort them by value.
    # Two divisors d_i, d_{i+1} are coprime iff their bitmask intersection is 0.
    
    # Optimization: Instead of sorting values, we can precompute the sorted order of masks?
    # No, the order depends on the specific prime values.
    # However, for a fixed set of primes, the relative order of divisors is fixed.
    
    # Let's stick to the efficient enumeration but optimize the inner loop.
    def tau_perp_squarefree(prime_list):
        k = len(prime_list)
        if k == 0:
            return 0
        
        # Generate divisors (value, mask)
        # Using a list of lists or iterative construction might be faster?
        # For k <= 15 (max for 10^18), 2^15 = 32768. Sorting 32k items is fast.
        
        divs = []
        # Bit 0 corresponds to prime_list[0]
        for mask in range(1 << k):
            d = 1
            # Multiplication by small primes
            for i in range(k):
                if mask & (1 << i):
                    d *= prime_list[i]
            divs.append((d, mask))
            
        divs.sort(key=lambda x: x[0])
        
        score = 0
        prev_mask = divs[0][1]
        for i in range(1, len(divs)):
            curr_mask = divs[i][1]
            if (prev_mask & curr_mask) == 0:
                score += 1
            prev_mask = curr_mask
            
        return score

    # 3. Find maximum number of distinct primes (primorial-based)
    # 2*3*5*...*p_k <= 10^18
    max_k = 0
    prod = 1
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1
        
    # Initial candidate: first max_k primes
    # Usually, smaller primes allow for more primes, which tends to increase the total number of divisors and potential coprime pairs.
    initial_primes = primes[:max_k]
    best_score = tau_perp_squarefree(initial_primes)
    best_primes = list(initial_primes)
    
    # 4. Structured Local Search (Hill Climbing)
    # The key improvement: We evaluate swaps more aggressively and efficiently.
    # We track the current product to quickly check feasibility.
    
    current_prod = 1
    for p in best_primes:
        current_prod *= p
        
    best_prime_set = set(best_primes)
    
    # Helper to check time
    def time_ok():
        return time.time() < deadline - 10.0
        
    # Hill climbing loop
    improved = True
    while improved and time_ok():
        improved = False
        best_primes.sort()
        best_prime_set = set(best_primes)
        
        # Try swapping each prime with a better candidate
        # We iterate through positions.
        for i in range(len(best_primes)):
            if not time_ok():
                break
                
            p_out = best_primes[i]
            # Max value for p_in such that current_prod * p_in / p_out <= 10^18
            max_p_in = int(10**18 * p_out / current_prod)
            
            # Candidate primes: not in set, <= max_p_in
            # We want to find a p_in that increases the score.
            # Heuristic: Try primes that are "structurally distinct".
            # In square-free numbers, the structure is determined by the set of primes.
            # Swapping a small prime for a large one reduces the number of divisors? No, count is same.
            # But it changes the values, thus the sorting order.
            
            # To speed up, we sample candidates.
            # If we have many candidates, sampling is better than exhaustive if we can't afford exhaustive.
            # However, for k~15, the number of primes up to 10^18/prod is small if prod is large.
            
            candidates = []
            for p in primes:
                if p > max_p_in:
                    break
                if p in best_prime_set:
                    continue
                candidates.append(p)
                
            if not candidates:
                continue
                
            # Sort candidates by a heuristic?
            # Maybe try smallest and largest first?
            # Or just iterate?
            
            # To be safe on time, if candidates > 50, sample 50.
            # But since primes are sorted, we can just take a few from start and end.
            if len(candidates) > 100:
                # Take first 50 and last 50
                sampled = candidates[:50] + candidates[-50:]
            else:
                sampled = candidates
                
            best_candidate = None
            best_candidate_score = best_score
            
            for p_in in sampled:
                if not time_ok():
                    break
                    
                new_primes = best_primes[:]
                new_primes[i] = p_in
                new_primes.sort()
                
                s = tau_perp_squarefree(new_primes)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = (p_in, s, new_primes)
                    
            if best_candidate is not None:
                p_in, new_score, new_primes = best_candidate
                best_score = new_score
                best_primes = new_primes
                best_prime_set = set(best_primes)
                current_prod = current_prod * p_in // p_out
                improved = True
                break  # Restart with new state to ensure consistency
                
    # 5. Random Perturbations (Simulated Annealing style, but simple restarts)
    # If stuck, jump to a random neighborhood.
    if time.time() < deadline - 5.0:
        for _ in range(100):
            if not time_ok():
                break
                
            curr_primes = best_primes[:]
            curr_prod = current_prod
            curr_prime_set = set(curr_primes)
            
            # Perform 2-5 random swaps
            num_swaps = random.randint(2, 5)
            valid = True
            for _ in range(num_swaps):
                i = random.randint(0, len(curr_primes) - 1)
                p_out = curr_primes[i]
                max_p_in = int(10**18 * p_out / curr_prod)
                
                candidates = [p for p in primes if p <= max_p_in and p not in curr_prime_set]
                if not candidates:
                    valid = False
                    break
                
                p_in = random.choice(candidates)
                curr_primes[i] = p_in
                curr_prod = curr_prod * p_in // p_out
                curr_prime_set.remove(p_out)
                curr_prime_set.add(p_in)
                curr_primes.sort()
                
            if valid:
                s = tau_perp_squarefree(curr_primes)
                if s > best_score:
                    best_score = s
                    best_primes = curr_primes
                    current_prod = curr_prod
                    
    # 6. Final Polish: Exhaustive single swaps from best solution
    # Use remaining time to do a thorough check of single swaps.
    if time.time() < deadline - 2.0:
        best_primes.sort()
        best_prime_set = set(best_primes)
        
        for i in range(len(best_primes)):
            if not time_ok():
                break
                
            p_out = best_primes[i]
            max_p_in = int(10**18 * p_out / current_prod)
            
            # Try all valid candidates
            for p_in in primes:
                if p_in > max_p_in:
                    break
                if p_in in best_prime_set:
                    continue
                    
                new_primes = best_primes[:]
                new_primes[i] = p_in
                new_primes.sort()
                
                s = tau_perp_squarefree(new_primes)
                if s > best_score:
                    best_score = s
                    best_primes = new_primes
                    best_prime_set = set(best_primes)
                    current_prod = current_prod * p_in // p_out

    # Construct final number
    result = 1
    for p in best_primes:
        result *= p

    return result

# Run construct to test if needed, but evaluator calls it.
if __name__ == "__main__":
    n = construct()
    print(n)
