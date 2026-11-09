import time
import math

def construct():
    """
    Returns a list of distinct ints from 1..256, pairwise coprime, maximizing sum.
    
    Strategy:
    1. Start with the base solution: {1} U {L_p for all primes p <= 256}, where L_p is the largest power of p <= 256.
    2. Identify "swaps": composite numbers c where c > sum(L_p for p|c).
    3. Solve the weighted set packing problem: select a subset of swaps such that no prime is used in more than one swap, maximizing total gain.
    4. Use backtracking with strong pruning (upper bound on remaining gain) to find the optimal set of swaps efficiently.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin (limit is 60s, keep 5s buffer)
    
    N = 256
    
    # 1. Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, math.isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Define base solution
    prime_power_map = {}
    base_sum = 1
    selected_base = {1: True}
    
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_power_map[p] = power
        base_sum += power
        selected_base[power] = True
        
    best_sum = base_sum
    best_set = list(selected_base.keys())
    
    # 3. Identify potential "swaps" (composites c where c > sum of L_p for p|c)
    candidates = []
    for c in range(4, N + 1):
        if is_prime[c]:
            continue
        # Find prime factors of c
        div_primes = []
        temp = c
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                div_primes.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            div_primes.append(temp)
            
        # Calculate gain
        base_val = sum(prime_power_map[p] for p in div_primes)
        gain = c - base_val
        if gain > 0:
            candidates.append((c, div_primes, gain))
            
    # Sort candidates by gain descending to help pruning
    candidates.sort(key=lambda x: -x[2])
    
    num_candidates = len(candidates)
    
    # Precompute suffix sums of gains for upper bound pruning
    # suffix_gain[i] = sum of gains of candidates from i to end
    suffix_gain = [0] * (num_candidates + 1)
    for i in range(num_candidates - 1, -1, -1):
        suffix_gain[i] = suffix_gain[i + 1] + candidates[i][2]
        
    # Map prime to bitmask for fast conflict checking
    # Since there are ~54 primes, we can use a Python int as a bitmask
    prime_to_bit = {p: (1 << i) for i, p in enumerate(primes)}
    
    # Convert candidate div_primes to bitmask
    candidate_masks = []
    for c, div_primes, gain in candidates:
        mask = 0
        for p in div_primes:
            mask |= prime_to_bit[p]
        candidate_masks.append(mask)
        
    def backtrack(idx, current_gain, used_mask, current_selections):
        nonlocal best_sum, best_set
        nonlocal deadline, start_time
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: if current_gain + upper_bound <= current_best_gain, prune
        # Note: We're comparing against best_sum - base_sum (the best gain found so far)
        if current_gain + suffix_gain[idx] <= best_sum - base_sum:
            return
            
        if idx == num_candidates:
            if current_gain > best_sum - base_sum:
                # Update best
                best_sum = base_sum + current_gain
                # Construct set
                new_set = set(selected_base.keys())
                for c, div_primes, g in current_selections:
                    # Remove prime powers
                    for p in div_primes:
                        lp = prime_power_map[p]
                        if lp in new_set:
                            new_set.remove(lp)
                    # Add composite
                    new_set.add(c)
                best_set = list(new_set)
            return
            
        # Option 1: Skip candidate at idx
        backtrack(idx + 1, current_gain, used_mask, current_selections)
        
        # Option 2: Take candidate at idx
        c, div_primes, gain = candidates[idx]
        mask = candidate_masks[idx]
        
        if (used_mask & mask) == 0:
            backtrack(idx + 1, current_gain + gain, used_mask | mask, current_selections + [(c, div_primes, gain)])
            
    # Run backtracking
    backtrack(0, 0, 0, [])
    
    # Final validation
    if time.time() < deadline:
        s = set(best_set)
        if len(s) != len(best_set):
            return [1] + [prime_power_map[p] for p in primes]
            
        for i in range(len(best_set)):
            for j in range(i + 1, len(best_set)):
                if math.gcd(best_set[i], best_set[j]) != 1:
                    return [1] + [prime_power_map[p] for p in primes]
                    
        if not all(1 <= x <= 256 for x in best_set):
            return [1] + [prime_power_map[p] for p in primes]
            
    # Ensure return type is list of ints
    return [int(x) for x in best_set]
