import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    N = 512
    
    # 1. Precomputation
    # Sieve for primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime

    is_prime = sieve(511)
    primes = [p for p in range(2, 512) if is_prime[p]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Prime factor masks for each number 1..511
    # mask[a] is a Python int (bitmask) of prime indices
    masks = [0] * 512
    for a in range(2, 512):
        m = a
        mask_val = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask_val |= (1 << prime_to_idx[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            # m is a prime
            mask_val |= (1 << prime_to_idx[m])
        masks[a] = mask_val
    masks[1] = 0 # 1 is coprime to everything, no prime factors

    # Weights
    weights = [0.0] * 512
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)
        
    # Candidates sorted by weight descending
    # We only care about 1..511
    candidates = list(range(511, 0, -1))
    candidates.sort(key=lambda x: weights[x], reverse=True)
    
    # Precompute for each prime, the list of candidates divisible by it, sorted by weight desc
    # This helps in calculating the Upper Bound
    prime_cands = [[] for _ in range(num_primes)]
    for a in candidates:
        m = masks[a]
        # Iterate set bits in m
        temp = m
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            prime_cands[idx].append(a)
            temp ^= lsb
            
    # Each prime_cands[i] is already sorted by weight desc because 'candidates' was sorted

    # 2. Greedy Initial Solution
    def greedy_solve():
        used_primes = 0
        score = 0.0
        sol = []
        for a in candidates:
            if (masks[a] & used_primes) == 0:
                sol.append(a)
                used_primes |= masks[a]
                score += weights[a]
        return sol, score

    initial_sol, initial_score = greedy_solve()
    best_score = initial_score
    best_sol = initial_sol[:]
    
    # Local Search on Greedy Solution
    def local_search(sol):
        sol = sol[:]
        score = sum(weights[a] for a in sol)
        improved = True
        while improved:
            improved = False
            # Try to add
            used = 0
            for a in sol:
                used |= masks[a]
            for a in candidates:
                if a in sol:
                    continue
                if (masks[a] & used) == 0:
                    # Add it
                    sol.append(a)
                    score += weights[a]
                    used |= masks[a]
                    improved = True
                    break # Restart loop to recalc used? Or just continue? 
                    # Since we added one, we might be able to add more? 
                    # No, adding one might block others. But it doesn't unblock.
                    # So we can just continue checking?
                    # Actually, if we add a, it might conflict with future candidates.
                    # So we should restart the scan or update used.
                    # Updating used is fine.
            # Try 1-for-1 swap
            for i in range(len(sol)):
                if time.time() > deadline - 5:
                    return sol, score
                a_out = sol[i]
                remaining = [x for j, x in enumerate(sol) if j != i]
                rem_used = 0
                for a in remaining:
                    rem_used |= masks[a]
                rem_score = score - weights[a_out]
                
                best_swap = -1
                best_gain = 0
                for a_in in candidates:
                    if a_in in remaining:
                        continue
                    if (masks[a_in] & rem_used) == 0:
                        gain = weights[a_in] - weights[a_out]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = a_in
                
                if best_swap != -1:
                    sol[i] = best_swap
                    score += best_gain
                    improved = True
                    break # Restart
            
        return sol, score

    sol, score = local_search(initial_sol)
    if score > best_score:
        best_score = score
        best_sol = sol[:]

    # 3. Branch and Bound
    # State: used_primes (int), current_score (float), avail_mask (int, 512 bits)
    # avail_mask bit i is 1 if candidate i is available (not excluded by branching)
    
    # Initialize avail_mask with all candidates 1..511
    full_avail_mask = 0
    for a in range(1, 512):
        full_avail_mask |= (1 << a)
        
    # Note: Python ints are arbitrary precision, so 1<<511 is fine.
    
    def calculate_ub(used_primes, avail_mask):
        ub = 0.0
        for i in range(num_primes):
            p = primes[i]
            # Check if prime p is used
            if used_primes & (1 << i):
                continue
            
            # Find best candidate for this prime
            # prime_cands[i] is sorted by weight desc
            found = False
            for a in prime_cands[i]:
                # Check if a is in avail_mask
                if not (avail_mask & (1 << a)):
                    continue
                # Check if a is compatible with used_primes
                if masks[a] & used_primes:
                    continue
                # This is the best available candidate using prime p
                ub += weights[a]
                found = True
                break
                
        return ub

    def bnb(used_primes, current_score, avail_mask):
        nonlocal best_score, best_sol
        
        if time.time() > deadline - 2:
            return
            
        # Update best
        if current_score > best_score:
            best_score = current_score
            # Reconstruct solution? 
            # We need to track the solution. 
            # Passing solution list is expensive. 
            # We can reconstruct from used_primes? No, multiple numbers can share no primes but used_primes is union.
            # We need to track the set of selected numbers.
            # Let's pass a tuple of selected numbers? 
            # Or, since we only update best_score, we can store the path.
            # But for now, let's just track score. We can reconstruct best_sol at the end 
            # by running a greedy on the final used_primes? No.
            # We must track the solution.
            # Let's modify bnb to take `sol` as a tuple/list.
            # To avoid copying, we can use a global stack or pass by reference and backtrack.
            # Given the depth, passing a list copy is likely okay if pruning is good.
            pass
            
    # Revised BNB with Solution Tracking
    def bnb_search(used_primes, current_score, avail_mask, sol):
        nonlocal best_score, best_sol
        
        if time.time() > deadline - 2:
            return
            
        if current_score > best_score:
            best_score = current_score
            best_sol = sol[:]
            
        if avail_mask == 0:
            return
            
        # Calculate Upper Bound
        ub = calculate_ub(used_primes, avail_mask)
        if current_score + ub <= best_score:
            return
            
        # Select a pivot candidate
        # Choose the highest weight candidate from avail_mask that is compatible with used_primes
        pivot = -1
        max_w = -1.0
        # Iterate through candidates sorted by weight
        for a in candidates:
            if not (avail_mask & (1 << a)):
                continue
            if masks[a] & used_primes:
                continue
            if weights[a] > max_w:
                max_w = weights[a]
                pivot = a
            # Since candidates are sorted, we can break early if we found one?
            # No, we want the highest weight. The first compatible one IS the highest weight.
            break
            
        if pivot == -1:
            return
            
        # Branch 1: Include pivot
        new_used = used_primes | masks[pivot]
        new_avail = avail_mask & ~(1 << pivot)
        # Remove all candidates conflicting with pivot
        # We can just let the compatibility check `(masks[a] & new_used) == 0` handle it in the next steps
        # But we must remove pivot itself from avail_mask.
        # The `calculate_ub` and next `bnb` call will check compatibility with `new_used`.
        
        new_sol = sol + [pivot]
        bnb_search(new_used, current_score + weights[pivot], new_avail, new_sol)
        
        if time.time() > deadline - 2:
            return

        # Branch 2: Exclude pivot
        new_avail_excl = avail_mask & ~(1 << pivot)
        bnb_search(used_primes, current_score, new_avail_excl, sol)

    # Run BNB
    # Start with empty set
    bnb_search(0, 0.0, full_avail_mask, [])
    
    return best_sol

# EVOLVE-BLOCK-END
