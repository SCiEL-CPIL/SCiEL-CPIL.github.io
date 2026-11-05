import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill
    
    # Precomputation
    N = 511
    
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
    
    # 1. Greedy Initial Solution
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
    
    # 2. Beam Search
    def beam_search(beam_width=5000):
        """
        Beam search over partial solutions.
        Each beam entry is (score, used_mask, frozenset_of_selected)
        We expand by adding each compatible candidate.
        """
        # Start with empty solution
        beam = [(0.0, 0, frozenset())]
        
        # Precompute for each candidate: its mask and weight
        # We'll iterate through candidates in order of weight (descending)
        
        for a in candidates:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            new_beam = []
            ma = masks[a]
            wa = weights[a]
            
            for (score, used, selected) in beam:
                # Option 1: Don't include a
                new_beam.append((score, used, selected))
                
                # Option 2: Include a if compatible
                if (ma & used) == 0:
                    new_score = score + wa
                    new_used = used | ma
                    new_selected = selected | {a}
                    new_beam.append((new_score, new_used, new_selected))
            
            # Keep top beam_width entries
            if len(new_beam) > beam_width:
                # Sort by score descending, take top beam_width
                new_beam.sort(key=lambda x: x[0], reverse=True)
                new_beam = new_beam[:beam_width]
            
            beam = new_beam
        
        # Extract best solution
        if beam:
            best = max(beam, key=lambda x: x[0])
            return list(best[2]), best[0]
        return [], 0.0

    # Run beam search
    bs_sol, bs_score = beam_search(beam_width=5000)
    if bs_score > best_score:
        best_score = bs_score
        best_sol = bs_sol[:]
    
    # 3. Local Search with 2-for-1 swaps (from inspiration)
    def local_search(sol):
        sol = sol[:]
        score = sum(weights[a] for a in sol)
        improved = True
        # Limit iterations to avoid getting stuck
        max_iters = 200
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            if time.time() - start_time > DEADLINE - 10:
                break

            # Try to add any compatible number
            used = 0
            for a in sol:
                used |= masks[a]
            
            added = False
            for a in candidates:
                if a in sol:
                    continue
                if (masks[a] & used) == 0:
                    sol.append(a)
                    score += weights[a]
                    used |= masks[a]
                    improved = True
                    added = True
                    break # Restart loop
            
            if added:
                continue

            # Try 1-for-1 swap
            swap_made = False
            for i in range(len(sol)):
                a_out = sol[i]
                remaining = [x for j, x in enumerate(sol) if j != i]
                rem_used = 0
                for a in remaining:
                    rem_used |= masks[a]
                
                best_swap = -1
                best_gain = 1e-9 # Small epsilon
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
                    swap_made = True
                    break # Restart loop
            
            if swap_made:
                continue

            # Try 2-for-1 swap: replace two numbers a1, a2 with one number a3
            found_2for1 = False
            # Iterate over pairs in sol
            for i in range(len(sol)):
                for j in range(i + 1, len(sol)):
                    a1 = sol[i]
                    a2 = sol[j]
                    
                    # Calculate used primes by rest of solution
                    rem_used = 0
                    for k in range(len(sol)):
                        if k != i and k != j:
                            rem_used |= masks[sol[k]]
                    
                    rem_score = score - weights[a1] - weights[a2]
                    
                    best_swap_in = -1
                    best_gain = 1e-9
                    for a_in in candidates:
                        if a_in == a1 or a_in == a2:
                            continue
                        if a_in in sol:
                            continue
                        # Must be coprime to rest
                        if (masks[a_in] & rem_used) != 0:
                            continue
                        
                        gain = weights[a_in] - weights[a1] - weights[a2]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap_in = a_in
                    
                    if best_swap_in != -1:
                        new_sol = []
                        for k in range(len(sol)):
                            if k != i and k != j:
                                new_sol.append(sol[k])
                        new_sol.append(best_swap_in)
                        sol = new_sol
                        score += best_gain
                        improved = True
                        found_2for1 = True
                        break # Restart loop
                if found_2for1:
                    break
            
            # Try 1-for-2 swap: replace one number a1 with two numbers a2, a3
            found_1for2 = False
            for i in range(len(sol)):
                a1 = sol[i]
                rem_used = 0
                for k in range(len(sol)):
                    if k != i:
                        rem_used |= masks[sol[k]]
                
                rem_score = score - weights[a1]
                
                # Find best pair a2, a3 such that:
                # 1. Coprime to rem_used
                # 2. Coprime to each other
                # 3. Distinct from a1
                # 4. Maximize weights[a2] + weights[a3] - weights[a1]
                
                compat = []
                for a in candidates:
                    if a == a1:
                        continue
                    if (masks[a] & rem_used) == 0:
                        compat.append(a)
                
                best_pair = None
                best_sum = -1e9
                top_compat = compat[:50]
                for x in range(len(top_compat)):
                    a2 = top_compat[x]
                    for y in range(x + 1, len(top_compat)):
                        a3 = top_compat[y]
                        if (masks[a2] & masks[a3]) == 0:
                            s = weights[a2] + weights[a3]
                            if s > best_sum:
                                best_sum = s
                                best_pair = (a2, a3)
                
                if best_pair:
                    gain = best_sum - weights[a1]
                    if gain > 1e-9:
                        new_sol = []
                        for k in range(len(sol)):
                            if k != i:
                                new_sol.append(sol[k])
                        new_sol.append(best_pair[0])
                        new_sol.append(best_pair[1])
                        sol = new_sol
                        score += gain
                        improved = True
                        found_1for2 = True
                        break # Restart loop
            
            if found_1for2:
                continue

        return sol, score

    # Polish the best solution with local search
    if best_sol:
        ls_sol, ls_score = local_search(best_sol)
        if ls_score > best_score:
            best_score = ls_score
            best_sol = ls_sol[:]

    # 4. Random restarts with local search if time permits
    if time.time() - start_time < DEADLINE - 20:
        import random
        random.seed(42)
        
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Perturb the best solution
            perturbed_sol = best_sol[:]
            for _ in range(random.randint(1, 3)):
                if perturbed_sol:
                    idx = random.randint(0, len(perturbed_sol) - 1)
                    perturbed_sol.pop(idx)
            
            p_sol, p_score = local_search(perturbed_sol)
            if p_score > best_score:
                best_score = p_score
                best_sol = p_sol[:]
                
    return best_sol
