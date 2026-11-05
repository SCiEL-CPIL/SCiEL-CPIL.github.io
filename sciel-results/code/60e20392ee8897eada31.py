import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

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
    cand_set = set(candidates)

    # Precompute for each prime, the list of candidates divisible by it, sorted by weight desc
    prime_cands = [[] for _ in range(num_primes)]
    for a in candidates:
        m = masks[a]
        temp = m
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            prime_cands[idx].append(a)
            temp ^= lsb
            
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
    
    # Helper to calculate used primes mask
    def get_used_mask(sol):
        used = 0
        for a in sol:
            used |= masks[a]
        return used

    # Helper to find best single add
    def find_best_add(used_mask, current_set):
        best_a = -1
        best_w = -1.0
        for a in candidates:
            if a in current_set:
                continue
            if (masks[a] & used_mask) == 0:
                if weights[a] > best_w:
                    best_w = weights[a]
                    best_a = a
        return best_a, best_w

    # Helper to find best pair add (a, b)
    # Optimization: only consider top K candidates for the pair search to keep it fast
    def find_best_pair(used_mask, current_set, k=100):
        valid_cands = []
        for a in candidates:
            if a in current_set:
                continue
            if (masks[a] & used_mask) == 0:
                valid_cands.append(a)
                if len(valid_cands) >= k:
                    break
        
        best_pair = None
        best_score_pair = -1.0
        # Pair search is O(K^2). K=100 -> 10k ops. 
        # If we do this for every element in sol (len ~ 50), that's 500k ops per full pass.
        # Acceptable.
        for i in range(len(valid_cands)):
            a = valid_cands[i]
            for j in range(i + 1, len(valid_cands)):
                b = valid_cands[j]
                if (masks[a] & masks[b]) != 0:
                    continue
                s = weights[a] + weights[b]
                if s > best_score_pair:
                    best_score_pair = s
                    best_pair = (a, b)
        return best_pair, best_score_pair

    # 3. Local Search with Merge (2-for-1) and Split (1-for-2) moves
    def local_search(sol):
        sol = sol[:]
        score = sum(weights[a] for a in sol)
        current_set = set(sol)
        
        improved = True
        while improved:
            improved = False
            used_mask = get_used_mask(sol)
            
            # A. Try Add (1-for-0)
            best_a, best_w = find_best_add(used_mask, current_set)
            if best_a != -1:
                sol.append(best_a)
                current_set.add(best_a)
                score += best_w
                improved = True
                continue # Restart loop with new state
            
            # B. Try 1-for-2 Swap (Split)
            # Remove one, add two.
            # This is useful if a large prime blocks two medium composites? 
            # Actually, usually we want to maximize size. 
            # If we have a small prime p, replacing it with two large composites a, b 
            # where gcd(a,b)=1 and gcd(a,p)=gcd(b,p)=1 is a win if w(a)+w(b) > w(p).
            # Since w is convex and increasing, large a,b are good.
            
            # C. Try 1-for-1 Swap
            # Standard.
            
            # D. Try 2-for-1 Swap (Merge)
            # Remove two, add one.
            # This is the key move found in inspirations.
            # If we have two small primes p, q, and a large composite c = p*q (or similar),
            # w(c) > w(p) + w(q) is possible?
            # Let's check: p=2, q=3. w(2)=1/510, w(3)=1/509. Sum ~ 1/255.
            # c=6. w(6)=1/506. 
            # 1/506 < 1/255. So simple product is worse.
            # But what if we replace a *large* prime with a composite?
            # The insight says: "replacing primes 17 and 29 with 493".
            # 17*29 = 493.
            # w(17) = 1/495. w(29) = 1/483. Sum = (483+495)/(495*483) = 978/239085 ~ 0.00409.
            # w(493) = 1/19 ~ 0.0526.
            # Huge gain! 
            # So we want to find pairs (a,b) in sol such that there exists c not in sol,
            # coprime to rest, with w(c) > w(a) + w(b).
            
            # Strategy for Merge:
            # Iterate over pairs (i, j) in sol.
            # Calculate remaining mask and score.
            # Find best c to add.
            # If w(c) > w(a_i) + w(a_j), do it.
            
            # To make this efficient, we can precompute or just brute force pairs if sol is small.
            # Sol size is typically < 100. Pairs < 5000.
            # For each pair, find best add. Find best add is O(N).
            # 5000 * 512 ~ 2.5M ops. This might be slow if done every iteration.
            # We can limit this to every few iterations or only if no other move found.
            
            # Let's structure the loop:
            # 1. Check Adds
            # 2. Check 1-for-1 Swaps
            # 3. Check 2-for-1 Merges (The powerful move)
            # 4. Check 1-for-2 Splits (Less common but possible)
            
            # We already did Add.
            
            # Let's try 1-for-1 Swap first as it's cheaper.
            found_swap = False
            for i in range(len(sol)):
                if time.time() > deadline - 10:
                    return sol, score
                a_out = sol[i]
                remaining = [x for j, x in enumerate(sol) if j != i]
                rem_set = set(remaining)
                rem_used = 0
                for a in remaining:
                    rem_used |= masks[a]
                
                # Find best add to remaining
                best_in, best_w_in = find_best_add(rem_used, rem_set)
                if best_in != -1:
                    gain = best_w_in - weights[a_out]
                    if gain > 1e-12:
                        sol[i] = best_in
                        current_set.remove(a_out)
                        current_set.add(best_in)
                        score += gain
                        improved = True
                        found_swap = True
                        break
            if found_swap:
                continue
            
            # Now try 2-for-1 Merge
            found_merge = False
            # To speed up, we can sort sol by weight ascending? 
            # Merging small items is more likely to be profitable.
            # Let's just iterate all pairs.
            for i in range(len(sol)):
                if time.time() > deadline - 10:
                    return sol, score
                a1 = sol[i]
                for j in range(i + 1, len(sol)):
                    a2 = sol[j]
                    remaining = [x for k, x in enumerate(sol) if k != i and k != j]
                    rem_set = set(remaining)
                    rem_used = 0
                    for a in remaining:
                        rem_used |= masks[a]
                    
                    # Find best add to remaining
                    best_in, best_w_in = find_best_add(rem_used, rem_set)
                    if best_in != -1:
                        gain = best_w_in - weights[a1] - weights[a2]
                        if gain > 1e-12:
                            # Execute merge
                            # Remove a1, a2
                            sol = remaining + [best_in]
                            current_set = rem_set | {best_in}
                            score += gain
                            improved = True
                            found_merge = True
                            break
                if found_merge:
                    break
            if found_merge:
                continue
            
            # Now try 1-for-2 Split
            # This is more expensive. Only try if nothing else found.
            found_split = False
            for i in range(len(sol)):
                if time.time() > deadline - 10:
                    return sol, score
                a_out = sol[i]
                remaining = [x for j, x in enumerate(sol) if j != i]
                rem_set = set(remaining)
                rem_used = 0
                for a in remaining:
                    rem_used |= masks[a]
                
                # Find best pair to add
                best_pair, best_w_pair = find_best_pair(rem_used, rem_set, k=50)
                if best_pair is not None:
                    gain = best_w_pair - weights[a_out]
                    if gain > 1e-12:
                        sol = remaining + list(best_pair)
                        current_set = rem_set | set(best_pair)
                        score += gain
                        improved = True
                        found_split = True
                        break
            if found_split:
                continue
            
            # If no improvement, break
            if not improved:
                break

        return sol, score

    sol, score = local_search(initial_sol)
    if score > best_score:
        best_score = score
        best_sol = sol[:]

    # 4. Diversified Restarts
    # Try a few different initializations to escape local optima.
    # Strategy: Randomly remove a few elements from the current best, then re-optimize.
    # Or: Randomly shuffle the order of greedy? 
    # Greedy is deterministic. 
    # Let's try "Perturbed Greedy": 
    # 1. Take best_sol.
    # 2. Remove 5-10 random elements.
    # 3. Run local_search on the remainder.
    
    import random
    rng = random.Random(42)
    
    for _ in range(10): # 10 restarts
        if time.time() > deadline - 20:
            break
        
        # Create a perturbed solution
        perturbed_sol = best_sol[:]
        # Remove 5 random elements
        num_remove = min(5, len(perturbed_sol))
        if num_remove > 0:
            to_remove = rng.sample(perturbed_sol, num_remove)
            remove_set = set(to_remove)
            perturbed_sol = [x for x in perturbed_sol if x not in remove_set]
        
        # Run local search
        new_sol, new_score = local_search(perturbed_sol)
        if new_score > best_score:
            best_score = new_score
            best_sol = new_sol[:]

    # 5. Final Polish
    # Run one more local search on the best found
    final_sol, final_score = local_search(best_sol)
    if final_score > best_score:
        best_score = final_score
        best_sol = final_sol[:]

    return best_sol

# EVOLVE-BLOCK-END
