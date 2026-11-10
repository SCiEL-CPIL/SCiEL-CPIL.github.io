"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(256)
    num_primes = len(primes)
    
    # Identify fixed primes (p > 128)
    fixed_primes = [p for p in primes if p > 128]
    var_primes = [p for p in primes if p <= 128]
    
    fixed_sum = sum(fixed_primes)
    
    # Map variable primes to indices 0..M-1
    var_prime_index = {p: i for i, p in enumerate(var_primes)}
    num_var_primes = len(var_primes)
    
    # Precompute prime factor masks for all numbers 1..256
    # mask bits correspond to var_primes
    # If a number has a prime factor not in var_primes (i.e., in fixed_primes), 
    # it can only be that prime itself. But we already handled fixed primes.
    # So we only consider numbers x such that all prime factors of x are in var_primes.
    # Note: 1 has no prime factors, mask 0.
    
    number_masks = {}
    for x in range(1, 257):
        mask = 0
        temp = x
        valid = True
        for p in var_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << var_prime_index[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is a prime factor
            if temp in var_prime_index:
                mask |= (1 << var_prime_index[temp])
            else:
                # temp is a fixed prime
                # If x == temp, it's a fixed prime (handled).
                # If x != temp, x is a multiple of a fixed prime, so x > fixed_prime.
                # Since 2*fixed_prime > 256, x must be fixed_prime itself.
                # But we are iterating x. If x is a fixed prime, we skip it in variable set.
                # If x is a multiple, it's invalid for variable set.
                valid = False
        
        if valid:
            number_masks[x] = mask

    # Group candidates by the smallest prime factor? Or just list all candidates.
    # For B&B, we need to branch on a prime.
    # For each variable prime p, list candidates x such that p divides x.
    # Sort candidates by value descending.
    
    candidates_by_prime = [[] for _ in range(num_var_primes)]
    for x, mask in number_masks.items():
        if x == 1:
            continue # 1 is added at the end, or handled separately. 
            # Actually, 1 is coprime to all. We should always include 1?
            # Yes, adding 1 increases sum by 1 and doesn't conflict.
            # So we can just add 1 to the final sum.
        
        # Find which primes divide x
        # We can just check bits in mask
        for i in range(num_var_primes):
            if mask & (1 << i):
                candidates_by_prime[i].append(x)
                
    # Sort candidates for each prime in descending order
    for i in range(num_var_primes):
        candidates_by_prime[i].sort(reverse=True)

    # Memoization for B&B
    memo = {}
    
    def solve(mask):
        if mask == 0:
            return 0
        if mask in memo:
            return memo[mask]
            
        # Upper Bound: Sum of max candidate for each available prime
        ub = 0
        temp_mask = mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            # Find max candidate for idx that is compatible with mask
            # A candidate x is compatible if its mask is a subset of mask
            # Since candidates_by_prime[idx] contains x where idx is a factor,
            # we need to check if all factors of x are in mask.
            # But we precomputed number_masks[x].
            # So we need to check if number_masks[x] & ~mask == 0.
            
            # Optimization: Precompute for each prime idx and each mask? No, too big.
            # Just iterate candidates.
            max_val = 0
            for x in candidates_by_prime[idx]:
                if number_masks[x] & ~mask == 0:
                    max_val = x
                    break # Since sorted desc, first valid is max
            ub += max_val
            temp_mask ^= lsb
            
        # If we have a global best, we could prune here, but let's rely on memoization and structure.
        # To improve pruning, we could pass current_sum and compare with global_best.
        # But for simplicity and correctness first, let's just do the branching.
        
        # Choose a prime to branch on.
        # Heuristic: Choose the prime with the fewest valid options?
        # Or the one with the highest max_val?
        # Let's choose the prime with the smallest index in the mask for determinism, 
        # or the one with the fewest options to reduce branching factor.
        
        best_idx = -1
        min_options = float('inf')
        temp_mask = mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            count = 0
            for x in candidates_by_prime[idx]:
                if number_masks[x] & ~mask == 0:
                    count += 1
                    # Early exit if count > min_options
                    if count >= min_options:
                        break
            if count < min_options:
                min_options = count
                best_idx = idx
            temp_mask ^= lsb
            
        if best_idx == -1:
            memo[mask] = 0
            return 0
            
        # Option 1: Skip best_idx
        # This means we don't pick any number divisible by best_idx.
        # So we remove best_idx from mask.
        val_skip = solve(mask & ~(1 << best_idx))
        
        # Option 2: Pick a number divisible by best_idx
        val_pick = 0
        for x in candidates_by_prime[best_idx]:
            if number_masks[x] & ~mask == 0:
                new_mask = mask & ~number_masks[x]
                val = x + solve(new_mask)
                if val > val_pick:
                    val_pick = val
                    # Optional: if val + ub_remaining > best_global, update?
                    # We don't track global best easily in this recursive structure without passing it.
                    # But memoization helps.
                
        best_val = max(val_skip, val_pick)
        memo[mask] = best_val
        return best_val

    all_var_mask = (1 << num_var_primes) - 1
    max_var_sum = solve(all_var_mask)
    
    # Construct the result
    # We need to recover the numbers.
    # We can modify solve to return (sum, list) or use a separate reconstruction.
    # Given the small size, we can just re-run or store decisions.
    
    decisions = {}
    
    def solve_decisions(mask):
        if mask == 0:
            return 0, []
        if mask in memo:
            # We need to recompute decisions if we didn't store them.
            # Let's store decisions in a dict during the first pass?
            # Or just re-run with a flag.
            pass
            
    # Let's modify solve to store the best choice.
    # We'll create a second function or modify the existing one to record choices.
    
    choice_memo = {}
    
    def solve_choice(mask):
        if mask == 0:
            return 0, []
        if mask in choice_memo:
            return choice_memo[mask]
            
        # Recalculate best_idx logic (same as above)
        best_idx = -1
        min_options = float('inf')
        temp_mask = mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            count = 0
            for x in candidates_by_prime[idx]:
                if number_masks[x] & ~mask == 0:
                    count += 1
                    if count >= min_options:
                        break
            if count < min_options:
                min_options = count
                best_idx = idx
            temp_mask ^= lsb
            
        if best_idx == -1:
            choice_memo[mask] = (0, [])
            return 0, []
            
        val_skip, list_skip = solve_choice(mask & ~(1 << best_idx))
        
        val_pick = 0
        list_pick = []
        best_x = -1
        for x in candidates_by_prime[best_idx]:
            if number_masks[x] & ~mask == 0:
                new_mask = mask & ~number_masks[x]
                val, lst = solve_choice(new_mask)
                total_val = x + val
                if total_val > val_pick:
                    val_pick = total_val
                    list_pick = [x] + lst
                    best_x = x
                    
        if val_skip >= val_pick:
            res = (val_skip, list_skip)
        else:
            res = (val_pick, list_pick)
            
        choice_memo[mask] = res
        return res

    _, var_numbers = solve_choice(all_var_mask)
    
    result = var_numbers + fixed_primes + [1]
    
    # Ensure types are int
    result = [int(x) for x in result]
    
    return result
# EVOLVE-BLOCK-END
