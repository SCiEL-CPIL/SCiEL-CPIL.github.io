"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    N = 256
    
    # 1. Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Calculate V_p: max power of p <= N
    V_p = {}
    for p in primes:
        max_pow = 1
        while max_pow * p <= N:
            max_pow *= p
        V_p[p] = max_pow
        
    baseline_sum = 1 + sum(V_p.values())
    
    # 3. Generate all composite numbers <= N and their prime factorizations
    # We only care about composites that could potentially be part of a merge.
    # A merge of primes S replaces sum(V_p for p in S) with C = prod(p^e for p in S).
    # Gain = C - sum(V_p for p in S).
    
    # We'll iterate through all numbers 2..N and find their prime factors.
    # For each number C, let S be its set of unique prime factors.
    # Calculate gain.
    
    merges = [] # List of (frozenset(S), gain)
    active_primes = set()
    
    for C in range(2, N + 1):
        if is_prime[C]:
            continue
        # Factorize C
        temp = C
        factors = set()
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
            
        if len(factors) < 2:
            continue # Not a composite with distinct prime factors (or prime power, but we already handle primes separately? No, prime powers are V_p. We don't merge a prime with itself.)
            
        # Calculate gain
        sum_V = sum(V_p[p] for p in factors)
        gain = C - sum_V
        
        if gain > 0:
            merges.append((frozenset(factors), gain))
            active_primes.update(factors)
            
    # 4. Solve Set Packing on active_primes
    # If active_primes is small, we can use DP or DFS.
    # Number of active primes is likely small.
    active_list = list(active_primes)
    if not active_list:
        # No merges found, return baseline
        return [1] + [V_p[p] for p in primes]
        
    # Map prime to index
    prime_to_idx = {p: i for i, p in enumerate(active_list)}
    num_active = len(active_list)
    
    # Convert merges to (mask, gain)
    merge_items = []
    for S, gain in merges:
        mask = 0
        for p in S:
            mask |= (1 << prime_to_idx[p])
        merge_items.append((mask, gain))
        
    # Sort by gain descending? Or just iterate.
    # Use DFS to find max sum of gains with disjoint masks.
    # Since num_active is small (likely < 20), 2^num_active is feasible for DP if we want, 
    # but DFS with pruning is fine too.
    
    # Let's use a simple recursive search with memoization or just branching.
    # Given the small size, we can even iterate over subsets of merges if count is low.
    # But merges can overlap.
    
    # Let's use a DP over bitmask of active primes.
    # dp[mask] = max gain achievable using only primes in 'mask'
    # This is O(3^k) or similar, but k is small.
    
    # Actually, standard set packing:
    # dp[mask] = max gain for subset of primes 'mask'
    # Init dp[0] = 0
    # For each mask, for each merge item (m_mask, m_gain):
    #   if mask & m_mask == 0:
    #      dp[mask | m_mask] = max(dp[mask | m_mask], dp[mask] + m_gain)
    # Wait, this is not quite right because we can pick multiple merges.
    # It is better to iterate over masks and update.
    
    size = 1 << num_active
    dp = [-1] * size
    dp[0] = 0
    
    # Optimization: Group merges by mask? No, multiple merges can have same mask?
    # No, each C has a unique set of prime factors.
    # So each mask corresponds to at most one merge item (the number C).
    # Wait, different C can have same prime factors?
    # E.g., C=6 (2,3) and C=12 (2,3).
    # For mask {2,3}, we want the max gain among all C with factors {2,3}.
    # So we should compress merge_items: for each mask, keep max gain.
    
    best_gain_for_mask = {}
    for mask, gain in merge_items:
        if mask not in best_gain_for_mask or gain > best_gain_for_mask[mask]:
            best_gain_for_mask[mask] = gain
            
    # Now we have a list of (mask, gain) with unique masks.
    items = list(best_gain_for_mask.items())
    
    # DP
    for mask in range(size):
        if dp[mask] < 0:
            continue
        for m_mask, m_gain in items:
            if mask & m_mask == 0:
                new_mask = mask | m_mask
                if dp[mask] + m_gain > dp[new_mask]:
                    dp[new_mask] = dp[mask] + m_gain
                    
    max_extra_gain = max(dp)
    
    # Determine which merges were selected to reconstruct the solution
    # We need to backtrack or store choices.
    # For simplicity, since we just need the sum, we can calculate the final sum.
    # Final Sum = baseline_sum + max_extra_gain
    
    # But we need to return the LIST of integers.
    # So we need to know which merges were picked.
    
    # Re-run DP storing parent pointers or just pick the optimal mask and decompose it.
    # Actually, we can just find the optimal mask 'opt_mask' and then find a set of items that covers it?
    # No, we need the specific items.
    
    # Let's modify DP to store the items chosen.
    # Or simpler: Since the number of active primes is small, we can just try all subsets of items?
    # Number of items is small (composites with positive gain).
    
    # Let's just iterate through all possible combinations of items if count is small (< 20).
    # If count is larger, we use the DP path.
    
    # Let's use a dictionary to store the best gain and the items used for each mask.
    # This might be memory heavy if size is large, but num_active is small.
    
    # Alternative: Just use the max gain value to compute the sum, and then 
    # reconstruct the list by checking which primes are "used" in the optimal solution.
    # But we don't know the solution structure from just the max gain.
    
    # Let's store the solution in the DP table.
    # dp_sol[mask] = list of item indices used to achieve dp[mask]
    
    dp_sol = [None] * size
    dp_sol[0] = []
    
    for mask in range(size):
        if dp_sol[mask] is None:
            continue
        for i, (m_mask, m_gain) in enumerate(items):
            if mask & m_mask == 0:
                new_mask = mask | m_mask
                new_gain = dp[mask] + m_gain
                if new_gain > dp[new_mask]:
                    dp[new_mask] = new_gain
                    dp_sol[new_mask] = dp_sol[mask] + [i]
                    
    # Find best mask
    best_mask = 0
    best_val = -1
    for m in range(size):
        if dp[m] > best_val:
            best_val = dp[m]
            best_mask = m
            
    selected_items = dp_sol[best_mask]
    
    # Construct result
    result = [1]
    
    # Mark primes used in merges
    used_primes = set()
    merge_values = []
    
    for i in selected_items:
        m_mask, m_gain = items[i]
        # Find the original number C for this mask
        # We need to map mask back to C.
        # We stored best_gain_for_mask. We need the C value.
        # Let's create a map mask -> C
        
    # We need to recover C from mask.
    # mask -> set of primes -> product? No, C is specific.
    # Let's store mask -> C in best_gain_for_mask creation.
    
    # Re-do step 4 with C storage
    best_C_for_mask = {}
    for S, gain in merges:
        mask = 0
        for p in S:
            mask |= (1 << prime_to_idx[p])
        if mask not in best_C_for_mask or gain > best_gain_for_mask[mask]:
            best_C_for_mask[mask] = C
            best_gain_for_mask[mask] = gain
            
    # Now recover
    for i in selected_items:
        m_mask, _ = items[i]
        c_val = best_C_for_mask[m_mask]
        merge_values.append(c_val)
        # Mark primes in m_mask as used
        for p in active_list:
            if m_mask & (1 << prime_to_idx[p]):
                used_primes.add(p)
                
    # Add V_p for primes not in used_primes
    for p in primes:
        if p not in used_primes:
            result.append(V_p[p])
            
    result.extend(merge_values)
    
    return result

# EVOLVE-BLOCK-END
