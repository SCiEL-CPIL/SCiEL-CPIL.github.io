import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precompute primes and masks
    N = 511
    # Sieve of Eratosthenes to find primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Calculate masks and weights for each number
    masks = [0] * (N + 1)
    weights = [0.0] * (N + 1)
    
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512 - a)
        m = 1  # Start with bit 0 set? No, use 0-based indexing for primes.
        # Actually, let's just use a Python int as a bitmask.
        mask = 0
        x = a
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            # x is a prime
            mask |= (1 << prime_to_idx[x])
        masks[a] = mask
    
    # 2. Greedy Initialization
    # Sort indices by weight descending
    indices = list(range(1, N + 1))
    indices.sort(key=lambda x: weights[x], reverse=True)
    
    chosen = []
    chosen_mask = 0
    for a in indices:
        if masks[a] & chosen_mask == 0:
            chosen.append(a)
            chosen_mask |= masks[a]
    
    # 3. Local Search
    # We will try to improve the set by swapping elements.
    # To make it efficient, we maintain a list of chosen indices and their masks.
    
    def get_score():
        return sum(weights[a] for a in chosen)
    
    best_score = get_score()
    
    # Helper to check if a list of candidates is compatible with a base mask
    def is_compatible(candidates, base_mask):
        # candidates: list of indices
        # Check pairwise coprime among candidates
        # Check coprime to base_mask
        for i in range(len(candidates)):
            mi = masks[candidates[i]]
            if mi & base_mask:
                return False
            for j in range(i + 1, len(candidates)):
                mj = masks[candidates[j]]
                if mi & mj:
                    return False
        return True

    # Precompute list of all indices not in chosen?
    # It's easier to just iterate all 1..511 and check if in chosen set.
    # But checking "in chosen" is O(K).
    # Let's use a set for chosen for O(1) lookup.
    
    chosen_set = set(chosen)
    
    # We will iterate a few times until no improvement
    # Or until time limit
    
    # To speed up, we can pre-sort indices by weight.
    # For 1-for-1 swap:
    # For each a in chosen:
    #   base_mask = chosen_mask ^ masks[a]
    #   For each b in indices:
    #     if b in chosen_set: continue
    #     if masks[b] & base_mask: continue
    #     if weights[b] > weights[a]:
    #        swap
    
    # For 1-for-2 swap:
    # For each a in chosen:
    #   base_mask = chosen_mask ^ masks[a]
    #   For each b in indices:
    #     if b in chosen_set: continue
    #     if masks[b] & base_mask: continue
    #     For each c in indices:
    #       if c in chosen_set or c == b: continue
    #       if masks[c] & base_mask: continue
    #       if masks[b] & masks[c]: continue
    #       if weights[b] + weights[c] > weights[a]:
    #          swap
    
    # Optimization:
    # We can limit the search for b and c to those with high weights.
    # Since weights decrease, we can break early if weights are too low.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1-for-1 swaps
        for a in list(chosen):
            base_mask = chosen_mask ^ masks[a]
            w_a = weights[a]
            # We only care about b with weight > w_a
            # Since indices is sorted by weight desc, we can break when weight <= w_a
            for b in indices:
                if weights[b] <= w_a:
                    break
                if b in chosen_set:
                    continue
                if masks[b] & base_mask:
                    continue
                
                # Swap a with b
                chosen.remove(a)
                chosen_set.remove(a)
                chosen.append(b)
                chosen_set.add(b)
                chosen_mask = base_mask | masks[b]
                improved = True
                break # Restart loop? Or just mark and break inner?
                # Breaking inner loop means we try next a.
                # But the set changed.
                # It's safer to restart the whole while loop if we make a swap.
                break
            if improved:
                break
        
        if improved:
            continue
            
        # 1-for-2 swaps
        for a in list(chosen):
            base_mask = chosen_mask ^ masks[a]
            w_a = weights[a]
            
            # We need w_b + w_c > w_a
            # Iterate b
            for i, b in enumerate(indices):
                if weights[b] <= w_a: # Even if w_c is max, sum won't exceed if w_b is small?
                    # Actually w_c <= w_b. So 2*w_b > w_a is a necessary condition for any pair starting with b?
                    # Not necessarily, but if w_b is very small, w_c is smaller.
                    # Let's just use the sum check inside.
                    pass
                
                if b in chosen_set:
                    continue
                if masks[b] & base_mask:
                    continue
                
                # Iterate c
                # c must be coprime to b and base_mask
                # And w_b + w_c > w_a => w_c > w_a - w_b
                for c in indices:
                    if weights[c] < w_a - weights[b]:
                        break
                    if c in chosen_set or c == b:
                        continue
                    if masks[c] & base_mask:
                        continue
                    if masks[b] & masks[c]:
                        continue
                    
                    # Swap a with b, c
                    chosen.remove(a)
                    chosen_set.remove(a)
                    chosen.append(b)
                    chosen.append(c)
                    chosen_set.add(b)
                    chosen_set.add(c)
                    chosen_mask = base_mask | masks[b] | masks[c]
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
        
    return chosen

# EVOLVE-BLOCK-END
