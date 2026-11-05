"""New program using Exact Branch and Bound search."""
# EVOLVE-BLOCK-START
def construct():
    import sys
    sys.setrecursionlimit(5000)
    
    MAX_VAL = 511
    N = 512
    
    # 1. Precompute prime factors as bitmasks
    primes = []
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, MAX_VAL + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, MAX_VAL + 1, i):
                is_prime[j] = False
                
    prime_index = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * (MAX_VAL + 1)
    for a in range(1, MAX_VAL + 1):
        m = a
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                masks[a] |= (1 << prime_index[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            if m in prime_index:
                masks[a] |= (1 << prime_index[m])
                
    # 2. Prepare candidates sorted by weight descending
    candidates = list(range(1, MAX_VAL + 1))
    weights = [1.0 / (N - a) for a in candidates]
    
    # Sort indices by weight descending
    sorted_indices = sorted(range(MAX_VAL), key=lambda i: weights[i], reverse=True)
    sorted_vals = [candidates[i] for i in sorted_indices]
    sorted_weights = [weights[i] for i in sorted_indices]
    sorted_masks = [masks[a] for a in sorted_vals]
    
    # Precompute suffix sums of weights for upper bound pruning
    n_candidates = len(sorted_vals)
    suffix_sums = [0.0] * (n_candidates + 1)
    for i in range(n_candidates - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_weights[i]
        
    best_score = 0.0
    best_set = []
    
    def search(idx, current_mask, current_score, current_set):
        nonlocal best_score, best_set
        
        # Pruning: if current + max possible remaining <= best, stop
        if current_score + suffix_sums[idx] <= best_score:
            return
            
        if idx == n_candidates:
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
            return
            
        # Option 1: Skip current candidate
        search(idx + 1, current_mask, current_score, current_set)
        
        # Option 2: Take current candidate if coprime
        val = sorted_vals[idx]
        mask = sorted_masks[idx]
        weight = sorted_weights[idx]
        
        if (current_mask & mask) == 0:
            current_set.append(val)
            search(idx + 1, current_mask | mask, current_score + weight, current_set)
            current_set.pop()
            
    search(0, 0, 0.0, [])
    
    return best_set
# EVOLVE-BLOCK-END
