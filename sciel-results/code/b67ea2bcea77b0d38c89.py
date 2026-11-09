import time
from math import gcd

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin for 120s limit
    
    N = 511
    # 1. Sieve of Eratosthenes to get primes and factor masks
    # We need prime masks for numbers 1..511
    # Primes < 512
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Precompute prime factor mask for each number 1..511
    # mask[a] is a bitmask where bit i is set if primes[i] divides a
    masks = [0] * (N + 1)
    for a in range(2, N + 1):
        m = a
        mask_val = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask_val |= (1 << prime_index[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            mask_val |= (1 << prime_index[m])
        masks[a] = mask_val
        
    # 2. Prepare candidates
    # We want to maximize sum 1/(512-a)
    # Candidates are a in 1..511
    # Sort by weight descending (a descending)
    # We can just iterate 511 down to 1
    
    # Precompute weights
    weights = [0.0] * (N + 1)
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512.0 - a)
        
    # B&B State
    best_score = 0.0
    best_set = []
    
    # To speed up bound calculation, we can precompute suffix sums of weights?
    # No, because compatibility depends on chosen set.
    # However, we can prune if the sum of ALL remaining weights (ignoring conflicts)
    # plus current score is <= best_score.
    # This is a weak bound but very fast.
    # Let's use a slightly tighter bound:
    # Sum of weights of remaining nodes that are coprime to the current mask.
    
    def solve(idx, current_mask, current_score, current_set):
        nonlocal best_score, best_set
        nonlocal start_time, TIME_LIMIT
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
            
        if idx > N:
            return
            
        # Weak Bound: Sum of all remaining weights
        # Precompute suffix sums of weights for fast O(1) bound
        # suffix_sum[i] = sum(weights[j] for j in range(i, N+1))
        # But we need to check compatibility.
        
        # Let's try a simple recursive step
        # Option 1: Include idx
        # Option 2: Exclude idx
        
        # Optimization: If current_score + sum(weights[j] for j in idx..N) <= best_score, prune
        # We need suffix sums.
        
    # Precompute suffix sums of weights (ignoring conflicts)
    # This provides a very fast upper bound.
    suffix_sums = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sums[i] = suffix_sums[i+1] + weights[i]
        
    # Stronger bound:
    # Iterate through remaining indices, sum weights of those coprime to current_mask.
    # This is O(N) per node.
    # Given N=511 and depth ~100, this is ~50k ops per level?
    # Might be too slow if branching factor is high.
    # But conflicts are high.
    
    def dfs(idx, current_mask, current_score, current_set):
        nonlocal best_score, best_set
        nonlocal start_time, TIME_LIMIT
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        # Pruning with weak bound
        if current_score + suffix_sums[idx] <= best_score:
            return
            
        if idx > N:
            return
            
        # Option 1: Include idx
        # Check compatibility
        if (current_mask & masks[idx]) == 0:
            # Include
            new_mask = current_mask | masks[idx]
            current_set.append(idx)
            dfs(idx + 1, new_mask, current_score + weights[idx], current_set)
            current_set.pop()
            
        # Option 2: Exclude idx
        # We can also prune here if even taking all remaining compatible ones doesn't help
        # But let's just recurse
        dfs(idx + 1, current_mask, current_score, current_set)
        
    # Initial call
    # Start with 0 mask, 0 score, empty set
    # We iterate idx from 511 down to 1?
    # My idx is 1..N.
    # weights are defined for a=1..511.
    # I want to process 511 first.
    # So I should map idx to a.
    # Let's create a list of a's sorted by weight desc.
    # a=511 is index 0 in this list?
    # Let's stick to a=511..1 order.
    
    # Re-index:
    # Let nodes be 0..510 corresponding to a=511..1
    # node i corresponds to a = 511 - i
    # weight[i] = 1 / (512 - (511-i)) = 1 / (1+i)
    # mask[i] = masks[511-i]
    
    nodes = list(range(511, 0, -1))
    node_weights = [1.0 / (512.0 - a) for a in nodes]
    node_masks = [masks[a] for a in nodes]
    
    # Suffix sums for this order
    n_nodes = len(nodes)
    suffix_sums = [0.0] * (n_nodes + 1)
    for i in range(n_nodes - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i+1] + node_weights[i]
        
    def dfs(idx, current_mask, current_score, current_set):
        nonlocal best_score, best_set
        nonlocal start_time, TIME_LIMIT
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if current_score + suffix_sums[idx] <= best_score:
            return
            
        if idx == n_nodes:
            return
            
        a = nodes[idx]
        
        # Option 1: Include a
        if (current_mask & node_masks[idx]) == 0:
            new_mask = current_mask | node_masks[idx]
            current_set.append(a)
            dfs(idx + 1, new_mask, current_score + node_weights[idx], current_set)
            current_set.pop()
            
        # Option 2: Exclude a
        dfs(idx + 1, current_mask, current_score, current_set)
        
    dfs(0, 0, 0.0, [])
    
    return best_set

# EVOLVE-BLOCK-END
