import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N_LIMIT = 10000
    
    # 1. Sieve of Eratosthenes
    is_prime = [True] * (N_LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N_LIMIT) + 1):
        if is_prime[i]:
            for j in range(i*i, N_LIMIT + 1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    
    if n < 2:
        return primes

    best_len = 0
    best_seq = []
    
    # Precompute next primes for each prime?
    # next_primes[i] = list of indices j > i
    # This is O(N^2) memory, ~1.5M ints. ~12MB. OK.
    # But we can just iterate j from i+1 to n.
    
    # Greedy initialization
    # Try to find a good initial solution to prune DFS
    # Greedy: Start with each prime, pick smallest valid next prime
    for start_idx in range(n):
        curr_idx = start_idx
        curr_gap = 0
        seq = [primes[start_idx]]
        while True:
            next_idx = -1
            for j in range(curr_idx + 1, n):
                gap = primes[j] - primes[curr_idx]
                if gap >= curr_gap:
                    next_idx = j
                    break
            if next_idx == -1:
                break
            seq.append(primes[next_idx])
            curr_gap = primes[next_idx] - primes[curr_idx]
            curr_idx = next_idx
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            
    # DFS with pruning
    # State: (curr_idx, prev_gap, depth, path)
    # We use a global best_len and best_seq
    
    # To speed up DFS, we can precompute for each i, the list of j > i
    # sorted by gap? No, gap depends on i and j.
    
    def dfs(curr_idx, prev_gap, depth, path):
        nonlocal best_len, best_seq
        if time.time() > deadline:
            return
        
        if depth > best_len:
            best_len = depth
            best_seq = path[:]
            
        # Pruning
        # Max possible additional steps
        # Next gap >= prev_gap
        # Min next prime = primes[curr_idx] + prev_gap
        # Max steps = (10000 - primes[curr_idx]) // prev_gap
        if prev_gap == 0:
             # First step, gap can be anything
             # But we are in dfs, so we have a prev_gap from the call
             # Actually, the first call has prev_gap=0?
             # Let's handle start separately or pass a flag.
             pass
        else:
             max_add = (N_LIMIT - primes[curr_idx]) // prev_gap
             if depth + max_add <= best_len:
                 return
        
        # Try next primes
        # We want to try primes that allow the longest continuation
        # Smaller gaps are generally better
        # Iterate j from curr_idx + 1 to n
        # Optimization: If primes[j] - primes[curr_idx] > N_LIMIT - primes[curr_idx] (impossible)
        
        # To avoid checking all j, we can break if gap is too large?
        # No, large gap means fewer steps, but we might still need to check if it leads to a valid solution
        # But if gap is large, max_add will be small, so pruning will kick in next level.
        
        for j in range(curr_idx + 1, n):
            gap = primes[j] - primes[curr_idx]
            if gap < prev_gap:
                continue
                
            # Pruning before recursing
            # If we take this step, new depth is depth + 1
            # New max_add = (10000 - primes[j]) // gap
            if depth + 1 + (N_LIMIT - primes[j]) // gap <= best_len:
                # Since gap increases or stays same, and primes[j] increases,
                # future gaps will be >= gap, so this bound is valid for this branch.
                # But wait, we might pick a larger gap later? No, gaps are non-decreasing.
                # So the *minimum* gap for future steps is `gap`.
                # So `max_add` calculated with `gap` is a valid upper bound for the *entire* remaining sequence.
                # Thus, if depth + 1 + max_add <= best_len, we can prune this branch.
                # And since j increases, primes[j] increases, gap increases.
                # So (10000 - primes[j]) // gap will decrease.
                # So we can break?
                # No, gap might increase slower than primes[j]?
                # primes[j] is roughly j * 10. gap is roughly constant or increasing.
                # (10000 - p) / g. As p increases, numerator decreases.
                # So the bound gets tighter.
                # If the bound fails for j, it might fail for j+1.
                # But gap for j+1 might be larger, making denominator larger.
                # So yes, we can likely break or continue.
                # Let's just continue to be safe, or break if we are confident.
                # Actually, if we break, we might miss a solution where a larger gap allows a different path?
                # No, larger gap reduces the number of steps.
                # So if we can't beat best_len with gap `g`, we can't beat it with gap `g' > g` starting from a larger prime.
                # So we can break.
                break
                
            path.append(primes[j])
            dfs(j, gap, depth + 1, path)
            path.pop()
            
    # Start DFS from each prime
    for start_idx in range(n):
        if time.time() > deadline:
            break
        # Initial gap is 0 (or -infinity)
        # But the first gap is just primes[next] - primes[start]
        # So we call dfs with prev_gap=0?
        # If prev_gap=0, any gap is valid.
        # Pruning with prev_gap=0:
        # max_add = (10000 - p) // 0 -> Error.
        # So handle start case separately or use a large negative prev_gap?
        # Better: The first step doesn't have a "previous gap" constraint.
        # So we just iterate j.
        
        # We can optimize the start by noticing that starting with a large prime
        # leaves less room.
        # But we must check all.
        
        # To improve pruning, we can sort start indices by prime value?
        # Smaller primes allow longer sequences.
        # So try small primes first.
        
        path = [primes[start_idx]]
        # Call dfs with prev_gap = 0
        # But dfs logic needs to handle prev_gap=0.
        # In dfs, if prev_gap == 0, we don't prune based on max_add?
        # No, we can prune based on max possible steps if we assume min gap 2?
        # But min gap is not 2, it's determined by the choice.
        # So for the first step, we just recurse.
        
        # Let's modify dfs to accept a flag or handle prev_gap=0.
        # If prev_gap == 0:
        #   max_add = (10000 - primes[curr_idx]) // 2  # Loose bound
        #   if depth + max_add <= best_len: return
        
        dfs(start_idx, 0, 1, path)

    return best_seq
