import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # 55s deadline for safety (limit is 60s)
    
    denoms = list(range(2, 49))
    N = len(denoms)
    fracs = [Fraction(1, d) for d in denoms]
    
    # Precompute suffix sums for pruning
    # suffix_sums[i] is the sum of fracs[i] + ... + fracs[N-1]
    suffix_sums = [Fraction(0)] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + fracs[i]
        
    # Greedy initialization to get a good lower bound
    best_sum = Fraction(0)
    best_idxs = []
    current_sum = Fraction(0)
    current_idxs = []
    for i in range(N):
        if current_sum + fracs[i] < 1:
            current_sum += fracs[i]
            current_idxs.append(i)
    best_sum = current_sum
    best_idxs = current_idxs[:]

    # DFS with Branch and Bound
    # State: index i, current sum s, list of indices
    # Pruning: if s + suffix_sums[i] <= best_sum, return
    
    # We use a nonlocal variable or a list to update bests
    # Since Python closures are tricky with assignment, we use a list wrapper or global-like structure
    # But we are inside a function, so we can use nonlocal if we nest dfs.
    
    def dfs(i, s, idxs):
        nonlocal best_sum, best_idxs
        if time.time() > deadline:
            return
        
        # Pruning: even if we take all remaining, we can't beat best_sum
        # Note: We must ensure strict inequality < 1 for the final sum.
        # The bound s + suffix_sums[i] <= best_sum is safe because 
        # any valid sum from this branch is <= s + suffix_sums[i].
        # If that max is <= best_sum, we can't improve.
        if s + suffix_sums[i] <= best_sum:
            return
            
        if i == N:
            if s > best_sum:
                best_sum = s
                best_idxs = idxs[:]
            return

        # Option 1: Include item i
        # Only if s + fracs[i] < 1
        if s + fracs[i] < 1:
            dfs(i + 1, s + fracs[i], idxs + [i])
            
        # Check time before exploring exclude branch
        if time.time() > deadline:
            return

        # Option 2: Exclude item i
        dfs(i + 1, s, idxs)

    dfs(0, Fraction(0), [])
    
    # Final check and return
    if not best_idxs:
        return [0]
        
    # Verify validity
    s_check = Fraction(0)
    for i in best_idxs:
        s_check += fracs[i]
    if s_check >= 1:
        # Fallback to greedy result if somehow invalid (should not happen)
        return current_idxs
        
    return best_idxs
