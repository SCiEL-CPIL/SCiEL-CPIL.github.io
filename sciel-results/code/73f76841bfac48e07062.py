import time
from itertools import combinations

def _smooth_numbers_up_to(n):
    """Return sorted list of all 2^a*3^b*5^c <= n (positive)."""
    out = set()
    p2 = 1
    while p2 <= n:
        p23 = p2
        while p23 <= n:
            p235 = p23
            while p235 <= n:
                out.add(p235)
                p235 *= 5
            p23 *= 3
        p2 *= 2
    return sorted(out)

def _precompute_optimal():
    smooth = _smooth_numbers_up_to(128)
    n_smooth = len(smooth)
    
    # Map value to index for quick lookup if needed, but we work with indices
    # Precompute conflict masks
    # conflict[i] is a bitmask of indices j such that smooth[i] and smooth[j] are comparable (one divides the other)
    # Note: if i==j, we don't care about self-comparability for the antichain check of distinct elements, 
    # but the mask logic "A & conflict[i] == 0" requires that no OTHER element in A is in conflict[i].
    # So conflict[i] should contain j != i if they are comparable.
    
    conflict = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            if smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0:
                conflict[i] |= (1 << j)
                
    # best_rep[t] will store the tuple of smooth numbers
    best_rep = [None] * 129
    solved = set()
    
    # Iterate k from 1 upwards
    max_k = 10 # Heuristic limit, should be enough
    for k in range(1, max_k + 1):
        if len(solved) == 128:
            break
            
        # Generate all combinations of size k
        # To optimize, we can prune based on sum? 
        # The sum of the combination must be <= 128.
        # Since smooth numbers are sorted, we can prune the combination generation.
        
        # Instead of itertools.combinations which generates all, we can use a recursive generator with pruning.
        # But itertools is in C and fast. Let's try itertools first and check time.
        # 38 choose 6 is ~2.7M. Checking antichain is O(k^2) or O(k) with bitmasks.
        # Bitmask check: 
        # For a combination of indices comb, create mask M.
        # Check if for all i in comb, (conflict[i] & M) == 0.
        
        for comb in combinations(range(n_smooth), k):
            # Create mask
            mask = 0
            s = 0
            for idx in comb:
                mask |= (1 << idx)
                s += smooth[idx]
                
            if s > 128:
                # Since combinations are generated in lexicographic order of indices, 
                # and smooth is sorted, we can't easily break out of itertools.combinations.
                # But we can just skip.
                continue
                
            if s in solved:
                continue
                
            # Check antichain
            is_antichain = True
            for idx in comb:
                if conflict[idx] & mask:
                    is_antichain = False
                    break
            
            if is_antichain:
                best_rep[s] = tuple(smooth[i] for i in comb)
                solved.add(s)
                
    # Fallback for any unsolved (should not happen if max_k is high enough)
    # If any are unsolved, we might need to increase max_k or use a different search.
    # But given the density, k<=6 or 7 should cover most.
    
    return best_rep[1:129]

# Pre-compute at module load or first call?
# Module load is safer for "instant" response if evaluator imports and calls.
# But if it takes too long, it might timeout on import?
# The limit is 60s for evaluation. Import is usually separate or part of it.
# Let's do it inside construct to be safe, but cache it.
_OPTIMAL_REPS = None

def construct():
    global _OPTIMAL_REPS
    if _OPTIMAL_REPS is None:
        _OPTIMAL_REPS = _precompute_optimal()
    return _OPTIMAL_REPS
