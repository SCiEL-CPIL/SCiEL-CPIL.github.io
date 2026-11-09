"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import permutations as _perms
from collections import Counter

# EVOLVE-BLOCK-START
def construct(n):
    """Greedy minimum-overlap superpermutation construction."""
    deadline = time.time() + 110.0
    
    if n <= 1:
        return "".join(map(str, range(1, n + 1)))
    
    # Generate all permutations
    all_perms = list(_perms(range(1, n + 1)))
    total_perms = len(all_perms)
    
    # Track which permutations have been included
    # We'll use a set of permutation indices
    included = [False] * total_perms
    
    # Start with the first permutation
    word = list(all_perms[0])
    included[0] = True
    count_included = 1
    
    # For efficiency, maintain a dict mapping suffix tuples to list of perm indices
    # that have that suffix
    # suffix_len can be 1..n-1
    
    # Build initial suffix index for all permutations
    # suffix_index[len][suffix_tuple] = list of perm indices
    suffix_index = {}
    for sl in range(1, n):
        suffix_index[sl] = {}
    
    for idx, perm in enumerate(all_perms):
        for sl in range(1, n):
            suffix = tuple(perm[-sl:])
            if suffix not in suffix_index[sl]:
                suffix_index[sl][suffix] = []
            suffix_index[sl][suffix].append(idx)
    
    # Greedy: at each step, find the permutation with maximum overlap
    # that hasn't been included yet
    while count_included < total_perms:
        if time.time() > deadline:
            break
        
        best_overlap = 0
        best_perm_idx = -1
        best_suffix_len = 0
        
        # Try all possible overlap lengths from n-1 down to 1
        for sl in range(n - 1, 0, -1):
            if sl > len(word):
                continue
            
            # Get the suffix of current word of length sl
            current_suffix = tuple(word[-sl:])
            
            # Find all permutations that have this suffix as their prefix
            # We need perm[:sl] == current_suffix
            # Use a prefix index instead
            # Let's build prefix index on the fly or precompute
            
            # Actually, let's use a different approach:
            # For each candidate permutation, check if its prefix matches word's suffix
            # But that's O(total_perms * n) per step which is too slow
            
            # Better: precompute prefix_index
            # prefix_index[sl][prefix_tuple] = list of perm indices
            
            # Let me restructure - build prefix index
            break  # Will rebuild below
        
        # This approach is getting complex. Let's use a simpler greedy:
        # At each step, try to extend by finding a permutation whose prefix
        # matches the suffix of the current word
        
        # Precompute prefix index
        if not hasattr(construct, '_prefix_index_cache'):
            construct._prefix_index_cache = {}
        
        if n not in construct._prefix_index_cache:
            prefix_index = {}
            for sl in range(1, n):
                prefix_index[sl] = {}
            for idx, perm in enumerate(all_perms):
                for sl in range(1, n):
                    prefix = tuple(perm[:sl])
                    if prefix not in prefix_index[sl]:
                        prefix_index[sl][prefix] = []
                    prefix_index[sl][prefix].append(idx)
            construct._prefix_index_cache[n] = prefix_index
        
        prefix_index = construct._prefix_index_cache[n]
        
        # Find best permutation to add
        best_sl = 0
        best_idx = -1
        
        for sl in range(n - 1, 0, -1):
            if sl > len(word):
                continue
            
            current_suffix = tuple(word[-sl:])
            
            if current_suffix in prefix_index[sl]:
                candidates = prefix_index[sl][current_suffix]
                # Find first not-yet-included candidate
                for idx in candidates:
                    if not included[idx]:
                        best_sl = sl
                        best_idx = idx
                        break
                if best_idx != -1:
                    break
        
        if best_idx == -1:
            # No overlap found, add a new permutation with no overlap
            # Find any not-included permutation
            for idx in range(total_perms):
                if not included[idx]:
                    best_idx = idx
                    break
        
        if best_idx == -1:
            break  # Shouldn't happen
        
        # Add the permutation with best_sl overlap
        perm = all_perms[best_idx]
        new_chars = list(perm[best_sl:])
        word.extend(new_chars)
        included[best_idx] = True
        count_included += 1
        
        # Periodically check time
        if count_included % 1000 == 0 and time.time() > deadline:
            break
    
    # If we didn't include all permutations, we have an invalid result
    # But let's try to fix by appending missing ones (will be longer but valid)
    if count_included < total_perms:
        missing = [all_perms[i] for i in range(total_perms) if not included[i]]
        for perm in missing:
            if time.time() > deadline:
                break
            # Try to add with maximum overlap
            best_sl = 0
            for sl in range(n - 1, 0, -1):
                if sl > len(word):
                    continue
                if tuple(word[-sl:]) == tuple(perm[:sl]):
                    best_sl = sl
                    break
            word.extend(perm[best_sl:])
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
