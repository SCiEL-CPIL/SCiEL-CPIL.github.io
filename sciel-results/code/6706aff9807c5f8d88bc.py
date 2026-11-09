"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    deadline = time.time() + 50  # 50 second safety margin
    
    # Generate all 2^n subsets as frozensets
    all_subsets = []
    for mask in range(1 << n):
        s = frozenset(b for b in range(n) if (mask >> b) & 1)
        all_subsets.append(s)
    
    # Precompute subset relationships
    # subset_of[i][j] = True if all_subsets[i] is subset of all_subsets[j]
    # We'll use bit masks for efficiency
    # For each pair, check if A ⊆ B via bitmask
    
    # Greedy construction: start with a good seed, then iteratively add best set
    # Strategy: build a family that maximizes comparable pairs
    
    # First, let's try a structured approach:
    # Select sets that form a dense partial order
    
    # Approach: Use a greedy algorithm
    # Start with empty set (comparable with everything)
    # Then greedily add the set that maximizes marginal gain
    
    # Precompute for each pair of masks whether one is subset of other
    # mask_a is subset of mask_b iff (mask_a & mask_b) == mask_a
    
    # For efficiency, precompute subset counts
    # For a given mask m, count how many existing sets it's comparable with
    
    selected = [0]  # empty set
    selected_set = {0}
    
    # Precompute: for each mask, its size
    sizes = [bin(mask).count('1') for mask in range(1 << n)]
    
    # Greedy: at each step, pick the mask that maximizes new comparable pairs
    # Comparable pairs added = number of existing sets that are subsets of new set
    #   + number of existing sets that are supersets of new set
    
    # For efficiency, we can track for each existing set, which new sets would be comparable
    
    # Let's use a different approach: precompute comparability matrix partially
    
    # For each candidate mask, compute marginal gain
    # marginal_gain(mask) = #{s in selected : s ⊆ mask or mask ⊆ s}
    
    # This is O(m * 2^n * n) per iteration which is 40 * 4096 * 12 ≈ 2M ops per iteration
    # 40 iterations = 80M ops, should be fast enough
    
    # But let's optimize: precompute for each mask, the list of masks that are subsets/supersets
    
    # Actually, let's just do it directly with bitmask operations
    
    for _ in range(m - 1):
        if time.time() > deadline:
            break
        
        best_mask = None
        best_gain = -1
        
        # Try all masks not yet selected
        for mask in range(1 << n):
            if mask in selected_set:
                continue
            
            # Count comparable pairs with existing selected sets
            gain = 0
            for s in selected:
                # Check if s ⊆ mask or mask ⊆ s
                if (s & mask) == s:  # s ⊆ mask
                    gain += 1
                elif (s & mask) == mask:  # mask ⊆ s
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        
        if best_mask is not None:
            selected.append(best_mask)
            selected_set.add(best_mask)
    
    # Convert to list of lists
    result = []
    for mask in selected:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result

# EVOLVE-BLOCK-END
