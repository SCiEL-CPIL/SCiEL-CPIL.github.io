import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) using an optimized Algorithm X (Exact Cover) 
    with bitmask operations.
    """
    START = time.time()
    TIME_LIMIT = 110.0
    
    points = list(range(12))
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    n5 = len(all_5_subsets)  # 792
    n6 = len(all_6_subsets)  # 924
    
    # Precompute masks for 6-subsets
    # Map 5-subset to index
    five_index = {s5: i for i, s5 in enumerate(all_5_subsets)}
    
    # six_to_mask[bi] is a bitmask of 5-subsets covered by all_6_subsets[bi]
    six_to_mask = [0] * n6
    # five_to_sixes[fi] is a list of 6-subset indices covering 5-subset fi
    five_to_sixes = [[] for _ in range(n5)]
    
    for bi, b in enumerate(all_6_subsets):
        mask = 0
        for s5 in combinations(b, 5):
            fi = five_index[s5]
            mask |= (1 << fi)
            five_to_sixes[fi].append(bi)
        six_to_mask[bi] = mask

    # Algorithm X with bitmasks
    # State: covered_mask (int), used_blocks (set of indices)
    # We want to find 132 blocks.
    
    def backtrack(covered_mask, used_blocks, depth):
        if time.time() - START > TIME_LIMIT:
            return None
        
        if depth == 132:
            return list(used_blocks)
        
        # Heuristic: Choose the uncovered 5-subset with the fewest valid 6-subsets
        # A 5-subset is uncovered if (covered_mask & (1 << fi)) == 0
        # A 6-subset is valid if (six_to_mask[bi] & covered_mask) == 0
        
        best_fi = -1
        best_count = n6 + 1
        best_candidates = []
        
        for fi in range(n5):
            if (covered_mask >> fi) & 1:
                continue # Already covered
                
            count = 0
            candidates = []
            for bi in five_to_sixes[fi]:
                if bi in used_blocks:
                    continue
                # Check if this 6-subset conflicts with currently covered 5-subsets
                if six_to_mask[bi] & covered_mask:
                    continue
                count += 1
                candidates.append(bi)
            
            if count < best_count:
                best_count = count
                best_fi = fi
                best_candidates = candidates
                
                if count == 0:
                    return None
                if count == 1:
                    break # Can't get better than 1
                    
        if best_fi == -1:
            # All 5-subsets covered?
            # If we haven't reached 132 blocks but all 5-subsets are covered, 
            # it's impossible to add more blocks without conflict, 
            # but S(5,6,12) covers exactly all 5-subsets with 132 blocks.
            # So if all are covered, we must have 132 blocks.
            if depth == 132:
                return list(used_blocks)
            return None
            
        if not best_candidates:
            return None
            
        for bi in best_candidates:
            new_covered = covered_mask | six_to_mask[bi]
            new_used = used_blocks | {bi}
            result = backtrack(new_covered, new_used, depth + 1)
            if result is not None:
                return result
                
        return None

    # Symmetry breaking: 
    # The automorphism group of S(5,6,12) is M12.
    # We can fix the first block to be {0, 1, 2, 3, 4, 5}.
    # This reduces the search space significantly.
    # Find the index of {0, 1, 2, 3, 4, 5}
    first_block = tuple(sorted([0, 1, 2, 3, 4, 5]))
    first_bi = all_6_subsets.index(first_block)
    
    initial_covered = six_to_mask[first_bi]
    initial_used = {first_bi}
    
    result_indices = backtrack(initial_covered, initial_used, 1)
    
    if result_indices is not None:
        return [list(all_6_subsets[bi]) for bi in result_indices]
        
    # Fallback: If symmetry breaking failed or timeout, try without or greedy
    # But it should work.
    return []
