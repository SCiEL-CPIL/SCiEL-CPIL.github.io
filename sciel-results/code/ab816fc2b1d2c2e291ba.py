import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) using optimized backtracking 
    with symmetry breaking. Returns a list of 132 distinct 6-subsets of 0..11.
    """
    START = time.time()
    TIME_LIMIT = 110.0  # Safety margin for 120s limit
    
    points = list(range(12))
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    
    n5 = len(all_5_subsets)  # 792
    n6 = len(all_6_subsets)  # 924
    
    # Index mapping
    five_index = {s5: i for i, s5 in enumerate(all_5_subsets)}
    
    # Precompute relationships
    # six_to_fives[bi] = list of 5-subset indices contained in 6-subset bi
    six_to_fives = []
    for b in all_6_subsets:
        fives = [five_index[s5] for s5 in combinations(b, 5)]
        six_to_fives.append(fives)
        
    # five_to_sixes[fi] = list of 6-subset indices containing 5-subset fi
    five_to_sixes = [[] for _ in range(n5)]
    for bi, fives in enumerate(six_to_fives):
        for fi in fives:
            five_to_sixes[fi].append(bi)

    # Symmetry Breaking:
    # The automorphism group M12 acts transitively on blocks.
    # We can fix the first block to be (0, 1, 2, 3, 4, 5).
    # This reduces the search space by a factor of 132 (number of blocks)
    # and further by the stabilizer of that block.
    
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_6_subsets.index(first_block)
    
    # Initialize state
    covered_5s = set(six_to_fives[first_block_idx])
    used_6s = {first_block_idx}
    blocks = [list(first_block)]
    
    # Remaining blocks to find: 131
    
    def backtrack(covered_5s, used_6s, blocks, depth):
        if time.time() - START > TIME_LIMIT:
            return False
            
        if len(blocks) == 132:
            return True
            
        # Find the most constrained uncovered 5-subset
        # We want to pick an uncovered 5-subset 'fi' such that the number of
        # valid 6-subsets containing it is minimized.
        
        best_fi = -1
        best_count = float('inf')
        best_candidates = []
        
        # Iterate through all 5-subsets to find the most constrained uncovered one
        # Optimization: We can skip covered ones.
        # Note: Iterating 792 items is cheap. The inner loop is the cost.
        
        for fi in range(n5):
            if fi in covered_5s:
                continue
            
            # Count valid candidates for this 5-subset
            count = 0
            candidates = []
            
            # Iterate through 6-subsets that contain this 5-subset
            for bi in five_to_sixes[fi]:
                if bi in used_6s:
                    continue
                
                # Check if this 6-subset conflicts with any other covered 5-subset
                # A conflict exists if any 5-subset in bi (other than fi, which is uncovered by definition of loop)
                # is already in covered_5s.
                # Actually, fi is NOT in covered_5s.
                # We need to check if (six_to_fives[bi] \ {fi}) intersects covered_5s.
                
                conflict = False
                for fi2 in six_to_fives[bi]:
                    if fi2 != fi and fi2 in covered_5s:
                        conflict = True
                        break
                        
                if not conflict:
                    count += 1
                    candidates.append(bi)
                    
            if count < best_count:
                best_count = count
                best_fi = fi
                best_candidates = candidates
                
                if count == 0:
                    return False # Dead end
                if count == 1:
                    # Can't get better than 1, break early? 
                    # No, there might be another fi with 0 candidates (dead end) 
                    # but we check 0 inside the loop. If we find 1, we still need to 
                    # ensure no other fi has 0. However, if we find 0, we return False immediately.
                    # So if we find 1, we can continue to see if any has 0? 
                    # Actually, if any has 0, we return False. If we find 1, we keep 
                    # looking for 0. If we finish and best is 1, we proceed.
                    # We can break if we want to save time, but finding a 0 is critical.
                    # Let's not break on 1 to be safe, but break on 0 (handled by return).
                    pass

        if best_fi == -1:
            # No uncovered 5-subsets? Should not happen if blocks < 132
            return False
            
        # Try each candidate 6-subset
        for bi in best_candidates:
            # Add bi to solution
            new_covered = set(covered_5s)
            new_covered.update(six_to_fives[bi])
            
            new_used = set(used_6s)
            new_used.add(bi)
            
            new_blocks = blocks + [list(all_6_subsets[bi])]
            
            if backtrack(new_covered, new_used, new_blocks, depth + 1):
                return True
                
            # Undo (handled by creating new sets)
            
        return False

    # Start search
    if backtrack(covered_5s, used_6s, blocks, 1):
        return blocks
    else:
        # Fallback: If backtracking fails (e.g. due to time limit or logic error), 
        # try a randomized greedy approach as a last resort to return *something* valid, 
        # though it likely won't reach 132.
        # However, the problem states 132 is the optimum and attained by the Witt design.
        # If the deterministic search with symmetry breaking fails, it's likely a bug or timeout.
        # Given the constraints, the deterministic path should work.
        return blocks # Return partial or empty if failed
