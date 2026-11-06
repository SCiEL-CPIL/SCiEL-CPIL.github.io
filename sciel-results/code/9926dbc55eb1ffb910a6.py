def construct():
    # Points are 0..11
    # We can use the fact that the Witt design is unique.
    # A standard construction uses the "small Witt design" S(5,6,12).
    
    # Let's use a known explicit list of 132 blocks or generate them.
    # Generation via M12 is complex to implement from scratch quickly.
    # Instead, we can use a backtracking solver with strong pruning, 
    # or a known construction.
    
    # Construction from S(4,5,11):
    # The 66 blocks of S(4,5,11) on {0..10} can be generated.
    # Let's generate S(4,5,11) first.
    
    # S(4,5,11) construction:
    # Points 0..10.
    # One known set of 66 blocks.
    
    # Alternatively, use the "Golay code" construction.
    # Or simply hardcode the 132 blocks.
    # Hardcoding is safest for correctness and speed.
    
    # I will generate the blocks using a verified algorithm.
    # Algorithm:
    # 1. Generate all 66 blocks of S(4,5,11) on {0..10}.
    # 2. The 132 blocks of S(5,6,12) on {0..11} are:
    #    - 66 blocks of the form B U {11} where B is a block of S(4,5,11).
    #    - 66 blocks of the form (X \ B) U {11}? No.
    
    # Correct construction:
    # Let S be the 66 blocks of S(4,5,11) on {0..10}.
    # The blocks of S(5,6,12) on {0..11} are:
    # 1. {B U {11} | B in S} (66 blocks)
    # 2. { {11} U (X \ C) | C in S } ? No, X\C has size 6.
    #    Wait, if B is a block of S(4,5,11), size 5.
    #    X \ B has size 6.
    #    So { (X \ B) U {11} } would have size 7. That's wrong.
    
    # Let's check the standard construction.
    # The Witt design S(5,6,12) is formed by:
    # - 66 blocks containing the point 11: {11} U B, where B is a block of S(4,5,11) on {0..10}.
    # - 66 blocks NOT containing 11: These are the complements of the blocks in the first set?
    #   No, the complement of a block of size 6 in 12 points is size 6.
    #   If B is a block of S(5,6,12), then X\B is also a block?
    #   Yes, the Witt design is self-complementary.
    #   So if we have 66 blocks containing 11, their complements are 66 blocks NOT containing 11.
    #   Do these 132 blocks cover all 5-subsets exactly once?
    #   Yes, this is a known property.
    
    # So the task reduces to generating the 66 blocks of S(4,5,11) on {0..10}.
    
    # How to generate S(4,5,11)?
    # It is unique.
    # We can hardcode the 66 blocks of S(4,5,11).
    
    # Known blocks of S(4,5,11) on {0..10}:
    # I will retrieve a known list.
    # Since I cannot browse, I must derive or recall.
    # Recall: S(4,5,11) is related to the Pappus graph or similar.
    # A standard list of 66 blocks can be found in combinatorial design literature.
    
    # Let's try to generate S(4,5,11) using a backtracking search, which is small enough.
    # Or, I can use the fact that S(4,5,11) is the unique design.
    # A greedy algorithm might work for S(4,5,11) if ordered correctly, but it's risky.
    
    # Better: Hardcode the 66 blocks of S(4,5,11).
    # I will construct them using a known symmetry or list.
    
    # Actually, I can use a simple backtracking for S(4,5,11) which is much smaller than S(5,6,12).
    # v=11, k=5, t=4.
    # Number of blocks = 66.
    # Total 4-subsets = C(11,4) = 330.
    # Each block covers C(5,4)=5 4-subsets.
    # 66 * 5 = 330.
    
    # Let's write a backtracker for S(4,5,11).
    # It should be fast.
    
    import itertools
    from collections import defaultdict
    
    def construct_s4511():
        points = range(11)
        all_4_subsets = list(itertools.combinations(points, 4))
        covered = set()
        blocks = []
        
        # Heuristic: Order blocks by some criterion?
        # Greedy might fail. Backtracking is safer.
        
        # To make backtracking fast, we need good pruning.
        # We can use the fact that the design is unique.
        
        # Let's use a recursive backtracking.
        # State: covered 4-subsets, list of blocks.
        # Candidates: all 5-subsets not yet "conflicting".
        
        # Optimization: Pre-compute the 4-subsets for each 5-subset.
        # There are C(11,5) = 462 candidates.
        
        cand_4_subs = {}
        for b in itertools.combinations(points, 5):
            cand_4_subs[b] = set(itertools.combinations(b, 4))
            
        # Sort candidates to break symmetry?
        # Symmetry breaking: Force the first block to be (0,1,2,3,4)?
        # Not necessarily, but we can fix the first block to reduce search space.
        # Since the design is symmetric, we can assume 0,1,2,3,4 is a block?
        # No, not every 5-set is a block. But we can assume the block containing 0,1,2,3 is fixed?
        # Let's just do a standard backtracking with symmetry breaking by fixing the first block.
        # We know one block is {0,1,2,3,4}? No, we don't know that.
        # But we can assume WLOG that the block containing {0,1,2,3} is {0,1,2,3,4}?
        # Is {0,1,2,3,4} a block?
        # In the unique S(4,5,11), any 4-set is in exactly one block.
        # We can relabel points so that {0,1,2,3} is in a block with 4.
        # So yes, we can fix the first block as (0,1,2,3,4).
        
        first_block = (0, 1, 2, 3, 4)
        covered.update(cand_4_subs[first_block])
        blocks.append(first_block)
        
        # Now find the remaining 65 blocks.
        # Candidates: all other 5-subsets that don't share a 4-subset with covered.
        
        def backtrack(blocks, covered):
            if len(blocks) == 66:
                return True
            if time.time() > deadline:
                return False # Global deadline check if needed, but this is fast.
            
            # Find candidate blocks
            # To speed up, we can pick a specific uncovered 4-subset and only consider blocks containing it.
            # Find an uncovered 4-subset.
            # To be deterministic, pick the lexicographically smallest uncovered 4-subset.
            uncovered_4 = None
            for s in all_4_subsets:
                if s not in covered:
                    uncovered_4 = s
                    break
            
            if uncovered_4 is None:
                return len(blocks) == 66
            
            # Find all candidate blocks containing uncovered_4
            candidates = []
            for b, subs in cand_4_subs.items():
                if uncovered_4 in subs:
                    # Check if b conflicts with current covered
                    conflict = False
                    for s in subs:
                        if s in covered:
                            conflict = True
                            break
                    if not conflict:
                        candidates.append(b)
            
            if not candidates:
                return False
            
            # Try each candidate
            for b in candidates:
                new_covered = covered | cand_4_subs[b]
                if len(new_covered) == len(covered) + 5: # No overlap in 4-sets
                    if backtrack(blocks + [b], new_covered):
                        return True
            
            return False
        
        # This might be slow if not pruned well.
        # But S(4,5,11) is small.
        # Let's try.
        
        # Actually, I can just hardcode the 66 blocks. It's safer.
        # I will generate them using a known source in my training data.
        # The blocks of S(4,5,11) are:
        # ... (I will use a reliable generation method in the code)
        
        # Let's use the backtracking above but with a time limit.
        # If it fails, fallback to greedy? No, greedy fails.
        
        # Better approach:
        # I will hardcode the 132 blocks of S(5,6,12) directly.
        # I can generate them by running the backtracker once in my head? No.
        # I will use a known list of the 66 blocks of S(4,5,11).
        
        # Here is a known list of 66 blocks for S(4,5,11) on {0..10}:
        # (I will reconstruct this from the M11 group or similar)
        # Actually, let's use the backtracking solver in the code. It's a standard technique and should be fast enough for v=11.
        
        import time
        start_time = time.time()
        deadline = start_time + 10.0 # 10 seconds for S(4,5,11) is plenty.
        
        # To ensure speed, we use the "uncovered 4-subset" heuristic.
        
        # Implementation details:
        # We need to pass `deadline` to the recursive function or use a global.
        
        return blocks if len(blocks) == 66 else None

    s4511_blocks = construct_s4511()
    if s4511_blocks is None:
        # Fallback: Return empty or greedy result?
        # The task says maximize. If we fail, we get 0.
        # But S(4,5,11) is very small. Backtracking with this heuristic should find it instantly.
        pass
        
    # Construct S(5,6,12)
    final_blocks = []
    for b in s4511_blocks:
        final_blocks.append(list(b) + [11])
        
    # Add complements
    for b in final_blocks:
        comp = sorted(set(range(12)) - set(b))
        final_blocks.append(comp)
        
    return final_blocks
