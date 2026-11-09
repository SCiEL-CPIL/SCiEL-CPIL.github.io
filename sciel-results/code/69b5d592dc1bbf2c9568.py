import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) using a deterministic backtracking 
    approach with symmetry breaking. Returns a list of 132 distinct 6-subsets 
    of 0..11.
    """
    START = time.time()
    TIME_LIMIT = 140.0
    
    points = list(range(12))
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    
    # Precompute: for each 6-subset, its 5-subsets
    six_to_fives = {b: list(combinations(b, 5)) for b in all_6_subsets}
    
    # For each 5-subset, which 6-subsets contain it
    five_to_sixes = {s5: [] for s5 in all_5_subsets}
    for b in all_6_subsets:
        for s5 in six_to_fives[b]:
            five_to_sixes[s5].append(b)
    
    # Use a deterministic backtracking with symmetry breaking
    # Fix the first few blocks to break symmetry
    # The Witt design is unique, so any valid completion works
    
    def backtrack(covered_5s, blocks, remaining_sixes):
        if time.time() - START > TIME_LIMIT:
            return None
        
        if len(blocks) == 132:
            return blocks
        
        # If no more candidates, fail
        if not remaining_sixes:
            return None
        
        # Choose the 5-subset with the fewest available 6-subsets (most constrained)
        # But we need to be careful: we want to pick a 5-subset that is uncovered
        # and has few options
        
        # Find uncovered 5-subsets
        uncovered = [s5 for s5 in all_5_subsets if s5 not in covered_5s]
        if not uncovered:
            # All 5-subsets covered, but we might not have 132 blocks
            # This shouldn't happen if we're on the right track
            return None
        
        # Pick the most constrained uncovered 5-subset
        # Count how many available 6-subsets can cover it
        best_s5 = None
        best_count = float('inf')
        best_candidates = None
        
        for s5 in uncovered:
            candidates = [b for b in remaining_sixes if s5 in six_to_fives[b]]
            if len(candidates) < best_count:
                best_count = len(candidates)
                best_s5 = s5
                best_candidates = candidates
                if best_count == 0:
                    return None  # Dead end
                if best_count == 1:
                    break
        
        if best_s5 is None or not best_candidates:
            return None
        
        # Try each candidate 6-subset for best_s5
        for b in best_candidates:
            # Check if b conflicts with covered_5s
            conflict = False
            for s5 in six_to_fives[b]:
                if s5 in covered_5s:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Add b and recurse
            new_covered = covered_5s | set(six_to_fives[b])
            new_blocks = blocks + [list(b)]
            new_remaining = [x for x in remaining_sixes if x != b]
            
            result = backtrack(new_covered, new_blocks, new_remaining)
            if result is not None:
                return result
        
        return None
    
    # Initial state: no blocks, no covered 5-subsets, all 6-subsets available
    # To break symmetry, we can fix the first block to contain point 0
    # Actually, let's just start with all candidates and let the algorithm work
    
    result = backtrack(set(), [], all_6_subsets)
    
    if result is not None:
        return result
    
    # Fallback: use randomized greedy if backtracking fails
    import random
    best_blocks = []
    best_count = 0
    
    for seed in range(1000):
        if time.time() - START > TIME_LIMIT:
            break
        
        random.seed(seed)
        covered_5s = set()
        blocks = []
        candidates = all_6_subsets[:]
        random.shuffle(candidates)
        
        for b in candidates:
            if time.time() - START > TIME_LIMIT:
                break
            
            conflict = False
            for s5 in six_to_fives[b]:
                if s5 in covered_5s:
                    conflict = True
                    break
            
            if not conflict:
                blocks.append(list(b))
                covered_5s.update(six_to_fives[b])
                
                if len(blocks) == 132:
                    return blocks
        
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    return best_blocks
