import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(11))
    all_5subsets = list(itertools.combinations(points, 5))
    
    # Map 4-subsets to blocks containing them
    # This allows us to find candidate blocks for a specific 4-set quickly
    
    # First, fix the first block
    first_block = (0, 1, 2, 3, 4)
    covered_4sets = set(itertools.combinations(first_block, 4))
    blocks_s4511 = [first_block]
    
    # Precompute all 4-subsets
    all_4sets = list(itertools.combinations(points, 4))
    
    # For each 4-set, list all 5-sets containing it
    # 5-set containing 4-set {a,b,c,d} must be {a,b,c,d,e} for e in {0..10}\{a,b,c,d}
    # There are 7 such 5-sets.
    
    # We need to select 65 more blocks.
    # We maintain a set of covered 4-sets.
    # At each step, pick the first uncovered 4-set (in lex order).
    # Try all 5-sets containing this 4-set that are compatible with current blocks.
    
    # To check compatibility, we need to ensure the new 5-set doesn't share
    # any 4-set with existing blocks.
    # Since we pick the first UNCOVERED 4-set, the new block will cover it.
    # It will also cover 4 other 4-sets. We must ensure those are also uncovered.
    
    def backtrack(current_blocks, covered):
        if time.time() > deadline:
            return False
        if len(current_blocks) == 66:
            return True
            
        # Find first uncovered 4-set
        target_4set = None
        for q in all_4sets:
            if q not in covered:
                target_4set = q
                break
                
        if target_4set is None:
            return len(current_blocks) == 66 # Should not happen if counts match
            
        # Find all 5-sets containing target_4set
        # target_4set is a tuple of 4 points
        # The 5th point can be any of the remaining 7 points
        used_points = set(target_4set)
        candidates = []
        for p in points:
            if p not in used_points:
                b = tuple(sorted(target_4set + (p,)))
                candidates.append(b)
                
        for b in candidates:
            # Check if b is compatible
            # b has 5 4-subsets. One is target_4set (uncovered).
            # The other 4 must be uncovered.
            subs = list(itertools.combinations(b, 4))
            valid = True
            for s in subs:
                if s in covered and s != target_4set:
                    valid = False
                    break
            if not valid:
                continue
                
            # Add b
            current_blocks.append(b)
            for s in subs:
                covered.add(s)
                
            if backtrack(current_blocks, covered):
                return True
                
            # Undo
            current_blocks.pop()
            for s in subs:
                covered.remove(s)
                
        return False

    if backtrack(blocks_s4511, covered_4sets):
        pass
    else:
        # If it fails, return what we have (should not happen)
        pass

    blocks_s5612 = []
    for b in blocks_s4511:
        blocks_s5612.append(b + (11,))
    for b in blocks_s4511:
        complement = tuple(p for p in range(11) if p not in b)
        blocks_s5612.append(complement)
        
    return [list(b) for b in blocks_s5612]
