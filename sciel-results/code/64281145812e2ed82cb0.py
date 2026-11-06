import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy: Use the known mathematical construction of the Witt design S(5,6,12)
    # derived from the Steiner system S(4,5,11). This is deterministic, O(1) in search time,
    # and guarantees the optimal 132 blocks if the construction is correct.
    # If verification fails, fall back to the fast backtracking search.
    
    # 1. Base blocks for S(4, 5, 11) on points 0..10
    # These are developed mod 11 to get 66 blocks.
    base_blocks = [
        (0, 1, 3, 7, 9),
        (0, 2, 3, 5, 8),
        (0, 1, 4, 6, 10),
        (0, 2, 4, 7, 10),
        (0, 1, 2, 5, 6),
        (0, 1, 5, 8, 10)
    ]
    
    # Generate all 66 blocks of S(4, 5, 11)
    s4511_blocks = []
    for base in base_blocks:
        for i in range(11):
            block = tuple(sorted((x + i) % 11 for x in base))
            s4511_blocks.append(block)
            
    # Remove duplicates just in case (should be unique)
    s4511_blocks = list(set(s4511_blocks))
    
    if len(s4511_blocks) == 66:
        # Construct S(5, 6, 12) blocks
        # Points are 0..11, where 11 is the point at infinity.
        # Type 1: {11} U B for B in S(4, 5, 11)
        # Type 2: {0..10} \ B for B in S(4, 5, 11)
        
        blocks = []
        points_0_10 = frozenset(range(11))
        
        for B in s4511_blocks:
            # Type 1
            b1 = tuple(sorted(list(B) + [11]))
            blocks.append(b1)
            
            # Type 2
            b2_set = points_0_10 - frozenset(B)
            b2 = tuple(sorted(b2_set))
            blocks.append(b2)
            
        # Verify the constructed design
        if len(blocks) == 132:
            # Check for duplicate blocks
            if len(set(blocks)) == 132:
                # Check that no 5-subset is repeated
                covered_5s = set()
                valid = True
                for b in blocks:
                    for s in combinations(b, 5):
                        if s in covered_5s:
                            valid = False
                            break
                        covered_5s.add(s)
                    if not valid:
                        break
                
                if valid and len(covered_5s) == 792:
                    return [list(b) for b in blocks]
    
    # Fallback: Fast backtracking search using precomputed structures
    points = range(12)
    
    # Precompute all 5-subsets and 6-subsets
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    
    # For each 6-subset, precompute its 5-subsets
    block_to_5s = {}
    for b in all_6_subsets:
        block_to_5s[b] = list(combinations(b, 5))
    
    # For each 5-subset, precompute which 6-subsets contain it
    five_to_sixes = {s: [] for s in all_5_subsets}
    for b in all_6_subsets:
        for s in block_to_5s[b]:
            five_to_sixes[s].append(b)
    
    covered = set()
    blocks = []
    
    # Symmetry breaking:
    # The first block must contain {0,1,2,3,4}. 
    # By symmetry, we can assume the 6th element is 5.
    # So Block 0 is {0,1,2,3,4,5}.
    
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_5s = block_to_5s[first_block]
    
    blocks.append(first_block)
    covered.update(first_block_5s)
    
    # Maintain a sorted list of uncovered 5-subsets for efficient access
    # Use a set for O(1) membership and a list for ordered iteration
    uncovered_5s = set(all_5_subsets)
    uncovered_5s.difference_update(first_block_5s)
    
    # Now we need 131 more blocks.
    # We will search for the next block.
    
    def backtrack():
        if len(blocks) == 132:
            return True
        
        if time.time() > deadline:
            return False
            
        if not uncovered_5s:
            # All covered, but we don't have 132 blocks?
            # This shouldn't happen if we stop at 132.
            return False
            
        # Find the first uncovered 5-subset in lexicographic order
        # Since all_5_subsets is in lex order, find min in uncovered_5s
        target_5 = min(uncovered_5s)
        
        # Find all 6-subsets that contain target_5
        candidates = five_to_sixes[target_5]
        
        # Filter candidates: no overlap in 5-subsets
        valid_candidates = []
        for b in candidates:
            # Check if this block is valid (no overlap in 5-subsets)
            block_5s = block_to_5s[b]
            if not any(s in covered for s in block_5s):
                valid_candidates.append(b)
                
        # Try each candidate
        for b in valid_candidates:
            # Add block
            blocks.append(b)
            block_5s = block_to_5s[b]
            covered.update(block_5s)
            uncovered_5s.difference_update(block_5s)
            
            if backtrack():
                return True
                
            # Undo
            blocks.pop()
            covered.difference_update(block_5s)
            uncovered_5s.update(block_5s)
            
        return False
        
    if backtrack():
        return [list(b) for b in blocks]
    else:
        # Fallback: Return what we have
        return [list(b) for b in blocks]
