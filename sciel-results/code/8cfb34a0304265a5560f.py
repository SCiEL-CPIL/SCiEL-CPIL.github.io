import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    points = range(12)
    
    # Precompute all 5-subsets and 6-subsets
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    
    # Index for 5-subsets
    five_idx = {s: i for i, s in enumerate(all_5_subsets)}
    n5 = len(all_5_subsets)  # 792
    n6 = len(all_6_subsets)  # 924
    
    # Map 6-subset to its index
    six_idx = {b: i for i, b in enumerate(all_6_subsets)}
    
    # For each 6-subset, precompute its 5-subset indices
    block_to_5idx = {}
    for b in all_6_subsets:
        block_to_5idx[b] = [five_idx[s] for s in combinations(b, 5)]
        
    # For each 5-subset index, precompute which 6-subsets contain it
    five_to_sixes = [[] for _ in range(n5)]
    for b in all_6_subsets:
        for idx in block_to_5idx[b]:
            five_to_sixes[idx].append(b)
            
    # Precompute block neighbors (blocks sharing a 5-subset)
    # We need this for conflict counting
    block_neighbors = [[] for _ in range(n6)]
    # Iterate over pairs of blocks? Too slow.
    # Iterate over 5-subsets. For each 5-subset, all 6-subsets containing it are neighbors to each other.
    for s_idx in range(n5):
        blocks_containing = five_to_sixes[s_idx]
        for i in range(len(blocks_containing)):
            for j in range(i+1, len(blocks_containing)):
                b1 = blocks_containing[i]
                b2 = blocks_containing[j]
                idx1 = six_idx[b1]
                idx2 = six_idx[b2]
                block_neighbors[idx1].append(idx2)
                block_neighbors[idx2].append(idx1)
                
    # State variables
    covered = [False] * n5
    blocks = []
    conflict_count = [0] * n6
    
    # Symmetry breaking
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = six_idx[first_block]
    first_block_5idx = block_to_5idx[first_block]
    
    blocks.append(first_block)
    for idx in first_block_5idx:
        covered[idx] = True
        
    # Initialize conflict counts for the first block
    for n_idx in block_neighbors[first_block_idx]:
        conflict_count[n_idx] += 1
    # The first block itself is "conflicted" with itself? 
    # We don't need to check if first_block is valid again.
    # But if we want consistency, conflict_count[first_block_idx] should reflect that it's chosen?
    # No, conflict_count is used to check if a *candidate* is valid.
    # A candidate is valid if conflict_count[cand] == 0.
    # The first block is already chosen, so it won't be a candidate.
    # But other blocks that share 5-subsets with it will have conflict_count > 0.
    
    # List of uncovered 5-subsets
    # We can maintain this list dynamically or just check covered[]
    # For MRV, we need to iterate uncovered 5-subsets.
    # n5 is 792, which is small.
    
    def backtrack():
        nonlocal covered, blocks, conflict_count
        
        if len(blocks) == 132:
            return True
            
        if time.time() > deadline:
            return False
            
        # MRV: Find uncovered 5-subset with fewest valid blocks
        min_count = float('inf')
        target_idx = -1
        valid_candidates_for_target = []
        
        # We can optimize by only checking uncovered 5-subsets
        # But we need to know which are uncovered.
        # covered[] is the source of truth.
        
        for s_idx in range(n5):
            if covered[s_idx]:
                continue
                
            # Count valid blocks containing s_idx
            count = 0
            valid_cands = []
            for b in five_to_sixes[s_idx]:
                b_idx = six_idx[b]
                if conflict_count[b_idx] == 0:
                    count += 1
                    valid_cands.append(b)
                    
            if count == 0:
                return False # Dead end
                
            if count < min_count:
                min_count = count
                target_idx = s_idx
                valid_candidates_for_target = valid_cands
                
                if min_count == 1:
                    break # Can't do better than 1
                    
        if target_idx == -1:
            # No uncovered 5-subsets? 
            # If len(blocks) < 132, this shouldn't happen if we haven't finished.
            # But if we covered all 792 5-subsets with < 132 blocks?
            # Max blocks is 132. If we used < 132, we must have uncovered some.
            # Wait, if we covered all 792, we are done?
            # 132 blocks * 6 = 792.
            # So if covered_count == 792, we must have 132 blocks.
            # So if len(blocks) < 132, there must be uncovered 5-subsets.
            return False
        
        # Order candidates
        # Heuristic: Try blocks that cover the most *other* uncovered 5-subsets?
        # Or just random/lexicographic?
        # With MRV, the order might matter less, but let's try to be smart.
        # Sort by number of uncovered 5-subsets in the block.
        valid_candidates_for_target.sort(key=lambda b: sum(1 for idx in block_to_5idx[b] if not covered[idx]), reverse=True)
        
        for b in valid_candidates_for_target:
            b_idx = six_idx[b]
            
            # Add block
            blocks.append(b)
            block_5idx = block_to_5idx[b]
            for idx in block_5idx:
                covered[idx] = True
                
            # Update conflict counts
            for n_idx in block_neighbors[b_idx]:
                conflict_count[n_idx] += 1
            # Also mark b itself? 
            # If we don't, b might be considered valid again in deeper levels?
            # No, b is in blocks, so it's not a candidate.
            # But if we backtrack, we undo.
            # It's safer to increment b's count too?
            # conflict_count[b_idx] += 1 ?
            # If we do, then when we check b (which we shouldn't), it's invalid.
            # But b is not in five_to_sixes for any *uncovered* 5-subset?
            # b's 5-subsets are now covered.
            # So b will not be considered as a candidate for any uncovered 5-subset.
            # So it doesn't matter.
            
            if backtrack():
                return True
                
            # Undo
            blocks.pop()
            for idx in block_5idx:
                covered[idx] = False
            for n_idx in block_neighbors[b_idx]:
                conflict_count[n_idx] -= 1
                
        return False
        
    if backtrack():
        return [list(b) for b in blocks]
    else:
        return [list(b) for b in blocks]
