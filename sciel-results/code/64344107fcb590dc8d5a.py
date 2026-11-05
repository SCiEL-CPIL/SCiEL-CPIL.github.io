from itertools import combinations
import sys

def construct():
    points = range(12)
    all_blocks = list(combinations(points, 6))
    n_blocks = len(all_blocks)
    
    # Pre-compute 5-sets for each block
    block_5sets = []
    for b in all_blocks:
        s5 = [frozenset(sub) for sub in combinations(b, 5)]
        block_5sets.append(s5)
        
    # Pre-compute for each 5-set, the list of blocks containing it
    # And initial cover count
    all_5sets = set()
    for s5 in block_5sets:
        all_5sets.update(s5)
        
    # cover_count[s] = number of blocks that contain s
    cover_count = {s: 0 for s in all_5sets}
    # block_indices_for_5set[s] = list of block indices containing s
    block_indices_for_5set = {s: [] for s in all_5sets}
    
    for i, s5 in enumerate(block_5sets):
        for s in s5:
            cover_count[s] += 1
            block_indices_for_5set[s].append(i)
            
    # Symmetry breaking: Fix first block (0,1,2,3,4,5)
    first_block = (0, 1, 2, 3, 4, 5)
    first_idx = all_blocks.index(first_block)
    
    used_5sets = set(block_5sets[first_idx])
    current_blocks = [list(first_block)]
    
    # Initialize available blocks
    # A block is available if it doesn't conflict with used_5sets
    # And we remove the first block itself.
    
    # We maintain a set of available block indices
    available = set(range(n_blocks))
    available.remove(first_idx)
    
    # Update cover_count and availability based on the first block selection
    # The first block is "used". 
    # Any block that shares a 5-set with the first block is NO LONGER available?
    # NO. The condition is "no 5-subset lies in two blocks".
    # So if block B shares a 5-set with block A, B is INVALID for the rest of the design.
    # So we must remove all blocks that conflict with the first block from `available`.
    
    for s in block_5sets[first_idx]:
        for b_idx in block_indices_for_5set[s]:
            if b_idx != first_idx:
                available.remove(b_idx)
                
    # Also, we need to update cover_count for the *remaining* 5-sets?
    # No, cover_count should reflect how many *available* blocks cover each 5-set.
    # Since we removed blocks, the cover counts for 5-sets covered by those removed blocks change.
    # Actually, simpler: 
    # cover_count[s] is the number of blocks in `available` that contain s.
    
    # Let's recompute cover_count based on `available`.
    # This is O(N * 6) which is fine.
    
    def recompute_cover_counts(available_set):
        counts = {s: 0 for s in all_5sets}
        for b_idx in available_set:
            for s in block_5sets[b_idx]:
                counts[s] += 1
        return counts
        
    current_cover_counts = recompute_cover_counts(available)
    
    # Check if any uncovered 5-set has 0 cover count
    # Uncovered 5-sets are those NOT in used_5sets.
    
    def is_consistent(used, cover_counts, available_set):
        for s in all_5sets:
            if s not in used:
                if cover_counts[s] == 0:
                    return False
        return True
        
    if not is_consistent(used_5sets, current_cover_counts, available):
        return [] # Should not happen
        
    def backtrack(available_set, used, cover_counts, blocks):
        if len(blocks) == 132:
            return True
            
        if not available_set:
            return False
            
        # Heuristic: Choose a 5-set that is uncovered and has the minimum cover count > 0.
        # Then, only consider blocks that contain this 5-set.
        # This is "Minimum Remaining Values" (MRV) on the 5-sets.
        
        min_count = float('inf')
        target_5set = None
        
        for s in all_5sets:
            if s not in used:
                c = cover_counts[s]
                if c == 0:
                    return False # Inconsistent
                if c < min_count:
                    min_count = c
                    target_5set = s
                    if min_count == 1:
                        break
                        
        if target_5set is None:
            # All 5-sets are used? 
            # If len(blocks) < 132, this shouldn't happen unless we have extra blocks?
            # No, if all 5-sets are used, we have covered all 792 5-sets.
            # 792 / 6 = 132 blocks. So len(blocks) should be 132.
            return len(blocks) == 132
            
        # Get candidate blocks that contain target_5set
        # These must be in available_set
        candidates = [b_idx for b_idx in block_indices_for_5set[target_5set] if b_idx in available_set]
        
        # Order candidates? 
        # Maybe by how many other 5-sets they cover that are "tight"?
        # For now, just iterate.
        
        for b_idx in candidates:
            # Select b_idx
            new_used = set(used)
            new_used.update(block_5sets[b_idx])
            
            new_blocks = blocks + [list(all_blocks[b_idx])]
            
            # Update available set
            # Remove b_idx
            # Remove any block that conflicts with b_idx
            # A block conflicts if it shares ANY 5-set with b_idx.
            
            new_available = set(available_set)
            new_available.remove(b_idx)
            
            # Remove conflicting blocks
            for s in block_5sets[b_idx]:
                for conflict_idx in block_indices_for_5set[s]:
                    if conflict_idx in new_available:
                        new_available.remove(conflict_idx)
                        
            # Update cover counts
            # We can recompute, or update incrementally.
            # Recomputing is O(924 * 6) ~ 5000 ops. Fast enough.
            new_cover_counts = recompute_cover_counts(new_available)
            
            # Check consistency
            if is_consistent(new_used, new_cover_counts, new_available):
                if backtrack(new_available, new_used, new_cover_counts, new_blocks):
                    return True
                    
        return False
        
    if backtrack(available, used_5sets, current_cover_counts, current_blocks):
        return current_blocks
    else:
        return []
