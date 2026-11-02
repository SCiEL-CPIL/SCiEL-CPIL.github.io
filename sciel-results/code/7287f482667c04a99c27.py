import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115.0
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # block_tsubs[i] = list of t-subset indices covered by block i
    block_tsubs = [None] * n_blocks
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
            
    # Convert tsub_blocks to lists of ints for faster iteration
    # (they are already lists of ints)

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        # gain[i] = number of uncovered t-subsets in block i
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi])
            
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
                
            # Find block with max gain
            max_gain = -1
            best_bi = -1
            # Random tie-breaking
            candidates = []
            for bi in range(n_blocks):
                if gain[bi] > max_gain:
                    max_gain = gain[bi]
                    candidates = [bi]
                elif gain[bi] == max_gain and gain[bi] > 0:
                    candidates.append(bi)
                    
            if max_gain <= 0:
                break
                
            # Pick random from candidates
            bi = candidates[rng.randint(0, len(candidates) - 1)]
            selected.append(bi)
            
            # Update gains
            for ts_idx in block_tsubs[bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Decrement gain for all blocks covering this ts_idx
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1
                        
            gain[bi] = -1 # Prevent re-selection
            
        return selected

    def local_search(blocks, seed):
        rng = random.Random(seed)
        # blocks is a list of indices
        # We work with a set for O(1) lookup
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
        # Initialize cover_count
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Remove redundant blocks
            # A block is redundant if all its t-subsets are covered by other blocks
            # i.e., cover_count[ts] > 1 for all ts in block
            to_remove = []
            for bi in block_set:
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)
                    
            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue # Restart loop as structure changed significantly
            
            # Phase 2: Try swaps
            # For each block, try to replace it with another that covers its unique subsets
            # and potentially allows further reduction
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                    
                # Find unique subsets for bi
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    # Should have been caught in Phase 1, but just in case
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart
                    
                # Find candidate blocks that cover all unique_tsubs
                # Candidate must not be in block_set
                # Intersection of tsub_blocks[ts] for ts in unique_tsubs
                # Start with the one with fewest blocks
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[ts_idx])
                    
                # Remove current block from candidates
                candidates.discard(bi)
                
                if not candidates:
                    continue
                    
                # Try candidates
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bj in cand_list[:20]: # Limit trials per block
                    if bj in block_set:
                        continue
                        
                    # Check if swapping bi -> bj is valid
                    # It is valid if bj covers all unique_tsubs of bi
                    # We already ensured this by intersection.
                    # But we must ensure that removing bi doesn't uncover any other subset
                    # that is NOT covered by bj.
                    # Actually, if bj covers all unique_tsubs of bi, then:
                    # - Subsets with count > 1 remain covered (by others).
                    # - Subsets with count == 1 (unique to bi) are now covered by bj.
                    # So the swap is always valid in terms of coverage!
                    
                    # Perform swap
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                        
                    block_set.add(bj)
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    swapped = True
                    break # Restart local search loop
                    
                if swapped:
                    break # Restart while loop
                    
        return list(block_set)

    def two_block_swap_search(blocks, seed):
        """
        Try to replace two blocks (bi, bj) with a single block bk that covers
        all unique t-subsets of bi and bj. This is a 2-for-1 improvement.
        """
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            # Try pairs of blocks
            # To keep it fast, we limit the number of pairs we check per pass
            # or just iterate through shuffled blocks and pick a random partner
            n_checked = 0
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                    
                # Pick a random partner bj from the remaining blocks
                if len(block_set) < 2:
                    break
                others = [b for b in block_set if b != bi]
                if not others:
                    continue
                bj = rng.choice(others)
                
                n_checked += 1
                if n_checked > 5000: # Limit checks per pass to avoid O(N^2) explosion
                    break
                
                # Find unique t-subsets for bi and bj
                # A t-subset is unique to the pair if it is covered ONLY by bi or ONLY by bj
                # Wait, the condition for removing bi and bj is that every t-subset
                # previously covered by bi or bj is now covered by some other block OR by the new block bk.
                
                # Let U_bi = {ts | cover_count[ts] == 1 and ts in bi}
                # Let U_bj = {ts | cover_count[ts] == 1 and ts in bj}
                # If a ts is in both bi and bj, and cover_count[ts] == 2, it's not unique to either alone.
                # If we remove bi and bj, we need a single block bk such that:
                # 1. bk covers all ts in U_bi
                # 2. bk covers all ts in U_bj
                # 3. bk covers any ts that was covered by bi or bj and had count 1?
                # Actually, if a ts is in bi and cover_count[ts] == 1, it MUST be covered by bk.
                # If a ts is in bj and cover_count[ts] == 1, it MUST be covered by bk.
                # If a ts is in bi and bj, and cover_count[ts] == 2, removing both makes count 0, so it MUST be covered by bk.
                # So the set of "required" t-subsets for bk is:
                # Required = {ts in bi | cover_count[ts] <= 1} U {ts in bj | cover_count[ts] <= 1}
                # Note: if ts in bi and bj, and count==2, it's in Required.
                # If ts in bi and bj, and count==1 (impossible if both are in set and count is sum of contributions? 
                # Wait, cover_count is global. If bi and bj both contain ts, cover_count[ts] >= 2.
                # So if cover_count[ts] == 1, it cannot be in both bi and bj.
                # So Required is simply the union of unique t-subsets of bi and unique t-subsets of bj.
                
                req_bi = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                req_bj = [ts for ts in block_tsubs[bj] if cover_count[ts] == 1]
                
                # Also, any ts in intersection of bi and bj with count == 2 must be covered by bk
                # Intersection of sets of indices
                # This is getting expensive to compute intersection of t-subsets.
                # However, if a ts is in bi and bj, it's in both lists.
                # Let's just take the union of req_bi and req_bj.
                # Is that sufficient?
                # If ts is in bi and bj, cover_count[ts] >= 2.
                # If cover_count[ts] == 2, it is NOT in req_bi (count!=1) and NOT in req_bj.
                # But if we remove bi and bj, cover_count[ts] becomes 0. So we MUST cover it.
                # So we need to include ts if (ts in bi OR ts in bj) AND (cover_count[ts] <= 2 AND (ts in bi XOR ts in bj OR cover_count[ts] == 2 AND ts in bi AND ts in bj)).
                # Simpler: 
                # Required = {ts | (ts in bi OR ts in bj) AND (cover_count[ts] - (1 if ts in bi else 0) - (1 if ts in bj else 0) == 0)}
                # i.e., ts is covered ONLY by bi and/or bj.
                
                # Efficient way:
                # Start with empty set.
                # For ts in bi: if cover_count[ts] == 1, add to Required.
                # For ts in bj: if cover_count[ts] == 1, add to Required.
                # For ts in bi AND bj: if cover_count[ts] == 2, add to Required.
                
                # To find intersection of bi and bj t-subsets, we can use a set.
                # But we don't have sets of t-subset indices precomputed for each block as a set.
                # We have lists.
                # Since block size is 7, t=4, there are 35 t-subsets per block.
                # Intersection of two lists of 35 elements is fast.
                
                set_bi = set(block_tsubs[bi])
                set_bj = set(block_tsubs[bj])
                
                required = set()
                for ts in req_bi:
                    required.add(ts)
                for ts in req_bj:
                    required.add(ts)
                
                # Check intersection
                inter = set_bi & set_bj
                for ts in inter:
                    if cover_count[ts] == 2:
                        required.add(ts)
                
                if not required:
                    # Both blocks are redundant? Phase 1 should have handled this.
                    continue
                
                # Find candidate blocks bk that cover all ts in required
                # Start with the rarest ts in required
                # To avoid O(N) scan for min, we can just pick one randomly or the first one
                # Actually, we can precompute the length of tsub_blocks for each ts?
                # tsub_blocks is a list of lists. len() is O(1).
                
                # Pick ts with fewest covering blocks
                min_ts = min(required, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])
                
                for ts in required:
                    if ts == min_ts:
                        continue
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[ts])
                    
                # Remove bi and bj from candidates
                candidates.discard(bi)
                candidates.discard(bj)
                
                if not candidates:
                    continue
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bk in cand_list[:10]: # Limit trials
                    if bk in block_set:
                        continue
                        
                    # Perform swap: remove bi, bj; add bk
                    block_set.remove(bi)
                    for ts in block_tsubs[bi]:
                        cover_count[ts] -= 1
                        
                    block_set.remove(bj)
                    for ts in block_tsubs[bj]:
                        cover_count[ts] -= 1
                        
                    block_set.add(bk)
                    for ts in block_tsubs[bk]:
                        cover_count[ts] += 1
                        
                    improved = True
                    swapped = True
                    break
                    
                if swapped:
                    break # Restart while loop
                    
        return list(block_set)

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy starts
    num_trials = 50
    for i in range(num_trials):
        if time.time() > deadline - 30:
            break
            
        blocks = greedy_cover(i)
        if not blocks:
            continue
            
        # Local search (single swaps)
        blocks = local_search(blocks, i)
        
        # Two-block swap search
        if time.time() < deadline - 15:
            blocks = two_block_swap_search(blocks, i + 1000)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # If we have time, try to improve the best solution further
    if best_blocks is not None and time.time() < deadline - 10:
        best_blocks = local_search(best_blocks, 99999)
        if time.time() < deadline - 5:
            best_blocks = two_block_swap_search(best_blocks, 88888)

    # Fallback if something went wrong
    if best_blocks is None:
        best_blocks = greedy_cover(0)
        
    return [all_blocks[i] for i in best_blocks]
