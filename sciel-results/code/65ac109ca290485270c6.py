import itertools
import random
import time
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback for non-target parameters
    if v != 19 or k != 7 or t != 4:
        # Simple greedy fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            best_block = None
            best_count = -1
            # This is very slow for large v, but only used if not 19,7,4
            for block in itertools.combinations(range(v), k):
                count = 0
                for tsub in itertools.combinations(block, t):
                    if tsub in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = block
            if best_block is None: break
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
            if time.time() > deadline:
                # Try to finish quickly or return invalid? Must return valid.
                # This fallback is risky. But task is 19,7,4.
                pass
        return blocks

    # --- Optimized Path for v=19, k=7, t=4 ---
    
    # 1. Precomputation
    points = range(v)
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_blocks = []
    block_tsubs = [] # list of lists of tsub ids
    tsub_to_blocks = [[] for _ in range(num_tsubs)] # inverse index
    
    for block_tuple in itertools.combinations(points, k):
        block_id = len(all_blocks)
        all_blocks.append(block_tuple)
        tsubs_in_block = []
        for tsub in itertools.combinations(block_tuple, t):
            tid = tsub_to_id[tsub]
            tsubs_in_block.append(tid)
            tsub_to_blocks[tid].append(block_id)
        block_tsubs.append(tsubs_in_block)
        
    num_blocks_total = len(all_blocks)
    
    # 2. Greedy Initialization
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    current_blocks = []
    in_cover = np.zeros(num_blocks_total, dtype=bool)
    
    def is_redundant(b_id):
        for tid in block_tsubs[b_id]:
            if cover_count[tid] <= 1:
                return False
        return True

    # Greedy Loop
    while True:
        if time.time() > deadline:
            break
        
        # Find block with max gain
        best_block = -1
        best_gain = -1
        
        # Optimization: Only check blocks not in cover?
        # Actually, adding a block already in cover is useless (gain 0 for uncovered).
        # So we can skip blocks in cover.
        
        # To speed up, we can precompute gains?
        # Gain of block b = sum(1 for tid in block_tsubs[b] if cover_count[tid] == 0)
        
        # Simple scan is ~1.7M ops.
        # Let's try to be smarter.
        # We can maintain a list of "active" blocks?
        # For now, simple scan with NumPy vectorization might be faster if we build a matrix.
        # But matrix is 50k x 3876. Sparse.
        # Let's stick to Python loop but optimized.
        
        # Actually, we can just iterate all blocks.
        for b_id in range(num_blocks_total):
            if in_cover[b_id]:
                continue
            gain = 0
            for tid in block_tsubs[b_id]:
                if cover_count[tid] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_id
                
        if best_block == -1 or best_gain == 0:
            break
            
        current_blocks.append(best_block)
        in_cover[best_block] = True
        for tid in block_tsubs[best_block]:
            cover_count[tid] += 1
            
        # Check if all covered
        if np.min(cover_count) > 0:
            break
            
    # 3. Local Search: Redundancy Removal + Swaps
    # We will iterate until no improvement or time limit.
    # Improvement: reduction in number of blocks.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Remove Redundant Blocks
        # Iterate backwards to safely remove
        for i in range(len(current_blocks) - 1, -1, -1):
            b_id = current_blocks[i]
            if is_redundant(b_id):
                # Remove
                for tid in block_tsubs[b_id]:
                    cover_count[tid] -= 1
                in_cover[b_id] = False
                current_blocks.pop(i)
                improved = True
                
        # Pass 2: Try Swaps
        # If no removal, try to swap blocks to enable future removals or just find a better structure.
        # A swap is valid if it maintains coverage.
        # Strategy: For each block b in current_blocks:
        #   1. Remove b
        #   2. Find uncovered tsubs U
        #   3. If U is empty, we would have removed it in Pass 1. So U is not empty.
        #   4. Find a block b' not in cover that covers all t in U.
        #   5. If found, swap b with b'.
        #   6. This swap is "neutral" in size, but might change the structure.
        #   7. We accept the swap if it leads to a state where *more* blocks are redundant in the next Pass 1?
        #   8. Or simply accept if it's valid, and rely on Pass 1 to clean up?
        #   9. Risk: Oscillation.
        #   10. Better: Only accept swap if it allows at least one block to become redundant immediately?
        #       Or if it reduces the "potential" for coverage?
        #       Let's just accept valid swaps and let Pass 1 clean up. To prevent oscillation, 
        #       we can keep a history of rejected swaps or just limit the number of swaps per outer loop.
        
        # Let's limit swaps to 10 per outer iteration to keep it fast.
        swaps_made = 0
        max_swaps = 10
        rng = random.Random(42)
        
        # Shuffle order of blocks to try to avoid bias
        indices = list(range(len(current_blocks)))
        rng.shuffle(indices)
        
        for idx in indices:
            if swaps_made >= max_swaps:
                break
            if time.time() > deadline:
                break
                
            b_id = current_blocks[idx]
            
            # Tentative removal
            for tid in block_tsubs[b_id]:
                cover_count[tid] -= 1
            
            # Find uncovered
            uncovered_tids = [tid for tid in range(num_tsubs) if cover_count[tid] == 0]
            
            if not uncovered_tids:
                # This block was redundant, why didn't Pass 1 catch it?
                # Maybe it became redundant due to previous swaps in this pass?
                # Pass 1 runs before swaps. So this shouldn't happen unless 
                # the logic is flawed.
                # Actually, if we swap b with b', and b' covers the same things, 
                # and then we remove b, it's redundant.
                # But we are in the "try swap" phase.
                # If uncovered_tids is empty, we can just keep it removed?
                # That would mean we found a redundant block.
                # Let's just accept the removal.
                in_cover[b_id] = False
                current_blocks.pop(idx)
                improved = True
                # Note: We don't restore cover_count because we removed it permanently.
                # But we are in a loop that assumes we might swap.
                # If we remove, we break the swap logic.
                # Let's handle this case: if redundant, remove and break swap loop for this idx.
                # But we already did Pass 1.
                # So this implies the state changed? No, Pass 1 didn't change state if it found no redundancies.
                # So this case should not occur.
                continue
            
            # Find candidate b' that covers all uncovered_tids
            # Intersect tsub_to_blocks for each tid in uncovered_tids
            candidates = None
            for tid in uncovered_tids:
                blocks_covering_tid = tsub_to_blocks[tid]
                if candidates is None:
                    candidates = set(blocks_covering_tid)
                else:
                    candidates.intersection_update(blocks_covering_tid)
                    
            if candidates:
                # Filter out blocks already in cover
                valid_candidates = [c for c in candidates if not in_cover[c]]
                if valid_candidates:
                    # Pick one
                    # Heuristic: Pick one that covers the most "highly covered" tsubs?
                    # Or random?
                    # Let's pick random to explore.
                    b_prime = rng.choice(valid_candidates)
                    
                    # Perform Swap
                    # Add b_prime
                    for tid in block_tsubs[b_prime]:
                        cover_count[tid] += 1
                    in_cover[b_prime] = True
                    
                    # Remove b_id from current_blocks
                    current_blocks[idx] = b_prime
                    in_cover[b_id] = False
                    # Note: cover_count for b_id's tsubs were already decremented.
                    # We added b_prime's tsubs.
                    # So net effect is correct.
                    
                    swaps_made += 1
                    # We don't set improved=True here because size didn't change.
                    # But structure changed.
                    # We will check for redundancy in next Pass 1.
                else:
                    # No valid candidate, revert
                    for tid in block_tsubs[b_id]:
                        cover_count[tid] += 1
            else:
                # No candidate covers all, revert
                for tid in block_tsubs[b_id]:
                    cover_count[tid] += 1

    # Final Redundancy Removal
    for i in range(len(current_blocks) - 1, -1, -1):
        b_id = current_blocks[i]
        if is_redundant(b_id):
            for tid in block_tsubs[b_id]:
                cover_count[tid] -= 1
            in_cover[b_id] = False
            current_blocks.pop(i)

    # Convert to lists
    final_blocks = [list(all_blocks[b_id]) for b_id in current_blocks]
    
    # Verify
    # (Optional, but good for safety)
    
    return final_blocks
