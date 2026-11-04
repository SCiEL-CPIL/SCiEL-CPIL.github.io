import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0
    
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    full_mask = (1 << n_subsets) - 1
    
    def get_block_mask(block):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = subset_to_index[sub]
            mask |= (1 << idx)
        return mask

    # 1. Greedy Phase
    uncovered = full_mask
    blocks = []
    block_masks = []
    
    while uncovered:
        if time.time() > deadline - 10:
            break
            
        lsb = uncovered & -uncovered
        if lsb == 0:
            break
        target_idx = lsb.bit_length() - 1
        target_subset = all_subsets[target_idx]
        
        rem_points = [p for p in range(v) if p not in target_subset]
        
        best_block = None
        best_gain = -1
        
        # Optimize: If k-t is small, iterate combinations.
        # Here k=7, t=3, so we pick 4 from 21.
        for combo in itertools.combinations(rem_points, k - t):
            block = tuple(sorted(target_subset + combo))
            
            # Calculate gain
            gain = 0
            for sub in itertools.combinations(block, t):
                idx = subset_to_index[sub]
                if (uncovered >> idx) & 1:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                if best_gain == 35:
                    break
        
        if best_block:
            blocks.append(best_block)
            mask = get_block_mask(best_block)
            block_masks.append(mask)
            uncovered &= ~mask

    # 2. Local Search Phase
    # Data structures:
    # blocks: list of tuples
    # block_masks: list of int masks
    # global_mask: int mask of union of all blocks
    
    def calc_global_mask():
        gm = 0
        for m in block_masks:
            gm |= m
        return gm

    global_mask = calc_global_mask()
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Step A: Remove redundant blocks
        n = len(blocks)
        for i in range(n):
            if time.time() > deadline - 5:
                break
            bi_mask = block_masks[i]
            # Check if bi is redundant
            # others_mask = global_mask & ~bi_mask
            # If (others_mask & bi_mask) == bi_mask, then redundant.
            others_mask = global_mask & ~bi_mask
            if (others_mask & bi_mask) == bi_mask:
                # Remove block i
                blocks.pop(i)
                block_masks.pop(i)
                global_mask = calc_global_mask()
                improved = True
                break # Restart loop
        
        if time.time() > deadline - 5:
            break
        
        # Step B: Try to improve by swapping
        # For each block, try to find a better replacement
        # This is expensive, so we limit attempts or use a simpler heuristic.
        
        # Let's try a simple random restart of greedy if we have time?
        # Or just stick to redundancy removal which is very effective.
        
        # Let's add a simple swap:
        # Pick a random block, try to replace it with a block that covers its unique parts
        # and has higher total coverage.
        
        if random.random() < 0.5: # 50% chance to try swap
            i = random.randint(0, len(blocks)-1)
            bi = blocks[i]
            bi_mask = block_masks[i]
            
            others_mask = global_mask & ~bi_mask
            unique_mask = bi_mask & ~others_mask
            
            if unique_mask == 0:
                continue # Should have been removed
            
            # Find points in unique_mask
            pts_in_unique = set()
            for idx in range(n_subsets):
                if (unique_mask >> idx) & 1:
                    pts_in_unique.update(all_subsets[idx])
            
            if len(pts_in_unique) > k:
                continue
                
            rem_pts = [p for p in range(v) if p not in pts_in_unique]
            need = k - len(pts_in_unique)
            
            if need < 0:
                continue
                
            best_block = None
            best_score = -1
            
            # Sample a few candidates to save time
            # Instead of all C(v-|U|, need), try 100 random ones
            candidates = []
            for _ in range(100):
                if need == 0:
                    candidates.append(tuple(sorted(pts_in_unique)))
                else:
                    combo = tuple(random.sample(rem_pts, need))
                    candidates.append(tuple(sorted(pts_in_unique.union(combo))))
            
            # Also add the original block to compare
            candidates.append(bi)
            
            for cand in candidates:
                if cand == bi:
                    continue
                cand_mask = get_block_mask(cand)
                if (cand_mask & unique_mask) != unique_mask:
                    continue
                
                # Score: Coverage of bi_mask
                score = bin(cand_mask & bi_mask).count('1')
                
                if score > best_score:
                    best_score = score
                    best_block = cand
                    if best_score == bin(bi_mask).count('1'):
                        break
                        
            if best_block and best_block != bi:
                # Accept swap
                blocks[i] = best_block
                block_masks[i] = get_block_mask(best_block)
                global_mask = calc_global_mask()
                improved = True

    return [list(b) for b in blocks]
