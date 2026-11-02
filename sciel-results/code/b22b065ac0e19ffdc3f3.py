import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute block_triples: list of lists of triple indices
    block_triples = []
    # Also build triple_to_blocks
    triple_to_blocks = [[] for _ in range(n_ts)]
    
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        triple_indices = [idx_map[sub] for sub in subs]
        block_triples.append(triple_indices)
        for tri_idx in triple_indices:
            triple_to_blocks[tri_idx].append(i)
            
    # Convert triple_to_blocks to numpy arrays for faster iteration? 
    # Lists are fine for iteration in Python if we don't do fancy slicing.
    # But for the greedy step, we need to iterate triple_to_blocks[t] frequently.
    # Let's keep them as lists of ints.
    
    # Precompute sets for fast subset check in local search
    # This might be memory heavy, but 346k sets of 35 ints is manageable.
    block_triples_set = [set(lst) for lst in block_triples]
    
    def fast_greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_ts, dtype=bool)
        blocks_indices = []
        
        # block_gain[i] = number of uncovered triples in block i
        block_gain = np.full(n_ks, 35, dtype=np.int32)
        
        # We need to know which blocks are still available.
        # block_gain > 0 implies available and has uncovered triples.
        
        while np.any(uncovered):
            # Find max gain
            max_gain = np.max(block_gain)
            if max_gain == 0:
                break # Should not happen if uncovered exists and we haven't covered all
            
            candidates = np.where(block_gain == max_gain)[0]
            best_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks_indices.append(best_idx)
            
            # Mark block as used
            block_gain[best_idx] = -1
            
            # Update gains for other blocks
            for tri_idx in block_triples[best_idx]:
                if uncovered[tri_idx]:
                    uncovered[tri_idx] = False
                    # Decrement gain for all blocks containing this triple
                    for other_block in triple_to_blocks[tri_idx]:
                        if block_gain[other_block] > 0:
                            block_gain[other_block] -= 1
        
        return blocks_indices

    def local_search(blocks_indices):
        current_blocks = list(blocks_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_triples[b_idx]] += 1
        
        # Phase 1: Removal
        improved = True
        while improved:
            improved = False
            new_current_blocks = []
            for b_idx in current_blocks:
                # Check if redundant
                subs = block_triples[b_idx]
                # If min coverage > 1, redundant
                if np.min(coverage_count[subs]) > 1:
                    # Remove
                    coverage_count[subs] -= 1
                    improved = True
                else:
                    new_current_blocks.append(b_idx)
            current_blocks = new_current_blocks
            
        # Phase 2: Pair Replacement
        # Time check
        if time.time() - start_time > time_limit * 0.7:
            return current_blocks
            
        replacement_improved = True
        while replacement_improved:
            if time.time() - start_time > time_limit * 0.9:
                break
                
            replacement_improved = False
            n_blocks = len(current_blocks)
            
            # Try all pairs
            for i in range(n_blocks):
                if time.time() - start_time > time_limit * 0.9:
                    break
                for j in range(i + 1, n_blocks):
                    if time.time() - start_time > time_limit * 0.9:
                        break
                        
                    b1 = current_blocks[i]
                    b2 = current_blocks[j]
                    
                    # Find critical triples
                    # Triples in b1 or b2 that will be uncovered if we remove both, 
                    # UNLESS covered by a replacement block.
                    # Actually, we need the set of triples that are covered ONLY by b1 or b2 
                    # (count 1) or by BOTH b1 and b2 (count 2).
                    
                    critical_set = set()
                    
                    for tri in block_triples[b1]:
                        if coverage_count[tri] == 1:
                            critical_set.add(tri)
                        
                    for tri in block_triples[b2]:
                        if coverage_count[tri] == 1:
                            critical_set.add(tri)
                        
                    # Triples in intersection with count 2
                    # If a triple is in both b1 and b2, and count is 2, 
                    # removing both makes count 0. So it must be covered by replacement.
                    # If count > 2, it's fine.
                    # If count == 1, it was already added above.
                    
                    # Efficient way:
                    # Iterate intersection?
                    # Or just check all in b1 and b2.
                    # Since b1 and b2 are small (35), we can just do:
                    for tri in block_triples[b1]:
                         if tri in block_triples_set[b2]:
                             if coverage_count[tri] == 2:
                                 critical_set.add(tri)
                             
                    if not critical_set:
                        # Both are redundant? Should have been removed in Phase 1.
                        # But maybe counts changed? No, counts are static here.
                        # If critical_set is empty, it means no triple is uniquely covered by b1 or b2 
                        # or exclusively by both.
                        # This implies b1 and b2 are redundant together?
                        # If a triple in b1 has count 2 (covered by b1 and X), and no triple in b1 has count 1,
                        # and same for b2, and no triple in intersection has count 2...
                        # Then removing b1 and b2 leaves all triples with count >= 1.
                        # So we can just remove both.
                        # Let's handle this case: remove both.
                        coverage_count[block_triples[b1]] -= 1
                        coverage_count[block_triples[b2]] -= 1
                        # Remove from list
                        # We need to be careful with indices if we modify list during iteration.
                        # We'll mark for removal after the loop or rebuild.
                        # For simplicity, let's just break and restart the pair loop with updated list.
                        new_blocks = []
                        for idx, b in enumerate(current_blocks):
                            if idx != i and idx != j:
                                new_blocks.append(b)
                        current_blocks = new_blocks
                        replacement_improved = True
                        break # Break inner j loop
                    else:
                        # We need a replacement block that covers all critical_set
                        # Pick a triple from critical_set
                        t0 = next(iter(critical_set))
                        
                        # Check candidates
                        found_replacement = False
                        for b3 in triple_to_blocks[t0]:
                            # b3 must not be b1 or b2
                            if b3 == b1 or b3 == b2:
                                continue
                                
                            # Check if critical_set is subset of b3
                            # Using set intersection or issubset
                            if critical_set.issubset(block_triples_set[b3]):
                                # Found replacement
                                # Update coverage counts
                                coverage_count[block_triples[b1]] -= 1
                                coverage_count[block_triples[b2]] -= 1
                                coverage_count[block_triples[b3]] += 1
                                
                                # Update current_blocks
                                new_blocks = []
                                for idx, b in enumerate(current_blocks):
                                    if idx != i and idx != j:
                                        new_blocks.append(b)
                                new_blocks.append(b3)
                                current_blocks = new_blocks
                                replacement_improved = True
                                found_replacement = True
                                break # Break candidate loop
                                
                        if found_replacement:
                            break # Break inner j loop
                            
                if replacement_improved:
                    break # Break outer i loop
                    
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    restart_count = 0
    while time.time() - start_time < time_limit * 0.6:
        blocks = fast_greedy_cover(restart_count)
        blocks = local_search(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
        restart_count += 1
        
        # If we hit the lower bound (69), we can stop early? 
        # But we don't know for sure. 70 is good.
        if best_len <= 70:
             # Maybe do one more polish?
             # But time is tight.
             pass

    if best_blocks is None:
        # Fallback
        best_blocks = fast_greedy_cover(0)
        best_blocks = local_search(best_blocks)
        
    # Convert indices back to blocks
    final_blocks = [list(all_ks[i]) for i in best_blocks]
    
    return final_blocks
