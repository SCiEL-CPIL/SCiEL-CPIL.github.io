import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    # 1. Precomputation
    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute bitsets for each block
    # Mask[i] is an integer where bit j is 1 if block i covers triple j
    block_masks = [0] * n_ks
    block_triple_indices = [None] * n_ks # Store list of triple indices for quick access
    
    for i, block in enumerate(all_ks):
        mask = 0
        indices = []
        for sub in itertools.combinations(block, t):
            ti = idx_map[sub]
            mask |= (1 << ti)
            indices.append(ti)
        block_masks[i] = mask
        block_triple_indices[i] = indices

    # Precompute TripleToBlocks: list of block indices covering each triple
    triple_to_blocks = [[] for _ in range(n_ts)]
    for i in range(n_ks):
        for ti in block_triple_indices[i]:
            triple_to_blocks[ti].append(i)

    # 2. Greedy Construction using Bitsets
    def greedy(seed):
        rng = random.Random(seed)
        uncovered = (1 << n_ts) - 1  # All triples uncovered initially
        blocks = []
        in_blocks = set()
        
        # To speed up greedy, we can pre-sort or just iterate.
        # Iterating 346k blocks * 76 steps is ~26M ops. 
        # bit_count is fast.
        
        while uncovered != 0:
            if time.time() - start_time > time_limit - 10:
                break
            
            best_gain = -1
            best_idx = -1
            
            # Optimization: If we have many blocks, iterating all is slow.
            # But 346k * 76 is manageable in C via numpy if vectorized, 
            # but bit_count on Python ints is not vectorized easily.
            # However, 346k iterations in Python takes ~0.05s. 
            # 76 * 0.05 = 3.8s per greedy run. 
            # With 10 seeds, that's 38s. Plus local search. Might be tight.
            
            # Let's try to optimize the search for best block.
            # We can skip blocks that are already in 'blocks'.
            
            for i in range(n_ks):
                if i in in_blocks:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break # Should not happen if uncovered != 0 and we have valid blocks
            
            blocks.append(best_idx)
            in_blocks.add(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return blocks

    # 3. Local Search
    def local_search(blocks_list):
        current = list(blocks_list)
        in_current = set(current)
        
        # Initialize counts
        counts = [0] * n_ts
        for b_idx in current:
            for ti in block_triple_indices[b_idx]:
                counts[ti] += 1
        
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 5:
                break
            improved = False
            
            # Phase 1: Remove redundant blocks
            new_current = []
            for b_idx in current:
                subs = block_triple_indices[b_idx]
                can_remove = True
                for ti in subs:
                    if counts[ti] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove b_idx
                    for ti in subs:
                        counts[ti] -= 1
                    in_current.remove(b_idx)
                    improved = True
                else:
                    new_current.append(b_idx)
            current = new_current
            
            # Phase 2: 1-for-1 Swaps
            # For each block i in current, try to replace with k not in current
            # such that k covers all critical triples of i.
            swapped = False
            for i in current:
                if time.time() - start_time > time_limit - 5:
                    break
                
                subs_i = block_triple_indices[i]
                critical = [ti for ti in subs_i if counts[ti] == 1]
                
                if not critical:
                    # Should have been removed in Phase 1, but just in case
                    for ti in subs_i:
                        counts[ti] -= 1
                    in_current.remove(i)
                    current.remove(i)
                    improved = True
                    swapped = True
                    break # Restart loop
                    
                if critical:
                    # Find candidate k that covers all critical
                    # Use the first critical triple to get candidates
                    first_crit = critical[0]
                    candidates = triple_to_blocks[first_crit]
                    
                    found_k = -1
                    for k in candidates:
                        if k in in_current:
                            continue
                        # Check if k covers all critical
                        valid = True
                        for ti in critical:
                            # Check if ti is in block k
                            # We can check if (block_masks[k] >> ti) & 1
                            if not (block_masks[k] >> ti) & 1:
                                valid = False
                                break
                        if valid:
                            found_k = k
                            break
                    
                    if found_k != -1:
                        # Perform swap
                        # Remove i
                        for ti in subs_i:
                            counts[ti] -= 1
                        in_current.remove(i)
                        
                        # Add k
                        subs_k = block_triple_indices[found_k]
                        for ti in subs_k:
                            counts[ti] += 1
                        in_current.add(found_k)
                        
                        current.remove(i)
                        current.append(found_k)
                        
                        improved = True
                        swapped = True
                        break # Restart loop to allow further removals/swaps
            
            if swapped:
                continue # Restart the while loop to re-evaluate removals
            
            # Phase 3: 2-for-1 Swaps (Limited)
            # Try to replace two blocks i, j with one block k
            # This is expensive. We limit the number of attempts.
            if time.time() - start_time < time_limit - 10:
                n_current = len(current)
                if n_current > 1:
                    # Sample a few pairs
                    pairs_checked = 0
                    max_pairs = 5000 # Limit to keep time bounded
                    
                    # Shuffle indices to randomize
                    indices = list(range(n_current))
                    random.shuffle(indices)
                    
                    for u in range(len(indices)):
                        if pairs_checked >= max_pairs:
                            break
                        for v in range(u + 1, len(indices)):
                            if pairs_checked >= max_pairs:
                                break
                            pairs_checked += 1
                            
                            i = current[indices[u]]
                            j = current[indices[v]]
                            
                            subs_i = block_triple_indices[i]
                            subs_j = block_triple_indices[j]
                            
                            # Find critical triples for the pair
                            # A triple is critical if count == 1 and it is in i or j
                            # Actually, if we remove i and j, any triple with count == 1 
                            # that is covered by i or j will become 0.
                            # So Req = {ti | ti in subs_i or ti in subs_j} and counts[ti] == 1
                            
                            req_set = set()
                            for ti in subs_i:
                                if counts[ti] == 1:
                                    req_set.add(ti)
                            for ti in subs_j:
                                if counts[ti] == 1:
                                    req_set.add(ti)
                                    
                            if not req_set:
                                # Both are redundant? Should have been removed.
                                continue
                                
                            # Find k that covers all in req_set
                            # Pick most constrained triple in req_set
                            # Smallest triple_to_blocks list
                            min_len = float('inf')
                            best_crit = None
                            for ti in req_set:
                                if len(triple_to_blocks[ti]) < min_len:
                                    min_len = len(triple_to_blocks[ti])
                                    best_crit = ti
                                    
                            candidates = triple_to_blocks[best_crit]
                            found_k = -1
                            for k in candidates:
                                if k in in_current:
                                    continue
                                valid = True
                                for ti in req_set:
                                    if not (block_masks[k] >> ti) & 1:
                                        valid = False
                                        break
                                if valid:
                                    found_k = k
                                    break
                                    
                            if found_k != -1:
                                # Perform swap
                                # Remove i
                                for ti in subs_i:
                                    counts[ti] -= 1
                                in_current.remove(i)
                                current.remove(i)
                                
                                # Remove j
                                for ti in subs_j:
                                    counts[ti] -= 1
                                in_current.remove(j)
                                current.remove(j)
                                
                                # Add k
                                subs_k = block_triple_indices[found_k]
                                for ti in subs_k:
                                    counts[ti] += 1
                                in_current.add(found_k)
                                current.append(found_k)
                                
                                improved = True
                                break # Restart loop
                        if improved:
                            break
                            
        return current

    # 4. Main Loop
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds
    # Greedy is the bottleneck. Let's run a few seeds.
    # Each greedy takes ~4s. Local search takes variable time.
    # We have 115s.
    # 10 seeds * 4s = 40s. Leaves 75s for local search.
    
    for seed in range(10):
        if time.time() - start_time > time_limit - 15:
            break
            
        greedy_blocks = greedy(seed)
        
        if not greedy_blocks:
            continue
            
        refined = local_search(greedy_blocks)
        
        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = refined
            
    if best_blocks is None:
        # Fallback: Return a valid cover if we found any, else empty
        # But we must return a valid cover.
        # If we didn't find any, we should try a deterministic greedy without time limit?
        # No, we must respect time.
        # If best_blocks is None, it means we timed out during greedy or local search failed.
        # We should return the best greedy result we had? 
        # The code above only updates best_blocks after local search.
        # Let's store the best greedy result as a fallback.
        
        # Re-run a quick greedy to ensure we have something
        # Actually, the loop above might have broken before updating best_blocks.
        # Let's ensure we have a valid cover.
        
        # Simple fallback: Run greedy with seed 0 and return it, even if not refined.
        greedy_blocks = greedy(0)
        if greedy_blocks:
            return [list(all_ks[i]) for i in greedy_blocks]
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_blocks]
    return final_blocks
