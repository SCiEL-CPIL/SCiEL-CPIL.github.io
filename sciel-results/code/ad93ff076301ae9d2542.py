import itertools
import time
import random
import heapq
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # block_mask[i] is an integer with bits set for t-subsets covered by block i
    block_mask = [0] * n_blocks
    # tsub_blocks[j] is a list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    # Precompute block masks and tsub_blocks
    for bi, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks[idx].append(bi)
        block_mask[bi] = mask
        
    # Helper to get list of indices from a mask
    def mask_to_indices(mask):
        indices = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            indices.append(idx)
            mask ^= lsb
        return indices

    def greedy_cover(seed):
        rng = random.Random(seed)
        covered_mask = 0
        selected = []
        selected_set = set()
        
        # gain[bi] = number of new tsubs covered by bi
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = block_mask[bi].bit_count()
            
        heap = [(-gain[bi], bi) for bi in range(n_blocks)]
        heapq.heapify(heap)
        
        while True:
            if time.time() > deadline - 10:
                break
                
            best_bi = -1
            best_gain = 0
            
            # Pop until valid
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                if g == gain[bi] and bi not in selected_set:
                    best_bi = bi
                    best_gain = g
                    break
                    
            if best_bi == -1 or best_gain == 0:
                break
                
            selected.append(best_bi)
            selected_set.add(best_bi)
            
            # Update covered mask
            new_bits = block_mask[best_bi] & ~covered_mask
            covered_mask |= new_bits
            
            # Update gains for other blocks
            # For each new bit covered, decrement gain of all blocks covering it
            # This is expensive if done naively. 
            # Optimization: Only update if gain > 0.
            # We can just recalculate gain for blocks that might be affected?
            # Or, since we are using a heap, we can just push updated gains.
            # But iterating over all blocks covering a tsub is O(degree).
            # Total work per step is sum of degrees of newly covered tsubs.
            
            indices_new = mask_to_indices(new_bits)
            for idx in indices_new:
                for bj in tsub_blocks[idx]:
                    if bj in selected_set:
                        continue
                    # This block covers a newly covered tsub, so its gain decreases by 1
                    gain[bj] -= 1
                    heapq.heappush(heap, (-gain[bj], bj))
                    
            gain[best_bi] = -1 # Mark as done
            
        return selected

    def local_search(blocks, seed):
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
        # Initialize cover counts
        for bi in block_set:
            m = block_mask[bi]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in block_set:
                m = block_mask[bi]
                redundant = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    to_remove.append(bi)
                    
            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    m = block_mask[bi]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                improved = True
                continue # Restart

            # Phase 2: 1-for-1 Swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > deadline - 5:
                    break
                if bi not in block_set:
                    continue
                    
                # Find unique tsubs for bi
                req_indices = []
                m = block_mask[bi]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        req_indices.append(idx)
                    m ^= lsb
                    
                if not req_indices:
                    # Can remove bi
                    block_set.remove(bi)
                    m = block_mask[bi]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    improved = True
                    break # Restart
                    
                # Find candidates covering all req_indices
                # Start with the rarest
                req_indices.sort(key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[req_indices[0]])
                
                for idx in req_indices[1:]:
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[idx])
                    
                candidates.discard(bi)
                
                if not candidates:
                    continue
                    
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bj in cand_list[:20]: # Limit candidates
                    if bj in block_set:
                        continue
                        
                    # Check if bj covers all req_indices (should be true by construction, but safe)
                    # Actually, intersection ensures it covers all.
                    
                    # Perform swap
                    block_set.remove(bi)
                    m = block_mask[bi]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                        
                    block_set.add(bj)
                    m = block_mask[bj]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        m ^= lsb
                        
                    improved = True
                    swapped = True
                    break # Restart
                    
                if swapped:
                    break # Restart while loop
                    
            if improved:
                continue
                
            # Phase 3: 2-for-1 Swaps
            # Try to replace two blocks with one
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            n_checked = 0
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 5:
                    break
                if bi not in block_set:
                    continue
                if len(block_set) < 2:
                    break
                    
                # Pick a random partner
                others = [b for b in block_set if b != bi]
                if not others:
                    continue
                bj = rng.choice(others)
                
                n_checked += 1
                if n_checked > 5000:
                    break
                    
                # Compute required mask for 2-for-1
                # Subsets in (bi U bj) that are covered only by bi or bj (count 1) 
                # or covered by both (count 2)
                req_indices = set()
                
                # Iterate bi
                m = block_mask[bi]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        req_indices.add(idx)
                    elif cover_count[idx] == 2 and (block_mask[bj] & (1 << idx)):
                        req_indices.add(idx)
                    m ^= lsb
                    
                # Iterate bj (only if not in bi to avoid double processing? No, just check)
                m = block_mask[bj]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        req_indices.add(idx)
                    elif cover_count[idx] == 2 and (block_mask[bi] & (1 << idx)):
                        req_indices.add(idx)
                    m ^= lsb
                    
                if not req_indices:
                    continue
                    
                req_list = list(req_indices)
                req_list.sort(key=lambda x: len(tsub_blocks[x]))
                
                candidates = set(tsub_blocks[req_list[0]])
                for idx in req_list[1:]:
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[idx])
                    
                candidates.discard(bi)
                candidates.discard(bj)
                
                if not candidates:
                    continue
                    
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bk in cand_list[:10]:
                    if bk in block_set:
                        continue
                        
                    # Perform swap
                    block_set.remove(bi)
                    m = block_mask[bi]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                        
                    block_set.remove(bj)
                    m = block_mask[bj]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                        
                    block_set.add(bk)
                    m = block_mask[bk]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        m ^= lsb
                        
                    improved = True
                    swapped = True
                    break
                    
                if swapped:
                    break # Restart while loop
                    
        return list(block_set)

    best_blocks = None
    best_len = float('inf')
    
    num_trials = 30
    
    for i in range(num_trials):
        if time.time() > deadline - 15:
            break
            
        blocks = greedy_cover(i)
        if not blocks:
            continue
            
        blocks = local_search(blocks, i)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # Final polish on the best solution
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = local_search(best_blocks, 99999)
        
    if best_blocks is None:
        best_blocks = greedy_cover(0)
        
    return [all_blocks[i] for i in best_blocks]
