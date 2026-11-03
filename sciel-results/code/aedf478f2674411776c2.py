import itertools
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Bitset representation
    tsub_mask = [0] * n_blocks
    tsub_blocks_mask = [0] * n_tsubsets
    
    # List of tsub indices for each block (for cover_count updates)
    block_tsubs = [[] for _ in range(n_blocks)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask
        
    full_tsub_mask = (1 << n_tsubsets) - 1
    
    # Precompute initial gains for greedy
    initial_gains = [(tsub_mask[b]).bit_count() for b in range(n_blocks)]
    
    def greedy_cover(seed):
        """
        Heap-based lazy greedy with precomputed initial gains.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        
        # Use precomputed initial gains for heap initialization
        heap = [(-initial_gains[b_idx], b_idx) for b_idx in range(n_blocks)]
        heapq.heapify(heap)
        
        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break
                
            while heap:
                neg_gain, b_idx = heapq.heappop(heap)
                current_gain = (tsub_mask[b_idx] & uncovered_mask).bit_count()
                
                if -neg_gain > current_gain:
                    heapq.heappush(heap, (-current_gain, b_idx))
                    continue
                    
                if current_gain == 0:
                    break
                
                selected.append(b_idx)
                uncovered_mask &= ~tsub_mask[b_idx]
                break
            
            if not heap or (heap and -heap[0][0] == 0):
                break

        return selected

    def local_search(blocks, seed):
        """
        Local search with swaps and deletions.
        Uses bitsets to find candidate replacement blocks quickly.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks
            
        block_set = set(blocks)
        
        # Initialize cover counts
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)
                    
            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue
            
            # Phase 2: Try swaps with increased candidate limit
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for b in current_blocks:
                if time.time() > deadline - 2.0:
                    break
                if b not in block_set:
                    continue
                    
                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break
                    
                cand_mask = (1 << n_blocks) - 1
                for ts_idx in unique_tsubs:
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                
                if cand_mask == 0:
                    continue
                    
                cand_mask &= ~(1 << b)
                
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 50: # Increased from 20
                        break
                        
                if not candidates:
                    continue
                    
                rng.shuffle(candidates)
                
                swapped = False
                for r in candidates:
                    if r in block_set:
                        continue
                        
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                        
                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    swapped = True
                    break
                    
                if swapped:
                    break
                    
        return list(block_set)

    def two_for_one_search(blocks, seed):
        """
        Try to replace two blocks with one.
        Optimized: limit required_mask size to avoid expensive ANDs.
        """
        rng = random.Random(seed)
        if len(blocks) < 2:
            return blocks
            
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            n_checks = 50 # Reduced from 100 to allow more restarts
            for _ in range(n_checks):
                if time.time() > deadline - 2.0:
                    break
                if len(block_set) < 2:
                    break
                    
                i, j = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[i]
                b2 = current_blocks[j]
                
                if b1 not in block_set or b2 not in block_set:
                    continue
                
                # Build required_mask but limit its size
                required_tsubs = []
                for ts_idx in block_tsubs[b1]:
                    if cover_count[ts_idx] <= 1:
                        required_tsubs.append(ts_idx)
                for ts_idx in block_tsubs[b2]:
                    if cover_count[ts_idx] <= 1:
                        required_tsubs.append(ts_idx)
                        
                if not required_tsubs:
                    continue
                
                # If too many required subsets, skip (unlikely to find a single block covering all)
                if len(required_tsubs) > 10:
                    continue
                    
                cand_mask = (1 << n_blocks) - 1
                for ts_idx in required_tsubs:
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                    
                if cand_mask == 0:
                    continue
                    
                cand_mask &= ~(1 << b1)
                cand_mask &= ~(1 << b2)
                
                # Get candidates
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 5:
                        break
                        
                if not candidates:
                    continue
                    
                rng.shuffle(candidates)
                r = candidates[0]
                
                block_set.remove(b1)
                for ts_idx in block_tsubs[b1]:
                    cover_count[ts_idx] -= 1
                    
                block_set.remove(b2)
                for ts_idx in block_tsubs[b2]:
                    cover_count[ts_idx] -= 1
                    
                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1
                    
                improved = True
                break
                
        return list(block_set)

    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    num_restarts = 25 # Increased since greedy is faster now
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        seed = 1000 + i
        greedy_sol = greedy_cover(seed)
        
        if not greedy_sol:
            continue
            
        ls_sol = local_search(greedy_sol, seed)
        
        t2o_sol = two_for_one_search(ls_sol, seed)
        
        if len(t2o_sol) < best_size:
            best_size = len(t2o_sol)
            best_solution = t2o_sol.copy()
            
    # Final polish: one more pass of local search on best solution
    if best_solution is not None and time.time() < deadline - 5.0:
        polished = local_search(best_solution, 99999)
        if len(polished) < best_size:
            best_size = len(polished)
            best_solution = polished
            
    if best_solution is None:
        best_solution = greedy_cover(0)
        
    return [list(all_blocks[idx]) for idx in best_solution]
