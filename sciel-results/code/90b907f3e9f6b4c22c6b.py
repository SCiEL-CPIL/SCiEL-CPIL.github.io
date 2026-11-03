import itertools
import time
import random
import numpy as np
from collections import defaultdict
import heapq

def construct(v, k, t):
    deadline = time.time() + 118.0
    
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
            
    # Precompute set versions of block_tsubs for faster intersection in local search
    block_tsub_sets = [set(tsubs) for tsubs in block_tsubs]
    
    # Precompute tsub_blocks as sets for faster intersection
    tsub_block_sets = [set(lst) for lst in tsub_blocks]
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi])
            
        # Use a max-heap: (-gain, bi)
        heap = [(-gain[bi], bi) for bi in range(n_blocks)]
        heapq.heapify(heap)
        
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
                
            # Pop until we find a valid (non-stale) entry
            best_bi = -1
            max_gain = 0
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                if g == gain[bi] and g > 0:
                    best_bi = bi
                    max_gain = g
                    break
                # Stale entry, discard
            else:
                break
                
            if max_gain <= 0:
                break
                
            selected.append(best_bi)
            
            # Update gains
            for ts_idx in block_tsubs[best_bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Decrement gain for all blocks covering this ts_idx
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1
                        # Push updated gain into heap
                        heapq.heappush(heap, (-gain[bj], bj))
                        
            gain[best_bi] = -1 # Prevent re-selection
            heapq.heappush(heap, (-gain[best_bi], best_bi))
            
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
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart
                    
                # Find candidate blocks that cover all unique_tsubs
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_block_sets[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]
                    
                # Remove current block from candidates
                candidates.discard(bi)
                
                if not candidates:
                    continue
                    
                # Try candidates - limit to a reasonable number
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bj in cand_list[:30]:
                    if bj in block_set:
                        continue
                        
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
            
            n_checked = 0
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 2:
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
                if n_checked > 8000:
                    break
                
                # Compute required t-subsets
                req_bi = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                req_bj = [ts for ts in block_tsubs[bj] if cover_count[ts] == 1]
                
                required = set(req_bi)
                required.update(req_bj)
                
                # Check intersection for count==2
                inter = block_tsub_sets[bi] & block_tsub_sets[bj]
                for ts in inter:
                    if cover_count[ts] == 2:
                        required.add(ts)
                
                if not required:
                    continue
                
                # Find candidate blocks that cover all ts in required
                min_ts = min(required, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_block_sets[min_ts])
                
                for ts in required:
                    if ts == min_ts:
                        continue
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts]
                    
                candidates.discard(bi)
                candidates.discard(bj)
                
                if not candidates:
                    continue
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bk in cand_list[:15]:
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

    def final_polish(blocks, seed):
        """
        Aggressive final polishing: repeatedly try all 1-for-1 and 2-for-1 swaps
        until no improvement is found.
        """
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
        
        # Try multiple passes
        for _ in range(3):
            if time.time() > deadline - 2:
                break
                
            # Remove redundant
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
                continue
            
            # Try 1-for-1 swaps
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                unique_tsubs = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                if not unique_tsubs:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_block_sets[min_ts])
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]
                candidates.discard(bi)
                if not candidates:
                    continue
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                for bj in cand_list[:20]:
                    if bj in block_set:
                        continue
                    block_set.remove(bi)
                    for ts in block_tsubs[bi]:
                        cover_count[ts] -= 1
                    block_set.add(bj)
                    for ts in block_tsubs[bj]:
                        cover_count[ts] += 1
                    improved = True
                    break
                if improved:
                    break
            
            if not improved and time.time() < deadline - 2:
                # Try 2-for-1 swaps
                current_blocks = list(block_set)
                rng.shuffle(current_blocks)
                n_checked = 0
                for bi in current_blocks:
                    if time.time() > deadline - 2:
                        break
                    if bi not in block_set:
                        continue
                    if len(block_set) < 2:
                        break
                    others = [b for b in block_set if b != bi]
                    if not others:
                        continue
                    bj = rng.choice(others)
                    n_checked += 1
                    if n_checked > 5000:
                        break
                    req_bi = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                    req_bj = [ts for ts in block_tsubs[bj] if cover_count[ts] == 1]
                    required = set(req_bi)
                    required.update(req_bj)
                    inter = block_tsub_sets[bi] & block_tsub_sets[bj]
                    for ts in inter:
                        if cover_count[ts] == 2:
                            required.add(ts)
                    if not required:
                        continue
                    min_ts = min(required, key=lambda x: len(tsub_blocks[x]))
                    candidates = set(tsub_block_sets[min_ts])
                    for ts in required:
                        if ts == min_ts:
                            continue
                        if not candidates:
                            break
                        candidates &= tsub_block_sets[ts]
                    candidates.discard(bi)
                    candidates.discard(bj)
                    if not candidates:
                        continue
                    cand_list = list(candidates)
                    rng.shuffle(cand_list)
                    for bk in cand_list[:10]:
                        if bk in block_set:
                            continue
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
                        break
                    if improved:
                        break
                if not improved:
                    break
                    
        return list(block_set)

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy starts
    num_trials = 40
    for i in range(num_trials):
        if time.time() > deadline - 25:
            break
            
        blocks = greedy_cover(i)
        if not blocks:
            continue
            
        # Local search (single swaps)
        blocks = local_search(blocks, i)
        
        # Two-block swap search
        if time.time() < deadline - 12:
            blocks = two_block_swap_search(blocks, i + 1000)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # Final polish on the best solution
    if best_blocks is not None and time.time() < deadline - 8:
        best_blocks = final_polish(best_blocks, 77777)
        
    # Fallback if something went wrong
    if best_blocks is None:
        best_blocks = greedy_cover(0)
        
    return [all_blocks[i] for i in best_blocks]
