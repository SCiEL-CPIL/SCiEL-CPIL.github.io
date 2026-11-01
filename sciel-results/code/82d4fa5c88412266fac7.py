import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute tset indices for each block
    block_tsets = []
    for b in all_blocks:
        tsets_in_b = []
        for ts in itertools.combinations(b, t):
            tsets_in_b.append(tset_to_idx[ts])
        block_tsets.append(np.array(tsets_in_b, dtype=np.int32))
    
    # Precompute block_tsets as sets for O(1) lookup
    block_tsets_sets = [set(block_tsets[i]) for i in range(num_blocks)]
    
    # Precompute tset_to_blocks
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx in range(num_blocks):
        for ts_idx in block_tsets[b_idx]:
            tset_to_blocks[ts_idx].append(b_idx)
    
    best_blocks = None
    best_len = float('inf')
    
    def greedy_cover_fast(seed_val):
        """Fast greedy with random tie-breaking."""
        random.seed(seed_val)
        uncovered = np.ones(num_tsets, dtype=bool)
        uncovered_set = set(range(num_tsets))
        blocks = []
        
        while uncovered_set:
            if time.time() > deadline:
                return None
            
            # Pick a random uncovered tset
            target_tset = random.choice(list(uncovered_set))
            candidate_blocks = tset_to_blocks[target_tset]
            
            if not candidate_blocks:
                break
                
            best_block = -1
            best_gain = -1
            
            for b_idx in candidate_blocks:
                gain = 0
                for ts_idx in block_tsets[b_idx]:
                    if uncovered[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                elif gain == best_gain and random.random() < 0.5:
                    best_block = b_idx
                    
            if best_block == -1:
                break
                
            blocks.append(best_block)
            
            for ts_idx in block_tsets[best_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_set.remove(ts_idx)
                    
        return blocks

    def local_search_remove(blocks):
        """Remove redundant blocks using coverage counts."""
        if not blocks:
            return blocks
            
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in blocks:
            for ts_idx in block_tsets[b_idx]:
                cover_count[ts_idx] += 1
                
        if np.any(cover_count == 0):
            return None
            
        blocks_list = blocks[:]
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            for i in range(len(blocks_list)):
                if time.time() > deadline:
                    break
                    
                b_idx = blocks_list[i]
                safe = True
                for ts_idx in block_tsets[b_idx]:
                    if cover_count[ts_idx] == 1:
                        safe = False
                        break
                        
                if safe:
                    for ts_idx in block_tsets[b_idx]:
                        cover_count[ts_idx] -= 1
                    blocks_list.pop(i)
                    improved = True
                    break
                    
        return blocks_list

    def pairwise_merge(blocks):
        """Try to replace two blocks with one that covers all uniquely-covered t-sets."""
        if len(blocks) < 2:
            return blocks
            
        # Build coverage count
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in blocks:
            for ts_idx in block_tsets[b_idx]:
                cover_count[ts_idx] += 1
                
        if np.any(cover_count == 0):
            return None
            
        blocks_list = blocks[:]
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            n = len(blocks_list)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    
                    b1, b2 = blocks_list[i], blocks_list[j]
                    
                    # Find t-sets uniquely covered by b1 or b2
                    unique_tsets = []
                    for ts_idx in block_tsets[b1]:
                        if cover_count[ts_idx] == 1:
                            unique_tsets.append(ts_idx)
                    for ts_idx in block_tsets[b2]:
                        if cover_count[ts_idx] == 1:
                            unique_tsets.append(ts_idx)
                            
                    if not unique_tsets:
                        continue
                        
                    # Find a block that covers all these unique t-sets
                    # and also covers the intersection of b1 and b2's t-sets
                    # Actually, we need a block that covers:
                    # - All t-sets uniquely covered by b1
                    # - All t-sets uniquely covered by b2
                    # - All t-sets covered by both b1 and b2 (these are covered twice, so removing both would uncover them)
                    
                    # The new block must cover: union of t-sets from b1 and b2 that are covered exactly once
                    # Plus all t-sets covered by both b1 and b2 (since they'd be uncovered if we remove both)
                    
                    required_tsets = set()
                    for ts_idx in block_tsets[b1]:
                        if cover_count[ts_idx] <= 1:
                            required_tsets.add(ts_idx)
                    for ts_idx in block_tsets[b2]:
                        if cover_count[ts_idx] <= 1:
                            required_tsets.add(ts_idx)
                            
                    # Find a block that covers all required_tsets
                    found = False
                    for cand_idx in range(num_blocks):
                        cand_set = block_tsets_sets[cand_idx]
                        if cand_set.issuperset(required_tsets):
                            # Check this block is not already in blocks_list
                            if cand_idx not in blocks_list:
                                # Remove b1 and b2, add cand
                                blocks_list.pop(j)  # Remove later index first
                                blocks_list.pop(i)
                                blocks_list.append(cand_idx)
                                improved = True
                                found = True
                                break
                                
                    if found:
                        break
                if improved:
                    break
                    
        return blocks_list

    # Run multiple attempts
    for attempt in range(30):
        if time.time() > deadline:
            break
            
        blocks = greedy_cover_fast(attempt)
        if blocks is None:
            continue
            
        # Local search to remove redundant blocks
        blocks = local_search_remove(blocks)
        
        if blocks is None:
            blocks = greedy_cover_fast(attempt)
            if blocks is None:
                continue
                
        # Pairwise merge
        merged = pairwise_merge(blocks)
        if merged is not None and len(merged) < len(blocks):
            blocks = merged
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
    if best_blocks is None:
        blocks = greedy_cover_fast(0)
        if blocks:
            return [list(all_blocks[b]) for b in blocks]
        return []
        
    return [list(all_blocks[b]) for b in best_blocks]
