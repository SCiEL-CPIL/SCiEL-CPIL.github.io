import itertools
import time
import sys
import numpy as np
import random

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Precompute block t-subset indices
    block_tsub_indices = []
    for b in all_kblocks:
        indices = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsub_indices.append(indices)
        
    # Precompute blocks covering each t-subset
    blocks_covering_ts = [[] for _ in range(num_tsubs)]
    for j, indices in enumerate(block_tsub_indices):
        for i in indices:
            blocks_covering_ts[i].append(j)

    def greedy_cover(randomize=False):
        uncovered = [True] * num_tsubs
        selected = []
        remaining_blocks = list(range(num_kblocks))
        
        while any(uncovered) and time.monotonic() < deadline - 20:
            best_gain = 0
            best_block = -1
            candidates = []
            
            for j in remaining_blocks:
                gain = 0
                for i in block_tsub_indices[j]:
                    if uncovered[i]:
                        gain += 1
                if gain > 0:
                    if gain > best_gain:
                        best_gain = gain
                        best_block = j
                        candidates = [j]
                    elif gain == best_gain:
                        candidates.append(j)
                        
            if best_gain == 0:
                break
            
            if randomize and len(candidates) > 1:
                best_block = random.choice(candidates)
                
            selected.append(best_block)
            for i in block_tsub_indices[best_block]:
                uncovered[i] = False
            remaining_blocks.remove(best_block)
            
        return selected

    def prune(selected):
        # Convert to set for fast lookup
        sel_set = set(selected)
        # Count coverage
        counts = [0] * num_tsubs
        for j in selected:
            for i in block_tsub_indices[j]:
                counts[i] += 1
                
        changed = True
        while changed:
            if time.monotonic() > deadline - 15:
                break
            changed = False
            new_selected = []
            for j in selected:
                redundant = True
                for i in block_tsub_indices[j]:
                    if counts[i] == 1:
                        redundant = False
                        break
                if redundant:
                    for i in block_tsub_indices[j]:
                        counts[i] -= 1
                    changed = True
                else:
                    new_selected.append(j)
            selected = new_selected
        return selected

    def local_search(selected):
        # selected is a list of block indices
        if not selected:
            return selected
            
        # Maintain counts and a set of selected blocks for fast check
        sel_set = set(selected)
        counts = [0] * num_tsubs
        for j in selected:
            for i in block_tsub_indices[j]:
                counts[i] += 1
                
        improved = True
        while improved and time.monotonic() < deadline - 10:
            improved = False
            
            # 1. Try 2-for-0 and 2-for-1
            # Get list of current blocks
            cur_blocks = list(selected)
            n = len(cur_blocks)
            
            for i_idx in range(n):
                for j_idx in range(i_idx + 1, n):
                    i = cur_blocks[i_idx]
                    j = cur_blocks[j_idx]
                    
                    # Compute U_ij: t-subsets covered only by i or j
                    U_ij = []
                    # Check t-subsets in i
                    for ts_idx in block_tsub_indices[i]:
                        # Check if all blocks covering ts_idx are in {i, j}
                        coverers = blocks_covering_ts[ts_idx]
                        # Optimization: if len(coverers) > 2, it can't be in U_ij
                        if len(coverers) <= 2:
                            if all(c == i or c == j for c in coverers):
                                U_ij.append(ts_idx)
                    # Check t-subsets in j
                    for ts_idx in block_tsub_indices[j]:
                        if ts_idx in U_ij: continue
                        coverers = blocks_covering_ts[ts_idx]
                        if len(coverers) <= 2:
                            if all(c == i or c == j for c in coverers):
                                U_ij.append(ts_idx)
                    
                    if not U_ij:
                        # 2-for-0: Remove both
                        selected.remove(i)
                        selected.remove(j)
                        sel_set.remove(i)
                        sel_set.remove(j)
                        for ts_idx in block_tsub_indices[i]:
                            counts[ts_idx] -= 1
                        for ts_idx in block_tsub_indices[j]:
                            counts[ts_idx] -= 1
                        improved = True
                        break
                    else:
                        # 2-for-1: Find best k covering U_ij
                        best_k = -1
                        best_cover = 0
                        # Candidates: blocks covering any ts in U_ij
                        candidates = set()
                        for ts_idx in U_ij:
                            candidates.update(blocks_covering_ts[ts_idx])
                        
                        for k in candidates:
                            if k in sel_set:
                                continue
                            cover = 0
                            for ts_idx in U_ij:
                                if ts_idx in block_tsub_indices[k]: # This check is slow
                                    cover += 1
                            # Faster: precompute set of tsub indices for k?
                            # Or just count intersection
                            # Since U_ij is small, this is okay.
                            if cover > best_cover:
                                best_cover = cover
                                best_k = k
                                
                        if best_k != -1 and best_cover == len(U_ij):
                            # Replace i, j with best_k
                            selected.remove(i)
                            selected.remove(j)
                            sel_set.remove(i)
                            sel_set.remove(j)
                            for ts_idx in block_tsub_indices[i]:
                                counts[ts_idx] -= 1
                            for ts_idx in block_tsub_indices[j]:
                                counts[ts_idx] -= 1
                            
                            selected.append(best_k)
                            sel_set.add(best_k)
                            for ts_idx in block_tsub_indices[best_k]:
                                counts[ts_idx] += 1
                                
                            improved = True
                            break
                            
                if improved:
                    break
                    
            if improved:
                continue
                
            # 2. Try 1-for-0 (Prune)
            for i in list(selected):
                redundant = True
                for ts_idx in block_tsub_indices[i]:
                    if counts[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    selected.remove(i)
                    sel_set.remove(i)
                    for ts_idx in block_tsub_indices[i]:
                        counts[ts_idx] -= 1
                    improved = True
                    break
                    
        return selected

    # Strategy:
    # 1. Deterministic Greedy
    # 2. Prune
    # 3. Local Search
    # 4. Random Greedy (if time)
    
    best_blocks = []
    best_count = float('inf')
    
    # Deterministic
    g1 = greedy_cover(randomize=False)
    p1 = prune(g1)
    l1 = local_search(p1)
    if len(l1) < best_count:
        best_count = len(l1)
        best_blocks = l1
        
    # Random Greedy
    if time.monotonic() < deadline - 30:
        for _ in range(5):
            if time.monotonic() > deadline - 20:
                break
            g_r = greedy_cover(randomize=True)
            p_r = prune(g_r)
            l_r = local_search(p_r)
            if len(l_r) < best_count:
                best_count = len(l_r)
                best_blocks = l_r
                
    return [list(all_kblocks[i]) for i in best_blocks]
