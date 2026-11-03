import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Data structures
    block_tsubs = [[] for _ in range(n_blocks)]
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    # Bitsets for fast set operations
    tsub_mask = [0] * n_blocks       # tsub_mask[b] has bits for tsubs covered by block b
    tsub_blocks_mask = [0] * n_tsubsets # tsub_blocks_mask[ts] has bits for blocks covering ts
    block_tsub_sets = [None] * n_blocks # Sets for local search intersection
    tsub_block_sets = [None] * n_tsubsets

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        tsub_set = set()
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
            tsub_set.add(idx)
        tsub_mask[b_idx] = mask
        block_tsub_sets[b_idx] = tsub_set

    for i in range(n_tsubsets):
        tsub_block_sets[i] = set()
        m = tsub_blocks_mask[i]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tsub_block_sets[i].add(idx)
            m ^= lsb

    full_tsub_mask = (1 << n_tsubsets) - 1

    def greedy_cover(seed):
        """
        Greedy set cover with random tie-breaking.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0
        
        # Initial gains
        gains = [0] * n_blocks
        for b_idx in range(n_blocks):
            gains[b_idx] = tsub_mask[b_idx].bit_count()
            
        # Use a list for heap, (neg_gain, random_priority, block_idx)
        # random_priority helps break ties randomly
        heap = [(-gains[b_idx], rng.random(), b_idx) for b_idx in range(n_blocks)]
        heapq.heapify(heap)
        
        while uncovered_mask:
            if time.time() > deadline - 10.0:
                break

            best_block = -1
            best_gain = 0

            while heap:
                neg_g, _, b_idx = heapq.heappop(heap)
                g = -neg_g
                if g == gains[b_idx] and not (selected_mask & (1 << b_idx)):
                    best_block = b_idx
                    best_gain = g
                    break

            if best_block == -1:
                break

            selected.append(best_block)
            selected_mask |= (1 << best_block)
            gains[best_block] = -1
            
            new_covered = tsub_mask[best_block] & uncovered_mask
            uncovered_mask &= ~tsub_mask[best_block]
            
            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                cm = tsub_blocks_mask[ts_idx]
                while cm:
                    lsb2 = cm & -cm
                    b_idx2 = lsb2.bit_length() - 1
                    if not (selected_mask & (1 << b_idx2)):
                        gains[b_idx2] -= 1
                        heapq.heappush(heap, (-gains[b_idx2], rng.random(), b_idx2))
                    cm ^= lsb2
                m ^= lsb

        return selected

    def local_search(blocks, seed):
        """
        Local search with 1-for-1 and 2-for-1 swaps.
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
        while improved and time.time() < deadline - 5.0:
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

            # Phase 2: 1-for-1 Swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 5.0:
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

                min_ts = min(unique_tsubs, key=lambda x: len(tsub_block_sets[x]))
                candidates = set(tsub_block_sets[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]

                if not candidates:
                    continue

                candidates.discard(b)
                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                if len(cand_list) > 10:
                    cand_list = cand_list[:10]

                swapped = False
                for r in cand_list:
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
            
            if improved:
                continue

            # Phase 3: 2-for-1 Swaps
            # Identify critical blocks (those with unique tsubs)
            critical_blocks = []
            unique_map = {}
            for b in block_set:
                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                if unique_tsubs:
                    critical_blocks.append(b)
                    unique_map[b] = unique_tsubs
            
            if len(critical_blocks) >= 2:
                # Limit pairs to check to avoid timeout
                # Shuffle to randomize
                critical_blocks_copy = critical_blocks[:]
                rng.shuffle(critical_blocks_copy)
                pairs_checked = 0
                max_pairs = 50
                
                for i in range(len(critical_blocks_copy)):
                    for j in range(i+1, len(critical_blocks_copy)):
                        if pairs_checked >= max_pairs:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        
                        b1 = critical_blocks_copy[i]
                        b2 = critical_blocks_copy[j]
                        
                        u1 = unique_map[b1]
                        u2 = unique_map[b2]
                        
                        # Union of unique tsubs
                        # If the union is large, it's unlikely a single block covers all
                        if len(u1) + len(u2) > 20: 
                            continue
                            
                        # Intersection of candidate sets
                        # Start with the smaller set
                        if len(u1) < len(u2):
                            cand_set = set(tsub_block_sets[u1[0]])
                            for ts in u1[1:]:
                                cand_set &= tsub_block_sets[ts]
                            for ts in u2:
                                cand_set &= tsub_block_sets[ts]
                        else:
                            cand_set = set(tsub_block_sets[u2[0]])
                            for ts in u2[1:]:
                                cand_set &= tsub_block_sets[ts]
                            for ts in u1:
                                cand_set &= tsub_block_sets[ts]
                                
                        cand_set.discard(b1)
                        cand_set.discard(b2)
                        
                        if not cand_set:
                            continue
                            
                        # Pick a candidate not in block_set
                        cand_list = [c for c in cand_set if c not in block_set]
                        if not cand_list:
                            continue
                            
                        rng.shuffle(cand_list)
                        r = cand_list[0]
                        
                        # Perform swap
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
                    if improved:
                        break

        return list(block_set)

    best_solution = None
    best_score = float('inf')

    # Run multiple restarts
    n_restarts = 5
    for s in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
            
        greedy_sol = greedy_cover(s)
        
        # Check if greedy finished (valid)
        # If not, we can try to repair or skip
        # Greedy usually finishes for this size
        
        current_sol = local_search(greedy_sol, s * 10 + 42)
        
        if len(current_sol) < best_score:
            best_score = len(current_sol)
            best_solution = current_sol
            
    if best_solution is None:
        # Fallback: return greedy from last seed
        best_solution = greedy_cover(n_restarts)
        
    # Final validation
    # Ensure all tsubs covered
    cover_count = [0] * n_tsubsets
    for b in best_solution:
        for ts_idx in block_tsubs[b]:
            cover_count[ts_idx] += 1
            
    if any(c == 0 for c in cover_count):
        # Repair: add blocks for missing tsubs
        missing_tsubs = [i for i, c in enumerate(cover_count) if c == 0]
        for ts_idx in missing_tsubs:
            # Find any block covering this tsub
            # tsub_block_sets[ts_idx] is not empty
            candidates = list(tsub_block_sets[ts_idx])
            # Pick one not in solution
            for c in candidates:
                if c not in best_solution:
                    best_solution.append(c)
                    # Update cover counts
                    for ts in block_tsubs[c]:
                        cover_count[ts] += 1
                    break
                    
    return [list(all_blocks[b]) for b in best_solution]
