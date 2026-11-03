"""
Combined strategy: Bitset-based precomputation for fast gain updates,
vectorized scoring with bias for greedy construction, and local search
(removal + swaps) for refinement using bitset intersections for speed.
"""
import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

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
    # tsub_mask[b] has bits for tsubs covered by block b
    # tsub_blocks_mask[ts] has bits for blocks covering ts
    tsub_mask = [0] * n_blocks
    tsub_blocks_mask = [0] * n_tsubsets

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask

    full_tsub_mask = (1 << n_tsubsets) - 1
    all_blocks_mask = (1 << n_blocks) - 1

    def greedy_cover(seed):
        """
        Greedy set cover using a max-heap with vectorized gain updates.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0

        # Initial gains: number of tsubs covered by each block
        gains = [0] * n_blocks
        for b_idx in range(n_blocks):
            gains[b_idx] = tsub_mask[b_idx].bit_count()

        heap = [(-gains[b_idx], b_idx) for b_idx in range(n_blocks)]
        heapq.heapify(heap)

        while uncovered_mask:
            if time.time() > deadline - 10.0:
                break

            best_block = -1
            best_gain = 0

            # Pop stale entries
            while heap:
                neg_g, b_idx = heapq.heappop(heap)
                g = -neg_g
                if g == gains[b_idx] and not (selected_mask & (1 << b_idx)):
                    best_block = b_idx
                    best_gain = g
                    break
                # Stale or already selected, discard

            if best_block == -1:
                break

            selected.append(best_block)
            selected_mask |= (1 << best_block)
            gains[best_block] = -1  # Mark as selected

            # Update gains for blocks covering newly covered tsubs
            new_covered = tsub_mask[best_block] & uncovered_mask
            uncovered_mask &= ~tsub_mask[best_block]

            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                # For every block covering this ts_idx, decrease its gain
                cm = tsub_blocks_mask[ts_idx]
                while cm:
                    lsb2 = cm & -cm
                    b_idx2 = lsb2.bit_length() - 1
                    if not (selected_mask & (1 << b_idx2)):
                        gains[b_idx2] -= 1
                        heapq.heappush(heap, (-gains[b_idx2], b_idx2))
                    cm ^= lsb2
                m ^= lsb

        return selected

    def local_search(blocks, seed):
        """
        Local search with swaps, deletions, and 2-for-2 exchanges.
        Uses bitset intersections for fast candidate generation.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)
        
        # cover_count[ts_idx] = number of blocks in block_set covering ts_idx
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        def try_remove(block_set, cover_count, b):
            """Try to remove block b if redundant. Returns True if removed."""
            for ts_idx in block_tsubs[b]:
                if cover_count[ts_idx] == 1:
                    return False
            block_set.remove(b)
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] -= 1
            return True

        def try_1for1_swap(block_set, cover_count, b, rng):
            """Try to swap block b with another block. Returns True if swapped."""
            unique_tsubs = []
            for ts_idx in block_tsubs[b]:
                if cover_count[ts_idx] == 1:
                    unique_tsubs.append(ts_idx)

            if not unique_tsubs:
                return try_remove(block_set, cover_count, b)

            # Find candidates using bitset intersection
            # Start with the most constrained t-subset (smallest block count)
            # to minimize the initial mask size, though bitset AND is fast regardless.
            # We just AND all masks corresponding to unique_tsubs.
            cand_mask = all_blocks_mask
            for ts_idx in unique_tsubs:
                cand_mask &= tsub_blocks_mask[ts_idx]
            
            # Exclude the current block b
            cand_mask &= ~(1 << b)
            
            # Exclude blocks already in the set? 
            # If we swap b with r, r must NOT be in block_set.
            # We can mask out existing blocks.
            existing_mask = 0
            for x in block_set:
                existing_mask |= (1 << x)
            cand_mask &= ~existing_mask
            
            if not cand_mask:
                return False

            # Collect candidates
            candidates = []
            m = cand_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                candidates.append(idx)
                m ^= lsb

            if not candidates:
                return False

            rng.shuffle(candidates)
            
            # Try candidates
            for r in candidates[:50]: # Limit to avoid long stalls if many candidates
                # Perform swap
                block_set.remove(b)
                for ts_idx in block_tsubs[b]:
                    cover_count[ts_idx] -= 1
                
                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1
                
                return True

            return False

        def try_2for2_exchange(block_set, cover_count, b1, b2, rng):
            """
            Try to replace blocks b1 and b2 with two other blocks.
            """
            # Find tsubs uniquely covered by b1 or b2
            unique_tsubs = []
            seen = set()
            for ts_idx in block_tsubs[b1]:
                if cover_count[ts_idx] == 1:
                    if ts_idx not in seen:
                        unique_tsubs.append(ts_idx)
                        seen.add(ts_idx)
            for ts_idx in block_tsubs[b2]:
                if cover_count[ts_idx] == 1:
                    if ts_idx not in seen:
                        unique_tsubs.append(ts_idx)
                        seen.add(ts_idx)

            if not unique_tsubs:
                # Both are redundant
                try_remove(block_set, cover_count, b1)
                try_remove(block_set, cover_count, b2)
                return True

            # Find candidate blocks that cover ALL unique_tsubs
            # A single block r1 must cover all unique_tsubs? 
            # No, the PAIR (r1, r2) must cover them.
            # This is a hitting set problem for the unique_tsubs with 2 blocks.
            # Heuristic: Find one block r1 that covers the "rarest" unique t-subset.
            # Then find r2 that covers the rest.
            
            # Sort unique_tsubs by rarity (number of blocks covering them)
            unique_tsubs.sort(key=lambda x: tsub_blocks_mask[x].bit_count())
            
            # The most constrained t-subset must be covered by r1 or r2.
            # Let's try to find r1 from the blocks covering the most constrained ts.
            ts_first = unique_tsubs[0]
            r1_candidates_mask = tsub_blocks_mask[ts_first]
            r1_candidates_mask &= ~(1 << b1)
            r1_candidates_mask &= ~(1 << b2)
            
            # Also exclude existing blocks? No, r1 and r2 are NEW blocks.
            existing_mask = 0
            for x in block_set:
                existing_mask |= (1 << x)
            r1_candidates_mask &= ~existing_mask
            
            if not r1_candidates_mask:
                return False
                
            r1_cands = []
            m = r1_candidates_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                r1_cands.append(idx)
                m ^= lsb
                
            if not r1_cands:
                return False
                
            rng.shuffle(r1_cands)
            
            # Try a limited number of r1 candidates
            for r1 in r1_cands[:20]:
                # Check which unique_tsubs are covered by r1
                r1_mask = tsub_mask[r1]
                remaining_unique = []
                for ts_idx in unique_tsubs:
                    if not (r1_mask & (1 << ts_idx)):
                        remaining_unique.append(ts_idx)
                        
                if not remaining_unique:
                    # r1 covers all unique tsubs. We need r2 to be any valid block 
                    # that doesn't break coverage of other tsubs?
                    # Actually, if r1 covers all unique tsubs, we just need r2 to be 
                    # a block that doesn't cause any tsub to drop to 0.
                    # Tsubs with count 1 are the unique ones. They are covered by r1.
                    # Tsubs with count 2 covered by b1/b2 will drop to 1 if b1/b2 removed.
                    # So any r2 is safe as long as it doesn't overlap in a way that 
                    # is problematic? No, adding r2 only increases counts.
                    # So we just need to pick ANY r2 that is not in block_set and not r1.
                    # But we must ensure the resulting set is valid.
                    # Since we only add blocks, and the "dangerous" tsubs (count 1) are covered by r1,
                    # the swap is safe for any r2.
                    # However, we want to minimize blocks. We are doing 2-for-2, so size stays same.
                    # We just need a valid r2.
                    r2_candidates_mask = all_blocks_mask
                    r2_candidates_mask &= ~(1 << b1)
                    r2_candidates_mask &= ~(1 << b2)
                    r2_candidates_mask &= ~(1 << r1)
                    r2_candidates_mask &= ~existing_mask
                    
                    if r2_candidates_mask:
                        lsb = r2_candidates_mask & -r2_candidates_mask
                        r2 = lsb.bit_length() - 1
                        
                        # Perform swap
                        block_set.remove(b1)
                        for ts_idx in block_tsubs[b1]:
                            cover_count[ts_idx] -= 1
                        block_set.remove(b2)
                        for ts_idx in block_tsubs[b2]:
                            cover_count[ts_idx] -= 1
                        block_set.add(r1)
                        for ts_idx in block_tsubs[r1]:
                            cover_count[ts_idx] += 1
                        block_set.add(r2)
                        for ts_idx in block_tsubs[r2]:
                            cover_count[ts_idx] += 1
                        return True
                    continue
                
                # r1 does not cover all. Find r2 that covers the remaining_unique.
                r2_candidates_mask = all_blocks_mask
                for ts_idx in remaining_unique:
                    r2_candidates_mask &= tsub_blocks_mask[ts_idx]
                
                r2_candidates_mask &= ~(1 << b1)
                r2_candidates_mask &= ~(1 << b2)
                r2_candidates_mask &= ~(1 << r1)
                r2_candidates_mask &= ~existing_mask
                
                if not r2_candidates_mask:
                    continue
                
                lsb = r2_candidates_mask & -r2_candidates_mask
                r2 = lsb.bit_length() - 1
                
                # Perform swap
                block_set.remove(b1)
                for ts_idx in block_tsubs[b1]:
                    cover_count[ts_idx] -= 1
                block_set.remove(b2)
                for ts_idx in block_tsubs[b2]:
                    cover_count[ts_idx] -= 1
                block_set.add(r1)
                for ts_idx in block_tsubs[r1]:
                    cover_count[ts_idx] += 1
                block_set.add(r2)
                for ts_idx in block_tsubs[r2]:
                    cover_count[ts_idx] += 1
                return True

            return False

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in list(block_set):
                if try_remove(block_set, cover_count, b):
                    to_remove.append(b)

            if to_remove:
                improved = True
                continue

            # Phase 2: Try 1-for-1 swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 5.0:
                    break
                if b not in block_set:
                    continue
                if try_1for1_swap(block_set, cover_count, b, rng):
                    improved = True
                    break

            if improved:
                continue

            # Phase 3: Try 2-for-2 exchanges
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for i in range(len(current_blocks)):
                if time.time() > deadline - 5.0:
                    break
                b1 = current_blocks[i]
                if b1 not in block_set:
                    continue
                for j in range(i + 1, len(current_blocks)):
                    b2 = current_blocks[j]
                    if b2 not in block_set:
                        continue
                    if try_2for2_exchange(block_set, cover_count, b1, b2, rng):
                        improved = True
                        break
                if improved:
                    break

        return list(block_set)

    # Main execution
    best_blocks = None
    best_len = float('inf')

    # Run multiple greedy restarts
    num_restarts = 10

    for i in range(num_restarts):
        if time.time() > deadline - 40:  # Keep 40s for local search
            break

        seed = i * 12345 + 7
        blocks = greedy_cover(seed)

        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    if not best_blocks:
        return []

    # Apply local search to the best solution
    if time.time() < deadline - 2:
        best_blocks = local_search(best_blocks, 42)

        # If local search improved, try another round with a different seed
        if time.time() < deadline - 2 and len(best_blocks) < best_len:
            best_len = len(best_blocks)
            best_blocks = local_search(best_blocks, 137)

    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[bi]) for bi in best_blocks]

    return final_blocks
