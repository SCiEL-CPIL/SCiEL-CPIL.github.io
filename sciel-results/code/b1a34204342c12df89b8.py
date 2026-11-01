import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}

    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, find the indices of the t-subsets it covers
    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)

    for i, block in enumerate(all_blocks):
        count = 0
        for sub in itertools.combinations(block, t):
            BlockTsets[i, count] = tset_to_idx[sub]
            count += 1

    # Precompute for each t-set, the list of blocks that cover it (inverted index)
    # This is crucial for efficient swap search.
    # tset_to_blocks[ts] = list of block indices covering ts
    # We can build this efficiently by iterating over blocks.
    tset_blocks = [[] for _ in range(num_tsets)]
    for b_idx in range(num_blocks):
        for ts in BlockTsets[b_idx]:
            tset_blocks[ts].append(b_idx)

    # 2. Polish function: Remove redundant blocks
    def polish_blocks(blocks):
        if not blocks:
            return blocks

        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1

        i = len(blocks) - 1
        while i >= 0:
            b = blocks[i]
            can_remove = True
            for ts in BlockTsets[b]:
                if cover_count[ts] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts in BlockTsets[b]:
                    cover_count[ts] -= 1
                blocks.pop(i)
            else:
                i -= 1

        return blocks

    # 3. 1-for-1 Swap Local Search
    # Try to replace a block in the current solution with another block
    # that covers more "essential" t-sets, potentially allowing removal of another block.
    # Simpler heuristic: Try to swap block A with block B if B covers all t-sets
    # that are uniquely covered by A, AND B covers at least one t-set that is
    # currently under-covered or helps in removing another block.
    # Actually, a simpler and effective move:
    # If we can remove block A and add block B, and the new solution is still valid,
    # and it allows us to remove another block C, then we reduce count by 1.
    # This is complex. Let's stick to a simpler "improvement" swap:
    # We want to minimize the number of blocks.
    # A swap (remove A, add B) is neutral in count.
    # But if after swap, we can polish and remove C, then we improved.

    def local_search_swap(blocks):
        if not blocks:
            return blocks

        # Convert to set for fast lookup
        block_set = set(blocks)
        n_blocks = len(blocks)

        # Build cover count
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle blocks to avoid bias
            np.random.shuffle(blocks)

            for i, b_out in enumerate(blocks):
                if time.time() > deadline - 5:
                    break

                # Identify t-sets uniquely covered by b_out
                # These are "essential" for b_out.
                essential_ts = []
                for ts in BlockTsets[b_out]:
                    if cover_count[ts] == 1:
                        essential_ts.append(ts)

                if not essential_ts:
                    # b_out is redundant? Polish should have caught it, but maybe not.
                    # Try to remove it.
                    # If we remove b_out, all its t-sets must still be covered.
                    # Since cover_count[ts] >= 1 for all ts in b_out, and if any is 1, it's essential.
                    # If no essential, it means all ts in b_out have cover_count >= 2.
                    # So we can remove b_out.
                    # But polish_blocks does this. Let's rely on polish.
                    continue

                # We need to find a block b_in that covers all essential_ts.
                # Candidate b_in must cover the intersection of tset_blocks[ts] for ts in essential_ts.
                # To be efficient, pick the rarest essential ts (smallest tset_blocks list)
                rarest_ts = min(essential_ts, key=lambda ts: len(tset_blocks[ts]))
                candidates = tset_blocks[rarest_ts]

                # Filter candidates: must not be b_out itself, and must cover all other essential_ts
                # Also, to potentially improve, b_in should be different from b_out.
                # If b_in == b_out, no change.

                # We also want b_in to be "useful".
                # A simple heuristic: try a few random candidates.
                # To allow improvement, we check if adding b_in and removing b_out
                # allows us to remove another block.
                # This is expensive.
                # Let's just try to find a b_in that covers essential_ts.
                # If found, we swap. Then we polish.
                # If polish reduces count, we improved.

                # Randomly sample candidates to avoid checking all
                num_candidates = len(candidates)
                if num_candidates == 0:
                    continue

                # Shuffle candidates
                cand_indices = np.random.permutation(num_candidates)
                # Try up to 10 random candidates
                for idx in cand_indices[:10]:
                    b_in = candidates[idx]
                    if b_in == b_out:
                        continue

                    # Check if b_in covers all essential_ts
                    # We already know it covers rarest_ts.
                    # Check others.
                    valid = True
                    for ts in essential_ts:
                        if ts != rarest_ts:
                            # Check if ts is in BlockTsets[b_in]
                            # This is O(t_subsets_per_block).
                            # We can speed this up with a set or boolean array.
                            # For now, linear search in BlockTsets[b_in] is slow.
                            # Let's precompute a set for b_in? No, too expensive per step.
                            # Let's use the inverted index check:
                            # Is b_in in tset_blocks[ts]?
                            # tset_blocks[ts] is a list. Checking membership is O(len).
                            # This might be slow.
                            pass

                    # Let's optimize the check.
                    # Instead of checking each ts, we can just try the swap and see if it's valid.
                    # But we need to ensure essential_ts are covered.
                    # If b_in doesn't cover an essential ts, the swap is invalid.

                    # Let's do a quick check using a temporary cover count update?
                    # No, let's just check membership.
                    # Optimization: Precompute a boolean matrix? Too big.
                    # Let's just iterate.

                    # Check if b_in covers all essential_ts
                    # We can create a set of tsets covered by b_in for fast lookup?
                    # b_in_tsets = set(BlockTsets[b_in])
                    # if all(ts in b_in_tsets for ts in essential_ts):
                    #     # Valid swap candidate
                    #     pass

                    # This set creation is O(126).
                    # essential_ts is small.
                    # Let's do it.
                    b_in_tsets = set(BlockTsets[b_in])
                    if all(ts in b_in_tsets for ts in essential_ts):
                        # Valid swap: replace b_out with b_in
                        # Update blocks list
                        blocks[i] = b_in

                        # Update cover_count
                        # Decrement for b_out
                        for ts in BlockTsets[b_out]:
                            cover_count[ts] -= 1
                        # Increment for b_in
                        for ts in BlockTsets[b_in]:
                            cover_count[ts] += 1

                        # Try to polish to see if we can remove another block
                        # We can't easily polish just one block in the list without re-building cover_count?
                        # Actually, we have the updated cover_count.
                        # Let's see if any block is now redundant.
                        # This is expensive to check all.
                        # Let's just mark improved and let the outer loop handle it?
                        # Or, we can check if b_in allows removal of some other block.
                        # Let's just set improved = True and break to restart the scan.
                        # This is a "move".
                        improved = True
                        break

                if improved:
                    break

        return blocks

    # 4. Greedy Search with Rarest-First Heuristic and Restart
    best_blocks = None
    best_len = float('inf')

    # Precompute for each t-set, the number of blocks covering it (global)
    # This is not needed for greedy, but might be useful for tie-breaking.

    while time.time() < deadline:
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []

        # Rarest-First Greedy
        # At each step, find the uncovered t-set with the fewest covering blocks.
        # Then pick a block that covers it, maximizing the number of other uncovered t-sets covered.

        # To make this efficient, we need to track uncovered t-sets.
        # num_tsets is 11628.
        # We can maintain a list of uncovered t-sets.

        # Initially, all are uncovered.
        # We can precompute for each t-set, the list of blocks that cover it (tset_blocks).
        # We need to find the t-set with min len(tset_blocks[ts]) among uncovered.
        # This is O(num_tsets) per step.
        # Greedy takes ~100-200 steps. 100 * 11628 = 1.1M ops. Fast.

        # We also need to update "uncovered" status.
        # And we need to calculate "gain" for candidate blocks.
        # Gain of block b = number of uncovered t-sets in BlockTsets[b].

        # Algorithm:
        # 1. Find rarest uncovered t-set ts*.
        # 2. Candidates = blocks that cover ts*.
        # 3. For each candidate, calculate gain (uncovered t-sets covered).
        # 4. Pick best candidate (max gain, tie-break random).
        # 5. Add to blocks, mark its t-sets as covered.

        # Optimization:
        # Instead of scanning all num_tsets to find rarest, we can maintain a priority queue?
        # But the "rarity" (number of covering blocks) is static?
        # No, the "uncovered" status changes, but the set of blocks covering a t-set is static.
        # The "rarity" in terms of options is static: len(tset_blocks[ts]).
        # So the rarest uncovered t-set is just the one with min len(tset_blocks[ts]) among uncovered.
        # We can pre-sort t-sets by len(tset_blocks[ts]).
        # Then iterate through this sorted list to find the first uncovered one.

        tset_order = np.argsort([len(tset_blocks[ts]) for ts in range(num_tsets)])

        while uncovered.any():
            # Find rarest uncovered t-set
            rarest_ts = -1
            for ts in tset_order:
                if uncovered[ts]:
                    rarest_ts = ts
                    break

            if rarest_ts == -1:
                break

            # Get candidates
            candidates = tset_blocks[rarest_ts]

            # Calculate gain for each candidate
            # Gain = sum(uncovered[BlockTsets[cand]])
            # Vectorized:
            # gains = uncovered[BlockTsets[candidates]].sum(axis=1)
            # But candidates is a list of indices.
            # BlockTsets[candidates] is a 2D array.
            # This might be slow if candidates is large.
            # Average len(tset_blocks[ts]) is (num_blocks * t_subsets_per_block) / num_tsets
            # = (75582 * 126) / 11628 ≈ 819.
            # So 819 candidates.
            # 819 * 126 = 103k ops per step.
            # 100 steps * 103k = 10M ops. Fast.

            cand_indices = np.array(candidates)
            gains = uncovered[BlockTsets[cand_indices]].sum(axis=1)

            max_gain = gains.max()
            if max_gain == 0:
                break

            best_cands = cand_indices[gains == max_gain]
            best_block_idx = best_cands[np.random.randint(len(best_cands))]

            blocks.append(best_block_idx)
            uncovered[BlockTsets[best_block_idx]] = False

        # Polish
        blocks = polish_blocks(blocks)

        # Local Search
        blocks = local_search_swap(blocks)
        blocks = polish_blocks(blocks)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]

    if best_blocks is None:
        return []

    result = [list(all_blocks[b]) for b in best_blocks]
    return result
