import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112  # Safety margin below 120s kill

    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)

    # Precompute block info
    block_masks = []
    block_ts_indices = []
    blocks_containing_subset = [[] for _ in range(num_t_subsets)]
    subset_frequencies = [0] * num_t_subsets

    for block in k_subsets:
        ts_indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            ts_indices.append(idx)
            blocks_containing_subset[idx].append(len(block_masks))
            subset_frequencies[idx] += 1
        block_ts_indices.append(ts_indices)

        mask = 0
        for idx in ts_indices:
            mask |= (1 << idx)
        block_masks.append(mask)

    full_mask = (1 << num_t_subsets) - 1

    # Weights for weighted greedy: rare subsets get higher weight
    weights = [1.0 / f for f in subset_frequencies]

    def weighted_greedy(seed):
        """Weighted greedy with random tie-breaking."""
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        # Shuffle block indices for tie-breaking diversity
        order = list(range(num_blocks))
        rng.shuffle(order)

        while uncovered:
            best_gain = -1.0
            best_block_idx = -1

            for i in order:
                # Compute weighted gain
                gain = 0.0
                # Use bit operations for speed
                u = uncovered & block_masks[i]
                if u == 0:
                    continue
                # Sum weights for set bits
                # This is the slow part. Optimization: precompute weighted sums?
                # Or just use bit_count if weights were uniform.
                # Since weights vary, we must iterate bits.
                # To speed up, we can skip if u has few bits?
                # Let's try a hybrid: if gain so far + max_possible < best_gain, skip?
                # max_possible is sum of all weights in block.
                # Precompute block_max_weight_sum?
                # For now, just iterate.
                # Optimization: use a lookup table for small integers?
                # Or just accept the cost.
                # Actually, iterating 35 bits is fast.
                for bit in block_ts_indices[i]:
                    if (u >> bit) & 1:
                        gain += weights[bit]

                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                elif gain == best_gain and gain > 0:
                    # Random tie-breaking: since order is shuffled, first max wins?
                    # No, we iterate all. If we want random tie-breaking,
                    # we can replace with probability 0.5?
                    # Or just keep the first one found in the shuffled order.
                    # Since we iterate 0..N in shuffled order, the first max encountered
                    # is effectively random among ties.
                    # So we do NOT replace on tie.
                    pass

            if best_gain <= 0:
                # Should not happen if uncovered is not empty and coverage exists
                break

            selected.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]

        return selected

    def prune_blocks(selected_indices):
        """Remove redundant blocks from a list of indices."""
        if not selected_indices:
            return selected_indices

        # Build counts
        counts = [0] * num_t_subsets
        for idx in selected_indices:
            for ts_idx in block_ts_indices[idx]:
                counts[ts_idx] += 1

        active = [True] * len(selected_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_indices)):
                if not active[i]:
                    continue
                redundant = True
                for ts_idx in block_ts_indices[selected_indices[i]]:
                    if counts[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    changed = True
                    for ts_idx in block_ts_indices[selected_indices[i]]:
                        counts[ts_idx] -= 1

        return [idx for i, idx in enumerate(selected_indices) if active[i]]

    def local_search(selected_indices, seed):
        """Local search: try to swap blocks to reduce count or improve coverage."""
        rng = random.Random(seed)
        current = list(selected_indices)
        if not current:
            return current

        # Build counts
        counts = [0] * num_t_subsets
        for idx in current:
            for ts_idx in block_ts_indices[idx]:
                counts[ts_idx] += 1

        # Helper to check if removing a block leaves coverage
        def is_redundant(pos):
            idx = current[pos]
            for ts_idx in block_ts_indices[idx]:
                if counts[ts_idx] == 1:
                    return False
            return True

        # Helper to remove
        def remove(pos):
            idx = current[pos]
            for ts_idx in block_ts_indices[idx]:
                counts[ts_idx] -= 1
            current.pop(pos)

        # Helper to add
        def add(idx):
            current.append(idx)
            for ts_idx in block_ts_indices[idx]:
                counts[ts_idx] += 1

        best_len = len(current)
        no_improve = 0
        max_no_improve = 50

        while time.time() < deadline:
            # Prune
            changed = True
            while changed:
                changed = False
                for i in range(len(current)):
                    if is_redundant(i):
                        remove(i)
                        changed = True
                        break # Restart scan after removal

            if len(current) < best_len:
                best_len = len(current)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve > max_no_improve:
                # Perturbation: add a random block
                if current:
                    rand_idx = rng.randint(0, num_blocks - 1)
                    if rand_idx not in current:
                        add(rand_idx)
                no_improve = 0
                continue

            # Try a swap: remove one, add one that covers its unique subsets
            if not current:
                break

            i = rng.randint(0, len(current) - 1)
            idx_i = current[i]

            # Find unique subsets covered by idx_i
            unique_bits = []
            for ts_idx in block_ts_indices[idx_i]:
                if counts[ts_idx] == 1:
                    unique_bits.append(ts_idx)

            if not unique_bits:
                # Redundant, should have been pruned
                remove(i)
                continue

            # Find a replacement block that covers all unique_bits
            # Intersect candidate sets
            cands = None
            for b in unique_bits:
                lst = blocks_containing_subset[b]
                if cands is None:
                    cands = set(lst)
                else:
                    cands.intersection_update(lst)
                if not cands:
                    break

            if cands:
                cands.discard(idx_i)
                if cands:
                    j = rng.choice(list(cands))
                    if j not in current:
                        remove(i)
                        add(j)
                        # Prune again after swap
                        changed = True
                        while changed:
                            changed = False
                            for k in range(len(current)):
                                if is_redundant(k):
                                    remove(k)
                                    changed = True
                                    break

        return current

    best_blocks = None
    best_count = float('inf')

    # Run multiple weighted greedy passes with different seeds
    # Then local search on the best ones
    # Time budget: ~100s for search, ~10s for precomp (already done), ~10s margin

    # Estimate time per greedy run.
    # num_blocks ~ 50k.
    # Inner loop: 50k * (bit ops + weight sum).
    # Weight sum: ~35 iterations.
    # Total ~ 50k * 35 * 150 blocks ~ 250M ops?
    # This might be too slow in Python.
    # Let's optimize the gain calculation.
    # Precompute for each block a list of (bit, weight) pairs?
    # Or use numpy?
    # Let's try a simpler greedy first: max bit_count.
    # Then weighted if time permits.

    def simple_greedy(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        order = list(range(num_blocks))
        rng.shuffle(order)

        while uncovered:
            best_gain = -1
            best_block_idx = -1

            for i in order:
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i

            if best_gain <= 0:
                break

            selected.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]

        return selected

    # Try simple greedy first as it's faster
    # Run 10 simple greedy passes
    # Then local search on the best 3

    candidates = []
    for i in range(10):
        if time.time() > deadline - 30:
            break
        sel = simple_greedy(i)
        pruned = prune_blocks(sel)
        candidates.append(pruned)

    # Sort by length
    candidates.sort(key=len)

    # Local search on top 5
    for i in range(min(5, len(candidates))):
        if time.time() > deadline - 10:
            break
        ls_res = local_search(candidates[i], 100 + i)
        if len(ls_res) < best_count:
            best_count = len(ls_res)
            best_blocks = [k_subsets[idx] for idx in ls_res]

    # If no improvement, try weighted greedy?
    # Or just return best found.

    if best_blocks is None:
        # Fallback: return first candidate
        if candidates:
            best_blocks = [k_subsets[idx] for idx in candidates[0]]
        else:
            return []

    return best_blocks
