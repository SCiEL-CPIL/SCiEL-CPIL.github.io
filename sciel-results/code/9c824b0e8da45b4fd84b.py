import itertools
import numpy as np
import time
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    # Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)

    # Precompute block_triples and triple_to_blocks
    block_triples = []
    triple_to_blocks = [[] for _ in range(n_triples)]

    for i, block in enumerate(all_blocks):
        b_triples = []
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            b_triples.append(t_idx)
            triple_to_blocks[t_idx].append(i)
        block_triples.append(b_triples)

    # Convert triple_to_blocks to numpy arrays
    triple_to_blocks_np = [np.array(b, dtype=np.int32) if b else np.array([], dtype=np.int32) for b in triple_to_blocks]

    # For faster access, create a flat array of block-triple memberships
    # and offsets for each block
    block_triple_counts = np.array([len(bt) for bt in block_triples], dtype=np.int32)
    block_triple_offsets = np.zeros(n_blocks + 1, dtype=np.int32)
    np.cumsum(block_triple_counts, out=block_triple_offsets[1:])
    total_block_triple_entries = block_triple_offsets[-1]

    # Build flat array
    block_triples_flat = np.zeros(total_block_triple_entries, dtype=np.int32)
    for i, bt in enumerate(block_triples):
        if bt:
            block_triples_flat[block_triple_offsets[i]:block_triple_offsets[i + 1]] = bt

    # Build a fast "which blocks contain triple t_idx" lookup
    # Also build triple coverage counts for pruning
    # Precompute for each triple: list of block indices
    # triple_to_blocks_np already has this

    def greedy_cover(random_seed, use_weighted=False):
        rng = np.random.RandomState(random_seed)

        # Initial gain for each block is k_t
        gains = np.full(n_blocks, k_t, dtype=np.float64)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_mask = np.zeros(n_blocks, dtype=bool)

        selected_indices = []
        remaining = n_triples

        # For weighted mode, use a secondary criterion: prefer blocks that
        # cover triples with fewer available blocks (more constrained triples)
        if use_weighted:
            # triple_weight[t] = 1 / len(triple_to_blocks_np[t]) -- higher for rarer triples
            tw = np.ones(n_triples, dtype=np.float64)
            for t_idx in range(n_triples):
                cnt = len(triple_to_blocks_np[t_idx])
                if cnt > 0:
                    tw[t_idx] = 1.0 / cnt
                else:
                    tw[t_idx] = 0.0

        while remaining > 0:
            if time.time() > deadline - 2:
                break

            # Find max gain among unselected blocks
            gains[selected_mask] = -1.0

            if use_weighted:
                # Weighted score: gain * (1 + 0.3 * max_triple_weight_in_block)
                # Approximate: just use gain for now, but tie-break by
                # number of "rare" triples covered
                max_gain = np.max(gains)
                if max_gain <= 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                if len(candidates) > 1:
                    # Score each candidate by sum of triple weights for newly-covered triples
                    best_score = -1
                    best_idx = candidates[0]
                    for c in candidates:
                        score = 0.0
                        for pos in range(block_triple_offsets[c], block_triple_offsets[c + 1]):
                            t_idx = block_triples_flat[pos]
                            if uncovered_mask[t_idx]:
                                score += tw[t_idx]
                        if score > best_score:
                            best_score = score
                            best_idx = c
                else:
                    best_idx = candidates[0]
            else:
                max_gain = np.max(gains)
                if max_gain <= 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = candidates[rng.randint(len(candidates))]

            # Select block
            selected_mask[best_idx] = True
            selected_indices.append(best_idx)

            # Update gains for affected blocks
            newly_covered = []
            for pos in range(block_triple_offsets[best_idx], block_triple_offsets[best_idx + 1]):
                t_idx = block_triples_flat[pos]
                if uncovered_mask[t_idx]:
                    uncovered_mask[t_idx] = False
                    newly_covered.append(t_idx)
                    remaining -= 1

            if newly_covered:
                for t_idx in newly_covered:
                    affected = triple_to_blocks_np[t_idx]
                    if len(affected) > 0:
                        gains[affected] -= 1.0

        return selected_indices

    def prune_blocks(selected_indices):
        """Remove redundant blocks from selected_indices."""
        if not selected_indices:
            return selected_indices

        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in selected_indices:
            for pos in range(block_triple_offsets[b_idx], block_triple_offsets[b_idx + 1]):
                t_idx = block_triples_flat[pos]
                coverage_count[t_idx] += 1

        active_blocks = list(selected_indices)

        changed = True
        passes = 0
        max_passes = 20
        rng = np.random.RandomState(42)

        while changed and passes < max_passes:
            if time.time() > deadline - 3:
                break
            changed = False
            passes += 1
            rng.shuffle(active_blocks)

            new_active_blocks = []
            for b_idx in active_blocks:
                # Check if any triple in this block has coverage_count == 1
                is_redundant = True
                for pos in range(block_triple_offsets[b_idx], block_triple_offsets[b_idx + 1]):
                    t_idx = block_triples_flat[pos]
                    if coverage_count[t_idx] == 1:
                        is_redundant = False
                        break

                if is_redundant:
                    for pos in range(block_triple_offsets[b_idx], block_triple_offsets[b_idx + 1]):
                        t_idx = block_triples_flat[pos]
                        coverage_count[t_idx] -= 1
                    changed = True
                else:
                    new_active_blocks.append(b_idx)

            active_blocks = new_active_blocks

        return active_blocks

    def swap_improve(selected_indices, max_iter=200):
        """Try to swap one selected block for a non-selected one if it reduces total."""
        if time.time() > deadline - 3:
            return selected_indices

        # Compute coverage counts
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in selected_indices:
            for pos in range(block_triple_offsets[b_idx], block_triple_offsets[b_idx + 1]):
                t_idx = block_triples_flat[pos]
                coverage_count[t_idx] += 1

        selected_set = set(selected_indices)

        # Find all triples with coverage_count == 1 (critical triples)
        critical_triples = set(np.where(coverage_count == 1)[0].tolist())

        # For each selected block, check if it's the sole cover of any critical triple
        # If a block is not the sole cover of any critical triple, it's a swap candidate (removable)
        # Then try to find a non-selected block that covers all the triples it was needed for

        # Actually, a simpler approach: try removing each block and see if coverage is still valid.
        # If valid, try to re-add the best non-selected block.

        # This is O(n_blocks * n_triples) per iteration, which might be too slow.
        # Instead, do a lighter version: just do the pruning pass again with more passes.

        # For now, just return as-is. The pruning is already done.
        return selected_indices

    # Run multiple strategies
    best_blocks = None
    best_len = float('inf')

    # Strategy 1: Simple greedy with pruning
    for i in range(50):
        if time.time() > deadline - 5:
            break
        sel = greedy_cover(i, use_weighted=False)
        pruned = prune_blocks(sel)
        if len(pruned) < best_len:
            best_len = len(pruned)
            best_blocks = [list(all_blocks[b]) for b in pruned]

    # Strategy 2: Weighted greedy with pruning
    for i in range(30):
        if time.time() > deadline - 5:
            break
        sel = greedy_cover(100 + i, use_weighted=True)
        pruned = prune_blocks(sel)
        if len(pruned) < best_len:
            best_len = len(pruned)
            best_blocks = [list(all_blocks[b]) for b in pruned]

    # Strategy 3: Greedy with random perturbation - start with a few random blocks, then complete
    for i in range(20):
        if time.time() > deadline - 5:
            break
        rng = np.random.RandomState(200 + i)

        # Start with 2-5 random blocks
        n_init = rng.randint(2, 6)
        init_blocks = set(rng.choice(n_blocks, size=n_init, replace=False).tolist())

        # Compute coverage from init blocks
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in init_blocks:
            for pos in range(block_triple_offsets[b_idx], block_triple_offsets[b_idx + 1]):
                t_idx = block_triples_flat[pos]
                coverage_count[t_idx] += 1

        # Find uncovered triples
        uncovered = np.where(coverage_count == 0)[0]
        remaining = len(uncovered)

        # Complete with greedy
        gains = np.zeros(n_blocks, dtype=np.float64)
        for b_idx in range(n_blocks):
            for pos in range(block_triple_offsets[b_idx], block_triple_offsets[b_idx + 1]):
                t_idx = block_triples_flat[pos]
                if coverage_count[t_idx] == 0:
                    gains[b_idx] += 1.0

        # Mark init blocks as selected
        selected_mask = np.zeros(n_blocks, dtype=bool)
        for b_idx in init_blocks:
            selected_mask[b_idx] = True

        selected_indices = list(init_blocks)

        while remaining > 0:
            if time.time() > deadline - 3:
                break
            gains[selected_mask] = -1.0
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            selected_mask[best_idx] = True
            selected_indices.append(best_idx)

            for pos in range(block_triple_offsets[best_idx], block_triple_offsets[best_idx + 1]):
                t_idx = block_triples_flat[pos]
                if coverage_count[t_idx] == 0:
                    coverage_count[t_idx] = 1
                    remaining -= 1
                    affected = triple_to_blocks_np[t_idx]
                    if len(affected) > 0:
                        gains[affected] -= 1.0

        pruned = prune_blocks(selected_indices)
        if len(pruned) < best_len:
            best_len = len(pruned)
            best_blocks = [list(all_blocks[b]) for b in pruned]

    # Fallback: ensure we have a valid result
    if best_blocks is None or best_len == float('inf'):
        sel = greedy_cover(0, use_weighted=False)
        best_blocks = [list(all_blocks[b]) for b in sel]

    return best_blocks
