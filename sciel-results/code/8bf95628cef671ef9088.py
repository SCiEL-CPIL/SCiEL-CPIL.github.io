import itertools
import random
import time
import numpy as np


def _local_search_polish(blocks, block_tsubs_map, tsub_blocks_map, counts, num_tsubs, deadline, rng):
    """Local search: 1-for-1 swaps and 2-for-2 swaps to reduce block count."""
    current_blocks = list(blocks)
    current_counts = counts.copy()
    current_set = set(current_blocks)
    improved = True

    while improved:
        if time.monotonic() > deadline:
            break
        improved = False

        # Pass 1: Remove redundant blocks (all tsubs have count > 1)
        to_remove = []
        for b_idx in current_blocks:
            tsubs = block_tsubs_map[b_idx]
            if np.all(current_counts[tsubs] > 1):
                to_remove.append(b_idx)
        if to_remove:
            for b_idx in to_remove:
                current_counts[block_tsubs_map[b_idx]] -= 1
                current_blocks.remove(b_idx)
                current_set.discard(b_idx)
            improved = True
            continue

        # Pass 2: 1-for-1 swap
        for b_idx in list(current_blocks):
            if time.monotonic() > deadline:
                break
            tsubs_removed = block_tsubs_map[b_idx]
            would_uncover = tsubs_removed[current_counts[tsubs_removed] == 1]

            if len(would_uncover) == 0:
                current_counts[tsubs_removed] -= 1
                current_blocks.remove(b_idx)
                current_set.discard(b_idx)
                improved = True
                break

            # Find replacement covering all would_uncover
            candidate_set = None
            for ts_idx in would_uncover:
                cands = set(tsub_blocks_map[ts_idx])
                if candidate_set is None:
                    candidate_set = cands
                else:
                    candidate_set &= cands
                if not candidate_set:
                    break

            if candidate_set is None or not candidate_set:
                continue
            candidate_set.discard(b_idx)
            candidates = [c for c in candidate_set if c not in current_set]
            if not candidates:
                continue

            best_repl = -1
            best_score = -1
            for c_idx in candidates:
                c_tsubs = block_tsubs_map[c_idx]
                score = np.sum(current_counts[c_tsubs] == 1)
                if score > best_score:
                    best_score = score
                    best_repl = c_idx

            if best_repl != -1 and best_score > 0:
                current_counts[tsubs_removed] -= 1
                current_counts[block_tsubs_map[best_repl]] += 1
                current_blocks.remove(b_idx)
                current_set.discard(b_idx)
                current_blocks.append(best_repl)
                current_set.add(best_repl)
                improved = True
                break

        if improved or time.monotonic() > deadline:
            continue

        # Pass 3: 2-for-2 swap (remove 2, add 2)
        n = len(current_blocks)
        if n < 4:
            break
        # Sample pairs to limit computation
        max_pairs = min(200, n * (n - 1) // 2)
        pairs = list(itertools.combinations(range(n), 2))
        if len(pairs) > max_pairs:
            pairs = rng.sample(pairs, max_pairs)

        for i, j in pairs:
            if time.monotonic() > deadline:
                break
            b1, b2 = current_blocks[i], current_blocks[j]
            tsubs1 = block_tsubs_map[b1]
            tsubs2 = block_tsubs_map[b2]

            # Simulate removal
            temp_counts = current_counts.copy()
            temp_counts[tsubs1] -= 1
            temp_counts[tsubs2] -= 1

            # Tsubs that would become uncovered
            would_uncover1 = tsubs1[temp_counts[tsubs1] == 0]
            would_uncover2 = tsubs2[temp_counts[tsubs2] == 0]
            all_uncover = np.concatenate([would_uncover1, would_uncover2])
            if len(all_uncover) == 0:
                # Both redundant - handled in pass 1, skip
                continue

            # Find 2 replacements covering all_uncover
            # For each tsub in all_uncover, intersect candidate sets
            candidate_set = None
            for ts_idx in all_uncover:
                cands = set(tsub_blocks_map[ts_idx])
                if candidate_set is None:
                    candidate_set = cands
                else:
                    candidate_set &= cands
                if not candidate_set:
                    break

            if candidate_set is None or not candidate_set:
                continue
            candidate_set.discard(b1)
            candidate_set.discard(b2)
            candidates = [c for c in candidate_set if c not in current_set]
            if len(candidates) < 2:
                continue

            # Score pairs of candidates
            best_pair = None
            best_score = -1
            cands_list = list(candidates)
            if len(cands_list) > 30:
                cands_list = rng.sample(cands_list, 30)
            for a, b in itertools.combinations(cands_list, 2):
                ta = block_tsubs_map[a]
                tb = block_tsubs_map[b]
                score = np.sum(temp_counts[ta] == 1) + np.sum(temp_counts[tb] == 1)
                if score > best_score:
                    best_score = score
                    best_pair = (a, b)

            if best_pair is not None and best_score > 0:
                a, b = best_pair
                current_counts[tsubs1] -= 1
                current_counts[tsubs2] -= 1
                current_counts[block_tsubs_map[a]] += 1
                current_counts[block_tsubs_map[b]] += 1
                current_blocks.remove(b1)
                current_blocks.remove(b2)
                current_set.discard(b1)
                current_set.discard(b2)
                current_blocks.append(a)
                current_blocks.append(b)
                current_set.add(a)
                current_set.add(b)
                improved = True
                break

    return current_blocks, current_counts


def _perturb_and_reoptimize(blocks, block_tsubs_map, tsub_blocks_map, counts, num_tsubs, deadline, rng, num_removals=3, num_additions=3):
    """Perturb the solution by removing some blocks and re-adding to cover gaps.
    Returns the perturbed solution as (blocks, counts).
    """
    current_blocks = list(blocks)
    current_counts = counts.copy()
    current_set = set(current_blocks)
    
    # Remove some random blocks
    n_remove = min(num_removals, len(current_blocks))
    if n_remove == 0:
        return current_blocks, current_counts
    
    to_remove = rng.sample(current_blocks, n_remove)
    for b_idx in to_remove:
        tsubs = block_tsubs_map[b_idx]
        current_counts[tsubs] -= 1
        current_blocks.remove(b_idx)
        current_set.discard(b_idx)
    
    # Check for uncovered tsubs
    uncovered = np.where(current_counts == 0)[0]
    if len(uncovered) == 0:
        # No gaps, just return the smaller solution
        return current_blocks, current_counts
    
    # Greedily add blocks to cover uncovered tsubs
    uncovered_mask = np.zeros(num_tsubs, dtype=bool)
    uncovered_mask[uncovered] = True
    num_uncovered = len(uncovered)
    
    additions_made = 0
    while num_uncovered > 0 and additions_made < num_additions * 2:
        if time.monotonic() > deadline:
            break
        
        # Pick a random uncovered tsub
        unc_idx = np.where(uncovered_mask)[0]
        if len(unc_idx) == 0:
            break
        anchor_tsub = unc_idx[rng.randrange(len(unc_idx))]
        
        candidates = tsub_blocks_map[anchor_tsub]
        if not candidates:
            break
        
        # Find best candidate
        best_candidate = -1
        best_gain = -1
        for b_idx in candidates:
            if b_idx in current_set:
                continue
            tsubs = block_tsubs_map[b_idx]
            gain = np.sum(uncovered_mask[tsubs])
            if gain > best_gain:
                best_gain = gain
                best_candidate = b_idx
        
        if best_candidate == -1 or best_gain <= 0:
            break
        
        current_blocks.append(best_candidate)
        current_set.add(best_candidate)
        tsubs = block_tsubs_map[best_candidate]
        current_counts[tsubs] += 1
        newly_covered = tsubs[uncovered_mask[tsubs]]
        uncovered_mask[tsubs] = False
        num_uncovered -= len(newly_covered)
        additions_made += 1
    
    # Remove redundant blocks
    current_blocks, current_counts = _remove_redundant_blocks(current_blocks, block_tsubs_map, current_counts)
    
    return current_blocks, current_counts


def _remove_redundant_blocks(block_indices, block_tsubs_map, counts):
    """Remove blocks that do not uniquely cover any t-subset.
    block_indices: list of block indices
    block_tsubs_map: list of numpy arrays of tsub indices for each block
    counts: numpy array of counts for each tsub
    Returns: (new_block_indices, updated_counts)
    """
    if not block_indices:
        return [], counts

    current_blocks = list(block_indices)
    changed = True
    
    while changed:
        changed = False
        to_remove = set()
        
        for b_idx in current_blocks:
            tsubs = block_tsubs_map[b_idx]
            # Check if all tsubs covered by this block have count > 1
            if np.all(counts[tsubs] > 1):
                to_remove.add(b_idx)
                changed = True
        
        if to_remove:
            for b_idx in to_remove:
                tsubs = block_tsubs_map[b_idx]
                counts[tsubs] -= 1
            
            current_blocks = [b for b in current_blocks if b not in to_remove]
            
    return current_blocks, counts


def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 110.0  # Safety margin

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Map each block to the list of tsub indices it covers
    block_tsubs_map = []
    for b in all_kblocks:
        tsubs = []
        for ts in itertools.combinations(b, t):
            tsubs.append(tsub_to_idx[ts])
        block_tsubs_map.append(np.array(tsubs, dtype=np.int32))
        
    # Map each tsub to the list of block indices that cover it
    tsub_blocks_map = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_tsubs_map):
        for ts_idx in tsubs:
            tsub_blocks_map[ts_idx].append(b_idx)

    best_blocks = None
    best_len = float('inf')

    rng = random.Random(42)

    # Precompute: for each tsub, the number of blocks covering it (degree)
    tsub_degree = np.array([len(c) for c in tsub_blocks_map], dtype=np.int32)

    num_restarts = 15

    for r in range(num_restarts):
        if time.monotonic() > deadline - 15.0:
            break

        counts = np.zeros(num_tsubs, dtype=np.int8)
        selected_blocks = []

        uncovered_mask = np.ones(num_tsubs, dtype=bool)
        num_uncovered = num_tsubs

        # MRV greedy: always pick the uncovered tsub with fewest candidate blocks
        # that haven't been used yet (or fewest total if that's not tracked)
        while num_uncovered > 0:
            if time.monotonic() > deadline - 3.0:
                break

            unc_idx = np.where(uncovered_mask)[0]
            if len(unc_idx) == 0:
                break

            # Pick the uncovered tsub with minimum degree (hardest to cover)
            # Among uncovered, find the one with fewest available blocks
            min_deg = np.inf
            anchor_tsub = -1
            # Sample if too many to check all
            check_idx = unc_idx
            if len(unc_idx) > 200:
                check_idx = rng.sample(unc_idx.tolist(), 200)
            for ts in check_idx:
                deg = tsub_degree[ts]
                if deg < min_deg:
                    min_deg = deg
                    anchor_tsub = ts

            if anchor_tsub == -1:
                break

            candidates = tsub_blocks_map[anchor_tsub]
            if not candidates:
                break

            # Pick the block with maximum gain among candidates
            best_candidate = -1
            best_gain = -1
            for b_idx in candidates:
                tsubs = block_tsubs_map[b_idx]
                gain = np.sum(uncovered_mask[tsubs])
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = b_idx

            if best_candidate == -1 or best_gain <= 0:
                break

            selected_blocks.append(best_candidate)
            tsubs = block_tsubs_map[best_candidate]
            counts[tsubs] += 1
            uncovered_mask[tsubs] = False
            num_uncovered -= best_gain

        current_blocks = selected_blocks

        # Refinement: Remove redundant blocks
        current_blocks, counts = _remove_redundant_blocks(current_blocks, block_tsubs_map, counts)

        # Local search polish
        if time.monotonic() < deadline - 8.0:
            current_blocks, counts = _local_search_polish(
                current_blocks, block_tsubs_map, tsub_blocks_map, counts, num_tsubs,
                deadline - 8.0, rng
            )

        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks[:]

    # ILS: Iterate on the best solution found so far
    if best_blocks is not None and time.monotonic() < deadline - 25.0:
        il_deadline = deadline - 3.0
        current_best = best_blocks[:]
        current_best_counts = np.zeros(num_tsubs, dtype=np.int8)
        for b_idx in current_best:
            current_best_counts[block_tsubs_map[b_idx]] += 1

        while time.monotonic() < il_deadline:
            # Perturb the best solution
            perturbed_blocks, perturbed_counts = _perturb_and_reoptimize(
                current_best, block_tsubs_map, tsub_blocks_map, current_best_counts,
                num_tsubs, il_deadline - 5.0, rng, num_removals=3, num_additions=3
            )

            # Verify validity
            if np.any(perturbed_counts == 0):
                continue

            # Local search polish on perturbed solution
            polished_blocks, polished_counts = _local_search_polish(
                perturbed_blocks, block_tsubs_map, tsub_blocks_map, perturbed_counts,
                num_tsubs, il_deadline - 5.0, rng
            )

            # Verify validity after polish
            if np.any(polished_counts == 0):
                continue

            if len(polished_blocks) < len(current_best):
                current_best = polished_blocks[:]
                current_best_counts = polished_counts.copy()
                if len(current_best) < best_len:
                    best_len = len(current_best)
                    best_blocks = current_best[:]

    # Final Check: Ensure we have a valid solution
    if best_blocks is None:
        if time.monotonic() < deadline - 1.0:
            counts = np.zeros(num_tsubs, dtype=np.int8)
            selected_blocks = []
            uncovered_mask = np.ones(num_tsubs, dtype=bool)
            num_uncovered = num_tsubs

            while num_uncovered > 0:
                if time.monotonic() > deadline - 0.5:
                    break
                unc_idx = np.where(uncovered_mask)[0]
                if len(unc_idx) == 0:
                    break
                anchor_tsub = unc_idx[0]
                candidates = tsub_blocks_map[anchor_tsub]
                if not candidates:
                    break

                best_candidate = -1
                best_gain = -1
                for b_idx in candidates:
                    tsubs = block_tsubs_map[b_idx]
                    gain = np.sum(uncovered_mask[tsubs])
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b_idx

                if best_candidate == -1:
                    break

                selected_blocks.append(best_candidate)
                tsubs = block_tsubs_map[best_candidate]
                counts[tsubs] += 1
                uncovered_mask[tsubs] = False
                num_uncovered -= best_gain

            current_blocks, counts = _remove_redundant_blocks(selected_blocks, block_tsubs_map, counts)
            best_blocks = current_blocks
            best_len = len(best_blocks)

    # Convert indices back to lists of points
    result_blocks = [list(all_kblocks[i]) for i in best_blocks]

    return result_blocks
