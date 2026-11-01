import itertools
import random
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute(v, k, t):
    """Precompute t-sets, blocks, and coverage mappings (vectorized where possible)."""
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # block_tsets: list of arrays of t-set indices covered by each block
    block_tsets = []
    for block in all_blocks:
        idxs = [tset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsets.append(np.array(idxs, dtype=np.int32))

    # tset_to_blocks: for each t-set, list of block indices covering it
    tset_to_blocks = defaultdict(list)
    for b_idx, arr in enumerate(block_tsets):
        for ts_idx in arr:
            tset_to_blocks[ts_idx].append(b_idx)

    return all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks


def _coverage_count(blocks_idx, block_tsets, n_tsets):
    """Return coverage count array and number of uncovered t-sets."""
    covered = np.zeros(n_tsets, dtype=np.int32)
    for b_idx in blocks_idx:
        covered[block_tsets[b_idx]] += 1
    uncovered = int(np.count_nonzero(covered == 0))
    return covered, uncovered


def _greedy_construct(v, k, t, n_tsets, block_tsets, tset_to_blocks, rng, time_limit, weighted=False):
    """
    Randomized greedy set cover.
    If weighted=True, use a soft weighting to diversify choices.
    Returns list of block indices.
    """
    deadline = time.time() + time_limit
    uncovered = np.ones(n_tsets, dtype=bool)
    blocks_chosen = []

    while np.any(uncovered) and time.time() < deadline:
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break

        # Pick a random uncovered t-set (or weighted pick)
        if weighted and len(uncovered_indices) > 1:
            # Prefer t-sets with fewer covering blocks (harder)
            weights = []
            for idx in uncovered_indices:
                weights.append(1.0 / (len(tset_to_blocks[idx]) + 1))
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_idx = uncovered_indices[0]
            for idx, w in zip(uncovered_indices, weights):
                cum += w
                if r <= cum:
                    chosen_idx = idx
                    break
        else:
            chosen_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]

        # Find best block covering this t-set
        best_block = -1
        best_gain = -1
        candidates = tset_to_blocks[chosen_idx]
        if weighted:
            # Randomly sample a subset of candidates to speed up
            if len(candidates) > 30:
                candidates = rng.sample(candidates, 30)

        for b_idx in candidates:
            gain = int(np.count_nonzero(uncovered[block_tsets[b_idx]]))
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx

        if best_block == -1 or best_gain <= 0:
            break

        blocks_chosen.append(best_block)
        uncovered[block_tsets[best_block]] = False

    return blocks_chosen


def _remove_redundant(blocks_idx, block_tsets, n_tsets, time_limit):
    """Remove blocks that don't uniquely cover any t-set."""
    deadline = time.time() + time_limit
    current = list(blocks_idx)
    if not current:
        return current

    covered, _ = _coverage_count(current, block_tsets, n_tsets)

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            b_idx = current[i]
            # Check if any t-set is only covered by this block
            if np.any(covered[block_tsets[b_idx]] == 1):
                continue
            # Redundant: remove
            del current[i]
            covered[block_tsets[b_idx]] -= 1
            changed = True
            break

    return current


def _try_swap_reduce(blocks_idx, block_tsets, n_tsets, all_blocks, n_blocks, tset_to_blocks, rng, time_limit):
    """
    Try to replace one block with another to maintain coverage while enabling
    a subsequent removal, or directly reduce size via 2-for-1 exchange.
    Returns possibly smaller list.
    """
    deadline = time.time() + time_limit
    current = list(blocks_idx)
    if len(current) <= 1:
        return current

    covered, _ = _coverage_count(current, block_tsets, n_tsets)

    # Try 1-for-1 swaps that allow a removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        M = len(current)
        # Shuffle positions for diversity
        positions = list(range(M))
        rng.shuffle(positions)

        for pos in positions:
            if time.time() > deadline:
                break
            old_b = current[pos]

            # Sample candidate new blocks
            n_samples = min(50, n_blocks)
            candidates = rng.sample(range(n_blocks), n_samples)

            for new_b in candidates:
                if new_b == old_b:
                    continue

                # Tentatively swap
                covered[block_tsets[old_b]] -= 1
                covered[block_tsets[new_b]] += 1

                # Check if coverage is still valid
                if np.any(covered == 0):
                    covered[block_tsets[new_b]] -= 1
                    covered[block_tsets[old_b]] += 1
                    continue

                # Now try to remove a redundant block
                for j in range(M):
                    if j == pos:
                        continue
                    b_j = current[j]
                    if np.any(covered[block_tsets[b_j]] == 1):
                        continue
                    # Remove b_j
                    covered[block_tsets[b_j]] -= 1
                    if np.any(covered == 0):
                        covered[block_tsets[b_j]] += 1
                        continue
                    # Success: swap + remove
                    current[pos] = new_b
                    del current[j]
                    improved = True
                    break
                if improved:
                    break

                # If no removal, revert swap
                covered[block_tsets[new_b]] -= 1
                covered[block_tsets[old_b]] += 1

    return current


def _alns_search(blocks_idx, block_tsets, n_tsets, all_blocks, n_blocks, tset_to_blocks, rng, time_limit):
    """
    Adaptive Large Neighborhood Search:
    - Destroy a random subset of blocks
    - Repair by adding blocks via weighted greedy to restore coverage
    - Accept if size doesn't increase (or with SA-like probability)
    """
    deadline = time.time() + time_limit
    current = list(blocks_idx)
    best = current[:]
    best_size = len(current)

    covered, _ = _coverage_count(current, block_tsets, n_tsets)
    current_size = len(current)

    temperature = 2.0
    cooling = 0.999
    min_destroy = 2
    max_destroy = 5

    while time.time() < deadline and temperature > 0.01:
        M = len(current)
        if M <= min_destroy:
            break

        # Destroy: remove random blocks
        n_destroy = rng.randint(min_destroy, min(max_destroy, M // 3))
        destroy_positions = rng.sample(range(M), n_destroy)
        destroyed = [current[i] for i in destroy_positions]

        # Remove them
        new_current = [b for i, b in enumerate(current) if i not in set(destroy_positions)]
        new_covered = np.zeros(n_tsets, dtype=np.int32)
        for b_idx in new_current:
            new_covered[block_tsets[b_idx]] += 1

        # Repair: add blocks to cover all t-sets
        uncovered = (new_covered == 0)
        repair_blocks = []
        while np.any(uncovered) and time.time() < deadline:
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            # Pick a random uncovered t-set
            ts_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            candidates = tset_to_blocks[ts_idx]
            if len(candidates) > 20:
                candidates = rng.sample(candidates, 20)

            best_b = -1
            best_gain = -1
            for b_idx in candidates:
                gain = int(np.count_nonzero(uncovered[block_tsets[b_idx]]))
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx

            if best_b == -1 or best_gain <= 0:
                break

            repair_blocks.append(best_b)
            new_covered[block_tsets[best_b]] += 1
            uncovered[block_tsets[best_b]] = False

        # Check if repair succeeded
        if not np.all(new_covered > 0):
            # Failed repair, revert
            temperature *= cooling
            continue

        candidate = new_current + repair_blocks
        candidate_size = len(candidate)
        delta = candidate_size - current_size

        # Acceptance
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            current = candidate
            current_size = candidate_size
            covered = new_covered
            if current_size < best_size:
                best_size = current_size
                best = current[:]

        temperature *= cooling

    return best


def _structural_seed(v, k, t, n_tsets, block_tsets, tset_to_blocks, all_blocks, n_blocks, rng, time_limit):
    """
    Build a structural seed: use cyclic shifts of a base block plus greedy fill.
    For v=24, k=7, we can try base blocks from a small pool and shift mod 24.
    """
    deadline = time.time() + time_limit
    best_blocks = None
    best_size = float('inf')

    # Try a few base blocks and their cyclic shifts
    base_candidates = [
        list(range(7)),
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 12],
        [0, 1, 2, 4, 5, 8, 12],
        [0, 1, 2, 3, 4, 6, 12],
        [0, 1, 2, 3, 5, 6, 12],
        [0, 1, 2, 4, 5, 6, 12],
    ]

    # Also add some random base blocks
    for _ in range(5):
        base_candidates.append(sorted(rng.sample(range(v), k)))

    for base in base_candidates:
        if time.time() > deadline - 2.0:
            break

        # Generate cyclic shifts
        blocks = []
        seen = set()
        for shift in range(v):
            shifted = sorted([(x + shift) % v for x in base])
            key = tuple(shifted)
            if key not in seen:
                seen.add(key)
                blocks.append(shifted)

        # Convert to block indices
        block_set = set()
        block_indices = []
        for b in blocks:
            # Find index in all_blocks
            # Use a dict for speed
            if b not in block_set:
                block_set.add(b)
                # Find index
                # For efficiency, precompute a mapping
                # But all_blocks is a list of tuples
                # We'll search
                found = False
                for idx, ab in enumerate(all_blocks):
                    if list(ab) == b:
                        block_indices.append(idx)
                        found = True
                        break
                if not found:
                    # Should not happen
                    pass

        # Greedy fill to cover remaining
        covered, _ = _coverage_count(block_indices, block_tsets, n_tsets)
        uncovered = (covered == 0)
        fill_blocks = _greedy_construct(v, k, t, n_tsets, block_tsets, tset_to_blocks, rng,
                                         max(0.0, deadline - time.time() - 5.0), weighted=True)

        candidate = block_indices + fill_blocks
        candidate = _remove_redundant(candidate, block_tsets, n_tsets, 1.0)

        if len(candidate) < best_size:
            best_size = len(candidate)
            best_blocks = candidate[:]

    # If no structural seed worked, use pure greedy
    if best_blocks is None:
        best_blocks = _greedy_construct(v, k, t, n_tsets, block_tsets, tset_to_blocks, rng,
                                         max(1.0, deadline - time.time()), weighted=False)
        best_blocks = _remove_redundant(best_blocks, block_tsets, n_tsets, 1.0)

    return best_blocks


def _verify_covering(blocks_idx, block_tsets, n_tsets):
    """Verify that the blocks cover all t-sets."""
    covered, _ = _coverage_count(blocks_idx, block_tsets, n_tsets)
    return bool(np.all(covered > 0))


def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0  # Safety margin before 150s limit

    # Precompute
    all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks = _precompute(v, k, t)

    rng = random.Random(42)

    # Phase 1: Structural seed
    seed_blocks = _structural_seed(v, k, t, n_tsets, block_tsets, tset_to_blocks,
                                    all_blocks, n_blocks, rng, 15.0)
    best_blocks = seed_blocks[:]
    best_size = len(seed_blocks)

    # Phase 2: ALNS search
    remaining = total_budget - (time.time() - start_time) - 10.0
    if remaining > 5.0:
        alns_result = _alns_search(seed_blocks, block_tsets, n_tsets, all_blocks, n_blocks,
                                     tset_to_blocks, rng, remaining)
        alns_result = _remove_redundant(alns_result, block_tsets, n_tsets, 2.0)
        if len(alns_result) < best_size:
            best_blocks = alns_result[:]
            best_size = len(alns_result)

    # Phase 3: Swap-reduce local search
    remaining = total_budget - (time.time() - start_time) - 5.0
    if remaining > 3.0:
        swap_result = _try_swap_reduce(best_blocks, block_tsets, n_tsets, all_blocks, n_blocks,
                                        tset_to_blocks, rng, remaining)
        swap_result = _remove_redundant(swap_result, block_tsets, n_tsets, 1.0)
        if len(swap_result) < best_size:
            best_blocks = swap_result[:]
            best_size = len(swap_result)

    # Phase 4: Final redundancy removal
    best_blocks = _remove_redundant(best_blocks, block_tsets, n_tsets, 2.0)

    # Verify
    if not _verify_covering(best_blocks, block_tsets, n_tsets):
        # Fallback: pure greedy
        rng_fallback = random.Random(789)
        best_blocks = _greedy_construct(v, k, t, n_tsets, block_tsets, tset_to_blocks,
                                         rng_fallback, 10.0, weighted=False)
        best_blocks = _remove_redundant(best_blocks, block_tsets, n_tsets, 5.0)

    # Convert to list of lists
    result = [list(all_blocks[idx]) for idx in best_blocks]

    return result

# EVOLVE-BLOCK-END
