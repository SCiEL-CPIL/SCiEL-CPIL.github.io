"""
Covering design C(19,8,5) using vectorized greedy set cover with random restarts
and aggressive redundancy removal.

Key improvements over the seed:
1. Fully vectorized gain computation (single numpy matrix-vector product per step).
2. Multiple random restarts with different tie-breaking to explore diverse solutions.
3. Aggressive iterative redundancy removal after each greedy pass.
4. Keep the best (smallest) solution found across all restarts within the time budget.
"""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # For each k-subset, precompute the list of t-subset indices it covers
    # C(8,5) = 56 t-subsets per block
    block_tsubs_list = []
    for block in all_ksubs:
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            tsubs_in_block.append(tsub_index[ts])
        block_tsubs_list.append(tsubs_in_block)

    # Store as a list of arrays (variable length not an issue here since all are 56)
    block_tsubs_arr = np.array(block_tsubs_list, dtype=np.int32)  # shape (n_ksubs, 56)

    # Also build a mapping from k-subset tuple to its index for fast lookup
    ksub_to_idx = {block: i for i, block in enumerate(all_ksubs)}

    def remove_redundant(blocks, cover_count):
        """Iteratively remove blocks whose t-subsets are all covered by >=2 blocks."""
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                b_idx = ksub_to_idx[b]
                tsubs = block_tsubs_arr[b_idx]
                if np.all(cover_count[tsubs] >= 2):
                    cover_count[tsubs] -= 1
                    blocks.pop(i)
                    changed = True
        return blocks

    def greedy_pass(rng):
        """One greedy pass with random tie-breaking. Returns list of blocks."""
        covered = np.zeros(n_tsubs, dtype=bool)
        blocks = []
        used_indices = set()

        while not covered.all():
            if time.time() > deadline:
                return None  # timeout

            # Vectorized gain: for each block, count uncovered t-subsets it covers
            uncov = (~covered).astype(np.int32)
            # gains[i] = sum of uncov over block_tsubs_arr[i]
            # Use matrix multiplication: block_tsubs_arr is (n_ksubs, 56), uncov is (n_tsubs,)
            # We need: for each row, sum of uncov at the indices in that row.
            # This is a gather + sum operation.
            # Efficient way: use np.take and then sum along axis 1
            gains = np.take(uncov, block_tsubs_arr, axis=0).sum(axis=1)

            # Exclude already-used blocks
            for idx in used_indices:
                gains[idx] = 0

            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break  # no improvement possible

            # Random tie-breaking: pick randomly among blocks with max gain
            max_gain = gains[best_idx]
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_idx = int(rng.choice(candidates))

            blocks.append(all_ksubs[best_idx])
            used_indices.add(best_idx)
            covered[block_tsubs_arr[best_idx]] = True

        return blocks

    def compute_cover_count(blocks):
        """Compute coverage count array for a list of blocks."""
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            b_idx = ksub_to_idx[b]
            cover_count[block_tsubs_arr[b_idx]] += 1
        return cover_count

    def is_valid(blocks):
        """Check if blocks form a valid covering."""
        if not blocks:
            return n_tsubs == 0
        cover_count = compute_cover_count(blocks)
        return bool(np.all(cover_count >= 1))

    # Try multiple greedy passes with different random seeds
    best_blocks = None
    best_len = float('inf')

    rng = random.Random(42)

    # First, do a deterministic greedy pass
    blocks = greedy_pass(rng)
    if blocks is not None and len(blocks) < best_len:
        # Remove redundant blocks
        cover_count = compute_cover_count(blocks)
        blocks = remove_redundant(blocks, cover_count)
        if is_valid(blocks):
            best_blocks = blocks
            best_len = len(blocks)

    # Now do randomized restarts
    num_restarts = 0
    while time.time() < deadline - 10:  # leave 10s for final cleanup
        num_restarts += 1
        # New random state for this pass
        local_rng = random.Random(num_restarts * 7919 + 13)
        blocks = greedy_pass(local_rng)
        if blocks is None:
            continue
        if len(blocks) >= best_len:
            continue  # not better, skip

        # Remove redundant blocks
        cover_count = compute_cover_count(blocks)
        blocks = remove_redundant(blocks, cover_count)

        if is_valid(blocks):
            best_blocks = blocks
            best_len = len(blocks)

    # Final cleanup: try more aggressive removal on the best solution
    if best_blocks is not None:
        cover_count = compute_cover_count(best_blocks)
        # Try multiple passes of removal
        for _ in range(5):
            prev_len = len(best_blocks)
            best_blocks = remove_redundant(best_blocks, cover_count)
            if len(best_blocks) == prev_len:
                break

    if best_blocks is None:
        # Fallback: should not happen, but return a trivial valid covering
        # This would be very large, but ensures validity
        best_blocks = []
        covered = np.zeros(n_tsubs, dtype=bool)
        while not covered.all():
            for i, block in enumerate(all_ksubs):
                if i in ksub_to_idx:
                    tsubs = block_tsubs_arr[i]
                    if not covered[tsubs].all():
                        best_blocks.append(block)
                        covered[tsubs] = True
                        break
            if covered.all():
                break

    return [list(b) for b in best_blocks]
