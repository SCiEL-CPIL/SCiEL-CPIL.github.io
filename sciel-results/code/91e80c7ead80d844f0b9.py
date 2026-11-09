"""
Strategy: Bitmask-based Vectorized Greedy Set Cover + Aggressive Local Search.

1. Precompute all C(19,4) = 3876 t-subsets and C(19,7) = 50388 k-blocks.
2. Build a coverage matrix (block -> t-subset) using bitmasks and NumPy.
3. Greedy selection: pick the block covering the most uncovered t-subsets.
4. Local search:
   - Removal: iteratively drop non-critical blocks.
   - 1-for-1 Swap: replace a critical block with a non-selected block that covers
     all its unique t-subsets, enabling further removals.
5. Multiple randomized restarts with a time budget to escape local optima.
"""

import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # ------------------------------------------------------------------
    # 1. Precompute all t-subsets and k-blocks with bitmasks
    # ------------------------------------------------------------------
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)

    t_masks = np.array([0] * num_t, dtype=np.int64)
    for i, sub in enumerate(t_subsets):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_masks[i] = mask

    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    k_masks = np.array([0] * num_k, dtype=np.int64)
    for i, block in enumerate(k_blocks):
        mask = 0
        for p in block:
            mask |= (1 << p)
        k_masks[i] = mask

    # ------------------------------------------------------------------
    # 2. Build coverage matrix (block -> t-subset) in chunks
    # ------------------------------------------------------------------
    chunk_size = 2048
    num_chunks = (num_k + chunk_size - 1) // chunk_size
    cover_list = []
    for c in range(num_chunks):
        s = c * chunk_size
        e = min(s + chunk_size, num_k)
        k_chunk = k_masks[s:e]
        # (chunk, num_t) boolean: block covers t-subset
        check = (t_masks[None, :] & ~k_chunk[:, None]) == 0
        cover_list.append(check.astype(np.uint8))

    # ------------------------------------------------------------------
    # 3. Single-run solver: greedy + local search
    # ------------------------------------------------------------------
    def _solve_once(rng, time_left_fn):
        def elapsed():
            return time.time() - start_time

        uncovered = np.ones(num_t, dtype=np.uint8)
        indices_selected = []

        # Greedy phase
        while np.sum(uncovered) > 0:
            if elapsed() > deadline - 15:
                break

            best_idx = -1
            best_gain = 0

            for c in range(num_chunks):
                gains = cover_list[c].dot(uncovered)
                mg = int(np.max(gains))
                if mg > best_gain:
                    best_gain = mg
                    best_idx = c * chunk_size + int(np.argmax(gains))

            if best_gain == 0:
                break

            indices_selected.append(best_idx)

            c = best_idx // chunk_size
            li = best_idx % chunk_size
            uncovered[cover_list[c][li] == 1] = 0

        # Build counts
        counts = np.zeros(num_t, dtype=np.int32)
        for idx in indices_selected:
            c = idx // chunk_size
            li = idx % chunk_size
            counts += cover_list[c][li]

        # Removal phase (repeated until no change)
        for _ in range(20):
            changed = False
            new_indices = []
            for idx in indices_selected:
                c = idx // chunk_size
                li = idx % chunk_size
                row = cover_list[c][li]
                if not np.any((row == 1) & (counts == 1)):
                    counts -= row
                    changed = True
                else:
                    new_indices.append(idx)
            indices_selected = new_indices
            if not changed:
                break

        # 1-for-1 swap phase
        if elapsed() < deadline - 5:
            selected_set = set(indices_selected)
            non_selected = [i for i in range(num_k) if i not in selected_set]

            max_swap_iters = 8000
            iters = 0
            while iters < max_swap_iters and elapsed() < deadline - 3:
                iters += 1
                if not indices_selected:
                    break

                s_pos = rng.randint(0, len(indices_selected) - 1)
                s_idx = indices_selected[s_pos]
                c_s = s_idx // chunk_size
                li_s = s_idx % chunk_size
                row_s = cover_list[c_s][li_s]

                unique_mask = (row_s == 1) & (counts == 1)
                n_unique = int(np.sum(unique_mask))

                if n_unique == 0:
                    # Removable
                    counts -= row_s
                    indices_selected.pop(s_pos)
                    continue

                # Sample candidate non-selected blocks
                sample_size = min(80, len(non_selected))
                candidates = rng.sample(non_selected, sample_size)

                found = False
                for n_idx in candidates:
                    c_n = n_idx // chunk_size
                    li_n = n_idx % chunk_size
                    row_n = cover_list[c_n][li_n]

                    # Check: row_n covers all unique subsets of s
                    if np.all(row_n[unique_mask]):
                        # Tentative swap
                        new_counts = counts - row_s + row_n
                        if np.any(new_counts == 0):
                            continue
                        # Accept
                        counts = new_counts
                        indices_selected[s_pos] = n_idx
                        found = True
                        break

                if not found:
                    continue

        # Final removal pass
        for _ in range(10):
            changed = False
            new_indices = []
            for idx in indices_selected:
                c = idx // chunk_size
                li = idx % chunk_size
                row = cover_list[c][li]
                if not np.any((row == 1) & (counts == 1)):
                    counts -= row
                    changed = True
                else:
                    new_indices.append(idx)
            indices_selected = new_indices
            if not changed:
                break

        # Validate
        final_counts = np.zeros(num_t, dtype=np.int32)
        for idx in indices_selected:
            c = idx // chunk_size
            li = idx % chunk_size
            final_counts += cover_list[c][li]

        if np.any(final_counts == 0):
            # Invalid; return empty to signal failure
            return []

        return [list(k_blocks[i]) for i in indices_selected]

    # ------------------------------------------------------------------
    # 4. Multi-restart loop
    # ------------------------------------------------------------------
    best_blocks = []
    best_len = float('inf')

    # First run: deterministic seed
    rng = random.Random(42)
    result = _solve_once(rng, lambda: time.time() - start_time)
    if result and len(result) < best_len:
        best_blocks = result
        best_len = len(result)

    # Subsequent runs with different seeds while time permits
    seed = 0
    while elapsed() < deadline - 20:
        seed += 1
        rng = random.Random(seed * 7919 + 13)
        result = _solve_once(rng, lambda: time.time() - start_time)
        if result and len(result) < best_len:
            best_blocks = result
            best_len = len(result)

    if not best_blocks:
        # Fallback: run one more deterministic greedy without local search
        rng = random.Random(0)
        uncovered = np.ones(num_t, dtype=np.uint8)
        indices_selected = []
        while np.sum(uncovered) > 0:
            best_idx = -1
            best_gain = 0
            for c in range(num_chunks):
                gains = cover_list[c].dot(uncovered)
                mg = int(np.max(gains))
                if mg > best_gain:
                    best_gain = mg
                    best_idx = c * chunk_size + int(np.argmax(gains))
            if best_gain == 0:
                break
            indices_selected.append(best_idx)
            c = best_idx // chunk_size
            li = best_idx % chunk_size
            uncovered[cover_list[c][li] == 1] = 0
        best_blocks = [list(k_blocks[i]) for i in indices_selected]

    return best_blocks
