import time
import random
import numpy as np
from itertools import combinations
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    all_points = list(range(v))
    all_t_subsets = list(combinations(all_points, t))
    n_t = len(all_t_subsets)
    t_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_subsets = list(combinations(all_points, k))
    n_k = len(all_k_subsets)

    c_k_t = comb(k, t)

    # cover_matrix[i] = array of t-indices covered by all_k_subsets[i]
    cover_matrix = np.zeros((n_k, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_k_subsets):
        for j, tri in enumerate(combinations(block, t)):
            cover_matrix[i, j] = t_idx[tri]

    # Map block tuple -> index in all_k_subsets for O(1) lookup
    block_to_idx = {b: i for i, b in enumerate(all_k_subsets)}

    def build_coverage(blocks):
        cov = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            idx = block_to_idx[b]
            cov[cover_matrix[idx]] += 1
        return cov

    def greedy_cover(rng):
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []
        while uncovered.any():
            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            chosen = candidates[rng.randint(0, len(candidates) - 1)]
            blocks.append(all_k_subsets[chosen])
            uncovered[cover_matrix[chosen]] = False
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        cov = build_coverage(blocks)
        changed = True
        while changed:
            changed = False
            to_remove = []
            for i, b in enumerate(blocks):
                idx = block_to_idx[b]
                if not np.any(cov[cover_matrix[idx]] == 1):
                    to_remove.append(i)
            if to_remove:
                remove_set = set(to_remove)
                blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
                cov = build_coverage(blocks)
                changed = True
        return blocks

    def swap_local_search(blocks, time_limit):
        """Try to swap blocks with unique triples to enable further removals."""
        if not blocks:
            return blocks
        cov = build_coverage(blocks)
        current_len = len(blocks)
        blocks_set = set(blocks)
        
        for _ in range(50):  # limited iterations
            if time.time() > time_limit:
                break
            improved = False
            cov = build_coverage(blocks)
            for i, b in enumerate(blocks):
                idx = block_to_idx[b]
                unique_tris = cover_matrix[idx][cov[cover_matrix[idx]] == 1]
                if len(unique_tris) == 0:
                    continue
                # Try to find a replacement block that covers all unique_tris
                # and is not in current set
                for j in range(n_k):
                    if all_k_subsets[j] in blocks_set:
                        continue
                    # Check if block j covers all unique_tris
                    block_j_tris = set(cover_matrix[j])
                    if all(tri in block_j_tris for tri in unique_tris):
                        # Tentatively swap
                        new_blocks = list(blocks)
                        new_blocks[i] = all_k_subsets[j]
                        # Check if this enables removal
                        new_cov = build_coverage(new_blocks)
                        # Try removing redundant
                        to_remove = []
                        for m, nb in enumerate(new_blocks):
                            nidx = block_to_idx[nb]
                            if not np.any(new_cov[cover_matrix[nidx]] == 1):
                                to_remove.append(m)
                        if len(to_remove) > 0:
                            remove_set = set(to_remove)
                            final_blocks = [nb for m, nb in enumerate(new_blocks) if m not in remove_set]
                            if len(final_blocks) < current_len:
                                blocks = final_blocks
                                current_len = len(blocks)
                                blocks_set = set(blocks)
                                improved = True
                                break
                        if improved:
                            break
            if not improved:
                break
        return blocks

    best_blocks = None
    best_len = float('inf')
    restart = 0

    while time.time() < deadline - 5:
        rng = random.Random(restart)
        blocks = greedy_cover(rng)
        blocks = remove_redundant(blocks)
        
        # Swap-based local search if time permits
        if time.time() < deadline - 10:
            blocks = swap_local_search(blocks, deadline - 10)
            blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
        
        restart += 1

    return [list(b) for b in best_blocks] if best_blocks is not None else []
