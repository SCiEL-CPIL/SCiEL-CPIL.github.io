"""Rewritten covering design C(19,7,4) solver with global greedy + MILP polish."""
import itertools
import time
import random
import heapq
from collections import defaultdict

import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    _HAS_MILP = True
except Exception:
    _HAS_MILP = False


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute t-subset indices for each k-block
    # C(19,7) = 50388 blocks, each has C(7,4)=35 t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks_total = len(all_blocks)

    # For each block, precompute which t-subsets it covers
    block_tsubs = []
    for block in all_blocks:
        idxs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_tsubs.append(idxs)

    # Build reverse index: for each t-subset, which blocks cover it
    # Each t-subset is in C(v-t, k-t) = C(15,3) = 455 blocks
    tsub_blocks = [[] for _ in range(n_t)]
    for bi, idxs in enumerate(block_tsubs):
        for idx in idxs:
            tsub_blocks[idx].append(bi)

    best_blocks = None
    best_count = float('inf')

    n_restarts = 30

    for restart in range(n_restarts):
        if not time_ok():
            break

        random.seed(12345 + restart * 7919)

        covered = [False] * n_t
        covered_count = 0
        selected_blocks = []
        # Track how many selected blocks cover each t-subset
        tsub_count = [0] * n_t

        # Global heap-based greedy: at each step pick the block with max marginal gain
        # Use a lazy heap (push all blocks initially, update as coverage changes)
        # For efficiency, we'll use a priority queue with lazy deletion

        # Initialize heap: (-gain, block_index)
        # Initial gain of each block = number of t-subsets it covers (all uncovered)
        # = len(block_tsubs[bi]) = C(k,t) = 35 for all blocks initially
        # So all have same initial gain; use random tie-breaking

        # We'll use a different approach: maintain for each block its current marginal gain
        # and use a max-heap. When we add a block, we need to update gains of other blocks.
        # This is expensive. Instead, use a batch approach:
        # - Compute marginal gain for all blocks, pick best, add it
        # - Repeat until all covered
        # But that's O(n_blocks * n_t) per step which is too slow.

        # Better: use the heap with lazy updates
        # For each t-subset, track how many selected blocks cover it
        # Marginal gain of block b = number of t-subsets in b that have tsub_count == 0

        # Let's use a simpler but effective approach:
        # Greedy with periodic full recomputation

        while covered_count < n_t and time_ok():
            # Compute marginal gain for each block
            # For speed, only consider blocks that could help
            best_bi = -1
            best_gain = 0

            # Sample-based approach for speed: evaluate a random subset of blocks
            # plus some targeted ones
            if len(selected_blocks) < 20:
                # Early: evaluate all blocks
                for bi in range(n_blocks_total):
                    gain = 0
                    for idx in block_tsubs[bi]:
                        if tsub_count[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_bi = bi
            else:
                # Later: sample blocks + blocks covering rare t-subsets
                # Find most constrained uncovered t-subsets
                uncovered = [i for i in range(n_t) if tsub_count[i] == 0]
                if not uncovered:
                    break

                # Pick a few random blocks and blocks covering rare uncovered t-subsets
                candidate_set = set()
                # Random sample
                for _ in range(200):
                    candidate_set.add(random.randint(0, n_blocks_total - 1))
                # Targeted: for a few uncovered t-subsets, add their covering blocks
                sample_uncovered = random.sample(uncovered, min(10, len(uncovered)))
                for idx in sample_uncovered:
                    for bi in tsub_blocks[idx]:
                        candidate_set.add(bi)

                for bi in candidate_set:
                    gain = 0
                    for idx in block_tsubs[bi]:
                        if tsub_count[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_bi = bi

            if best_bi == -1 or best_gain == 0:
                # Fallback: pick a random block
                best_bi = random.randint(0, n_blocks_total - 1)

            # Add the block
            selected_blocks.append(best_bi)
            for idx in block_tsubs[best_bi]:
                tsub_count[idx] += 1
                if tsub_count[idx] == 1:
                    covered_count += 1

        if covered_count < n_t:
            # Shouldn't happen with full block set, but just in case
            continue

        # Deletion phase: remove redundant blocks
        if time_ok():
            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(selected_blocks) - 1, -1, -1):
                    if not time_ok():
                        break
                    bi = selected_blocks[i]
                    removable = True
                    for idx in block_tsubs[bi]:
                        if tsub_count[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        for idx in block_tsubs[bi]:
                            tsub_count[idx] -= 1
                        selected_blocks.pop(i)
                        changed = True
                        break

        # Local search: try to replace a block with a better one
        if time_ok():
            for _ in range(200):
                if not time_ok():
                    break
                idx = random.randint(0, len(selected_blocks) - 1)
                old_bi = selected_blocks[idx]
                unique = [i for i in block_tsubs[old_bi] if tsub_count[i] == 1]
                if not unique:
                    # Redundant, remove
                    for i in block_tsubs[old_bi]:
                        tsub_count[i] -= 1
                    selected_blocks.pop(idx)
                    continue
                # Try to find a replacement block covering all unique t-subsets
                # The replacement must contain all points from unique t-subsets
                union_pts = set()
                for i in unique:
                    union_pts.update(all_t_subsets[i])
                if len(union_pts) > k:
                    continue
                # Find blocks that contain all points in union_pts
                found = False
                for bi in range(n_blocks_total):
                    if bi == old_bi:
                        continue
                    # Check if block bi contains all points in union_pts
                    if all(pt in all_blocks[bi] for pt in union_pts):
                        # Check if it covers all unique t-subsets
                        valid = all(
                            all(pt in all_blocks[bi] for pt in all_t_subsets[i])
                            for i in unique
                        )
                        if valid:
                            # Replace
                            for i in block_tsubs[old_bi]:
                                tsub_count[i] -= 1
                            for i in block_tsubs[bi]:
                                tsub_count[i] += 1
                            selected_blocks[idx] = bi
                            found = True
                            break
                if not found and len(union_pts) == k:
                    # The union itself is a block
                    union_tuple = tuple(sorted(union_pts))
                    if union_tuple in all_blocks:
                        bi = all_blocks.index(union_tuple)
                        if bi != old_bi:
                            for i in block_tsubs[old_bi]:
                                tsub_count[i] -= 1
                            for i in block_tsubs[bi]:
                                tsub_count[i] += 1
                            selected_blocks[idx] = bi

        if len(selected_blocks) < best_count:
            best_count = len(selected_blocks)
            best_blocks = [all_blocks[bi] for bi in selected_blocks]

        if best_count <= 145:
            break

    # MILP polish
    if _HAS_MILP and best_blocks is not None and time_ok():
        best_blocks = _milp_polish(v, k, t, best_blocks, t_sub_to_idx,
                                   all_t_subsets, block_tsubs, deadline)

    if best_blocks is None:
        # Fallback: return all blocks (shouldn't happen)
        best_blocks = [list(b) for b in all_blocks]

    return [list(b) for b in best_blocks]


def _milp_polish(v, k, t, blocks, t_sub_to_idx, all_t_subsets, block_tsubs, deadline):
    n_t = len(all_t_subsets)
    n_blocks = len(blocks)
    if n_blocks == 0:
        return blocks

    # Build candidate pool
    tsub_block_count = [0] * n_t
    for b in blocks:
        for idx in block_tsubs(b):
            tsub_block_count[idx] += 1

    scores = []
    for i, b in enumerate(blocks):
        s = 0
        for idx in block_tsubs(b):
            if tsub_block_count[idx] == 1:
                s += 1
        scores.append(s)

    order = sorted(range(n_blocks), key=lambda i: -scores[i])
    top = order[:min(n_blocks, 300)]
    extra = random.sample(range(n_blocks), min(n_blocks, 50))
    pool = list(set(top) | set(extra))
    pool = sorted(pool)
    m = len(pool)

    from scipy.sparse import lil_matrix
    A = lil_matrix((n_t, m), dtype=np.int8)
    for j, bi in enumerate(pool):
        for idx in block_tsubs(blocks[bi]):
            A[idx, j] = 1
    A = A.tocsr()

    c = np.ones(m)
    constraint = LinearConstraint(A, 1, None)
    bounds = Bounds(0, 1)

    try:
        res = milp(c, integrality=np.ones(m), constraints=constraint,
                   bounds=bounds, options={'time_limit': 40, 'mip_rel_gap': 0.0,
                                           'mip_absgap': 0.0, 'verbose': 0})
    except Exception:
        return blocks

    if res.x is not None:
        sel = [pool[j] for j in range(m) if res.x[j] > 0.5]
        new_blocks = [blocks[i] for i in sel]
        if len(new_blocks) < len(blocks):
            cov = [False] * n_t
            for b in new_blocks:
                for idx in block_tsubs(b):
                    cov[idx] = True
            if all(cov):
                return new_blocks

    return blocks
