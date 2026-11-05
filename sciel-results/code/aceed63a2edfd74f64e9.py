"""Improved covering design C(24,7,3): fast vectorized greedy + spectral redundancy removal + local search."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_id = {ts: i for i, ts in enumerate(all_tsets)}

    # Precompute for each t-set, the list of (t-1)-subsets it contains
    tset_to_subs = {}
    for ts in all_tsets:
        tset_to_subs[ts] = list(itertools.combinations(ts, t - 1))

    # Precompute: for each (t-1)-subset, which t-sets contain it
    sub_to_tsets = {}
    for ts in all_tsets:
        for sub in tset_to_subs[ts]:
            if sub not in sub_to_tsets:
                sub_to_tsets[sub] = []
            sub_to_tsets[sub].append(ts)

    # For each (t-1)-subset, precompute the t-set IDs
    sub_to_tset_ids = {}
    for sub, tsets in sub_to_tsets.items():
        sub_to_tset_ids[sub] = np.array([tset_to_id[ts] for ts in tsets], dtype=np.int32)

    def block_tset_ids(block):
        return [tset_to_id[ts] for ts in itertools.combinations(block, t)]

    def block_tset_ids_array(block):
        return np.array(block_tset_ids(block), dtype=np.int32)

    # Fast greedy: vectorized gain computation
    def greedy_construct(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsets, dtype=bool)
        blocks = []

        while uncovered.any():
            if time.time() > deadline - 60:
                break
            unc_idx = np.where(uncovered)[0]
            start_id = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            start_ts = all_tsets[start_id]

            block = list(start_ts)
            block_set = set(block)

            while len(block) < k:
                candidates = [p for p in range(v) if p not in block_set]
                subs_in_block = list(itertools.combinations(block, t - 1))

                # Vectorized gain: for each candidate, sum uncovered counts over (t-1)-subs
                best_point = -1
                max_gain = -1
                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = tset_to_id[ts]
                        if uncovered[tid]:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p

                if max_gain == 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)
            for tid in block_tset_ids(block):
                uncovered[tid] = False

        return blocks

    # Phase 1: Multiple greedy restarts
    best_blocks = None
    best_count = float('inf')

    for seed in range(40):
        if time.time() > deadline - 50:
            break
        blocks = greedy_construct(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    # Phase 2: Local search with redundancy detection
    count = np.zeros(n_tsets, dtype=np.int32)

    def update_counts(blocks_list, delta=1):
        for block in blocks_list:
            for tid in block_tset_ids(block):
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)
    current_count = len(current_blocks)

    def is_redundant(block):
        for tid in block_tset_ids(block):
            if count[tid] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        for i in range(n):
            if time.time() > deadline - 20:
                return blocks, False
            for j in range(i + 1, n):
                b1 = blocks[i]
                b2 = blocks[j]
                ids_b1 = set(block_tset_ids(b1))
                ids_b2 = set(block_tset_ids(b2))
                check_ids = ids_b1 | ids_b2

                u_need = []
                for tid in check_ids:
                    c = count[tid]
                    cov_b1 = 1 if tid in ids_b1 else 0
                    cov_b2 = 1 if tid in ids_b2 else 0
                    if c == cov_b1 + cov_b2:
                        u_need.append(all_tsets[tid])

                if not u_need:
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True

                p_need = set()
                for ts in u_need:
                    p_need.update(ts)

                if len(p_need) <= k:
                    new_block = list(p_need)
                    rng = random.Random(i * 1000 + j)
                    while len(new_block) < k:
                        p = rng.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()

                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    def try_swap_1_for_1(blocks):
        n = len(blocks)
        if n < 1:
            return blocks, False

        for i in range(n):
            if time.time() > deadline - 20:
                return blocks, False

            b = blocks[i]
            ids_b = set(block_tset_ids(b))

            u_need = []
            for tid in ids_b:
                if count[tid] == 1:
                    u_need.append(all_tsets[tid])

            if not u_need:
                update_counts([b], -1)
                return [x for idx, x in enumerate(blocks) if idx != i], True

            p_need = set()
            for ts in u_need:
                p_need.update(ts)

            if len(p_need) <= k:
                new_block = list(p_need)
                rng = random.Random(i * 7919)
                while len(new_block) < k:
                    p = rng.randint(0, v - 1)
                    if p not in new_block:
                        new_block.append(p)
                new_block.sort()

                if new_block != b:
                    update_counts([b], -1)
                    update_counts([new_block], 1)
                    new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    # Spectral redundancy removal: build co-occurrence matrix, find clusters
    def spectral_redundancy_removal(blocks):
        n = len(blocks)
        if n < 3:
            return blocks

        # Build co-occurrence matrix: M[i,j] = number of shared triples between block i and j
        block_ids = [set(block_tset_ids(b)) for b in blocks]
        M = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                shared = len(block_ids[i] & block_ids[j])
                M[i, j] = shared
                M[j, i] = shared

        # Compute similarity: high shared triples = more redundant
        # Use eigendecomposition to find partition
        try:
            eigvals, eigvecs = np.linalg.eigh(M)
            # Use top few eigenvectors for clustering
            n_clusters = min(3, n)
            top_vecs = eigvecs[:, -n_clusters:]
            # Simple k-means on top eigenvectors
            from numpy import argmax
            # Normalize
            norms = np.linalg.norm(top_vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            top_vecs_norm = top_vecs / norms

            # K-means
            rng = random.Random(123)
            centroids = top_vecs_norm[rng.sample(range(n), n_clusters)].copy()
            labels = np.zeros(n, dtype=np.int32)
            for _ in range(10):
                dists = np.zeros((n, n_clusters))
                for c in range(n_clusters):
                    dists[:, c] = np.sum((top_vecs_norm - centroids[c]) ** 2, axis=1)
                new_labels = np.argmin(dists, axis=1)
                if np.array_equal(labels, new_labels):
                    break
                labels = new_labels
                for c in range(n_clusters):
                    mask = labels == c
                    if mask.any():
                        centroids[c] = top_vecs_norm[mask].mean(axis=0)
        except Exception:
            return blocks

        # For each cluster, try to merge all blocks into one
        new_blocks = []
        for c in range(n_clusters):
            cluster_blocks = [blocks[i] for i in range(n) if labels[i] == c]
            if len(cluster_blocks) <= 1:
                new_blocks.extend(cluster_blocks)
                continue

            # Try to find a single block that covers all essential triples
            # Essential triples = those covered only by blocks in this cluster
            essential = set()
            for b in cluster_blocks:
                for tid in block_tset_ids(b):
                    # Check if covered by any block outside cluster
                    covered_outside = False
                    for idx, ob in enumerate(blocks):
                        if labels[idx] != c:
                            if tid in block_ids[idx]:
                                covered_outside = True
                                break
                    if not covered_outside:
                        essential.add(tid)

            if not essential:
                # All triples in cluster are covered elsewhere; remove all
                continue

            # Find points needed
            p_need = set()
            for tid in essential:
                p_need.update(all_tsets[tid])

            if len(p_need) <= k:
                new_block = list(p_need)
                rng = random.Random(c * 31 + 7)
                while len(new_block) < k:
                    p = rng.randint(0, v - 1)
                    if p not in new_block:
                        new_block.append(p)
                new_block.sort()
                new_blocks.append(new_block)
            else:
                # Can't merge; keep all blocks in cluster
                new_blocks.extend(cluster_blocks)

        return new_blocks

    # Global greedy refinement: try to replace a block with a better one
    def global_greedy_refine(blocks):
        """Try to improve by replacing individual blocks with better alternatives."""
        n = len(blocks)
        improved = False
        for i in range(n):
            if time.time() > deadline - 25:
                break
            b = blocks[i]
            ids_b = set(block_tset_ids(b))
            # Find triples only covered by this block
            unique_triples = []
            for tid in ids_b:
                if count[tid] == 1:
                    unique_triples.append(all_tsets[tid])

            if not unique_triples:
                # Block is fully redundant
                update_counts([b], -1)
                new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                return new_blocks, True

            # Try to find a better block that covers unique_triples and maximizes other coverage
            p_need = set()
            for ts in unique_triples:
                p_need.update(ts)

            if len(p_need) > k:
                continue

            # Try several candidate blocks containing p_need
            best_new_block = None
            best_gain = -1
            rng = random.Random(i * 131 + 17)
            for attempt in range(20):
                cand = list(p_need)
                while len(cand) < k:
                    p = rng.randint(0, v - 1)
                    if p not in cand:
                        cand.append(p)
                cand.sort()
                # Compute gain: how many additional triples does this cover?
                cand_ids = set(block_tset_ids(cand))
                # Gain = triples in cand not in b (that are uncovered or have low count)
                gain = 0
                for tid in cand_ids:
                    if tid not in ids_b:
                        if count[tid] == 0:
                            gain += 1
                        elif count[tid] <= 2:
                            gain += 0.5
                if gain > best_gain:
                    best_gain = gain
                    best_new_block = cand

            if best_new_block is not None and best_gain > 0 and best_new_block != b:
                update_counts([b], -1)
                update_counts([best_new_block], 1)
                new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                new_blocks.append(best_new_block)
                return new_blocks, True

        return blocks, False

    # Iterative local search
    improved = True
    max_rounds = 200
    for _ in range(max_rounds):
        if time.time() > deadline - 40:
            break
        if not improved:
            break
        improved = False

        new_blocks, rem = try_remove_redundant(current_blocks)
        if rem:
            current_blocks = new_blocks
            improved = True

        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True

        new_blocks, swp = try_swap_1_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True

        new_blocks, swp = global_greedy_refine(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True

    # Spectral redundancy removal
    if time.time() < deadline - 30 and len(current_blocks) > 5:
        try:
            spec_blocks = spectral_redundancy_removal(current_blocks)
            if len(spec_blocks) < len(current_blocks):
                # Verify coverage
                covered = set()
                for b in spec_blocks:
                    for tid in block_tset_ids(b):
                        covered.add(tid)
                if len(covered) == n_tsets:
                    current_blocks = spec_blocks
                    # Update count
                    count[:] = 0
                    for b in current_blocks:
                        for tid in block_tset_ids(b):
                            count[tid] += 1
                    # Run local search again
                    improved = True
                    for _ in range(50):
                        if time.time() > deadline - 35:
                            break
                        if not improved:
                            break
                        improved = False
                        new_blocks, rem = try_remove_redundant(current_blocks)
                        if rem:
                            current_blocks = new_blocks
                            improved = True
                        new_blocks, swp = try_swap_2_for_1(current_blocks)
                        if swp:
                            current_blocks = new_blocks
                            improved = True
                        new_blocks, swp = try_swap_1_for_1(current_blocks)
                        if swp:
                            current_blocks = new_blocks
                            improved = True
        except Exception:
            pass

    # Final: try more aggressive 2-for-1 passes
    for _ in range(20):
        if time.time() > deadline - 15:
            break
        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if not swp:
            break
        current_blocks = new_blocks

    # Final verification
    covered = set()
    for block in current_blocks:
        for tid in block_tset_ids(block):
            covered.add(tid)

    if len(covered) < n_tsets:
        return best_blocks

    return current_blocks
