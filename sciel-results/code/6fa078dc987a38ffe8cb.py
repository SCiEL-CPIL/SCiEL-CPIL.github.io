import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before the 150s kill

    # ---------- Precomputation ----------
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = list(all_triplets)

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # block_masks: integer bitset of t-subsets in each block
    block_masks = [0] * n_blocks
    block_triplet_ids = [None] * n_blocks
    for bi, block in enumerate(all_blocks):
        mask = 0
        ids = []
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
            ids.append(tid)
        block_masks[bi] = mask
        block_triplet_ids[bi] = ids

    # point -> list of block indices containing it
    point_to_blocks = defaultdict(list)
    for bi, block in enumerate(all_blocks):
        for p in block:
            point_to_blocks[p].append(bi)

    def get_blocks_containing(points):
        points = set(points)
        if len(points) > k:
            return []
        if not points:
            return list(range(n_blocks))
        first = next(iter(points))
        cand = set(point_to_blocks[first])
        for p in points:
            if p == first:
                continue
            cand &= set(point_to_blocks[p])
            if not cand:
                return []
        return list(cand)

    # ---------- Greedy construction ----------
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        blocks = []
        used = set()

        while uncovered:
            idx = -1
            for _ in range(20):
                cand = random.randint(0, n_triplets - 1)
                if (uncovered >> cand) & 1:
                    idx = cand
                    break
            if idx == -1:
                lsb = uncovered & -uncovered
                idx = lsb.bit_length() - 1
            start_triplet = id_to_triplet[idx]

            best_bi = -1
            best_ratio = -1.0
            best_mask = 0
            for bi in get_blocks_containing(start_triplet):
                if bi in used:
                    continue
                b_mask = block_masks[bi]
                gain = (b_mask & uncovered).bit_count()
                if gain == 0:
                    continue
                cost = b_mask.bit_count()
                ratio = gain / cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_bi = bi
                    best_mask = b_mask

            if best_bi == -1:
                for bi in get_blocks_containing(start_triplet):
                    if (block_masks[bi] & uncovered).bit_count() > 0:
                        best_bi = bi
                        best_mask = block_masks[bi]
                        break
            if best_bi == -1:
                break

            blocks.append(all_blocks[best_bi])
            used.add(best_bi)
            uncovered &= ~best_mask

        return blocks

    # ---------- Local search optimizer (remove / 2-for-1 / 1-for-1) ----------
    def optimize(blocks, time_limit):
        counts = np.zeros(n_triplets, dtype=np.int32)

        def recalc(b_list):
            counts[:] = 0
            for b in b_list:
                for triplet in itertools.combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1

        recalc(blocks)
        no_improve = 0
        max_no_improve = 30

        while time.time() < time_limit:
            improved = False
            n = len(blocks)

            # 1. Remove redundant
            for i in range(n):
                b = blocks[i]
                redundant = True
                for triplet in itertools.combinations(b, t):
                    if counts[triplet_to_id[triplet]] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    for triplet in itertools.combinations(b, t):
                        counts[triplet_to_id[triplet]] -= 1
                    improved = True
                    break

            # 2. 2-for-1 replacement
            if not improved and n >= 2:
                unique_loads = []
                for i in range(n):
                    b = blocks[i]
                    load = 0
                    for triplet in itertools.combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            load += 1
                    unique_loads.append((load, i))
                unique_loads.sort(reverse=True)

                tried = set()
                for load1, i in unique_loads[:15]:
                    if load1 == 0:
                        continue
                    for load2, j in unique_loads[:15]:
                        if i == j or i > j or (i, j) in tried:
                            continue
                        tried.add((i, j))
                        b1 = blocks[i]
                        b2 = blocks[j]
                        ids_b1 = set(triplet_to_id[tr] for tr in itertools.combinations(b1, t))
                        ids_b2 = set(triplet_to_id[tr] for tr in itertools.combinations(b2, t))
                        all_involved = ids_b1 | ids_b2
                        needed = []
                        pts = set()
                        for tid in all_involved:
                            if counts[tid] == 1:
                                needed.append(tid)
                                pts.update(id_to_triplet[tid])
                        if not needed:
                            blocks.pop(j)
                            for tr in itertools.combinations(b2, t):
                                counts[triplet_to_id[tr]] -= 1
                            blocks.pop(i)
                            for tr in itertools.combinations(b1, t):
                                counts[triplet_to_id[tr]] -= 1
                            improved = True
                            break
                        if len(pts) > k:
                            continue
                        for bi in get_blocks_containing(pts):
                            nb = all_blocks[bi]
                            if set(nb) == set(b1) or set(nb) == set(b2):
                                continue
                            new_ids = set(block_triplet_ids[bi])
                            if all(tid in new_ids for tid in needed):
                                blocks.pop(j)
                                for tr in itertools.combinations(b2, t):
                                    counts[triplet_to_id[tr]] -= 1
                                blocks.pop(i)
                                for tr in itertools.combinations(b1, t):
                                    counts[triplet_to_id[tr]] -= 1
                                blocks.append(nb)
                                for tid in block_triplet_ids[bi]:
                                    counts[tid] += 1
                                improved = True
                                break
                        if improved:
                            break
                        if time.time() > time_limit - 3:
                            break

            # 3. 1-for-1 replacement
            if not improved and n >= 1:
                i = random.randint(0, n - 1)
                b1 = blocks[i]
                ids_b1 = set(triplet_to_id[tr] for tr in itertools.combinations(b1, t))
                needed = []
                pts = set()
                for tid in ids_b1:
                    if counts[tid] == 1:
                        needed.append(tid)
                        pts.update(id_to_triplet[tid])
                if needed and len(pts) <= k:
                    for bi in get_blocks_containing(pts):
                        nb = all_blocks[bi]
                        if set(nb) == set(b1):
                            continue
                        new_ids = set(block_triplet_ids[bi])
                        if all(tid in new_ids for tid in needed):
                            blocks.pop(i)
                            for tr in itertools.combinations(b1, t):
                                counts[triplet_to_id[tr]] -= 1
                            blocks.append(nb)
                            for tid in block_triplet_ids[bi]:
                                counts[tid] += 1
                            improved = True
                            break

            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
            else:
                no_improve = 0

        return blocks

    # ---------- Weighted independent-set (Max-IS) refinement ----------
    # Treat candidate blocks as nodes; weight = marginal coverage gain.
    # Greedily select a dense, non-redundant core that covers all triples.
    def max_is_refine(blocks, time_limit):
        if time.time() > time_limit:
            return blocks
        # candidate pool = current blocks + blocks that cover many uncovered triplets
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for tr in itertools.combinations(b, t):
                counts[triplet_to_id[tr]] += 1

        cand = list(range(len(blocks)))
        # add high-gain external blocks (those that cover the most currently uncovered triplets)
        # "uncovered" = triplets with count 0; but counts are all >=1 if valid. Use marginal gain.
        ext_candidates = []
        for bi in range(n_blocks):
            if bi in set(cand):
                continue
            # marginal gain = triplets covered by bi that are covered exactly once by the pool
            gain = 0
            for tid in block_triplet_ids[bi]:
                if counts[tid] == 1:
                    gain += 1
            if gain > 0:
                ext_candidates.append((gain, bi))
        ext_candidates.sort(reverse=True)
        for _, bi in ext_candidates[:60]:
            cand.append(bi)

        # candidate block tuples
        cand_blocks = [all_blocks[bi] for bi in cand]
        m = len(cand_blocks)
        if m == 0:
            return blocks

        # weight = marginal gain (how many triplets this block uniquely helps)
        weights = [0] * m
        for i, bi in enumerate(cand):
            w = 0
            for tid in block_triplet_ids[bi]:
                if counts[tid] == 1:
                    w += 1
            weights[i] = w

        # conflict: two candidates conflict if they share a t-subset (overlap)
        # Build adjacency lazily via masks.
        masks = [block_masks[bi] for bi in cand]

        def covers_all(selected):
            # check every triplet covered at least once
            cov = np.zeros(n_triplets, dtype=np.int32)
            for si in selected:
                for tid in block_triplet_ids[cand[si]]:
                    cov[tid] += 1
            return int(np.min(cov)) > 0

        # greedy weighted Max-IS: repeatedly pick node with best weight/degree,
        # remove it and all its neighbors (conflicts), until all triplets covered.
        selected = []
        remaining = list(range(m))
        # we need coverage of all triplets, so we iterate until covers_all
        # Use a heuristic: pick node maximizing (weight - alpha * overlap_penalty)
        alpha = 0.5
        guard = 0
        while not covers_all(selected) and remaining and guard < 400:
            guard += 1
            if time.time() > time_limit:
                break
            best = -1
            best_score = -1
            for ri, si in enumerate(remaining):
                # overlap penalty: how many of its triplets are already covered by selected
                overlap = 0
                sm = masks[si]
                sel_mask = 0
                for sj in selected:
                    sel_mask |= masks[sj]
                overlap = (sm & sel_mask).bit_count()
                score = weights[si] - alpha * overlap
                if score > best_score:
                    best_score = score
                    best = ri
            if best == -1:
                break
            si = remaining.pop(best)
            selected.append(si)
            # remove conflicting nodes (sharing any t-subset)
            new_remaining = []
            for rj in remaining:
                if masks[rj] & masks[si]:
                    continue
                new_remaining.append(rj)
            remaining = new_remaining

        if covers_all(selected):
            new_blocks = [cand_blocks[si] for si in selected]
            if len(new_blocks) < len(blocks):
                return new_blocks
        return blocks

    # ---------- Driver ----------
    best_blocks = None
    best_len = float('inf')

    n_restarts = 8
    for seed in range(n_restarts):
        if time.time() > deadline - 12:
            break

        init_blocks = greedy_construct(seed)
        opt_blocks = optimize(init_blocks, deadline - 6)

        # Max-IS refinement on a copy
        try:
            refined = max_is_refine(list(opt_blocks), deadline - 4)
            if len(refined) < len(opt_blocks):
                opt_blocks = refined
        except Exception:
            pass

        # Verify validity
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for tr in itertools.combinations(b, t):
                counts[triplet_to_id[tr]] += 1

        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    return list(best_blocks)


if __name__ == "__main__":
    pass
