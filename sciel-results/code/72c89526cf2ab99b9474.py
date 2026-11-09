# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = 615.0  # safety margin before 630s limit

    # A 1-intersecting family of 3-subsets of {0..10} (no two share exactly 1 element)
    # yields 8*|F| integer points with entries in {-1,0,1}, norm^2 = 3,
    # and pairwise squared distance >= 3 = max norm^2.
    best_blocks = None
    best_size = 0

    def build_family(blocks):
        """Greedy + local search + simulated annealing to maximize a 1-intersecting family."""
        nonlocal best_blocks, best_size
        blocks = set(blocks)
        cur = set(blocks)
        cur_size = len(cur)
        if cur_size > best_size:
            best_size = cur_size
            best_blocks = list(cur)

        # Precompute conflict (share exactly 1) for all pairs of 3-subsets
        # Represent each 3-subset as a frozenset; build a dict of conflict neighbors on the fly.
        # For speed, map each block to its element set.
        # We'll compute conflict lazily using a precomputed table indexed by integer encoding.

        # Encode each 3-subset as a tuple of 3 sorted ints.
        # Build full conflict table over all C(11,3)=165 blocks.
        all_blocks = list(itertools.combinations(range(11), 3))
        block_index = {b: i for i, b in enumerate(all_blocks)}
        n_all = len(all_blocks)
        # conflict_mask[i] = bitmask of j with |all_blocks[i] & all_blocks[j]| == 1
        # 165 < 512, use Python int as bitset.
        conflict_mask = [0] * n_all
        for i in range(n_all):
            bi = set(all_blocks[i])
            for j in range(i + 1, n_all):
                bj = set(all_blocks[j])
                if len(bi & bj) == 1:
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)

        # Represent current family as set of indices into all_blocks.
        def to_idx(blocks_set):
            return {block_index[b] for b in blocks_set}
        cur_idx = to_idx(cur)

        # Greedy add: repeatedly add a block with zero conflicts in cur.
        def greedy_add(cur_idx, order):
            for idx in order:
                if idx in cur_idx:
                    continue
                if conflict_mask[idx] & _mask_of(cur_idx) == 0:
                    cur_idx.add(idx)
            return cur_idx

        def _mask_of(idx_set):
            m = 0
            for x in idx_set:
                m |= (1 << x)
            return m

        # Greedy rounds
        for _ in range(30):
            order = list(range(n_all))
            random.shuffle(order)
            new_idx = greedy_add(set(cur_idx), order)
            if len(new_idx) > len(cur_idx):
                cur_idx = new_idx
                cur_size = len(cur_idx)
                if cur_size > best_size:
                    best_size = cur_size
                    best_blocks = [all_blocks[i] for i in cur_idx]

        # Simulated annealing: moves = add a non-conflicting block, or swap (remove one, add one).
        cur_mask = _mask_of(cur_idx)
        T = 3.0
        step = 0
        while time.time() < deadline - 5.0:
            step += 1
            if step % 500 == 0:
                T *= 0.995
                if T < 0.05:
                    break
            # Random move
            r = random.random()
            if r < 0.5:
                # add a random block
                cand = random.randrange(n_all)
                if cand in cur_idx:
                    continue
                # check conflict
                if conflict_mask[cand] & cur_mask:
                    # delta "cost" = number of conflicts
                    cost = bin(conflict_mask[cand] & cur_mask).count("1")
                    if random.random() < np.exp(-cost / max(T, 1e-6)):
                        # accept: remove conflicting blocks
                        conf = conflict_mask[cand] & cur_mask
                        while conf:
                            lsb = conf & (-conf)
                            j = lsb.bit_length() - 1
                            cur_idx.discard(j)
                            cur_mask &= ~(1 << j)
                            conf ^= lsb
                        cur_idx.add(cand)
                        cur_mask |= (1 << cand)
            else:
                # swap: pick a current block to remove and a random block to add
                if not cur_idx:
                    continue
                rem = random.choice(list(cur_idx))
                cand = random.randrange(n_all)
                if cand == rem or cand in cur_idx:
                    continue
                # new conflicts of cand with cur \ {rem}
                cur_mask_without_rem = cur_mask & ~(1 << rem)
                cost = bin(conflict_mask[cand] & cur_mask_without_rem).count("1")
                # removing rem may free up, net change in size = 0
                if cost == 0:
                    cur_idx.discard(rem)
                    cur_idx.add(cand)
                    cur_mask = (cur_mask & ~(1 << rem)) | (1 << cand)
                else:
                    if random.random() < np.exp(-cost / max(T, 1e-6)):
                        # remove rem and all conflicting
                        conf = conflict_mask[cand] & cur_mask_without_rem
                        cur_idx.discard(rem)
                        cur_mask &= ~(1 << rem)
                        while conf:
                            lsb = conf & (-conf)
                            j = lsb.bit_length() - 1
                            cur_idx.discard(j)
                            cur_mask &= ~(1 << j)
                            conf ^= lsb
                        cur_idx.add(cand)
                        cur_mask |= (1 << cand)

            cur_size = len(cur_idx)
            if cur_size > best_size:
                best_size = cur_size
                best_blocks = [all_blocks[i] for i in cur_idx]

    # Initial seed: a star (all blocks containing 0) = 45 blocks. Plus try random starts.
    star = [b for b in itertools.combinations(range(11), 3) if 0 in b]
    build_family(set(star))

    # A few random restarts with small random initial families
    for restart in range(6):
        if time.time() > deadline - 10.0:
            break
        init = set()
        for _ in range(random.randint(5, 20)):
            b = tuple(sorted(random.sample(range(11), 3)))
            init.add(b)
        build_family(init)

    if best_blocks is None or len(best_blocks) < 1:
        best_blocks = [star[0]]

    # Build points from the 1-intersecting family
    points = []
    for block in best_blocks:
        for signs in itertools.product([-1, 1], repeat=3):
            v = np.zeros(d, dtype=np.int64)
            for pos, s in zip(block, signs):
                v[pos] = s
            points.append(v)

    if len(points) < 2:
        # Fallback: basis vectors
        pts = []
        for i in range(d):
            a = np.zeros(d, dtype=np.int64); a[i] = 1; pts.append(a)
            b = np.zeros(d, dtype=np.int64); b[i] = -1; pts.append(b)
        points = pts

    result = np.array(points, dtype=np.int64)
    result = np.around(result).astype(np.int64)

    # Validate
    k = len(result)
    if k >= 2 and result.shape[1] == d:
        norms_sq = np.sum(result ** 2, axis=1)
        if np.all(norms_sq > 0):
            max_norm_sq = int(np.max(norms_sq))
            min_dist_sq = np.inf
            for i in range(k):
                for j in range(i + 1, k):
                    diff = result[i] - result[j]
                    dist_sq = int(np.sum(diff ** 2))
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
            if min_dist_sq >= max_norm_sq:
                return result

    # Fallback valid config: 2 points
    return np.array([np.ones(d), -np.ones(d)]).astype(np.int64)

# EVOLVE-BLOCK-END
