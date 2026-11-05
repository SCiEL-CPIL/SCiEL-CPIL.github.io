import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0

    # Precompute all t-subsets and their indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, which t-subsets does it cover?
    block_tsets = []
    for b in all_blocks:
        ts_indices = [tset_index[ts] for ts in itertools.combinations(b, t)]
        block_tsets.append(ts_indices)

    # For each t-subset, which blocks cover it?
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)

    # Greedy set cover with random tie-breaking
    def greedy_cover(rng=None):
        if rng is None:
            rng = random.Random()
        uncovered = np.ones(n_tsub, dtype=bool)
        gain = np.zeros(n_blocks, dtype=np.int32)
        for bi in range(n_blocks):
            gain[bi] = len(block_tsets[bi])

        selected = []
        remaining = n_tsub

        while remaining > 0:
            if time.time() > deadline - 40.0:
                break
            # Add small random noise to break ties
            scores = gain.astype(np.float64).copy()
            scores += rng.random() * 0.1
            bi = int(np.argmax(scores))
            if gain[bi] == 0:
                break
            selected.append(bi)
            gain[bi] = 0
            for ti in block_tsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        gain[bj] -= 1

        return selected

    selected = greedy_cover(random.Random(42))

    # Reduction
    def reduce_solution(sel):
        if not sel:
            return []
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1
        new_sel = []
        for bi in sel:
            essential = False
            for ti in block_tsets[bi]:
                if cov[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets[bi]:
                    cov[ti] -= 1
            else:
                new_sel.append(bi)
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel

    selected = reduce_solution(selected)

    # Build coverage array
    cov = np.zeros(n_tsub, dtype=np.int32)
    for bi in selected:
        for ti in block_tsets[bi]:
            cov[ti] += 1

    best_selected = list(selected)
    best_size = len(selected)

    # Local search with multiple strategies
    T = 3.0
    cooling_rate = 0.997
    rng = random.Random(123)

    while time.time() < deadline - 5.0 and T > 0.05:
        if not selected:
            break

        move_type = rng.random()

        if move_type < 0.4:
            # Kick: remove one block, greedily refill
            idx = rng.randint(0, len(selected) - 1)
            bi_kick = selected[idx]
            saved_selected = list(selected)
            saved_cov = cov.copy()

            selected.pop(idx)
            for ti in block_tsets[bi_kick]:
                cov[ti] -= 1

            uncovered_tsets = [ti for ti in block_tsets[bi_kick] if cov[ti] == 0]

            if not uncovered_tsets:
                selected = reduce_solution(selected)
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    for ti in block_tsets[bi]:
                        cov[ti] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                T *= cooling_rate
                continue

            # Greedy refill
            new_blocks = []
            uncovered_set = set(uncovered_tsets)
            while uncovered_set and time.time() < deadline - 5.0:
                best_bj = -1
                max_cover = 0
                candidates = set()
                for ti in uncovered_set:
                    candidates.update(tset_to_blocks[ti])
                for bj in candidates:
                    cnt = sum(1 for ti in block_tsets[bj] if ti in uncovered_set)
                    if cnt > max_cover:
                        max_cover = cnt
                        best_bj = bj
                if best_bj == -1:
                    break
                new_blocks.append(best_bj)
                for ti in block_tsets[best_bj]:
                    if ti in uncovered_set:
                        uncovered_set.remove(ti)

            for bj in new_blocks:
                selected.append(bj)
                for ti in block_tsets[bj]:
                    cov[ti] += 1

            selected = reduce_solution(selected)
            delta = len(selected) - len(saved_selected)
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    for ti in block_tsets[bi]:
                        cov[ti] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
            else:
                selected = saved_selected
                cov = saved_cov

        elif move_type < 0.7:
            # Swap: remove one, add one
            if len(selected) < 2:
                T *= cooling_rate
                continue
            idx_remove = rng.randint(0, len(selected) - 1)
            bi_remove = selected[idx_remove]
            single_covered = [ti for ti in block_tsets[bi_remove] if cov[ti] == 1]
            if single_covered:
                candidates = set()
                for ti in single_covered:
                    candidates.update(tset_to_blocks[ti])
                candidates.discard(bi_remove)
                if not candidates:
                    T *= cooling_rate
                    continue
                bi_add = rng.choice(list(candidates))
            else:
                bi_add = rng.randint(0, n_blocks - 1)
                if bi_add == bi_remove:
                    T *= cooling_rate
                    continue
            if bi_add in selected:
                T *= cooling_rate
                continue
            saved_selected = list(selected)
            saved_cov = cov.copy()
            selected[idx_remove] = bi_add
            for ti in block_tsets[bi_remove]:
                cov[ti] -= 1
            for ti in block_tsets[bi_add]:
                cov[ti] += 1
            valid = all(cov[ti] > 0 for ti in block_tsets[bi_remove])
            if not valid:
                selected = saved_selected
                cov = saved_cov
                T *= cooling_rate
                continue
            selected = reduce_solution(selected)
            delta = len(selected) - len(saved_selected)
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    for ti in block_tsets[bi]:
                        cov[ti] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
            else:
                selected = saved_selected
                cov = saved_cov

        else:
            # Multi-kick: remove 2 blocks, refill
            if len(selected) < 3:
                T *= cooling_rate
                continue
            idxs = rng.sample(range(len(selected)), 2)
            bi1, bi2 = selected[idxs[0]], selected[idxs[1]]
            saved_selected = list(selected)
            saved_cov = cov.copy()

            # Remove both
            for idx in sorted(idxs, reverse=True):
                selected.pop(idx)
            for bi in [bi1, bi2]:
                for ti in block_tsets[bi]:
                    cov[ti] -= 1

            uncovered_tsets = [ti for ti in set(block_tsets[bi1]) | set(block_tsets[bi2]) if cov[ti] == 0]
            if not uncovered_tsets:
                selected = reduce_solution(selected)
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    for ti in block_tsets[bi]:
                        cov[ti] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                T *= cooling_rate
                continue

            new_blocks = []
            uncovered_set = set(uncovered_tsets)
            while uncovered_set and time.time() < deadline - 5.0:
                best_bj = -1
                max_cover = 0
                candidates = set()
                for ti in uncovered_set:
                    candidates.update(tset_to_blocks[ti])
                for bj in candidates:
                    cnt = sum(1 for ti in block_tsets[bj] if ti in uncovered_set)
                    if cnt > max_cover:
                        max_cover = cnt
                        best_bj = bj
                if best_bj == -1:
                    break
                new_blocks.append(best_bj)
                for ti in block_tsets[best_bj]:
                    if ti in uncovered_set:
                        uncovered_set.remove(ti)

            for bj in new_blocks:
                selected.append(bj)
                for ti in block_tsets[bj]:
                    cov[ti] += 1

            selected = reduce_solution(selected)
            delta = len(selected) - len(saved_selected)
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    for ti in block_tsets[bi]:
                        cov[ti] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
            else:
                selected = saved_selected
                cov = saved_cov

        T *= cooling_rate

    best_selected = reduce_solution(best_selected)
    return [list(all_blocks[bi]) for bi in best_selected]
