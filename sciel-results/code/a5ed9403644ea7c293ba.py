import numpy as np
from scipy.sparse import csr_matrix
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # ---------- Precomputation ----------
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)

    t_sub_index = {ts: i for i, ts in enumerate(all_t_subs)}

    # block -> list of t-subset indices it covers
    block_to_subsets = [[] for _ in range(num_blocks)]
    # subset -> list of block indices covering it
    subset_to_blocks = [[] for _ in range(num_t_subs)]

    rows = []
    cols = []
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            j = t_sub_index[ts]
            block_to_subsets[i].append(j)
            subset_to_blocks[j].append(i)
            rows.append(i)
            cols.append(j)

    data = np.ones(len(rows), dtype=np.int32)
    M = csr_matrix((data, (rows, cols)), shape=(num_blocks, num_t_subs))

    # Precompute dense row vectors for fast count updates
    # Only for blocks we might touch; store lazily is fine, but let's precompute
    # the row as a list of (col, 1) pairs via the subset lists.
    def row_vector(b_idx):
        # returns a sparse-ish dense addition: we add 1 to each subset in block_to_subsets
        return block_to_subsets[b_idx]

    # ---------- Greedy (randomized) ----------
    def run_greedy(seed):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(num_t_subs, dtype=np.int8)
        # gain[b] = number of uncovered subsets in block b
        gains = np.zeros(num_blocks, dtype=np.int32)
        for b in range(num_blocks):
            gains[b] = len(block_to_subsets[b])

        selected = []
        while True:
            if time.time() > deadline - 15.0:
                break
            if gains.max() == 0:
                break
            # randomized tie-breaking via small noise
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best = int(np.argmax(scores))
            if gains[best] == 0:
                break
            selected.append(best)
            # update gains for all blocks sharing subsets with best
            for s in block_to_subsets[best]:
                if uncovered[s]:
                    uncovered[s] = 0
                    for b in subset_to_blocks[s]:
                        gains[b] -= 1
        return selected

    # ---------- Reduction (randomized redundant removal) ----------
    def reduce_blocks(blocks, seed):
        rng = random.Random(seed)
        cur = list(blocks)
        if not cur:
            return cur
        cover = np.zeros(num_t_subs, dtype=np.int32)
        for b in cur:
            for s in block_to_subsets[b]:
                cover[s] += 1

        improved = True
        while improved:
            if time.time() > deadline - 15.0:
                break
            improved = False
            redundant = []
            for i, b in enumerate(cur):
                is_red = True
                for s in block_to_subsets[b]:
                    if cover[s] == 1:
                        is_red = False
                        break
                if is_red:
                    redundant.append(i)
            if not redundant:
                break
            pos = redundant[rng.randint(0, len(redundant) - 1)]
            b = cur[pos]
            for s in block_to_subsets[b]:
                cover[s] -= 1
            cur.pop(pos)
            improved = True
        return cur

    # ---------- Repair (partial greedy on uncovered) ----------
    def repair(uncovered_mask, seed):
        rng = np.random.default_rng(seed)
        unc = uncovered_mask.copy()
        if not np.any(unc):
            return []
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s in np.where(unc)[0]:
            for b in subset_to_blocks[s]:
                gains[b] += 1
        selected = []
        while True:
            if time.time() > deadline - 15.0:
                break
            if gains.max() == 0:
                break
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best = int(np.argmax(scores))
            if gains[best] == 0:
                break
            selected.append(best)
            for s in block_to_subsets[best]:
                if unc[s]:
                    unc[s] = 0
                    for b in subset_to_blocks[s]:
                        gains[b] -= 1
        return selected

    # ---------- Search loop ----------
    best = None
    best_len = float('inf')
    trial = 0
    while time.time() < deadline - 20.0:
        trial += 1
        # Strategy 1: full greedy + reduce
        g = run_greedy(trial)
        r = reduce_blocks(g, trial)
        if len(r) < best_len:
            best_len = len(r)
            best = r

        # Strategy 2: perturb best + repair + reduce
        if best is not None and len(best) > 10 and time.time() < deadline - 25.0:
            rng2 = random.Random(trial * 7 + 1)
            perturbed = list(best)
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                rem_idx = rng2.sample(range(len(perturbed)), num_remove)
                rem_set = set(perturbed[i] for i in rem_idx)
                # recompute uncovered from perturbed minus removed
                cover = np.zeros(num_t_subs, dtype=np.int32)
                for b in perturbed:
                    if b in rem_set:
                        continue
                    for s in block_to_subsets[b]:
                        cover[s] += 1
                unc_mask = (cover == 0).astype(np.int8)
                rep = repair(unc_mask, trial * 13 + 5)
                new_set = [b for b in perturbed if b not in rem_set]
                new_set.extend(rep)
                # de-duplicate
                new_set = list(dict.fromkeys(new_set))
                r2 = reduce_blocks(new_set, trial * 17 + 3)
                if len(r2) < best_len:
                    best_len = len(r2)
                    best = r2

    if best is None:
        best = reduce_blocks(run_greedy(0), 0)

    return [list(all_blocks[i]) for i in best]
