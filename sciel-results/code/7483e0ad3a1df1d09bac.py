import itertools
import time
import random
import numpy as np
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 132.0  # safety margin before 150s kill

    # ---------- Precomputation ----------
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    all_t_subs = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subs)

    t_sub_index = {ts: i for i, ts in enumerate(all_t_subs)}

    # Incidence matrix M: M[b, s] = 1 if block b covers t-subset s
    rows = []
    cols = []
    # Also keep dense per-block index lists for fast coverage updates
    block_to_subsets = [[] for _ in range(num_blocks)]
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            s_idx = t_sub_index[ts]
            rows.append(b_idx)
            cols.append(s_idx)
            block_to_subsets[b_idx].append(s_idx)

    M = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_blocks, num_t))
    M = M.tocsr()

    # subset -> blocks covering it (for repair / pivot selection)
    subset_to_blocks = [[] for _ in range(num_t)]
    for b_idx, idxs in enumerate(block_to_subsets):
        for s in idxs:
            subset_to_blocks[s].append(b_idx)

    best_blocks_idx = None
    best_len = float('inf')

    def check_time():
        return time.time() > deadline

    def reduce_blocks(blocks_idx, coverage):
        """Remove redundant blocks. coverage is a mutable array."""
        if not blocks_idx:
            return blocks_idx, coverage
        changed = True
        while changed and not check_time():
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                idxs = block_to_subsets[b_idx]
                # redundant if all covered subsets have coverage > 1
                if np.all(coverage[idxs] > 1):
                    coverage[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks
        return blocks_idx, coverage

    def reduce_random_order(blocks_idx, coverage, rng):
        """Reduction with randomized order for diversity."""
        if not blocks_idx:
            return blocks_idx, coverage
        changed = True
        while changed and not check_time():
            changed = False
            order = list(blocks_idx)
            rng.shuffle(order)
            new_blocks = []
            for b_idx in order:
                idxs = block_to_subsets[b_idx]
                if np.all(coverage[idxs] > 1):
                    coverage[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks
        return blocks_idx, coverage

    def greedy_construct(seed, vectorized=True):
        """Global greedy: pick block covering most uncovered t-subsets."""
        rng = random.Random(seed)
        coverage = np.zeros(num_t, dtype=np.int32)
        uncovered = num_t
        blocks_idx = []

        while uncovered > 0 and not check_time():
            if vectorized:
                mask = (coverage == 0).astype(np.int8)
                gains = M.dot(mask)
                max_gain = int(np.max(gains))
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                b_idx = int(rng.choice(candidates.tolist()))
            else:
                # Fallback: pivot on a random uncovered subset
                # find an uncovered subset
                uncov = np.where(coverage == 0)[0]
                if len(uncov) == 0:
                    break
                pivot = int(rng.choice(uncov.tolist()))
                cands = subset_to_blocks[pivot]
                if not cands:
                    break
                best_b = -1
                best_gain = -1
                for b_idx in cands:
                    gain = 0
                    for s in block_to_subsets[b_idx]:
                        if coverage[s] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b < 0:
                    break
                b_idx = best_b

            blocks_idx.append(b_idx)
            idxs = block_to_subsets[b_idx]
            coverage[idxs] += 1
            uncovered = int(np.sum(coverage == 0))

        return blocks_idx, coverage

    def random_subset_construct(seed, p):
        """LLL-inspired: include each block with prob p, repair uncovered, reduce."""
        rng = random.Random(seed)
        coverage = np.zeros(num_t, dtype=np.int32)
        selected = [False] * num_blocks
        blocks_idx = []

        # Random inclusion
        for b_idx in range(num_blocks):
            if rng.random() < p:
                selected[b_idx] = True
                blocks_idx.append(b_idx)
                idxs = block_to_subsets[b_idx]
                coverage[idxs] += 1

        # Repair: while uncovered subsets remain, add a block covering one
        while not check_time():
            uncov = np.where(coverage == 0)[0]
            if len(uncov) == 0:
                break
            # pick a random uncovered subset
            pivot = int(rng.choice(uncov.tolist()))
            cands = subset_to_blocks[pivot]
            if not cands:
                break
            # among candidates, prefer one with most new coverage
            best_b = -1
            best_gain = -1
            # sample a few if too many
            if len(cands) > 30:
                sample = rng.sample(cands, 30)
            else:
                sample = cands
            for b_idx in sample:
                if selected[b_idx]:
                    continue
                gain = 0
                for s in block_to_subsets[b_idx]:
                    if coverage[s] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b < 0:
                break
            selected[best_b] = True
            blocks_idx.append(best_b)
            idxs = block_to_subsets[best_b]
            coverage[idxs] += 1

        return blocks_idx, coverage, selected

    def swap_local_search(blocks_idx, coverage, seed, max_iters=800):
        """Try swaps: remove a non-essential block, add one that covers gaps."""
        rng = random.Random(seed)
        if not blocks_idx:
            return blocks_idx, coverage

        sel_set = set(blocks_idx)
        for _ in range(max_iters):
            if check_time():
                break
            rem_pos = rng.randint(0, len(blocks_idx) - 1)
            rem_idx = blocks_idx[rem_pos]
            rem_ts = block_to_subsets[rem_idx]

            # would removing it uncover anything?
            would_uncover = False
            for s in rem_ts:
                if coverage[s] == 1:
                    would_uncover = True
                    break
            if would_uncover:
                continue

            # temporarily remove
            for s in rem_ts:
                coverage[s] -= 1

            # find a block (not selected) that covers most currently-uncovered
            best_add = -1
            best_gain = 0
            n_sample = min(80, num_blocks)
            candidates = rng.sample(range(num_blocks), n_sample)
            for b_idx in candidates:
                if b_idx in sel_set:
                    continue
                gain = 0
                for s in block_to_subsets[b_idx]:
                    if coverage[s] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_add = b_idx

            if best_add < 0 or best_gain == 0:
                for s in rem_ts:
                    coverage[s] += 1
                continue

            # perform swap
            for s in block_to_subsets[best_add]:
                coverage[s] += 1
            sel_set.discard(rem_idx)
            sel_set.add(best_add)
            blocks_idx[rem_pos] = best_add

        return blocks_idx, coverage

    def deletion_based_reduce(seed):
        """Start with all blocks, remove redundant in random order."""
        rng = random.Random(seed)
        coverage = np.zeros(num_t, dtype=np.int32)
        # initialize counts
        for b_idx in range(num_blocks):
            coverage[block_to_subsets[b_idx]] += 1
        blocks_idx = list(range(num_blocks))
        blocks_idx, coverage = reduce_random_order(blocks_idx, coverage, rng)
        return blocks_idx, coverage

    # ---------- Run construction families ----------
    # Family A: vectorized greedy + reduction (multiple seeds)
    for seed in range(6):
        if check_time():
            break
        blocks_idx, coverage = greedy_construct(seed * 101 + 7, vectorized=True)
        blocks_idx, coverage = reduce_blocks(blocks_idx, coverage)
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks_idx = blocks_idx[:]
            best_coverage = coverage.copy()

    # Family B: randomized subset + repair + reduction (LLL-flavored)
    for seed in range(8):
        if check_time():
            break
        # vary p: start high to get a quick covering, then lower for sparser
        p = 0.15 + 0.02 * (seed % 5)
        blocks_idx, coverage, selected = random_subset_construct(seed * 211 + 13, p)
        blocks_idx, coverage = reduce_random_order(blocks_idx, coverage, random.Random(seed * 31 + 3))
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks_idx = blocks_idx[:]
            best_coverage = coverage.copy()

    # Family C: deletion-from-all (strong baseline, converges to small covering)
    for seed in range(4):
        if check_time():
            break
        blocks_idx, coverage = deletion_based_reduce(seed * 7777 + 1)
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks_idx = blocks_idx[:]
            best_coverage = coverage.copy()

    # Family D: swap local search polish on the best solution found so far
    if best_blocks_idx is not None and not check_time():
        for seed in range(6):
            if check_time():
                break
            sol_copy = best_blocks_idx[:]
            cov_copy = best_coverage.copy()
            sol_copy, cov_copy = swap_local_search(sol_copy, cov_copy, seed * 503 + 11, max_iters=600)
            # reduce again after swaps
            sol_copy, cov_copy = reduce_blocks(sol_copy, cov_copy)
            if len(sol_copy) < best_len:
                best_len = len(sol_copy)
                best_blocks_idx = sol_copy[:]
                best_coverage = cov_copy.copy()

    # Final reduction pass on the best
    if best_blocks_idx is not None and not check_time():
        best_blocks_idx, best_coverage = reduce_blocks(best_blocks_idx, best_coverage)

    # Convert to lists of points
    final_blocks = []
    if best_blocks_idx:
        for b_idx in best_blocks_idx:
            final_blocks.append(list(all_blocks[b_idx]))
    else:
        # Fallback: greedy from scratch
        blocks_idx, coverage = greedy_construct(0, vectorized=True)
        for b_idx in blocks_idx:
            final_blocks.append(list(all_blocks[b_idx]))

    return final_blocks
