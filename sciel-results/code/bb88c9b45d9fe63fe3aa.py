import itertools
import random
import time
import numpy as np

_PRECOMPUTED = {}

def _precompute(v, k, t):
    if (v, k, t) in _PRECOMPUTED:
        return _PRECOMPUTED[(v, k, t)]

    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, the indices of the t-sets it contains (as a numpy array)
    block_tsets = []
    for block in all_blocks:
        idxs = [tset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsets.append(np.array(idxs, dtype=np.int64))

    # For each t-set, the indices of blocks containing it
    tset_to_block_indices = [[] for _ in range(n_tsets)]
    for b_idx, arr in enumerate(block_tsets):
        for ts_idx in arr:
            tset_to_block_indices[ts_idx].append(b_idx)

    # Symmetric "dense" pool: blocks with high mutual overlap (good starting material)
    # We just rank blocks by average overlap with others is expensive; instead build a
    # pool of a random sample plus a few highly-overlapping ones for the initial seed.
    data = {
        'all_blocks': all_blocks,
        'n_blocks': n_blocks,
        'n_tsets': n_tsets,
        'block_tsets': block_tsets,
        'tset_to_block_indices': tset_to_block_indices,
        'tset_to_idx': tset_to_idx,
        'all_tsets': all_tsets,
    }
    _PRECOMPUTED[(v, k, t)] = data
    return data


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin vs 150s wall-clock

    data = _precompute(v, k, t)
    all_blocks = data['all_blocks']
    n_blocks = data['n_blocks']
    n_tsets = data['n_tsets']
    block_tsets = data['block_tsets']
    tset_to_block_indices = data['tset_to_block_indices']

    # Precompute, for each t-set, the list of blocks as a numpy array (for fast membership via set)
    tset_blocks_arr = [np.array(b, dtype=np.int64) for b in tset_to_block_indices]

    def _greedy_density(rng, time_limit):
        """Greedy that maximizes a density objective: prefer blocks that lift the
        lowest coverage counts (reduces sum of 1/coverage)."""
        dl = time.time() + time_limit
        covered = np.zeros(n_tsets, dtype=np.int32)
        chosen = []
        # objective: sum over t-sets of f(covered) where f is high when covered is low
        # We track sum_inv = sum(1/max(1,covered)); adding a block that covers many
        # low-coverage t-sets reduces sum_inv the most.

        while time.time() < dl:
            # pick the most "starved" uncovered t-set
            zero = np.where(covered == 0)[0]
            if len(zero) == 0:
                break
            # among starved, prefer one whose candidate blocks collectively cover many
            # low-coverage t-sets. Sample a few starved t-sets and evaluate.
            sample = zero[rng.choice(len(zero), size=min(6, len(zero)), replace=False)]
            best_block = -1
            best_score = -1.0
            for ts in sample:
                for b_idx in tset_blocks_arr[ts]:
                    arr = block_tsets[b_idx]
                    # score = reduction in sum_inv if we add this block
                    c = covered[arr]
                    # delta_inv = sum(1/c_old - 1/c_new) over t-sets in block
                    c_old = np.maximum(1, c)
                    c_new = np.maximum(1, c + 1)
                    score = float(np.sum(1.0 / c_old - 1.0 / c_new))
                    # tie-break: prefer covering more currently-zero
                    if score > best_score or (score == best_score and best_block == -1):
                        best_score = score
                        best_block = b_idx
            if best_block == -1:
                break
            chosen.append(best_block)
            covered[block_tsets[best_block]] += 1

        return chosen

    def _coverage_of(blocks_idx):
        covered = np.zeros(n_tsets, dtype=np.int32)
        for b_idx in blocks_idx:
            covered[block_tsets[b_idx]] += 1
        return covered

    def _density_obj(covered):
        c = np.maximum(1, covered)
        return float(np.sum(1.0 / c))

    def _local_search(blocks_idx, rng, time_limit):
        """Local search: repeatedly remove blocks that are fully redundant, and swap
        a block for one that increases redundancy (improves density objective)."""
        dl = time.time() + time_limit
        current = list(blocks_idx)
        covered = _coverage_of(current)
        best = current[:]
        best_obj = _density_obj(covered)

        improved = True
        while time.time() < dl and improved:
            improved = False

            # Pass 1: remove redundant blocks (those whose removal keeps full coverage)
            for i in range(len(current)):
                if time.time() > dl:
                    break
                b_idx = current[i]
                arr = block_tsets[b_idx]
                if np.all(covered[arr] > 1):
                    # safe to remove
                    covered[arr] -= 1
                    current.pop(i)
                    improved = True
                    obj = _density_obj(covered)
                    if obj < best_obj:
                        best_obj = obj
                        best = current[:]
                    # restart scan (indices shifted)
                    i = -1
            if time.time() > dl:
                break

            # Pass 2: try swapping each block for a better one (improve density)
            for i in range(len(current)):
                if time.time() > dl:
                    break
                old = current[i]
                old_arr = block_tsets[old]
                # t-sets that would become starved if we drop old
                critical = old_arr[covered[old_arr] == 1]
                if len(critical) == 0:
                    # fully redundant
                    covered[old_arr] -= 1
                    current.pop(i)
                    improved = True
                    obj = _density_obj(covered)
                    if obj < best_obj:
                        best_obj = obj
                        best = current[:]
                    i = -1
                    continue
                # candidate replacements must cover all critical t-sets
                cand = set(tset_blocks_arr[critical[0]].tolist())
                for ts in critical[1:]:
                    cand &= set(tset_blocks_arr[ts].tolist())
                cand.discard(old)
                cand -= set(current)
                if not cand:
                    continue
                cand = list(cand)
                rng.shuffle(cand)
                # score each candidate by resulting density
                best_new = -1
                best_new_obj = best_obj
                for nb in cand[:40]:
                    new_arr = block_tsets[nb]
                    # simulate
                    covered[old_arr] -= 1
                    covered[new_arr] += 1
                    obj = _density_obj(covered)
                    # restore
                    covered[new_arr] -= 1
                    covered[old_arr] += 1
                    if obj < best_new_obj:
                        best_new_obj = obj
                        best_new = nb
                if best_new != -1:
                    covered[old_arr] -= 1
                    covered[block_tsets[best_new]] += 1
                    current[i] = best_new
                    improved = True
                    if best_new_obj < best_obj:
                        best_obj = best_new_obj
                        best = current[:]

            # Pass 3: random shake if stuck
            if not improved and time.time() < dl:
                if len(current) >= 2:
                    i, j = rng.sample(range(len(current)), 2)
                    covered[block_tsets[current[i]]] -= 1
                    covered[block_tsets[current[j]]] -= 1
                    current[i] = rng.randrange(n_blocks)
                    current[j] = rng.randrange(n_blocks)
                    covered[block_tsets[current[i]]] += 1
                    covered[block_tsets[current[j]]] += 1
                    improved = True  # loop again

        return best

    best_blocks = None
    best_size = float('inf')

    # Several greedy restarts with density objective
    for seed in range(8):
        if time.time() > deadline - 30.0:
            break
        rng = random.Random(seed)
        blocks = _greedy_density(rng, 4.0)
        if blocks and len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]

    if best_blocks is None:
        rng = random.Random(42)
        best_blocks = _greedy_density(rng, 8.0)
        best_size = len(best_blocks)

    # Local search polish
    if time.time() < deadline - 20.0:
        rng = random.Random(123)
        remaining = min(35.0, deadline - (time.time() - start_time) - 20.0)
        improved = _local_search(best_blocks, rng, remaining)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)

    # Final local search pass
    if time.time() < deadline - 5.0:
        rng = random.Random(789)
        remaining = min(12.0, deadline - (time.time() - start_time) - 5.0)
        improved = _local_search(best_blocks, rng, remaining)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)

    # Verify full coverage; if not, top up with greedy
    covered = _coverage_of(best_blocks)
    if not np.all(covered > 0):
        rng = random.Random(2000)
        # add blocks until covered
        guard = 0
        while not np.all(covered > 0) and guard < n_blocks:
            zero = np.where(covered == 0)[0]
            if len(zero) == 0:
                break
            ts = zero[rng.randrange(len(zero))]
            # pick block covering ts with max new coverage
            best_b = -1
            best_gain = -1
            for b_idx in tset_blocks_arr[ts]:
                gain = int(np.sum(covered[block_tsets[b_idx]] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1:
                break
            best_blocks.append(best_b)
            covered[block_tsets[best_b]] += 1
            guard += 1

    result = [list(all_blocks[idx]) for idx in best_blocks]
    return result
