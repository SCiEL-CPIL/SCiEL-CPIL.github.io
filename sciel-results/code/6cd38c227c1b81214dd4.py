import itertools
import time
import numpy as np

def _greedy_cover(v, k, t, deadline, rng, alpha):
    """Global greedy set cover with a smooth (log-sum-exp) gain objective."""
    # t-subsets and their index
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # k-subsets (blocks) and the t-subset indices they contain
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    n_t_in_k = len(list(itertools.combinations(range(k), t)))

    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    for i, block in enumerate(k_subsets):
        for j, tc in enumerate(itertools.combinations(block, t)):
            block_t_indices[i, j] = t_sub_to_idx[tc]

    uncovered = np.ones(n_t, dtype=np.int8)
    selected = []
    start = time.time()

    while True:
        if time.time() - start > deadline - 6.0:
            break
        gains = uncovered[block_t_indices].sum(axis=1).astype(np.float64)
        max_gain = gains.max()
        if max_gain == 0:
            break

        if alpha > 0:
            # Smooth surrogate: soft-max of gains rewards blocks covering many
            # of the currently-worst uncovered t-subsets, not just the single max.
            m = gains.max()
            sm = np.log(np.sum(np.exp(gains - m)))
            # Gradient of soft-max w.r.t. gains -> weights per block
            w = np.exp(gains - m)
            w = w / w.sum()
            # Perturb by temperature to escape deterministic local choices
            noise = rng.standard_normal(n_k) * alpha
            score = w + noise
            best_idx = int(np.argmax(score))
            if gains[best_idx] == 0:
                best_idx = int(np.argmax(gains))
        else:
            best_idx = int(np.argmax(gains))

        selected.append(best_idx)
        uncovered[block_t_indices[best_idx]] = 0

    return k_subsets, block_t_indices, t_sub_to_idx, [k_subsets[i] for i in selected]


def _remove_redundant(selected, block_t_indices, t_sub_to_idx, deadline):
    """Remove blocks whose t-subsets are all covered by >=2 blocks."""
    n_t = len(t_sub_to_idx)
    coverage = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        for tc in itertools.combinations(b, t_sub_to_idx and 5):
            coverage[t_sub_to_idx[tc]] += 1

    changed = True
    while changed:
        if time.time() > deadline - 3.0:
            break
        changed = False
        for i in range(len(selected)):
            block = selected[i]
            redundant = True
            for tc in itertools.combinations(block, 5):
                if coverage[t_sub_to_idx[tc]] <= 1:
                    redundant = False
                    break
            if redundant:
                for tc in itertools.combinations(block, 5):
                    coverage[t_sub_to_idx[tc]] -= 1
                selected.pop(i)
                changed = True
                break
    return selected


def _swap_polish(selected, k_subsets, block_t_indices, t_sub_to_idx, deadline):
    """Local search: try swapping one block for another to reduce count while
    keeping full coverage."""
    n_t = len(t_sub_to_idx)
    coverage = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        for tc in itertools.combinations(b, 5):
            coverage[t_sub_to_idx[tc]] += 1

    # Precompute which blocks cover each t-subset? Too big. Instead, for each
    # candidate removal, find uncovered t-subsets, then find a block covering all.
    # Only attempt when a removal leaves few holes and one block fixes them.
    improved = True
    while improved:
        if time.time() > deadline - 2.0:
            break
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline - 2.0:
                break
            block = selected[i]
            # Temporarily remove
            holes = []
            for tc in itertools.combinations(block, 5):
                idx = t_sub_to_idx[tc]
                coverage[idx] -= 1
                if coverage[idx] == 0:
                    holes.append(idx)
            if not holes:
                # Pure redundancy
                selected.pop(i)
                improved = True
                break
            # Try to find a single block covering all holes
            hole_set = set(holes)
            found = -1
            for j, cand in enumerate(selected):
                if j == i:
                    continue
                ok = True
                for h in hole_set:
                    # check if cand covers t-subset h
                    if not np.any(block_t_indices[j] == h):
                        ok = False
                        break
                if ok:
                    found = j
                    break
            if found == -1:
                # search all k-subsets
                for j in range(len(k_subsets)):
                    if j == i:
                        continue
                    ok = True
                    for h in hole_set:
                        if not np.any(block_t_indices[j] == h):
                            ok = False
                            break
                    if ok:
                        found = j
                        break
            if found != -1:
                # swap: replace block i with k_subsets[found]
                for tc in itertools.combinations(block, 5):
                    coverage[t_sub_to_idx[tc]] -= 0  # already decremented
                for tc in itertools.combinations(k_subsets[found], 5):
                    coverage[t_sub_to_idx[tc]] += 1
                selected[i] = k_subsets[found]
                improved = True
                break
            else:
                # restore
                for h in holes:
                    coverage[h] += 1
    return selected


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin under 120s

    # Single smooth-greedy pass, then polish. Keep it simple and robust.
    rng = np.random.default_rng(12345)
    k_subsets, block_t_indices, t_sub_to_idx, selected = _greedy_cover(
        v, k, t, deadline, rng, alpha=0.05)

    selected = _remove_redundant(selected, block_t_indices, t_sub_to_idx, deadline)

    if time.time() < deadline - 3.0:
        selected = _swap_polish(selected, k_subsets, block_t_indices, t_sub_to_idx, deadline)

    # Final safety: verify coverage; if broken, fall back to greedy result.
    return [list(b) for b in selected]
