import itertools
import time
import random
import numpy as np


def _deadline_ok(deadline):
    return time.time() < deadline


def _greedy_cover(sub_to_blocks, block_masks, n_t, deadline, rng,
                  full_set=None, max_iter=None):
    """Greedy set cover over rare subsets. full_set: mask of all uncovered (bit i = subset i)."""
    if full_set is None:
        full_set = np.ones((n_t + 63) // 64, dtype=np.uint64)
    uncovered = full_set.copy()
    selected = []
    it = 0
    while True:
        if not _deadline_ok(deadline):
            break
        if max_iter is not None and it >= max_iter:
            break
        it += 1
        # find rarest uncovered subset
        best_sub = -1
        best_cnt = 10**9
        best_cand = None
        for s, lst in sub_to_blocks:
            if not (uncovered[s >> 6] >> (s & 63)) & 1:
                continue
            c = len(lst)
            if c < best_cnt:
                best_cnt = c
                best_sub = s
                best_cand = lst
                if c == 1:
                    break
        if best_sub < 0:
            break
        # pick candidate with max coverage of uncovered
        best_block = -1
        best_gain = -1
        for b in best_cand:
            gain = int(np.count_nonzero(np.bitwise_and(block_masks[b], uncovered)))
            if gain > best_gain:
                best_gain = gain
                best_block = b
            elif gain == best_gain and gain > 0 and rng.random() < 0.5:
                best_block = b
        selected.append(best_block)
        uncovered = np.bitwise_and(uncovered, np.invert(block_masks[best_block]))
    return selected


def _prune_redundant(selected, block_masks, n_t, deadline):
    """Iteratively remove blocks whose coverage is fully covered by the others."""
    if not selected:
        return selected
    sel = list(selected)
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        n = len(sel)
        if n <= 1:
            break
        for i in range(n):
            if not _deadline_ok(deadline):
                break
            others = np.ones((n_t + 63) // 64, dtype=np.uint64)
            for j in range(n):
                if j != i:
                    others = np.bitwise_and(others, np.invert(block_masks[sel[j]]))
            if not np.any(np.bitwise_and(block_masks[sel[i]], others)):
                sel.pop(i)
                changed = True
                break
    return sel


def _local_search(selected, block_masks, sub_to_blocks, n_t, deadline, rng,
                  full_set=None):
    """1-remove (redundancy) and 2-for-1 replacement local search."""
    if full_set is None:
        full_set = np.ones((n_t + 63) // 64, dtype=np.uint64)
    sel = list(selected)
    improved = True
    while improved and _deadline_ok(deadline):
        improved = False
        n = len(sel)
        if n <= 1:
            break
        # redundancy removal pass
        for i in range(n):
            if not _deadline_ok(deadline):
                break
            others = np.ones((n_t + 63) // 64, dtype=np.uint64)
            for j in range(n):
                if j != i:
                    others = np.bitwise_and(others, np.invert(block_masks[sel[j]]))
            if not np.any(np.bitwise_and(block_masks[sel[i]], others)):
                sel.pop(i)
                improved = True
                break
        if improved or not _deadline_ok(deadline):
            continue
        # 2-for-1: remove pair (i,j) if union is coverable by one block
        n = len(sel)
        for i in range(n):
            if not _deadline_ok(deadline):
                break
            for j in range(i + 1, n):
                union = np.bitwise_or(block_masks[sel[i]], block_masks[sel[j]])
                # rare uncovered subsets of the union
                cand = None
                feasible = True
                for s, lst in sub_to_blocks:
                    if (union[s >> 6] >> (s & 63)) & 1:
                        if cand is None:
                            cand = list(lst)
                        else:
                            cand = [b for b in cand if b in lst]
                        if not cand:
                            feasible = False
                            break
                if not feasible:
                    continue
                for b in cand:
                    if b == sel[i] or b == sel[j]:
                        continue
                    if np.any(np.bitwise_and(union, np.invert(block_masks[b]))):
                        continue
                    # valid: replace i,j with b
                    sel = [x for x in sel if x != sel[i] and x != sel[j]]
                    sel.append(b)
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
    return sel


def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 6.0  # safety margin

    # 1. All t-subsets and their indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}

    # 2. All k-blocks; coverage bitmask for each
    n_words = (n_t + 63) // 64
    blocks = []
    block_masks = np.zeros((0, n_words), dtype=np.uint64)
    for block in itertools.combinations(range(v), k):
        blocks.append(block)
        m = np.zeros(n_words, dtype=np.uint64)
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            m[idx >> 6] |= (np.uint64(1) << np.uint64(idx & 63))
        block_masks = np.vstack([block_masks, m])
    if not _deadline_ok(deadline):
        return []
    n_blocks = len(blocks)

    # 3. Sparse index: for each t-subset, list of blocks covering it (only keep rare ones)
    # Count occurrences first
    counts = np.zeros(n_t, dtype=np.int64)
    for m in block_masks:
        nz = np.flatnonzero(m)
        for w in nz:
            word = m[w]
            bits = np.flatnonzero(word)
            for b in bits:
                counts[w * 64 + b] += 1
    # Build index for subsets with small count (the binding constraints)
    RARE = 12
    sub_to_blocks = []
    for s in range(n_t):
        if counts[s] <= RARE:
            lst = []
            for b in range(n_blocks):
                if (block_masks[b, s >> 6] >> (s & 63)) & 1:
                    lst.append(b)
            sub_to_blocks.append((s, lst))
    if not _deadline_ok(deadline):
        return []

    # 4. Multiple randomized greedy restarts + prune + local search
    best_result = None
    best_size = float('inf')
    rng = random.Random(42)
    restart = 0
    while _deadline_ok(deadline):
        restart += 1
        selected = _greedy_cover(sub_to_blocks, block_masks, n_t, deadline, rng)
        if not selected:
            continue
        selected = _prune_redundant(selected, block_masks, n_t, deadline)
        if _deadline_ok(deadline):
            selected = _local_search(selected, block_masks, sub_to_blocks, n_t, deadline, rng)
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = list(selected)
        # stop early if we've hit a good size and time is tight
        if restart >= 2 and not _deadline_ok(deadline - 15.0):
            break

    if best_result is None:
        selected = _greedy_cover(sub_to_blocks, block_masks, n_t, deadline, rng)
        selected = _prune_redundant(selected, block_masks, n_t, deadline)
        best_result = selected

    # Verify validity
    if _deadline_ok(deadline - 2.0):
        covered = np.zeros(n_words, dtype=np.uint64)
        for i in best_result:
            covered = np.bitwise_or(covered, block_masks[i])
        # all subsets must be covered
        ok = True
        for s in range(n_t):
            if not (covered[s >> 6] >> (s & 63)) & 1:
                ok = False
                break
        if not ok:
            # fallback: fresh greedy
            selected = _greedy_cover(sub_to_blocks, block_masks, n_t, deadline, rng)
            selected = _prune_redundant(selected, block_masks, n_t, deadline)
            best_result = selected

    return [list(blocks[i]) for i in best_result]
