import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 112.0


def _generic_greedy(v, k, t, start_time):
    """Generic randomized greedy for small instances."""
    points = list(range(v))
    all_t = list(itertools.combinations(points, t))
    n_t = len(all_t)
    idx_map = {s: i for i, s in enumerate(all_t)}
    all_blocks = list(itertools.combinations(points, k))
    n_b = len(all_blocks)

    cov = np.zeros((n_b, n_t), dtype=np.int8)
    for b, blk in enumerate(all_blocks):
        for sub in itertools.combinations(blk, t):
            cov[b, idx_map[sub]] = 1

    uncovered = np.ones(n_t, dtype=bool)
    chosen = []
    deadline = start_time + DEADLINE_SECONDS

    while uncovered.any() and time.time() < deadline:
        gains = cov[:, uncovered].sum(axis=1)
        if gains.max() <= 0:
            break
        b = int(np.argmax(gains))
        chosen.append(b)
        uncovered[cov[b] == 1] = False

    return [list(all_blocks[i]) for i in chosen]


def construct(v, k, t):
    start_time = time.time()

    if (v, k, t) != (19, 8, 5):
        return _generic_greedy(v, k, t, start_time)

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)  # 11628
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    n_b = len(all_blocks)  # 75582

    # Coverage matrix (float32 for fast einsum)
    cov = np.zeros((n_b, n_t), dtype=np.float32)
    block_subsets = []  # per-block list of covered t-subset indices
    for b, blk in enumerate(all_blocks):
        subs = [idx_map[sub] for sub in itertools.combinations(blk, t)]
        block_subsets.append(subs)
        if subs:
            cov[b, subs] = 1.0

    # Blocks that cover each t-subset
    blocks_for_subset = [np.where(cov[:, s] == 1.0)[0] for s in range(n_t)]

    rng = random.Random(12345)
    deadline = start_time + DEADLINE_SECONDS

    def coverage_counts(chosen):
        counts = np.zeros(n_t, dtype=np.int32)
        for b in chosen:
            subs = block_subsets[b]
            if subs:
                counts[subs] += 1
        return counts

    def prune(chosen):
        if not chosen:
            return []
        counts = coverage_counts(chosen)
        new_chosen = []
        order = list(range(len(chosen)))
        rng.shuffle(order)
        for i in order:
            b = chosen[i]
            subs = block_subsets[b]
            if subs and np.any(counts[subs] == 1):
                new_chosen.append(b)
            else:
                if subs:
                    counts[subs] -= 1
        return new_chosen

    def local_swap(chosen, counts, max_iters=200):
        """Try to swap one block for another to reduce size; keep improvement."""
        chosen = list(chosen)
        counts = counts.copy()
        for _ in range(max_iters):
            if time.time() > deadline - 1.0:
                break
            # Find a block that is "redundant-ish": covers only already-covered subsets
            # (i.e., for all its subsets counts>1) -> candidate to remove.
            removable = None
            for b in chosen:
                subs = block_subsets[b]
                if subs and not np.any(counts[subs] == 1):
                    removable = b
                    break
            if removable is None:
                break
            # Removing it leaves some subsets at count 0? No: by definition all its
            # subsets have count>=2, so removing keeps all covered. So it's freely removable.
            subs_r = block_subsets[removable]
            counts[subs_r] -= 1
            chosen.remove(removable)
        return chosen, counts

    def greedy_randomized():
        uncovered = np.ones(n_t, dtype=np.float32)
        chosen = []
        while uncovered.any():
            unc_idx = np.where(uncovered > 0.5)[0]
            if len(unc_idx) == 0:
                break
            s = int(rng.choice(unc_idx))
            cands = blocks_for_subset[s]
            if len(cands) == 0:
                break
            gains = np.einsum('ij,j->i', cov[cands], uncovered)
            noise = rng.random() * 0.5
            gains_perturbed = gains + noise
            best_local = int(np.argmax(gains_perturbed))
            b = int(cands[best_local])
            chosen.append(b)
            uncovered[cov[b] == 1.0] = 0.0
        return chosen

    def greedy_greedy():
        """Pure max-gain greedy (deterministic-ish)."""
        uncovered = np.ones(n_t, dtype=np.float32)
        chosen = []
        while uncovered.any():
            gains = np.einsum('ij,j->i', cov, uncovered)
            b = int(np.argmax(gains))
            if gains[b] <= 0:
                break
            chosen.append(b)
            uncovered[cov[b] == 1.0] = 0.0
        return chosen

    best_blocks = None
    best_count = float('inf')

    def consider(chosen):
        nonlocal best_blocks, best_count
        counts = coverage_counts(chosen)
        if np.any(counts == 0):
            return  # invalid
        pruned = prune(chosen)
        pruned_counts = coverage_counts(pruned)
        if np.any(pruned_counts == 0):
            pruned = chosen  # keep valid one
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_blocks = pruned[:]

    # Initial deterministic greedy
    consider(greedy_greedy())

    # Multiple randomized restarts
    while time.time() < deadline - 8.0:
        chosen = greedy_randomized()
        consider(chosen)
        # local improvement via free removals
        if best_blocks is not None:
            counts = coverage_counts(best_blocks)
            improved, counts2 = local_swap(best_blocks, counts, max_iters=100)
            if len(improved) < best_count:
                consider(improved)

    if best_blocks is None:
        best_blocks = prune(greedy_greedy())

    return [list(all_blocks[i]) for i in best_blocks]
