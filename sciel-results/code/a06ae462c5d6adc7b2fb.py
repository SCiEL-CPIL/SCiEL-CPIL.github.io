"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: fast bitset greedy construction + aggressive local search (removal/swap/add).
"""
import time
import itertools
import random

_DEADLINE = 115.0  # seconds, safety margin under 120s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets as ints, and for every k-subset its coverage bitmap."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    # For each k-subset, the set of t-subset indices it covers
    k_subsets = list(itertools.combinations(range(v), k))
    # coverage: list of frozenset of t-indices for each k-subset (as a set for speed)
    coverage = []
    for ks in k_subsets:
        cov = set()
        for ts in itertools.combinations(ks, t):
            cov.add(t_index[ts])
        coverage.append(cov)
    return all_t, t_index, k_subsets, coverage


def _greedy_construct(v, k, t, rng):
    """Greedy set cover: repeatedly add the k-subset covering the most uncovered t-subsets."""
    all_t, t_index, k_subsets, coverage = _precompute(v, k, t)
    n_t = len(all_t)
    # uncovered as a set of t-indices
    uncovered = set(range(n_t))
    blocks = []  # list of k-subset index
    # We'll maintain coverage as sets; recompute gain each iteration (expensive but ok for moderate)
    # Optimization: only consider k-subsets that share at least one point with uncovered? Hard.
    # For v=19,k=7: C(19,7)=50388 k-subsets. Each has C(7,4)=35 t-subsets.
    # Gain = len(coverage[i] & uncovered). Do this with set intersection.
    while uncovered and _elapsed() < _DEADLINE - 20:
        best_i = -1
        best_gain = 0
        for i, cov in enumerate(coverage):
            g = len(cov & uncovered)
            if g > best_gain:
                best_gain = g
                best_i = i
                if best_gain == 0:
                    break
        if best_i < 0:
            break
        blocks.append(best_i)
        uncovered -= coverage[best_i]
    return blocks, all_t, t_index, k_subsets, coverage


def _coverage_of_blocks(blocks, coverage):
    """Union of coverage sets of the given block indices."""
    cov = set()
    for b in blocks:
        cov |= coverage[b]
    return cov


def _is_covered(blocks, coverage, n_t):
    cov = set()
    for b in blocks:
        cov |= coverage[b]
    return len(cov) == n_t


def _local_search(blocks, coverage, n_t, k_subsets, rng, time_budget):
    """Removal + swap + add local search to minimize |blocks| while staying a cover."""
    blocks = list(blocks)
    n = len(blocks)

    def union_cov(exclude=None):
        cov = set()
        for idx, b in enumerate(blocks):
            if exclude is not None and idx == exclude:
                continue
            cov |= coverage[b]
        return cov

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            # Check if removing b still covers everything
            cov = set()
            for idx, bb in enumerate(blocks):
                if idx == i:
                    continue
                cov |= coverage[bb]
            if len(cov) == n_t:
                blocks.pop(i)
                improved = True
                continue
            i += 1
        # --- Swap pass: replace block i with some j not in blocks if it improves (same size, but reduces redundancy) ---
        # Actually swap keeps size; we want to reduce size via removal. Swap helps find better removals.
        # Do a few random swap attempts to escape local minima, then removal again.
        for _ in range(30):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(len(k_subsets))
            if j in blocks:
                continue
            # Try swapping: does the new set still cover? If yes, keep (same size) — only keep if it enables later removals.
            # We'll tentatively swap and see if removal then possible.
            old = blocks[i]
            blocks[i] = j
            # quick removal check
            removed = False
            for idx, bb in enumerate(blocks):
                cov = set()
                for idx2, bb2 in enumerate(blocks):
                    if idx2 == idx:
                        continue
                    cov |= coverage[bb2]
                if len(cov) == n_t:
                    blocks.pop(idx)
                    removed = True
                    break
            if removed:
                improved = True
            else:
                blocks[i] = old  # revert
        # --- Add pass: if not covered (shouldn't happen), add best block ---
        if len(blocks) > 0:
            cov = set()
            for b in blocks:
                cov |= coverage[b]
            if len(cov) < n_t:
                # add the block covering most missing
                missing = set(range(n_t)) - cov
                best_i = -1
                best_g = 0
                for idx, c in enumerate(coverage):
                    g = len(c & missing)
                    if g > best_g:
                        best_g = g
                        best_i = idx
                if best_i >= 0:
                    blocks.append(best_i)
    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, coverage = _precompute(v, k, t)
    n_t = len(all_t)

    # Greedy construction
    blocks, _, _, _, _ = _greedy_construct(v, k, t, rng)
    # Convert to list of block indices (already are)
    # Local search
    time_budget = _DEADLINE - 5
    blocks = _local_search(blocks, coverage, n_t, k_subsets, rng, time_budget)

    # Safety: ensure valid; if not, do a quick repair
    cov = set()
    for b in blocks:
        cov |= coverage[b]
    if len(cov) < n_t:
        # repair with greedy add
        while len(cov) < n_t and _elapsed() < _DEADLINE - 2:
            missing = set(range(n_t)) - cov
            best_i = -1
            best_g = 0
            for idx, c in enumerate(coverage):
                g = len(c & missing)
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i < 0:
                break
            blocks.append(best_i)
            cov |= coverage[best_i]

    # Return as list of k-subsets (lists of ints)
    result = [list(k_subsets[b]) for b in blocks]
    # Deduplicate
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
