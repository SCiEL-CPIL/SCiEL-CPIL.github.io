"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: Cyclic orbit greedy initialization + integer-bitset local search.
All coverage is tracked as Python int bitmasks (C(19,4)=3876 bits fits in ~500 bytes).
"""
import time
import itertools
import random

_DEADLINE = 115.0
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute t-subset index map and integer-bitmask coverage for every k-subset."""
    t_index = {}
    idx = 0
    for c in itertools.combinations(range(v), t):
        t_index[c] = idx
        idx += 1
    n_t = idx
    all_t = list(t_index.keys())

    k_subsets = list(itertools.combinations(range(v), k))
    cov = []
    for ks in k_subsets:
        mask = 0
        for ts in itertools.combinations(ks, t):
            mask |= (1 << t_index[ts])
        cov.append(mask)
    return all_t, t_index, k_subsets, cov


def _cyclic_orbit_greedy(v, k, t, rng):
    """Greedy construction exploiting Z_v cyclic symmetry (v=19 prime, free action)."""
    all_t, t_index, k_subsets, cov = _precompute(v, k, t)
    n_t = len(all_t)
    full = (1 << n_t) - 1

    k_tuples = [tuple(ks) for ks in k_subsets]
    tuple_to_idx = {tp: i for i, tp in enumerate(k_tuples)}

    # Build orbits under Z_v
    orbit_reps = {}
    for i, ks in enumerate(k_subsets):
        rotations = []
        for shift in range(v):
            rot = tuple(sorted((x + shift) % v for x in ks))
            rotations.append(rot)
        canonical = min(rotations)
        if canonical not in orbit_reps:
            orbit = []
            for rot in rotations:
                if rot in tuple_to_idx:
                    orbit.append(tuple_to_idx[rot])
            orbit_reps[canonical] = orbit

    # Compute union coverage mask for each orbit
    orbit_cov = {}
    for canonical, orbit in orbit_reps.items():
        m = 0
        for idx in orbit:
            m |= cov[idx]
        orbit_cov[canonical] = (orbit, m)

    # Greedy: select orbits maximizing uncovered coverage
    uncovered = full
    selected_blocks = []

    while uncovered and _elapsed() < _DEADLINE - 35:
        best_orbit = None
        best_gain = 0
        for canonical, (orbit, m) in orbit_cov.items():
            g = (m & uncovered).bit_count()
            if g > best_gain:
                best_gain = g
                best_orbit = canonical
        if best_orbit is None or best_gain == 0:
            break
        orbit, m = orbit_cov[best_orbit]
        selected_blocks.extend(orbit)
        uncovered &= ~m

    return selected_blocks, all_t, t_index, k_subsets, cov, full


def _local_search(blocks, cov, n_t, k_subsets, full, rng, time_budget):
    """Removal + swap + add local search using integer bitmasks."""
    blocks = list(blocks)

    # Precompute union of all blocks
    def compute_union(blks):
        u = 0
        for b in blks:
            u |= cov[b]
        return u

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            # Union of all blocks except i
            u = 0
            for idx, bb in enumerate(blocks):
                if idx != i:
                    u |= cov[bb]
            if u == full:
                blocks.pop(i)
                improved = True
                continue
            i += 1

        # --- Swap pass: try replacing random blocks to enable further removals ---
        for _ in range(50):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(len(k_subsets))
            if j in blocks:
                continue
            old = blocks[i]
            blocks[i] = j
            # Try removal after swap
            removed = False
            for idx in range(len(blocks)):
                u = 0
                for idx2, bb2 in enumerate(blocks):
                    if idx2 != idx:
                        u |= cov[bb2]
                if u == full:
                    blocks.pop(idx)
                    removed = True
                    improved = True
                    break
            if not removed:
                blocks[i] = old

        # --- Repair if needed ---
        u = compute_union(blocks)
        if u != full:
            missing = full & ~u
            while missing and _elapsed() < time_budget:
                best_i = -1
                best_g = 0
                for idx, c in enumerate(cov):
                    g = (c & missing).bit_count()
                    if g > best_g:
                        best_g = g
                        best_i = idx
                if best_i < 0:
                    break
                blocks.append(best_i)
                missing &= ~cov[best_i]
                improved = True

    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    # Stage 1: Cyclic orbit greedy
    blocks, all_t, t_index, k_subsets, cov, full = _cyclic_orbit_greedy(v, k, t, rng)
    n_t = len(all_t)

    # Stage 2: Local search
    time_budget = _DEADLINE - 5
    blocks = _local_search(blocks, cov, n_t, k_subsets, full, rng, time_budget)

    # Safety repair
    u = 0
    for b in blocks:
        u |= cov[b]
    if u != full:
        missing = full & ~u
        while missing and _elapsed() < _DEADLINE - 2:
            best_i = -1
            best_g = 0
            for idx, c in enumerate(cov):
                g = (c & missing).bit_count()
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i < 0:
                break
            blocks.append(best_i)
            missing &= ~cov[best_i]

    # Return as list of k-subsets
    result = [list(k_subsets[b]) for b in blocks]
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
