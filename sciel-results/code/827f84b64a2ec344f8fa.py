"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: Two-stage cyclic construction + aggressive local search with bitset coverage.
Stage 1: Greedy on orbits under Z19 (free action, orbit size 19).
Stage 2: Local search on individual blocks using integer bitsets for speed.
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


def _cyclic_orbit_greedy(v, k, t, rng):
    """
    Stage 1: Greedy construction exploiting Z_v cyclic symmetry.
    Since v=19 is prime, the action is free: each orbit has exactly v blocks.
    We select orbits (sets of v blocks) greedily to maximize uncovered t-subset coverage.
    """
    all_t, t_index, k_subsets, coverage = _precompute(v, k, t)
    n_t = len(all_t)
    
    # Generate all orbits under Z_v: orbit of a k-subset S = {S, S+1, ..., S+v-1} mod v
    # Canonical representative: lexicographically smallest rotation
    orbit_reps = {}  # canonical tuple -> orbit (list of k-subset indices)
    for i, ks in enumerate(k_subsets):
        # Generate all rotations
        rotations = []
        for shift in range(v):
            rot = tuple(sorted((x + shift) % v for x in ks))
            rotations.append(rot)
        canonical = min(rotations)
        if canonical not in orbit_reps:
            # Build the orbit: find all k-subsets in this orbit
            orbit = []
            for shift in range(v):
                rot = tuple(sorted((x + shift) % v for x in ks))
                # Find index of this rotation in k_subsets
                # We need a mapping from tuple to index
                orbit.append(i)  # This is wrong; need proper mapping
            orbit_reps[canonical] = []
    
    # Better approach: build orbit representatives properly
    k_subsets_tuple = [tuple(ks) for ks in k_subsets]
    tuple_to_idx = {tup: i for i, tup in enumerate(k_subsets_tuple)}
    
    orbit_reps = {}
    for i, ks in enumerate(k_subsets):
        rotations = []
        for shift in range(v):
            rot = tuple(sorted((x + shift) % v for x in ks))
            rotations.append(rot)
        canonical = min(rotations)
        if canonical not in orbit_reps:
            # Collect all blocks in this orbit
            orbit = []
            for shift in range(v):
                rot = tuple(sorted((x + shift) % v for x in ks))
                if rot in tuple_to_idx:
                    orbit.append(tuple_to_idx[rot])
            orbit_reps[canonical] = orbit
    
    # For each orbit, compute the union of coverage of all v blocks in the orbit
    orbit_coverages = {}
    for canonical, orbit in orbit_reps.items():
        cov = set()
        for idx in orbit:
            cov |= coverage[idx]
        orbit_coverages[canonical] = (orbit, cov)
    
    # Greedy: select orbits to maximize uncovered coverage
    uncovered = set(range(n_t))
    selected_orbits = []
    selected_blocks = []
    
    while uncovered and _elapsed() < _DEADLINE - 30:
        best_orbit = None
        best_gain = 0
        for canonical, (orbit, cov) in orbit_coverages.items():
            g = len(cov & uncovered)
            if g > best_gain:
                best_gain = g
                best_orbit = canonical
                if best_gain == 0:
                    break
        if best_orbit is None:
            break
        orbit, cov = orbit_coverages[best_orbit]
        selected_orbits.append(best_orbit)
        selected_blocks.extend(orbit)
        uncovered -= cov
    
    return selected_blocks, all_t, t_index, k_subsets, coverage


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    # Stage 1: Cyclic orbit greedy
    blocks, all_t, t_index, k_subsets, coverage = _cyclic_orbit_greedy(v, k, t, rng)
    n_t = len(all_t)
    
    # Stage 2: Local search on individual blocks
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
