"""Kissing number in dimension 11 - greedy set packing with compatibility precomputation."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def _build_candidates():
    """Generate candidate vectors: weight-2 (264) and weight-11 all-sign (2048)."""
    cands = []
    # Weight-2 vectors: two nonzero ±1 entries
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                cands.append(v)
    # Weight-11 vectors: all entries ±1
    for mask in range(1 << D):
        v = []
        for i in range(D):
            v.append(1 if (mask >> i) & 1 else -1)
        cands.append(v)
    return cands

def _build_compatibility(cands):
    """Precompute pairwise compatibility: compatible iff |u-v|^2 >= max(||u||^2, ||v||^2).
    Returns list of bitsets (as Python ints) where bit j is 1 if cands[i] is compatible with cands[j]."""
    n = len(cands)
    norms = [sum(x*x for x in v) for v in cands]
    compat = [0] * n
    for i in range(n):
        ni = norms[i]
        mask = 0
        for j in range(i + 1, n):
            nj = norms[j]
            # distance squared
            d2 = 0
            ui = cands[i]; uj = cands[j]
            for k in range(D):
                diff = ui[k] - uj[k]
                d2 += diff * diff
            if d2 >= ni and d2 >= nj:
                mask |= (1 << j)
                compat[j] |= (1 << i)
        compat[i] = mask
    return compat, norms

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    random.seed(12345)

    cands = _build_candidates()
    n = len(cands)
    compat, norms = _build_compatibility(cands)

    # Greedy set packing: repeatedly pick a candidate compatible with all selected
    # Order: prioritize weight-11 vectors first (they're mutually compatible and
    # compatible with weight-2), then weight-2.
    # Actually, let's be smart: weight-11 vectors are mutually compatible (all have
    # same norm 11, pairwise distance = 2*(number of differing signs) >= 2*? Let's check:
    # Two all-±1 vectors in R^11: distance^2 = sum of (a-b)^2 where a,b in {±1}.
    # If they differ in t positions, distance^2 = 2t. Max norm^2 = 11.
    # Need 2t >= 11, so t >= 6 (since t integer, 2*6=12>=11).
    # So two weight-11 vectors are compatible iff they differ in >= 6 positions.
    # They are NOT all mutually compatible. So we can't just add all 2048.
    # But the known result says 22 Type A vectors work. Let's use the compatibility graph.

    # Strategy: Greedy with random restarts, pick candidate with fewest conflicts among unselected
    # to maximize packing.

    def greedy_packing(order):
        """Greedy: iterate through order, add candidate if compatible with all selected."""
        selected = []
        selected_set = set()
        for idx in order:
            # Check compatibility with all selected
            ok = True
            for s in selected:
                if not (compat[idx] >> s) & 1:
                    ok = False
                    break
            if ok:
                selected.append(idx)
                selected_set.add(idx)
        return selected

    # Build orders: weight-11 first, then weight-2 (weight-11 have norm 11, weight-2 have norm 2)
    # Actually, let's try multiple strategies:
    # Strategy 1: weight-11 first, then weight-2
    w11 = [i for i in range(n) if norms[i] == 11]
    w2 = [i for i in range(n) if norms[i] == 2]

    best = []
    best_count = 0

    # Try: all weight-11 in random order, then all weight-2 in random order
    for restart in range(200):
        if time.time() > deadline:
            break
        order11 = w11[:]
        random.shuffle(order11)
        order2 = w2[:]
        random.shuffle(order2)
        selected = greedy_packing(order11 + order2)
        if len(selected) > best_count:
            best_count = len(selected)
            best = selected[:]

        # Also try: weight-2 first, then weight-11
        order2b = w2[:]
        random.shuffle(order2b)
        order11b = w11[:]
        random.shuffle(order11b)
        selected2 = greedy_packing(order2b + order11b)
        if len(selected2) > best_count:
            best_count = len(selected2)
            best = selected2[:]

    # If no selection found (shouldn't happen), fall back to seed
    if not best:
        # Return weight-2 vectors only (seed behavior)
        best = w2[:]

    result = np.array([cands[i] for i in best], dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
