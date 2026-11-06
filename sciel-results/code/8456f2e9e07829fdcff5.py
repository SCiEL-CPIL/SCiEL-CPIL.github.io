"""Maximize cardinality of a dissociated reciprocal subset of [1,22].

A set S is dissociated iff all 2^|S| reciprocal subset sums (including the
empty sum 0) are distinct. We search with fast float64 arithmetic (a
systematic backtracking search with pruning) to explore the combinatorial
space, then verify the best candidate exactly with Fraction arithmetic.
"""
import time
import random

_DEADLINE = 78.0  # seconds; safety margin before the 90 s evaluator kill
_START = time.time()

def _deadline_hit():
    return (time.time() - _START) > _DEADLINE

def _float_dissociated(recip, tol=1e-12):
    """Fast float check: are all 2^k subset sums of `recip` distinct?"""
    k = len(recip)
    n = 1 << k
    sums = [0.0] * n
    # sums[mask] = sum of recip[i] for i in mask
    # Build incrementally
    for i in range(k):
        r = recip[i]
        bit = 1 << i
        for mask in range(bit):
            sums[bit | mask] = sums[mask] + r
    # Check all distinct
    seen = set()
    for s in sums:
        key = round(s, 11)
        if key in seen:
            return False
        seen.add(key)
    return True

def _exact_verify(chosen):
    """Exact Fraction verification that all subset sums are distinct."""
    from fractions import Fraction
    k = len(chosen)
    n = 1 << k
    sums = [Fraction(0)] * n
    for i in range(k):
        r = Fraction(1, chosen[i])
        bit = 1 << i
        for mask in range(bit):
            sums[bit | mask] = sums[mask] + r
    return len(set(sums)) == n

def _backtrack_search(order, best_k, tol=1e-12):
    """Backtracking search over elements in `order` (indices 0..n-1).
    Tries to build a set of size > best_k with all subset sums distinct.
    Returns a list of chosen elements (the actual values, not indices)."""
    n = len(order)
    recip = [1.0 / order[i] for i in range(n)]
    chosen_idx = []
    # sums[mask] for the current chosen set
    # We'll maintain the list of subset sums for the chosen set.
    
    # For efficiency, maintain the set of current subset sums as a list.
    cur_sums = [0.0]
    
    def bt(pos, depth, cur_sums, chosen_idx):
        if _deadline_hit():
            return None
        # Pruning: even if we add all remaining, can we beat best_k?
        remaining = n - pos
        if depth + remaining <= best_k:
            return None
        if pos == n:
            if depth > best_k:
                return list(chosen_idx)
            return None
        # Option 1: skip order[pos]
        res = bt(pos + 1, depth, cur_sums, chosen_idx)
        if res is not None:
            return res
        # Option 2: take order[pos]
        r = recip[pos]
        # Check collision: for each s in cur_sums, s+r must not be in cur_sums
        # Build set of current sums
        cur_set = set(round(s, 11) for s in cur_sums)
        ok = True
        new_sums = cur_sums[:]
        for s in cur_sums:
            ns = s + r
            key = round(ns, 11)
            if key in cur_set:
                ok = False
                break
            new_sums.append(ns)
        if ok:
            chosen_idx.append(pos)
            res = bt(pos + 1, depth + 1, new_sums, chosen_idx)
            if res is not None:
                return res
            chosen_idx.pop()
        return None
    
    result = bt(0, 0, cur_sums, chosen_idx)
    if result is None:
        return None
    return [order[i] for i in result]

def _greedy(order):
    """Simple greedy: add elements in order if no collision (float)."""
    cur_sums = [0.0]
    cur_set = {0.0}
    chosen = []
    for a in order:
        r = 1.0 / a
        ok = True
        new_sums = cur_sums[:]
        for s in cur_sums:
            ns = s + r
            key = round(ns, 11)
            if key in cur_set:
                ok = False
                break
            new_sums.append(ns)
        if ok:
            cur_sums = new_sums
            for s in new_sums[len(cur_sums) - len(new_sums):]:
                pass
            # update cur_set
            cur_set = set(round(s, 11) for s in new_sums)
            chosen.append(a)
    return chosen

def construct():
    global _START
    _START = time.time()
    random.seed()
    full = list(range(1, 23))

    # Deterministic seed baseline (known valid, 9 elements).
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best = seed
    best_k = len(seed)

    # Strategy: random restarts. For each restart, do a greedy to get a
    # decent baseline, then backtracking search to try to beat it.
    # Also try a systematic backtracking with a fixed order sorted by
    # descending value (larger reciprocals first tend to be more "spread out").

    # First: try systematic backtracking with order sorted descending.
    order_desc = sorted(full, reverse=True)
    cand = _backtrack_search(order_desc, best_k)
    if cand and len(cand) > best_k:
        best_k = len(cand)
        best = cand

    # Also try ascending.
    order_asc = sorted(full)
    cand = _backtrack_search(order_asc, best_k)
    if cand and len(cand) > best_k:
        best_k = len(cand)
        best = cand

    # Random restarts with backtracking.
    restarts = 0
    while not _deadline_hit():
        restarts += 1
        order = full[:]
        random.shuffle(order)
        # Quick greedy to estimate potential
        g = _greedy(order)
        if len(g) <= best_k:
            # Greedy couldn't beat best; backtracking might still, but
            # only if it has a chance. Try backtracking if remaining time.
            if _deadline_hit():
                break
            cand = _backtrack_search(order, best_k)
            if cand and len(cand) > best_k:
                best_k = len(cand)
                best = cand
                if best_k >= 14:
                    break
        else:
            # Greedy beat best; try to improve with backtracking.
            best_k = len(g)
            best = g
            cand = _backtrack_search(order, best_k)
            if cand and len(cand) > best_k:
                best_k = len(cand)
                best = cand
                if best_k >= 14:
                    break

    # Final exact verification of the best candidate.
    if best is None or len(best) == 0:
        best = [1]
    # Verify with exact arithmetic; if it fails (float false positive),
    # fall back to the known-valid seed.
    if not _exact_verify(best):
        best = seed
    return list(best)
