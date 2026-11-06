import time
import numpy as np
from itertools import combinations

_DEADLINE = 85.0  # seconds wall-clock safety margin

def _any_triple_divisible(arr, d):
    """Return True if any 3 distinct elements of arr sum to a multiple of d."""
    m = len(arr)
    if m < 3:
        return False
    # suffix residue sets: res_set[i] = set of arr[k] % d for k in range(i, m)
    # We'll iterate i (smallest), j (middle), and check if needed residue exists among k>j.
    suffix = [None] * (m + 1)
    suffix[m] = frozenset()
    for i in range(m - 1, -1, -1):
        suffix[i] = suffix[i + 1] | {arr[i] % d}
    for i in range(m - 2):
        ai = arr[i]
        for j in range(i + 1, m - 1):
            need = (-ai - arr[j]) % d
            if need in suffix[j + 1]:
                return True
    return False


def _can_add(S, x):
    """S is a list/set of selected ints (not containing x). Check if adding x keeps validity."""
    S_list = list(S)
    m = len(S_list)
    # 1) For a = x: any triple of S_list sum divisible by x?
    if _any_triple_divisible(S_list, x):
        return False
    # 2) For each a in S_list: triples {x, u, v} with u,v in S_list\{a}
    #    need (x+u+v) % a != 0 for all u<v.
    #    i.e. no pair (u,v) in S_list\{a} with (u+v)%a == (-x)%a.
    #    Optimized: precompute residue sets per a in O(m) using a frequency dict.
    for a in S_list:
        need = (-x) % a
        # We need: exists u,v in S_list\{a} with (u+v) % a == need
        # Equivalent: exists r1, r2 in residues of S_list\{a} such that (r1+r2)%a == need
        # Build residue frequency for S_list excluding a
        freq = {}
        for v in S_list:
            if v == a:
                continue
            r = v % a
            freq[r] = freq.get(r, 0) + 1
        # Check all pairs of residues
        # For each residue r1, we need r2 = (need - r1) % a
        # If r1 != r2: need freq[r2] >= 1
        # If r1 == r2: need freq[r1] >= 2
        for r1, c1 in freq.items():
            r2 = (need - r1) % a
            if r1 == r2:
                if c1 >= 2:
                    return False
            else:
                if r2 in freq:
                    return False
    return True


def _is_valid(S):
    """Full validation of set S."""
    S_list = list(S)
    m = len(S_list)
    for a in S_list:
        others = [v for v in S_list if v != a]
        if _any_triple_divisible(others, a):
            return False
    return True


def _greedy(order, deadline):
    """Greedy add in given order, respecting deadline."""
    selected = []
    sel_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(sel_set, x):
            selected.append(x)
            sel_set.add(x)
    return selected


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    n = 192
    best = []

    # Seed 1: descending (matches original, known good)
    cand = _greedy(range(n, 0, -1), deadline)
    if len(cand) > len(best):
        best = cand

    # Seed 2: ascending
    if time.time() < deadline:
        cand = _greedy(range(1, n + 1), deadline)
        if len(cand) > len(best):
            best = cand

    # Seed 3: multiples / large-odd mix — try a few heuristics
    if time.time() < deadline:
        # prefer numbers with few divisors among range: use a fixed pseudo-random-ish order
        import random
        rng = random.Random(42)
        order = list(range(1, n + 1))
        rng.shuffle(order)
        cand = _greedy(order, deadline)
        if len(cand) > len(best):
            best = cand

    # Seed 4: high-then-low — start with large numbers (harder to divide), then try small ones
    if time.time() < deadline:
        order = list(range(n, n // 2, -1)) + list(range(1, n // 2 + 1))
        cand = _greedy(order, deadline)
        if len(cand) > len(best):
            best = cand

    # Seed 5: another random seed with different seed value
    if time.time() < deadline:
        import random
        rng = random.Random(123)
        order = list(range(1, n + 1))
        rng.shuffle(order)
        cand = _greedy(order, deadline)
        if len(cand) > len(best):
            best = cand

    # Local search: try to improve best via swaps/adds
    best_set = set(best)
    
    # Pass 1: ascending add + swap
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # try adding any unselected x
        for x in range(1, n + 1):
            if time.time() > deadline - 2.0:
                break
            if x not in best_set:
                if _can_add(best_set, x):
                    best_set.add(x)
                    improved = True
        # try swaps: remove one a, add one x (x not in set) if still valid and same size -> no gain;
        # only useful if enables later adds. Do a simple 1-out-1-in that keeps size, then re-run add loop.
        # To gain cardinality we mainly rely on the add loop; swaps help escape local optima.
        for a in list(best_set):
            if time.time() > deadline - 2.0:
                break
            trial = best_set - {a}
            for x in range(1, n + 1):
                if time.time() > deadline - 2.0:
                    break
                if x in trial:
                    continue
                if _can_add(trial, x):
                    # size unchanged; accept if it allows a new add afterwards
                    trial2 = trial | {x}
                    # try to add one more
                    for y in range(1, n + 1):
                        if y in trial2:
                            continue
                        if _can_add(trial2, y):
                            # net +1
                            best_set = trial2 | {y}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # Pass 2: try removing larger elements first (different swap order to escape local optima)
    if time.time() < deadline - 2.0:
        improved2 = True
        while improved2 and time.time() < deadline - 2.0:
            improved2 = False
            # try adding any unselected x
            for x in range(1, n + 1):
                if time.time() > deadline - 2.0:
                    break
                if x not in best_set:
                    if _can_add(best_set, x):
                        best_set.add(x)
                        improved2 = True
            # try swaps: remove larger elements first
            for a in sorted(list(best_set), reverse=True):
                if time.time() > deadline - 2.0:
                    break
                trial = best_set - {a}
                for x in range(1, n + 1):
                    if time.time() > deadline - 2.0:
                        break
                    if x in trial:
                        continue
                    if _can_add(trial, x):
                        trial2 = trial | {x}
                        for y in range(1, n + 1):
                            if y in trial2:
                                continue
                            if _can_add(trial2, y):
                                best_set = trial2 | {y}
                                improved2 = True
                                break
                        if improved2:
                            break
                    if improved2:
                        break
                if improved2:
                    break

    result = list(best_set)
    # Final safety validation
    if _is_valid(result):
        return result
    # Fallback: rebuild greedily descending (fast) and validate
    fb = _greedy(range(n, 0, -1), start + _DEADLINE)
    if _is_valid(fb):
        return fb
    return [n]  # last resort, always valid
