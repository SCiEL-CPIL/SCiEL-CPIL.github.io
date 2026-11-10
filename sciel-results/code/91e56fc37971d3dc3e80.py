"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import bisect
from functools import reduce

def _is_dissociated(elems):
    """Exact check: all reciprocal subset sums (incl. empty) distinct, via integer LCM scaling."""
    if not elems:
        return True
    if len(elems) == 1:
        return True
    n = len(elems)
    try:
        L = reduce(math.lcm, elems)
    except (ValueError, OverflowError):
        return False
    weights = [L // e for e in elems]
    sums = {0}
    for w in weights:
        new_sums = {s + w for s in sums}
        if sums & new_sums:
            return False
        sums |= new_sums
    return len(sums) == (1 << n)

def _build_sums(elems):
    """Return (L, weights, sums) for a dissociated set, else None."""
    if not elems:
        return (1, [], {0})
    L = reduce(math.lcm, elems)
    weights = [L // e for e in elems]
    sums = {0}
    for w in weights:
        new_sums = {s + w for s in sums}
        if sums & new_sums:
            return None
        sums |= new_sums
    return (L, weights, sums)

def _collision_check(sums, new_w):
    """True iff some s and s' in sums satisfy s + new_w == s' (a collision)."""
    if len(sums) > 4096:
        arr = sorted(sums)
        for s in arr:
            t = s + new_w
            i = bisect.bisect_left(arr, t)
            if i < len(arr) and arr[i] == t:
                return True
        return False
    else:
        for s in sums:
            if (s + new_w) in sums:
                return True
        return False

def _can_add(L, weights, sums, e):
    """Try to add element e. Returns (new_L, new_weights, new_sums) or None."""
    new_L = math.lcm(L, e)
    scale = new_L // L
    new_w = new_L // e
    if scale == 1:
        if _collision_check(sums, new_w):
            return None
        new_weights = weights + [new_w]
        new_sums = sums | {s + new_w for s in sums}
        return (new_L, new_weights, new_sums)
    else:
        scaled = {s * scale for s in sums}
        if _collision_check(scaled, new_w):
            return None
        new_weights = [w * scale for w in weights] + [new_w]
        new_sums = scaled | {s + new_w for s in scaled}
        return (new_L, new_weights, new_sums)

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin within 90s wall clock
    universe = list(range(1, 23))

    def time_left():
        return deadline - time.time()

    def greedy(order):
        L, weights, sums, res = 1, [], {0}, []
        for e in order:
            r = _can_add(L, weights, sums, e)
            if r is not None:
                L, weights, sums = r
                res.append(e)
        return res

    def local_add(s, L, weights, sums):
        improved = True
        while improved and time_left() > 3:
            improved = False
            s_set = set(s)
            rem = [e for e in universe if e not in s_set]
            random.shuffle(rem)
            for e in rem:
                r = _can_add(L, weights, sums, e)
                if r is not None:
                    L, weights, sums = r
                    s = s + [e]
                    improved = True
                    break
        return s, L, weights, sums

    def local_swap(s, L, weights, sums):
        improved = True
        while improved and time_left() > 3:
            improved = False
            s_set = set(s)
            out = [e for e in universe if e not in s_set]
            random.shuffle(out)
            for u in list(s):
                if time_left() < 3:
                    break
                for v in out:
                    if time_left() < 3:
                        break
                    cand = [x for x in s if x != u] + [v]
                    r = _build_sums(cand)
                    if r is not None:
                        L, weights, sums = r
                        s = cand
                        s, L, weights, sums = local_add(s, L, weights, sums)
                        improved = True
                        break
                if improved:
                    break
        return s, L, weights, sums

    def local_remove_add2(s, L, weights, sums):
        """Remove one element, then try to add two different ones (net +1)."""
        improved = True
        while improved and time_left() > 3:
            improved = False
            s_set = set(s)
            out = [e for e in universe if e not in s_set]
            random.shuffle(out)
            for u in list(s):
                if time_left() < 3:
                    break
                elems = [x for x in s if x != u]
                r = _build_sums(elems)
                if r is None:
                    continue
                L2, w2, s2 = r
                added = 0
                for v in out:
                    if time_left() < 3:
                        break
                    r2 = _can_add(L2, w2, s2, v)
                    if r2 is not None:
                        L2, w2, s2 = r2
                        elems.append(v)
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    s, L, weights, sums = elems, L2, w2, s2
                    s, L, weights, sums = local_add(s, L, weights, sums)
                    improved = True
                    break
        return s, L, weights, sums

    # ---- Phase 1: deterministic greedy orderings ----
    best_set, best_L, best_w, best_sums = [], 1, [], {0}
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        list(universe),
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 2, 3, 5, 7, 11, 13, 17, 19],
    ]
    for order in orders:
        if time_left() < 8:
            break
        res = greedy(order)
        if len(res) > len(best_set):
            r = _build_sums(res)
            if r:
                best_set, best_L, best_w, best_sums = res, r[0], r[1], r[2]
    best_set, best_L, best_w, best_sums = local_add(best_set, best_L, best_w, best_sums)

    # ---- Phase 2: branch-and-bound over element orderings ----
    def branch_and_bound(remaining, L, weights, sums, chosen, best_len):
        """Recursively try to extend chosen by picking an element from remaining.
        Returns (chosen, L, weights, sums, len) for the best found in this subtree."""
        if not remaining or time_left() < 3:
            return chosen, L, weights, sums, len(chosen)
        if len(chosen) + len(remaining) <= best_len[0]:
            return chosen, L, weights, sums, len(chosen)
        # Try candidates that can be added, preferring larger (more separating) ones
        cands = [e for e in remaining if _can_add(L, weights, sums, e) is not None]
        if not cands:
            return chosen, L, weights, sums, len(chosen)
        cands.sort(reverse=True)
        best_local = (chosen, L, weights, sums, len(chosen))
        for e in cands:
            if time_left() < 3:
                break
            r = _can_add(L, weights, sums, e)
            if r is None:
                continue
            new_chosen = chosen + [e]
            sub = branch_and_bound(
                [x for x in remaining if x != e],
                r[0], r[1], r[2], new_chosen, best_len)
            if len(sub[4]) > best_local[4]:
                best_local = sub
                best_len[0] = best_local[4]
        return best_local

    # Run B&B from a few starting seeds (including the current best and a fresh greedy)
    seeds = [list(best_set)]
    seeds.append(greedy(sorted(universe, reverse=True)))
    for seed in seeds:
        if time_left() < 10:
            break
        r = _build_sums(seed)
        if not r:
            continue
        L0, w0, s0 = r
        in_seed = set(seed)
        rem = [e for e in universe if e not in in_seed]
        best_len = [len(seed)]
        res = branch_and_bound(rem, L0, w0, s0, list(seed), best_len)
        if res[4] > len(best_set):
            r2 = _build_sums(res[0])
            if r2:
                best_set, best_L, best_w, best_sums = res[0], r2[0], r2[1], r2[2]
        best_set, best_L, best_w, best_sums = local_add(best_set, best_L, best_w, best_sums)

    # ---- Phase 3: local improvement loop (add / swap / remove-add2) ----
    for _ in range(6):
        if time_left() < 5:
            break
        before = len(best_set)
        best_set, best_L, best_w, best_sums = local_add(best_set, best_L, best_w, best_sums)
        if len(best_set) > before or time_left() < 5:
            before = len(best_set)
            best_set, best_L, best_w, best_sums = local_swap(best_set, best_L, best_w, best_sums)
            if len(best_set) > before or time_left() < 5:
                before = len(best_set)
                best_set, best_L, best_w, best_sums = local_remove_add2(best_set, best_L, best_w, best_sums)
        if len(best_set) == before and time_left() < 5:
            break

    # ---- Phase 4: final exact add pass in descending order ----
    if time_left() > 3:
        s_set = set(best_set)
        for e in sorted([e for e in universe if e not in s_set], reverse=True):
            if time_left() < 3:
                break
            r = _can_add(best_L, best_w, best_sums, e)
            if r is not None:
                best_L, best_w, best_sums = r
                best_set.append(e)

    # ---- Final verification; fallback if invalid ----
    if best_set and _is_dissociated(best_set):
        return sorted(best_set)
    # Fallback to a known valid set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
