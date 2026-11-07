"""Hypergraph independent set for the 6-set forbidden constraint.
Combines precomputed hypergraph greedy (from current) with a fast direct
validity check (from inspiration) plus a richer local-search / swap machinery."""
import time
from collections import defaultdict

N = 128
DEADLINE = 85.0  # safety margin under 90s

# Precompute forbidden 6-sets {a,b,c,a+b,a+c,b+c}, a<b<c, b+c <= N
_TRIPLES = []
for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        for c in range(b + 1, N + 1):
            s3 = b + c
            if s3 <= N:
                _TRIPLES.append((a, b, c, a + b, a + c, s3))

_NUM_TRIPLES = len(_TRIPLES)

_val_to_triples = defaultdict(list)
for idx, t in enumerate(_TRIPLES):
    for v in t:
        if 1 <= v <= N:
            _val_to_triples[v].append(idx)


def _build_counts(sel):
    counts = [0] * _NUM_TRIPLES
    for v in sel:
        for ti in _val_to_triples[v]:
            counts[ti] += 1
    return counts


def _is_safe(v, counts):
    for ti in _val_to_triples[v]:
        if counts[ti] == 5:
            return False
    return True


def _can_add_direct(S, x):
    """Direct check: can we add x to set S?"""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        if (x + b) > N or (x + b) not in S:
            continue
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            if (x + c) > N or (x + c) not in S:
                continue
            bc = b + c
            if (bc in S) or (bc == x):
                return False
    return True


def _greedy_dynamic(remaining, counts, deadline):
    sel = set()
    result = []
    while remaining and time.time() < deadline:
        best_v = None
        best_danger = None
        for v in remaining:
            danger = 0
            for ti in _val_to_triples[v]:
                if counts[ti] == 5:
                    danger += 1
            if best_danger is None or danger < best_danger:
                best_danger = danger
                best_v = v
            if danger == 0:
                break
        if best_v is None:
            break
        if best_danger > 0:
            found = False
            for v in remaining:
                if _is_safe(v, counts):
                    best_v = v
                    found = True
                    break
            if not found:
                break
        sel.add(best_v)
        result.append(best_v)
        remaining.discard(best_v)
        for ti in _val_to_triples[best_v]:
            counts[ti] += 1
    return result


def _greedy_weighted(remaining, counts, deadline):
    sel = set()
    result = []
    while remaining and time.time() < deadline:
        best_v = None
        best_cost = None
        for v in remaining:
            safe = True
            cost = 0
            for ti in _val_to_triples[v]:
                c = counts[ti]
                if c == 5:
                    safe = False
                    break
                cost += c
                if c >= 3:
                    cost += (c - 2) * 3
            if not safe:
                continue
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_v = v
            if cost == 0:
                break
        if best_v is None:
            break
        sel.add(best_v)
        result.append(best_v)
        remaining.discard(best_v)
        for ti in _val_to_triples[best_v]:
            counts[ti] += 1
    return result


def _local_search(sel, counts, deadline):
    """Add any safe value, then remove-one-to-enable. Repeat until no improvement."""
    best_set = set(sel)
    best_size = len(sel)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Phase 1: add any safe value
        for v in range(1, N + 1):
            if v in sel:
                continue
            if _is_safe(v, counts):
                sel.add(v)
                for ti in _val_to_triples[v]:
                    counts[ti] += 1
                changed = True
        # Phase 2: remove-one-to-enable
        if not changed:
            improved = False
            for v_remove in list(sel):
                if time.time() > deadline:
                    break
                sel.discard(v_remove)
                for ti in _val_to_triples[v_remove]:
                    counts[ti] -= 1
                added = []
                for v_add in range(1, N + 1):
                    if v_add in sel:
                        continue
                    if _is_safe(v_add, counts):
                        sel.add(v_add)
                        added.append(v_add)
                        for ti in _val_to_triples[v_add]:
                            counts[ti] += 1
                new_size = len(sel)
                if new_size > best_size:
                    best_set = set(sel)
                    best_size = new_size
                    improved = True
                else:
                    for v_add in added:
                        sel.discard(v_add)
                        for ti in _val_to_triples[v_add]:
                            counts[ti] -= 1
                    sel.add(v_remove)
                    for ti in _val_to_triples[v_remove]:
                        counts[ti] += 1
            if improved:
                changed = True
                sel = set(best_set)
                counts = _build_counts(sel)
    return best_set


def _swap_polish(sel, counts, deadline):
    """2-swap polish: try removing one and adding another (net 0 or +1 size)."""
    best_size = len(sel)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v_remove in list(sel):
            if time.time() > deadline:
                break
            sel.discard(v_remove)
            for ti in _val_to_triples[v_remove]:
                counts[ti] -= 1
            added = []
            for v_add in range(1, N + 1):
                if v_add in sel:
                    continue
                if _is_safe(v_add, counts):
                    sel.add(v_add)
                    added.append(v_add)
                    for ti in _val_to_triples[v_add]:
                        counts[ti] += 1
            if len(sel) > best_size:
                best_size = len(sel)
                improved = True
            else:
                for v_add in added:
                    sel.discard(v_add)
                    for ti in _val_to_triples[v_add]:
                        counts[ti] -= 1
                sel.add(v_remove)
                for ti in _val_to_triples[v_remove]:
                    counts[ti] += 1
    return best_size


def _greedy_order(order, deadline):
    sel = set()
    result = []
    counts = [0] * _NUM_TRIPLES
    for v in order:
        if v in sel:
            continue
        if not _is_safe(v, counts):
            continue
        sel.add(v)
        result.append(v)
        for ti in _val_to_triples[v]:
            counts[ti] += 1
        if time.time() > deadline:
            break
    return result


def construct():
    import random
    start = time.time()
    deadline = start + DEADLINE
    best = []
    best_size = 0

    def _try(sel):
        nonlocal best, best_size
        if not sel:
            return
        counts = _build_counts(sel)
        sel2 = _local_search(set(sel), counts, deadline)
        if len(sel2) > best_size:
            best = sel2
            best_size = len(sel2)

    # Strategy 1: dynamic danger greedy
    remaining = set(range(1, N + 1))
    counts = [0] * _NUM_TRIPLES
    res = _greedy_dynamic(remaining, counts, deadline)
    _try(res)

    # Strategy 2: Max-SAT weighted greedy
    remaining = set(range(1, N + 1))
    counts = [0] * _NUM_TRIPLES
    res = _greedy_weighted(remaining, counts, deadline)
    _try(res)

    # Strategy 3: fixed-order heuristics
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(_val_to_triples[x])),
        sorted(range(1, N + 1), key=lambda x: -len(_val_to_triples[x])),
        list(range(N // 2 + 1, N + 1)) + list(range(1, N // 2 + 1)),
    ]
    for order in orders:
        if time.time() > deadline - 2:
            break
        res = _greedy_order(order, deadline)
        _try(res)

    # Strategy 4: randomized restarts
    rng = random.Random(12345)
    while time.time() < deadline - 3:
        remaining = set(range(1, N + 1))
        counts = [0] * _NUM_TRIPLES
        if rng.random() < 0.5:
            res = _greedy_dynamic(remaining, counts, deadline)
        else:
            res = _greedy_weighted(remaining, counts, deadline)
        _try(res)
        if best_size >= 95:
            break

    # Final swap polish on best
    if best and time.time() < deadline - 2:
        counts = _build_counts(best)
        _swap_polish(best, counts, deadline)

    # Final validity check
    if best:
        sel_set = set(best)
        valid = True
        for (a, b, c, s1, s2, s3) in _TRIPLES:
            if (a in sel_set and b in sel_set and c in sel_set
                    and s1 in sel_set and s2 in sel_set and s3 in sel_set):
                valid = False
                break
        if not valid:
            best = list(range(N // 2 + 1, N + 1))

    return [int(x) for x in best]
