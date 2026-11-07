"""Hypergraph independent set for the 6-set forbidden constraint.
Combined: dynamic conflict-degree greedy + incremental counts + remove-to-enable local search
+ base-set decomposition (small numbers added to a large base) for more reliable >64 sets."""
import time
import random
from collections import defaultdict

N = 128
DEADLINE = 55.0  # safety margin under 90s kill

# Precompute forbidden 6-sets: {a, b, c, a+b, a+c, b+c} where a<b<c and b+c <= N
_TRIPLES = []
for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        for c in range(b + 1, N + 1):
            s1 = a + b
            s2 = a + c
            s3 = b + c
            if s3 <= N:
                _TRIPLES.append((a, b, c, s1, s2, s3))

_NUM_TRIPLES = len(_TRIPLES)

# Index: for each value, which triple indices it belongs to
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


def _greedy_dynamic(remaining, counts, deadline, rng=None):
    """Pick each next value by minimum danger (fewest near-complete triples)."""
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


def _greedy_weighted(remaining, counts, deadline, rng=None):
    """Max-SAT-style: minimize sum of clause weights over affected clauses."""
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
    """Phase 1: add any safe value. Phase 2: remove-one-to-enable. Repeat."""
    best_set = set(sel)
    best_size = len(sel)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for v in range(1, N + 1):
            if v in sel:
                continue
            if _is_safe(v, counts):
                sel.add(v)
                for ti in _val_to_triples[v]:
                    counts[ti] += 1
                changed = True
        if not changed:
            improved = False
            for v_remove in list(sel):
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
                if time.time() > deadline:
                    for v_add in added:
                        sel.discard(v_add)
                        for ti in _val_to_triples[v_add]:
                            counts[ti] -= 1
                    sel.add(v_remove)
                    for ti in _val_to_triples[v_remove]:
                        counts[ti] += 1
                    break
            if improved:
                changed = True
                sel = set(best_set)
                counts = _build_counts(sel)
    return best_set


def _swap_polish(sel, counts, deadline):
    """2-swap polish: try removing one and adding another."""
    best_size = len(sel)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v_remove in list(sel):
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
            if time.time() > deadline:
                for v_add in added:
                    sel.discard(v_add)
                    for ti in _val_to_triples[v_add]:
                        counts[ti] -= 1
                sel.add(v_remove)
                for ti in _val_to_triples[v_remove]:
                    counts[ti] += 1
                break
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


def _base_set_decomposition(deadline):
    """Strategy: start with a base set of large numbers (65..128), then greedily
    add small numbers that don't create violations. This reduces the effective
    hypergraph size and allows more reliable construction."""
    base = set(range(65, 129))
    counts = _build_counts(base)
    sel = set(base)
    
    # Try to add small numbers in various orders
    small_numbers = list(range(1, 65))
    
    # Order 1: ascending
    for v in small_numbers:
        if time.time() > deadline:
            break
        if _is_safe(v, counts):
            sel.add(v)
            for ti in _val_to_triples[v]:
                counts[ti] += 1
    
    # Local improvement
    sel = _local_search(sel, _build_counts(sel), deadline)
    return sel


def _simulated_annealing(sel, counts, deadline, rng):
    """Simulated annealing to escape local optima.
    Accept moves that decrease size with probability exp(-delta/T)."""
    T = 1.0
    cooling = 0.995
    best_set = set(sel)
    best_size = len(sel)
    current_size = len(sel)
    
    while time.time() < deadline:
        # Pick a random move: remove one, add one, or swap
        move_type = rng.random()
        
        if move_type < 0.3:
            # Remove one
            if not sel:
                break
            v_remove = rng.choice(list(sel))
            sel.discard(v_remove)
            for ti in _val_to_triples[v_remove]:
                counts[ti] -= 1
            delta = -1
            # Try to add some safe values
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
            delta = new_size - current_size
            current_size = new_size
            
            if new_size > best_size:
                best_set = set(sel)
                best_size = new_size
            elif delta < 0 and rng.random() > 0.5:
                # Reject
                for v_add in added:
                    sel.discard(v_add)
                    for ti in _val_to_triples[v_add]:
                        counts[ti] -= 1
                sel.add(v_remove)
                for ti in _val_to_triples[v_remove]:
                    counts[ti] += 1
                current_size += 1
        
        elif move_type < 0.6:
            # Add one (if safe)
            candidates = []
            for v in range(1, N + 1):
                if v not in sel and _is_safe(v, counts):
                    candidates.append(v)
            if candidates:
                v_add = rng.choice(candidates)
                sel.add(v_add)
                for ti in _val_to_triples[v_add]:
                    counts[ti] += 1
                current_size += 1
                if current_size > best_size:
                    best_set = set(sel)
                    best_size = current_size
        
        else:
            # Swap: remove one, add one
            if not sel:
                break
            v_remove = rng.choice(list(sel))
            sel.discard(v_remove)
            for ti in _val_to_triples[v_remove]:
                counts[ti] -= 1
            
            # Find a safe value to add
            candidates = []
            for v in range(1, N + 1):
                if v not in sel and _is_safe(v, counts):
                    candidates.append(v)
            
            if candidates:
                v_add = rng.choice(candidates)
                sel.add(v_add)
                for ti in _val_to_triples[v_add]:
                    counts[ti] += 1
                # Size unchanged
                if current_size > best_size:
                    best_set = set(sel)
                    best_size = current_size
            else:
                # Can't add, revert
                sel.add(v_remove)
                for ti in _val_to_triples[v_remove]:
                    counts[ti] += 1
        
        T *= cooling
        if T < 0.01:
            break
    
    return best_set


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

    # Strategy 1: base set decomposition (large base + small additions)
    if time.time() < deadline - 5:
        res = _base_set_decomposition(deadline)
        _try(res)

    # Strategy 2: dynamic danger greedy
    if time.time() < deadline - 5:
        remaining = set(range(1, N + 1))
        counts = [0] * _NUM_TRIPLES
        res = _greedy_dynamic(remaining, counts, deadline)
        _try(res)

    # Strategy 3: Max-SAT weighted greedy
    if time.time() < deadline - 5:
        remaining = set(range(1, N + 1))
        counts = [0] * _NUM_TRIPLES
        res = _greedy_weighted(remaining, counts, deadline)
        _try(res)

    # Strategy 4: fixed-order heuristics
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(_val_to_triples[x])),
        sorted(range(1, N + 1), key=lambda x: -len(_val_to_triples[x])),
        list(range(N // 2 + 1, N + 1)) + list(range(1, N // 2 + 1)),
    ]
    for order in orders:
        if time.time() > deadline - 3:
            break
        res = _greedy_order(order, deadline)
        _try(res)

    # Strategy 5: randomized restarts
    rng = random.Random(12345)
    while time.time() < deadline - 2:
        remaining = set(range(1, N + 1))
        counts = [0] * _NUM_TRIPLES
        if rng.random() < 0.5:
            res = _greedy_dynamic(remaining, counts, deadline, rng=rng)
        else:
            res = _greedy_weighted(remaining, counts, deadline, rng=rng)
        _try(res)
        if best_size >= 95:
            break

    # Strategy 6: simulated annealing on the best set found so far
    if best and time.time() < deadline - 2:
        counts = _build_counts(best)
        sa_result = _simulated_annealing(set(best), counts, deadline, rng)
        if len(sa_result) > best_size:
            best = sa_result
            best_size = len(sa_result)

    # Final swap polish on best
    if best and time.time() < deadline - 1:
        counts = _build_counts(best)
        _swap_polish(best, counts, deadline)

    # Final validity check
    if best:
        sel_set = set(best)
        for (a, b, c, s1, s2, s3) in _TRIPLES:
            if (a in sel_set and b in sel_set and c in sel_set
                    and s1 in sel_set and s2 in sel_set and s3 in sel_set):
                best = list(range(N // 2 + 1, N + 1))
                break

    return [int(x) for x in best]
