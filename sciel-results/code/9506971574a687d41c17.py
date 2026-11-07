"""Combined: bitset-accelerated greedy + dynamic conflict-degree + 1-swap local search."""
import time
import random

N = 128
MAX_TIME = 85.0  # explicit deadline guard (90s wall-clock limit)

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

NUM_TRIPLES = len(_TRIPLES)

# Index: for each value v in 1..N, list of triple indices containing v
_val_to_triples = [[] for _ in range(N + 1)]
for idx, t in enumerate(_TRIPLES):
    for v in t:
        if 1 <= v <= N:
            _val_to_triples[v].append(idx)

# Degree: number of triples each value participates in
_degree = [len(_val_to_triples[v]) for v in range(N + 1)]


def _greedy_order(order, deadline):
    """Build a set by adding values in 'order', skipping any that would complete a forbidden 6-set."""
    sel = set()
    result = []
    counts = [0] * NUM_TRIPLES
    for v in order:
        if v in sel:
            continue
        ok = True
        for ti in _val_to_triples[v]:
            if counts[ti] == 5:
                ok = False
                break
        if not ok:
            continue
        sel.add(v)
        result.append(v)
        for ti in _val_to_triples[v]:
            counts[ti] += 1
        if time.time() > deadline:
            break
    return result


def _greedy_dynamic(deadline, rng=None):
    """Dynamic greedy: at each step, pick the unselected value with the fewest
    near-complete conflicts (triples where 5 of 6 are already selected)."""
    sel = set()
    result = []
    counts = [0] * NUM_TRIPLES
    remaining = set(range(1, N + 1))

    while remaining and time.time() < deadline:
        best_v = None
        best_danger = None
        tie_candidates = []

        for v in remaining:
            danger = 0
            for ti in _val_to_triples[v]:
                if counts[ti] == 5:
                    danger += 1
            if best_danger is None or danger < best_danger:
                best_danger = danger
                best_v = v
                tie_candidates = [v]
            elif danger == best_danger:
                tie_candidates.append(v)
            if danger == 0:
                # Can't do better than 0; but keep scanning to collect all zero-danger for tie-breaking
                pass

        if best_v is None:
            break

        if best_danger > 0:
            # No safe value exists; stop
            break

        # Among zero-danger candidates, pick one with minimum degree (most flexible) or random tie-break
        if len(tie_candidates) > 1:
            if rng is not None:
                # Random tie-break among zero-danger, weighted by inverse degree
                best_v = min(tie_candidates, key=lambda x: (_degree[x], rng.random()))
            else:
                best_v = min(tie_candidates, key=lambda x: _degree[x])

        sel.add(best_v)
        result.append(best_v)
        remaining.discard(best_v)
        for ti in _val_to_triples[best_v]:
            counts[ti] += 1

    return result


def _local_search(sel_list, deadline, rng=None):
    """Aggressive local search: add any safe value; if stuck, try 1-swap (remove one, add many)."""
    if not sel_list:
        return []

    sel = set(sel_list)
    counts = [0] * NUM_TRIPLES
    for v in sel:
        for ti in _val_to_triples[v]:
            counts[ti] += 1

    best_set = set(sel)
    best_size = len(sel)

    changed = True
    while changed and time.time() < deadline:
        changed = False

        # Phase 1: Add any safe value
        for v in range(1, N + 1):
            if v in sel:
                continue
            safe = True
            for ti in _val_to_triples[v]:
                if counts[ti] == 5:
                    safe = False
                    break
            if safe:
                sel.add(v)
                for ti in _val_to_triples[v]:
                    counts[ti] += 1
                changed = True

        if not changed:
            # Phase 2: Try 1-swap: remove one selected value, then greedily add safe values
            improved = False
            # Sort selected by degree ascending (remove least-constraining first? No, remove most-constraining to free most)
            # Actually, removing a high-degree value frees more triples. Try all.
            sel_list_sorted = sorted(sel, key=lambda x: -_degree[x])
            for v_remove in sel_list_sorted:
                if time.time() > deadline:
                    break
                # Tentatively remove v_remove
                sel.discard(v_remove)
                for ti in _val_to_triples[v_remove]:
                    counts[ti] -= 1

                # Try to add as many safe values as possible
                added = []
                for v_add in range(1, N + 1):
                    if v_add in sel:
                        continue
                    safe = True
                    for ti in _val_to_triples[v_add]:
                        if counts[ti] == 5:
                            safe = False
                            break
                    if safe:
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
                    # Revert: remove the added values and put back v_remove
                    for v_add in added:
                        sel.discard(v_add)
                        for ti in _val_to_triples[v_add]:
                            counts[ti] -= 1
                    sel.add(v_remove)
                    for ti in _val_to_triples[v_remove]:
                        counts[ti] += 1

            if improved:
                changed = True
                # Reset sel to best_set and rebuild counts
                sel = set(best_set)
                counts = [0] * NUM_TRIPLES
                for v in sel:
                    for ti in _val_to_triples[v]:
                        counts[ti] += 1

    return list(best_set)


def _validate(sel):
    """Final validity check."""
    sel_set = set(sel)
    for (a, b, c, s1, s2, s3) in _TRIPLES:
        if a in sel_set and b in sel_set and c in sel_set and s1 in sel_set and s2 in sel_set and s3 in sel_set:
            return False
    return True


def construct():
    start = time.time()
    deadline = start + MAX_TIME
    best = []
    best_size = 0

    # Strategy 1: Deterministic greedy with several heuristic orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: _degree[x]),  # least restrictive first
        sorted(range(1, N + 1), key=lambda x: -_degree[x]),  # most restrictive first
        list(range(N // 2 + 1, N + 1)) + list(range(1, N // 2 + 1)),
    ]
    for order in orders:
        if time.time() > deadline - 3:
            break
        res = _greedy_order(order, deadline)
        res = _local_search(res, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)

    # Strategy 2: Dynamic conflict-degree greedy
    if time.time() < deadline - 3:
        res = _greedy_dynamic(deadline)
        res = _local_search(res, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)

    # Strategy 3: Randomized restarts
    rng = random.Random(12345)
    while time.time() < deadline - 1:
        # Random order with bias toward low-degree values
        order = list(range(1, N + 1))
        rng.shuffle(order)
        if rng.random() < 0.5:
            order.sort(key=lambda x: (_degree[x], rng.random()))
        else:
            order.sort(key=lambda x: (-_degree[x], rng.random()))

        res = _greedy_order(order, deadline)
        res = _local_search(res, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)
            if best_size >= 95:
                break

        # Also try dynamic greedy with random tie-breaking
        if time.time() < deadline - 1:
            res2 = _greedy_dynamic(deadline, rng=rng)
            res2 = _local_search(res2, deadline)
            if len(res2) > best_size:
                best = res2
                best_size = len(res2)
                if best_size >= 95:
                    break

    # Final validity check
    if best and not _validate(best):
        # Fallback to a known valid set
        best = list(range(N // 2 + 1, N + 1))

    # Ensure types: list of Python ints
    return [int(x) for x in best]
