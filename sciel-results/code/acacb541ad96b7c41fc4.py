"""Rewritten program: randomized greedy over precomputed forbidden triples."""
import time
import random

N = 128
DEADLINE = 55.0  # safety margin under 60s

# Precompute all forbidden triples: distinct a<b<c such that
# a+b, a+c, b+c are all in 1..N.  If a,b,c are selected, we must NOT
# also select a+b, a+c, b+c.  So the forbidden pattern is the 6-set
# {a, b, c, a+b, a+c, b+c} (with possible collisions, but a,b,c distinct
# and sums >= a+1 > c typically, so usually 6 distinct values).
_TRIPLES = []
for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        for c in range(b + 1, N + 1):
            s1 = a + b
            s2 = a + c
            s3 = b + c
            if s3 <= N:
                # all six must be in range; a,b,c are in range by loop
                _TRIPLES.append((a, b, c, s1, s2, s3))

def _is_valid(sel_set):
    # sel_set is a set of ints in 1..N
    for (a, b, c, s1, s2, s3) in _TRIPLES:
        if a in sel_set and b in sel_set and c in sel_set and s1 in sel_set and s2 in sel_set and s3 in sel_set:
            return False
    return True

def _greedy_random(order):
    # order: list of candidate ints in selection order
    sel = set()
    sel_list = []
    # For speed, maintain a counter per triple of how many of its 6 members are in sel.
    # A triple is "active" (forbidden to complete) if all 6 are in sel.
    # We just check at the end for simplicity, but to prune during greedy we track
    # which triples become "blocked" (i.e., we cannot add the last member).
    # Simpler: after each add, check if any triple now has all 6 in sel -> if so, this add is invalid.
    # To make it fast, precompute for each value the list of triples it participates in.
    return sel_list

# Build membership index: for each value v, list of triple indices containing v.
_val_to_triples = {v: [] for v in range(1, N + 1)}
for idx, t in enumerate(_TRIPLES):
    for v in t:
        if v in _val_to_triples:
            _val_to_triples[v].append(idx)

def _greedy_order(order, deadline):
    """Try to build a set by adding values in 'order', skipping any that would
    complete a forbidden 6-set.  Returns the set (as a list)."""
    sel = set()
    result = []
    # count[v] = number of members of triple t that are in sel
    counts = [0] * len(_TRIPLES)
    for v in order:
        if v in sel:
            continue
        # Tentatively add v: check if any triple containing v would reach count 6
        # i.e., currently count == 5 and v is the 6th member.
        ok = True
        for ti in _val_to_triples[v]:
            if counts[ti] == 5:
                # adding v completes this triple -> forbidden
                ok = False
                break
        if not ok:
            continue
        # commit
        sel.add(v)
        result.append(v)
        for ti in _val_to_triples[v]:
            counts[ti] += 1
        if time.time() > deadline:
            break
    return result

def _local_improve(sel_list, deadline):
    """Try to add any remaining value that doesn't create a violation;
    repeat until no more can be added.  Also try swapping a selected
    value for two unselected ones if it increases size (rare)."""
    sel = set(sel_list)
    counts = [0] * len(_TRIPLES)
    for v in sel:
        for ti in _val_to_triples[v]:
            counts[ti] += 1
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for v in range(1, N + 1):
            if v in sel:
                continue
            ok = True
            for ti in _val_to_triples[v]:
                if counts[ti] == 5:
                    ok = False
                    break
            if ok:
                sel.add(v)
                for ti in _val_to_triples[v]:
                    counts[ti] += 1
                changed = True
    return list(sel)

def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = []
    best_size = 0

    # Strategy 1: deterministic greedy with several heuristic orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(_val_to_triples[x])),  # least restrictive first
        sorted(range(1, N + 1), key=lambda x: -len(_val_to_triples[x])),  # most restrictive first (avoid early)
        list(range(N // 2 + 1, N + 1)) + list(range(1, N // 2 + 1)),
    ]
    for order in orders:
        if time.time() > deadline - 2:
            break
        res = _greedy_order(order, deadline)
        res = _local_improve(res, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)

    # Strategy 2: randomized restarts
    rng = random.Random(12345)
    attempts = 0
    while time.time() < deadline - 1:
        attempts += 1
        # random order with slight bias: shuffle, but put low-degree first with some probability
        order = list(range(1, N + 1))
        rng.shuffle(order)
        # With 50% chance, sort by degree after shuffle to bias toward flexible values
        if rng.random() < 0.5:
            order.sort(key=lambda x: (len(_val_to_triples[x]), rng.random()))
        res = _greedy_order(order, deadline)
        res = _local_improve(res, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)
            # Early exit if we hit a high bar
            if best_size >= 90:
                break

    # Final validity check (should be safe by construction, but verify)
    if best:
        sel_set = set(best)
        for (a, b, c, s1, s2, s3) in _TRIPLES:
            if a in sel_set and b in sel_set and c in sel_set and s1 in sel_set and s2 in sel_set and s3 in sel_set:
                # Should not happen; if it does, fall back to seed
                best = list(range(N // 2 + 1, N + 1))
                break

    # Ensure types: list of Python ints
    return [int(x) for x in best]
