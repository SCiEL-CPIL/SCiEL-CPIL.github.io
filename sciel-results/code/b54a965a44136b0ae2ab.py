import time
import random
from fractions import Fraction

_N = 256
_KILL = 90.0
_MARGIN = 5.0
_DEADLINE = time.time() + _KILL - _MARGIN

# Precompute reciprocals (float) for fast comparison
_RECIP = [0.0] + [1.0 / i for i in range(1, _N + 1)]

# Precompute, for each x, the list of y (y != x) such that adding both
# creates a 3-AP conflict (i.e., they would block a third element together).
# block_pairs[x] = set of y where {x,y} is "conflicting" (would force a 3rd).
# A set S is 3-AP-free iff for all x in S, (block_pairs[x] & S) is empty.
_BLOCK_PAIRS = [set() for _ in range(_N + 1)]
for _a in range(1, _N + 1):
    for _b in range(_a + 1, _N + 1):
        # middle
        s = _a + _b
        if s % 2 == 0:
            m = s // 2
            if 1 <= m <= _N:
                _BLOCK_PAIRS[_a].add(_b)
                _BLOCK_PAIRS[_b].add(_a)
        # a,b, 2b-a
        c = 2 * _b - _a
        if 1 <= c <= _N and c != _a and c != _b:
            _BLOCK_PAIRS[_a].add(_c) if False else None
            _BLOCK_PAIRS[_a].add(c)
            _BLOCK_PAIRS[c].add(_a)
        # 2a-b, a, b
        c = 2 * _a - _b
        if 1 <= c <= _N and c != _a and c != _b:
            _BLOCK_PAIRS[_b].add(c)
            _BLOCK_PAIRS[c].add(_b)
# NOTE: the pair (a,b) itself is only a problem if the THIRD element is in S.
# So block_pairs[x] should be the set of y such that x and y together
# require a third element to be excluded. Actually for the "conflict" test
# we need: x in S and y in S is invalid iff the third element t (with x,y,t
# forming a 3-AP) is ALSO in S. That's a 3-way condition, not pairwise.
# So pairwise block_pairs is NOT sufficient for validity. We keep the
# blocked-set approach for correctness, but use block_pairs to speed
# blocked maintenance.

# Rebuild the clean, correct per-element "would-block" list:
# For each x, block_list[x] = set of elements that x blocks (i.e., elements
# that cannot coexist with x because they'd form a 3-AP with x and some
# third element). Actually x alone blocks nothing. x and y together block
# the third. So the blocked set for S = union over pairs {x,y} in S of
# the third element. Equivalently: an element z is blocked iff there exist
# x,y in S with {x,y,z} a 3-AP.
# We'll maintain blocked via: when adding x, for each y already in S,
# add the third element(s) formed by x,y to blocked.

def _thirds(x, y):
    """Yield the third element(s) t such that {x,y,t} is a 3-AP (t != x,y)."""
    a, b = x, y
    s = a + b
    if s % 2 == 0:
        m = s // 2
        if m != a and m != b and 1 <= m <= _N:
            yield m
    c = 2 * b - a
    if c != a and c != b and 1 <= c <= _N:
        yield c
    c = 2 * a - b
    if c != a and c != b and 1 <= c <= _N:
        yield c


def _get_blocked(S):
    blocked = set()
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            for t in _thirds(a, b):
                blocked.add(t)
    return blocked


def _add_to_blocked(blocked, S, x):
    for a in S:
        if a == x:
            continue
        for t in _thirds(a, x):
            blocked.add(t)


def _validate(S):
    """Check S is 3-AP-free."""
    blocked = _get_blocked(S)
    return not (blocked & S)


def _greedy_ascending():
    S = set()
    blocked = set()
    for x in range(1, _N + 1):
        if x not in blocked:
            S.add(x)
            _add_to_blocked(blocked, S, x)
    return S


def _greedy_descending():
    S = set()
    blocked = set()
    for x in range(_N, 0, -1):
        if x not in blocked:
            S.add(x)
            _add_to_blocked(blocked, S, x)
    return S


def _random_init(rng, size_range=(30, 80)):
    S = set()
    blocked = set()
    elements = list(range(1, _N + 1))
    rng.shuffle(elements)
    target_size = rng.randint(*size_range)
    for x in elements:
        if len(S) >= target_size:
            break
        if x not in blocked:
            S.add(x)
            _add_to_blocked(blocked, S, x)
    return S


def _local_search(S, deadline, rng):
    S = set(S)
    blocked = _get_blocked(S)
    weight = sum(_RECIP[x] for x in S)

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Step 1: Greedy additions
        added_any = False
        for x in range(1, _N + 1):
            if x not in S and x not in blocked:
                S.add(x)
                weight += _RECIP[x]
                _add_to_blocked(blocked, S, x)
                added_any = True
                improved = True
        if added_any:
            continue

        # Step 2: Single swaps (remove a, add b, 1/b > 1/a)
        best_swap_gain = -1.0
        best_a = None
        best_b = None
        S_list = sorted(S, reverse=True)
        for a in S_list:
            if time.time() > deadline:
                break
            S_temp = S - {a}
            blocked_temp = _get_blocked(S_temp)
            for b in range(1, a):
                if b in S_temp or b in blocked_temp:
                    continue
                gain = _RECIP[b] - _RECIP[a]
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_a = a
                    best_b = b
                break
        if best_a is not None and best_swap_gain > 1e-12:
            S.remove(best_a)
            S.add(best_b)
            weight += best_swap_gain
            blocked = _get_blocked(S)
            improved = True
            continue

        # Step 3: Two-for-one swaps (remove a,b; add c)
        S_list_all = sorted(S)
        n = len(S_list_all)
        found = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                a = S_list_all[i]
                b = S_list_all[j]
                S_temp = S - {a, b}
                blocked_temp = _get_blocked(S_temp)
                for c in range(1, _N + 1):
                    if c in S_temp or c in blocked_temp:
                        continue
                    gain = _RECIP[c] - _RECIP[a] - _RECIP[b]
                    if gain > 0:
                        S.remove(a)
                        S.remove(b)
                        S.add(c)
                        weight += gain
                        blocked = _get_blocked(S)
                        improved = True
                        found = True
                        break
                if found:
                    break
            if found:
                break

        # Step 4: One-for-two swaps (remove a; add c,d)
        if not improved and not found and time.time() < deadline:
            for a in S_list_all:
                if time.time() > deadline:
                    break
                S_temp = S - {a}
                blocked_temp = _get_blocked(S_temp)
                found_two = []
                for c in range(1, _N + 1):
                    if len(found_two) >= 2:
                        break
                    if c in S_temp or c in blocked_temp:
                        continue
                    found_two.append(c)
                    _add_to_blocked(blocked_temp, S_temp, c)
                if len(found_two) == 2:
                    gain = _RECIP[found_two[0]] + _RECIP[found_two[1]] - _RECIP[a]
                    if gain > 0:
                        S.remove(a)
                        S.add(found_two[0])
                        S.add(found_two[1])
                        weight += gain
                        blocked = _get_blocked(S)
                        improved = True
                        found = True
                        break

    return S


def _branch_and_bound(deadline, rng, max_nodes=200000):
    """Branch-and-bound over the decision to include/exclude each x in
    ascending order (small x = large reciprocal, so include early).
    Bound: current_weight + sum of reciprocals of all remaining candidates
    that are not yet blocked and not in S (a loose but valid upper bound).
    """
    order = list(range(1, _N + 1))  # ascending
    best_set = set()
    best_weight = -1.0
    nodes = 0

    # Precompute suffix reciprocal sums for a quick bound.
    suffix = [0.0] * (_N + 2)
    for i in range(_N, 0, -1):
        suffix[i] = suffix[i + 1] + _RECIP[i]

    # State: index into order, S, blocked, weight
    stack = [(0, set(), set(), 0.0)]
    # Use iterative DFS with a priority by potential (weight + bound)
    # For simplicity, DFS in include-first order.
    while stack and time.time() < deadline:
        idx, S, blocked, weight = stack.pop()
        nodes += 1
        if nodes % 5000 == 0 and time.time() > deadline:
            break

        # Update best
        if weight > best_weight:
            best_weight = weight
            best_set = set(S)

        if idx == _N:
            continue

        x = order[idx]

        # Upper bound: weight + sum of reciprocals of candidates from idx+1
        # that are not blocked and not in S. Use suffix as a loose bound.
        bound = weight + suffix[idx + 1]
        if bound <= best_weight + 1e-12:
            continue  # prune

        # Option 1: include x (if not blocked)
        if x not in blocked:
            new_S = S | {x}
            new_blocked = set(blocked)
            _add_to_blocked(new_blocked, S, x)
            stack.append((idx + 1, new_S, new_blocked, weight + _RECIP[x]))

        # Option 2: exclude x
        stack.append((idx + 1, S, blocked, weight))

    return best_set


def _polish(S, deadline):
    """Final exact polish: repeated single-swap and two-for-one until no
    improvement, with exact Fraction check for validity."""
    S = set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        blocked = _get_blocked(S)

        # single swaps
        for a in sorted(S, reverse=True):
            if time.time() > deadline:
                break
            S_temp = S - {a}
            blocked_temp = _get_blocked(S_temp)
            for b in range(1, a):
                if b in S_temp or b in blocked_temp:
                    continue
                new_S = S_temp | {b}
                if _validate(new_S):
                    S = new_S
                    improved = True
                    break
            if improved:
                break

        # two-for-one
        if not improved and time.time() < deadline:
            S_list = sorted(S)
            n = len(S_list)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    a, b = S_list[i], S_list[j]
                    S_temp = S - {a, b}
                    blocked_temp = _get_blocked(S_temp)
                    for c in range(1, _N + 1):
                        if c in S_temp or c in blocked_temp:
                            continue
                        new_S = S_temp | {c}
                        if _validate(new_S):
                            S = new_S
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return S


def _reciprocal_sum_exact(S):
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total


def construct():
    start_time = time.time()
    deadline = start_time + _KILL - _MARGIN
    rng = random.Random(20240517)

    best_set = None
    best_weight = -1.0

    def consider(S):
        nonlocal best_set, best_weight
        if not S:
            return
        if not _validate(S):
            return
        w = sum(_RECIP[x] for x in S)
        if w > best_weight:
            best_weight = w
            best_set = set(S)

    # Diverse seeds
    seeds = []
    seeds.append(_greedy_ascending())
    seeds.append(_greedy_descending())
    for i in range(8):
        seeds.append(_random_init(rng, size_range=(20, 100)))

    # Local search on each seed
    for S_init in seeds:
        if time.time() > deadline:
            break
        result = _local_search(S_init, deadline, rng)
        consider(result)

    # Branch-and-bound to explore exclusions
    if time.time() < deadline:
        bb_set = _branch_and_bound(deadline, rng)
        consider(bb_set)
        # Polish the B&B result
        if time.time() < deadline:
            consider(_polish(bb_set, deadline))

    # Perturbation restarts from best
    for restart in range(60):
        if time.time() > deadline:
            break
        S = set(best_set)
        n_remove = rng.randint(1, min(6, len(S)))
        to_remove = rng.sample(list(S), n_remove)
        for x in to_remove:
            S.discard(x)
        available = [x for x in range(1, _N + 1) if x not in S]
        if available:
            n_add = rng.randint(0, min(3, len(available)))
            to_add = rng.sample(available, n_add)
            for x in to_add:
                if x not in _get_blocked(S):
                    S.add(x)
        result = _local_search(S, deadline, rng)
        consider(result)

    # Final polish
    if time.time() < deadline:
        consider(_polish(best_set, deadline))

    return list(best_set)
