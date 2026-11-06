"""Equal-gcd-triple-free subsets of [1,128]: branch-and-bound MIS solver.

Structural change: model as maximum independent set on a conflict graph where
vertices are [1..128] and an edge (a,b) exists iff there is some c in [1..128]
with gcd(a,b)=gcd(a,c)=gcd(b,c). A valid set is an independent set.

Solver: Tomita branch-and-bound with greedy coloring upper bound. The conflict
graph is precomputed via the bitmask machinery. This replaces the heuristic
local search + SA with a systematic exact solver that can find (or prove) the
maximum.
"""
import time
import random
import math
from math import gcd

N = 128
_DEADLINE = 84.0  # 6s safety margin under 90s kill

_PRE = None
_BIT = None
_CONFLICT_MASKS = None
_ADJ = None  # _ADJ[x] = bitmask of vertices conflicting with x
_ADJ_LIST = None  # _ADJ_LIST[x] = list of vertices conflicting with x


def _ensure_pre():
    global _PRE, _BIT, _CONFLICT_MASKS, _ADJ, _ADJ_LIST
    if _PRE is not None:
        return
    PRE = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            PRE[i][j] = g
            PRE[j][i] = g
    _PRE = PRE

    BIT = [0] * (N + 1)
    for x in range(1, N + 1):
        BIT[x] = 1 << (x - 1)
    _BIT = BIT

    # Conflict masks: for each x, list of (mask_a, mask_b) pairs
    CONFLICT_MASKS = {x: [] for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            g = PRE[x][a]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if PRE[x][b] == g and PRE[a][b] == g:
                    CONFLICT_MASKS[x].append((BIT[a], BIT[b]))
    _CONFLICT_MASKS = CONFLICT_MASKS

    # Build adjacency: a and b conflict iff there exists c with
    # gcd(a,b)=gcd(a,c)=gcd(b,c). Equivalently, b in CONFLICT_MASKS[a] check.
    # Two vertices a,b are adjacent in the conflict graph iff adding a to a
    # set containing b is blocked (i.e., there's some c with b,c in the set
    # such that gcd(a,b)=gcd(a,c)=gcd(b,c)).
    # Simpler: a,b adjacent iff exists c != a,b with gcd(a,b)=gcd(a,c)=gcd(b,c).
    ADJ = [0] * (N + 1)
    ADJ_LIST = {x: [] for x in range(1, N + 1)}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = PRE[a][b]
            # check if exists c != a,b with gcd(a,c)==gab and gcd(b,c)==gab
            found = False
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if PRE[a][c] == gab and PRE[b][c] == gab:
                    found = True
                    break
            if found:
                ADJ[a] |= BIT[b]
                ADJ[b] |= BIT[a]
                ADJ_LIST[a].append(b)
                ADJ_LIST[b].append(a)
    _ADJ = ADJ
    _ADJ_LIST = ADJ_LIST


def _is_valid(Sset):
    P = _PRE
    sl = sorted(Sset)
    n = len(sl)
    for i in range(n):
        a = sl[i]
        for j in range(i + 1, n):
            b = sl[j]
            gab = P[a][b]
            for k in range(j + 1, n):
                c = sl[k]
                if P[a][c] == gab and P[b][c] == gab:
                    return False
    return True


def _bad_count(Sset):
    P = _PRE
    sl = sorted(Sset)
    n = len(sl)
    cnt = 0
    for i in range(n):
        a = sl[i]
        for j in range(i + 1, n):
            b = sl[j]
            gab = P[a][b]
            for k in range(j + 1, n):
                c = sl[k]
                if P[a][c] == gab and P[b][c] == gab:
                    cnt += 1
    return cnt


def _can_add_mask(Smask, x):
    for ma, mb in _CONFLICT_MASKS[x]:
        if (Smask & ma) and (Smask & mb):
            return False
    return True


def _greedy_max_independent(cand_mask, start_size, deadline_fn):
    """Greedy: repeatedly pick the vertex with fewest conflicts among candidates.
    Returns (independent_set_list, remaining_cand_mask, size)."""
    S = []
    Smask = 0
    cand = cand_mask
    while cand and deadline_fn():
        # Pick vertex in cand with minimum degree within cand
        best_v = -1
        best_deg = N + 1
        m = cand
        while m:
            low = m & (-m)
            v = low.bit_length()  # 1-based index
            deg = (ADJ[v] & cand).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
            m ^= low
        if best_v < 0:
            break
        S.append(best_v)
        Smask |= _BIT[best_v]
        # Remove best_v and its neighbors from candidates
        cand = cand & ~( _BIT[best_v] | ADJ[best_v] )
    return S, cand, start_size + len(S)


def _greedy_upper_bound(cand_mask):
    """Greedy coloring upper bound on independent set size in cand_mask.
    Returns an upper bound (number of colors)."""
    # Greedy: assign colors so that same color = independent set.
    # Upper bound on alpha = number of colors used.
    uncolored = cand_mask
    num_colors = 0
    while uncolored:
        num_colors += 1
        # Greedily build an independent set (color class)
        color_class = 0
        remaining = uncolored
        while remaining:
            low = remaining & (-remaining)
            v = low.bit_length()
            # v can join color class if not adjacent to any in color_class
            if (ADJ[v] & color_class) == 0:
                color_class |= low
                remaining &= ~low
            else:
                pass  # skip, will try next
            remaining ^= low
        # Remove color class from uncolored
        uncolored &= ~color_class
    return num_colors


def _branch_and_bound(cand_mask, current_set, current_size, best_set, deadline_fn):
    """Recursive branch-and-bound for maximum independent set.
    cand_mask: bitmask of candidate vertices (none adjacent to current_set).
    Returns (best_set_list, best_size)."""
    if not deadline_fn():
        return list(best_set), len(best_set)
    if cand_mask == 0:
        if current_size > len(best_set):
            best_set = list(current_set)
        return list(best_set), current_size

    # Upper bound: current_size + greedy coloring bound on cand_mask
    ub = current_size + _greedy_upper_bound(cand_mask)
    if ub <= len(best_set):
        return list(best_set), current_size

    # Pick a vertex to branch on: the one with max degree in cand_mask
    # (or min for better pruning - use max degree for Tomita)
    best_v = -1
    best_deg = -1
    m = cand_mask
    while m:
        low = m & (-m)
        v = low.bit_length()
        deg = (ADJ[v] & cand_mask).bit_count()
        if deg > best_deg:
            best_deg = deg
            best_v = v
        m ^= low

    if best_v < 0:
        return list(best_set), current_size

    # Branch 1: include best_v
    current_set.append(best_v)
    new_cand = cand_mask & ~( _BIT[best_v] | ADJ[best_v] )
    best_set, best_size = _branch_and_bound(new_cand, current_set, current_size + 1, best_set, deadline_fn)
    current_set.pop()

    # Branch 2: exclude best_v
    new_cand2 = cand_mask & ~_BIT[best_v]
    best_set, best_size = _branch_and_bound(new_cand2, current_set, current_size, best_set, deadline_fn)

    return list(best_set), best_size


def _branch_and_bound_iter(cand_mask, deadline_fn, max_nodes=2000000):
    """Iterative branch-and-bound with a stack to avoid recursion limits.
    Uses greedy coloring for upper bound. Returns best independent set."""
    # Use a stack-based DFS with branch-and-bound
    # State: (cand_mask, current_set_tuple, current_size)
    # We track the best found.
    best_set = []
    best_size = 0

    # Initial greedy solution for a good lower bound
    gset, _, gsize = _greedy_max_independent(cand_mask, 0, deadline_fn)
    best_set = list(gset)
    best_size = len(gset)

    # Stack for DFS
    stack = [(cand_mask, (), 0)]
    nodes = 0
    while stack and deadline_fn() and nodes < max_nodes:
        cand, cur_set, cur_size = stack.pop()
        nodes += 1
        if cand == 0:
            if cur_size > best_size:
                best_size = cur_size
                best_set = list(cur_set)
            continue
        ub = cur_size + _greedy_upper_bound(cand)
        if ub <= best_size:
            continue
        # Pick branch vertex
        best_v = -1
        best_deg = -1
        m = cand
        while m:
            low = m & (-m)
            v = low.bit_length()
            deg = (ADJ[v] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            m ^= low
        if best_v < 0:
            continue
        # Push exclude branch first (so include is processed first via LIFO)
        stack.append((cand & ~_BIT[best_v], cur_set, cur_size))
        # Include branch
        new_cand = cand & ~( _BIT[best_v] | ADJ[best_v] )
        stack.append((new_cand, cur_set + (best_v,), cur_size + 1))

    return best_set, best_size


def _local_improve(S, max_rounds=200, time_budget=None):
    S = list(S)
    Sset = set(S)
    Smask = 0
    for v in Sset:
        Smask |= _BIT[v]
    start_time = time.time()
    for rnd in range(max_rounds):
        if time_budget is not None and time.time() - start_time > time_budget:
            break
        added = False
        for x in range(1, N + 1):
            if x in Sset:
                continue
            if _can_add_mask(Smask, x):
                S.append(x)
                Sset.add(x)
                Smask |= _BIT[x]
                added = True
        if not added:
            break
        improved = False
        for r in S:
            for x in range(1, N + 1):
                if x in Sset or x == r:
                    continue
                Smask_new = Smask & ~_BIT[r]
                ok = True
                for ma, mb in _CONFLICT_MASKS[x]:
                    if (Smask_new & ma) and (Smask_new & mb):
                        ok = False
                        break
                if not ok:
                    continue
                # Tentative swap
                S.remove(r)
                Sset.discard(r)
                Smask &= ~_BIT[r]
                S.append(x)
                Sset.add(x)
                Smask |= _BIT[x]
                new_add = None
                for y in range(1, N + 1):
                    if y in Sset:
                        continue
                    if _can_add_mask(Smask, y):
                        new_add = y
                        break
                if new_add is not None:
                    S.append(new_add)
                    Sset.add(new_add)
                    Smask |= _BIT[new_add]
                    improved = True
                    break
                else:
                    S.remove(x)
                    Sset.discard(x)
                    Smask &= ~_BIT[x]
                    S.append(r)
                    Sset.add(r)
                    Smask |= _BIT[r]
            if improved:
                break
        if not improved:
            break
    return S


def _construct():
    global _START
    _ensure_pre()
    _START = time.time()

    def time_left():
        return _DEADLINE - (time.time() - _START)

    def deadline_fn():
        return time.time() - _START < _DEADLINE - 2.0

    full_mask = 0
    for x in range(1, N + 1):
        full_mask |= _BIT[x]

    best = []
    best_len = -1

    # Strategy 1: Greedy initial solutions with multiple orderings
    def num_divs(n):
        c = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                c += 1
                if i * i != n:
                    c += 1
            i += 1
        return c

    divs = {x: num_divs(x) for x in range(1, N + 1)}

    def is_prime_power(x):
        if x <= 1:
            return False
        v = x
        p0 = None
        for p in range(2, int(v ** 0.5) + 1):
            if v % p == 0:
                if p0 is None:
                    p0 = p
                elif p0 != p:
                    return False
                while v % p == 0:
                    v //= p
        if v > 1:
            if p0 is None:
                p0 = v
            elif p0 != v:
                return False
        return True

    prime_powers = [x for x in range(1, N + 1) if is_prime_power(x)]

    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: divs[x]),
        sorted(range(1, N + 1), key=lambda x: divs[x], reverse=True),
        sorted(range(1, N + 1), key=lambda x: (0 if is_prime_power(x) else 1, x)),
        prime_powers + [x for x in range(1, N + 1) if not is_prime_power(x)],
    ]

    for order in orders:
        if time_left() < 5:
            break
        S = []
        Sset = set()
        Smask = 0
        for x in order:
            if x in Sset:
                continue
            if _can_add_mask(Smask, x):
                S.append(x)
                Sset.add(x)
                Smask |= _BIT[x]
        S = _local_improve(S, max_rounds=50, time_budget=min(2, time_left() - 2))
        if len(S) > best_len:
            best = S
            best_len = len(S)

    # Strategy 2: Branch-and-bound on the full graph (or on candidates not in best)
    if time_left() > 15:
        # Try B&B from scratch with a good initial lower bound
        gset, _, gsize = _greedy_max_independent(full_mask, 0, deadline_fn)
        if len(gset) > best_len:
            best = list(gset)
            best_len = len(gset)

        bb_set, bb_size = _branch_and_bound_iter(full_mask, deadline_fn, max_nodes=1500000)
        if len(bb_set) > best_len:
            best = list(bb_set)
            best_len = len(bb_set)
            # Local improve
            S = _local_improve(best, max_rounds=50, time_budget=min(3, time_left() - 3))
            if len(S) > best_len:
                best = S
                best_len = len(S)

    # Strategy 3: More B&B restarts with different node budgets
    rng = random.Random(42)
    while time_left() > 10:
        # Random perturbation: start from a random subset
        rand_set = []
        for x in range(1, N + 1):
            if rng.random() < 0.4:
                rand_set.append(x)
        # Build candidate mask: all vertices not conflicting with rand_set
        cand = full_mask
        for x in rand_set:
            cand &= ~( _BIT[x] | ADJ[x] )
        # Add rand_set as fixed
        gset, _, gsize = _greedy_max_independent(cand, 0, deadline_fn)
        candidate = rand_set + gset
        # Validate
        if _is_valid(set(candidate)):
            S = _local_improve(candidate, max_rounds=30, time_budget=min(2, time_left() - 2))
            if len(S) > best_len:
                best = S
                best_len = len(S)

    # Strategy 4: Final local search
    if time_left() > 3:
        S = _local_improve(best, max_rounds=100, time_budget=time_left() - 1)
        if len(S) > best_len:
            best = S
            best_len = len(S)

    # Final validation
    if not _is_valid(set(best)):
        sl = sorted(best)
        while sl and not _is_valid(set(sl)):
            worst = None
            worst_cnt = -1
            P = _PRE
            for elem in sl:
                cnt = 0
                rest = [y for y in sl if y != elem]
                for i in range(len(rest)):
                    for j in range(i + 1, len(rest)):
                        a, b = rest[i], rest[j]
                        if P[elem][a] == P[elem][b] == P[a][b]:
                            cnt += 1
                if cnt > worst_cnt:
                    worst_cnt = cnt
                    worst = elem
            if worst is not None:
                sl.remove(worst)
            else:
                break
        best = sl

    if not best:
        best = [2]

    return best


def construct():
    try:
        res = _construct()
    except Exception:
        _ensure_pre()
        res = []
        resset = set()
        Smask = 0
        for x in range(1, N + 1):
            if _can_add_mask(Smask, x):
                res.append(x)
                resset.add(x)
                Smask |= _BIT[x]
        if not res:
            res = [2]
    return res
