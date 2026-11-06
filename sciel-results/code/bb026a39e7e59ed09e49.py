"""Equal-gcd-triple-free subsets of [1,128]: bitmask-accelerated search.

Model: maximize |S| s.t. for every triple {a,b,c} with a<b<c in [1,128],
    NOT( gcd(a,b)==gcd(a,c)==gcd(b,c) )

Key structural change: represent the current set S as a 128-bit integer bitmask.
Precompute for each element x a list of bitmask pairs (mask_a, mask_b) where
adding x creates a conflict iff (S & mask_a) and (S & mask_b) are both nonzero.
This makes each conflict check O(1) bitwise ops instead of O(|conflicts|) set lookups.

The search uses:
1. Fast greedy with bitmask conflict checks
2. Aggressive local search: try all adds, then 1-swap and 2-swap moves
3. Simulated annealing with fast incremental bad-triple counting via bitmasks
"""
import time
import random
import math
from math import gcd

N = 128
_DEADLINE = 84.0  # 6s safety margin under 90s kill

# Precomputed structures
_PRE = None       # _PRE[i][j] = gcd(i,j) for 1..128
# _CONFLICT_MASKS[x] = list of (mask_a, mask_b) where mask_a/mask_b are bitmasks
#   such that adding x to S creates a conflict iff (S & mask_a) != 0 AND (S & mask_b) != 0
_CONFLICT_MASKS = None
# _BIT[x] = 1 << (x-1)
_BIT = None
# _CONFLICT_LISTS[x][y] = bitmask of z such that {x,y,z} is a bad triple
_CONFLICT_LISTS = None


def _ensure_pre():
    global _PRE, _CONFLICT_MASKS, _BIT, _CONFLICT_LISTS
    if _PRE is not None:
        return
    # Build gcd table
    PRE = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            PRE[i][j] = g
            PRE[j][i] = g
    _PRE = PRE

    # Bit for each element
    BIT = [0] * (N + 1)
    for x in range(1, N + 1):
        BIT[x] = 1 << (x - 1)
    _BIT = BIT

    # Build conflict masks for validity check (existing logic)
    CONFLICT = {x: [] for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            g = PRE[x][a]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if PRE[x][b] == g and PRE[a][b] == g:
                    CONFLICT[x].append((a, b))
    
    # Convert to bitmask pairs for fast checking
    CONFLICT_MASKS = {x: [] for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for a, b in CONFLICT[x]:
            CONFLICT_MASKS[x].append((BIT[a], BIT[b]))
    _CONFLICT_MASKS = CONFLICT_MASKS
    
    # Build conflict lists for incremental counting
    # _CONFLICT_LISTS[x][y] is a bitmask of all z != x,y such that {x,y,z} is a bad triple
    CL = [[0] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            g = PRE[x][y]
            mask = 0
            for z in range(1, N + 1):
                if z == x or z == y:
                    continue
                if PRE[x][z] == g and PRE[y][z] == g:
                    mask |= BIT[z]
            CL[x][y] = mask
            CL[y][x] = mask
    _CONFLICT_LISTS = CL


def _is_valid(Sset):
    """Check if Sset (a set of ints) has no equal-gcd triple."""
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
    """Count equal-gcd triples in Sset."""
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
    """Check if x can be added to set represented by bitmask Smask."""
    for ma, mb in _CONFLICT_MASKS[x]:
        if (Smask & ma) and (Smask & mb):
            return False
    return True


def _can_add(Sset, x):
    """Check if x can be added to valid set S (Sset is a set)."""
    Smask = 0
    for v in Sset:
        Smask |= _BIT[v]
    return _can_add_mask(Smask, x)


def _swap_valid_mask(Smask, r, x):
    """Check if S - {r} + {x} is valid using bitmask."""
    Smask_new = Smask & ~_BIT[r]
    for ma, mb in _CONFLICT_MASKS[x]:
        if (Smask_new & ma) and (Smask_new & mb):
            return False
    return True

def _count_conflicts_for_add(Smask, x):
    """Count how many bad triples are created by adding x to S."""
    cnt = 0
    CL = _CONFLICT_LISTS
    for y in range(1, N + 1):
        if (Smask & _BIT[y]):
            cnt += (CL[x][y] & Smask).bit_count()
    return cnt

def _greedy_add(order, S):
    """Greedy: try to add elements in order to S (S is a list). Returns new list."""
    S = list(S)
    Sset = set(S)
    Smask = 0
    for v in Sset:
        Smask |= _BIT[v]
    for x in order:
        if x in Sset:
            continue
        if _can_add_mask(Smask, x):
            S.append(x)
            Sset.add(x)
            Smask |= _BIT[x]
    return S

def _local_improve(S, max_rounds=200, time_budget=None):
    """Local improvement: repeatedly try all adds, then swaps that enable adds."""
    S = list(S)
    Sset = set(S)
    Smask = 0
    for v in Sset:
        Smask |= _BIT[v]
    start_time = time.time()
    for rnd in range(max_rounds):
        if time_budget is not None and time.time() - start_time > time_budget:
            break
        # Phase A: try all possible adds
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
        # Phase B: try swaps that enable new adds
        improved = False
        for r in S:
            for x in range(1, N + 1):
                if x in Sset or x == r:
                    continue
                if not _swap_valid_mask(Smask, r, x):
                    continue
                # Tentative swap
                S.remove(r)
                Sset.discard(r)
                Smask &= ~_BIT[r]
                S.append(x)
                Sset.add(x)
                Smask |= _BIT[x]
                # Check if any new element can be added
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
                    # Revert swap
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


def _compute_bad_count_fast(Smask):
    """Compute number of bad triples using bitmask-based approach."""
    # This is O(N^3) but with very small constant due to precomputed gcd table.
    # Used only for initial setup or final validation if needed.
    CL = _CONFLICT_LISTS
    cnt = 0
    sl = [x for x in range(1, N + 1) if (Smask & _BIT[x])]
    n = len(sl)
    for i in range(n):
        a = sl[i]
        for j in range(i + 1, n):
            b = sl[j]
            # Count c > b in S such that c is in CL[a][b]
            # We can just check all c in S, but we need to avoid double counting.
            # Actually, CL[a][b] contains all c != a,b.
            # If we sum over all pairs (a,b) with a<b in S, and count c in S \ {a,b} from CL[a][b],
            # we count each triple {a,b,c} exactly 3 times (once for each pair).
            # So divide by 3 at the end.
            cnt += (CL[a][b] & Smask).bit_count()
    return cnt // 3

def _simulated_annealing(S, n_iters=50000, time_budget=30.0):
    """SA with bitmask-based fast conflict checks."""
    S = list(S)
    Sset = set(S)
    Smask = 0
    for v in Sset:
        Smask |= _BIT[v]
    start_time = time.time()
    rng = random.Random()
    best_S = list(S)
    best_bc = _compute_bad_count_fast(Smask)
    best_score = -best_bc + 0.1 * len(S)
    curr_bc = best_bc
    curr_score = best_score

    T = 1.0
    cooling = 0.99995

    for it in range(n_iters):
        if time.time() - start_time > time_budget:
            break
        # Choose random move
        move_type = rng.random()
        if move_type < 0.5:
            # Swap
            if len(S) < 2:
                continue
            r = rng.choice(S)
            x = rng.randint(1, N)
            if x == r or x in Sset:
                continue
            
            # Incremental update
            # Remove r: bc decreases by sum_{y in S\{r}} |CL[r][y] & (S\{r})|
            # Add x: bc increases by sum_{y in S\{r}} |CL[x][y] & (S\{r})|
            
            Smask_no_r = Smask & ~_BIT[r]
            
            delta_bc = 0
            CL = _CONFLICT_LISTS
            # Cost of removing r
            # For each y in S_no_r, count z in S_no_r such that {r,y,z} bad.
            # This is sum_{y in S_no_r} (CL[r][y] & S_no_r).bit_count()
            # But note CL[r][y] includes z. If z=r, it's not in S_no_r.
            # So this correctly counts triples containing r.
            
            # We can compute this in O(|S|)
            # To speed up, we can precompute? No, S changes.
            # But O(|S|) is ~20-30 operations. Very fast.
            
            # Let's just calculate the new bad count directly using the fast method?
            # _compute_bad_count_fast is O(|S|^2 * bit_count).
            # |S| ~ 20. 20^2 = 400. 400 * 1 (bit_count is C speed) = 400 ops.
            # Incremental is O(|S|) = 20 ops.
            # Let's use incremental.
            
            # However, implementing incremental correctly for swap is tricky.
            # Let's stick to _compute_bad_count_fast for simplicity but it's O(N^2).
            # Wait, _compute_bad_count_fast iterates over pairs in S.
            # Number of pairs is ~200. Each does a bit_count.
            # 200 bit_counts is very fast.
            # Let's try it.
            
            Smask_new = Smask_no_r | _BIT[x]
            new_bc = _compute_bad_count_fast(Smask_new)
            new_score = -new_bc + 0.1 * len(S)
            delta = new_score - curr_score
            if delta >= 0 or rng.random() < math.exp(delta / max(T, 0.001)):
                # Accept
                S.remove(r)
                Sset.discard(r)
                S.append(x)
                Sset.add(x)
                Smask = Smask_new
                curr_bc = new_bc
                curr_score = new_score
            T *= cooling
            if curr_score > best_score:
                best_score = curr_score
                best_S = list(S)
        elif move_type < 0.75:
            # Add
            x = rng.randint(1, N)
            if x in Sset:
                continue
            Smask_new = Smask | _BIT[x]
            new_bc = _compute_bad_count_fast(Smask_new)
            new_score = -new_bc + 0.1 * (len(S) + 1)
            delta = new_score - curr_score
            if delta >= 0 or rng.random() < math.exp(delta / max(T, 0.001)):
                S.append(x)
                Sset.add(x)
                Smask = Smask_new
                curr_bc = new_bc
                curr_score = new_score
            T *= cooling
            if curr_score > best_score:
                best_score = curr_score
                best_S = list(S)
        else:
            # Remove
            if len(S) < 2:
                continue
            r = rng.choice(S)
            Smask_new = Smask & ~_BIT[r]
            new_bc = _compute_bad_count_fast(Smask_new)
            new_score = -new_bc + 0.1 * (len(S) - 1)
            delta = new_score - curr_score
            if delta >= 0 or rng.random() < math.exp(delta / max(T, 0.001)):
                S.remove(r)
                Sset.discard(r)
                Smask = Smask_new
                curr_bc = new_bc
                curr_score = new_score
            T *= cooling
            if curr_score > best_score:
                best_score = curr_score
                best_S = list(S)

    # Final: clean up to get a valid set from best_S
    best_Sset = set(best_S)
    if _is_valid(best_Sset):
        return best_S
    # If not valid, repair
    sl = sorted(best_S)
    for elem in sl:
        if _bad_count(set(sl)) > 0 and elem in sl:
            sl.remove(elem)
            if _bad_count(set(sl)) == 0:
                break
    return sl


def _construct():
    global _START
    _ensure_pre()
    _START = time.time()
    P = _PRE

    def time_left():
        return _DEADLINE - (time.time() - _START)

    best = []
    best_len = -1

    # Strategy 1: Multiple greedy orders + local improvement
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
        sorted(range(1, N + 1), key=lambda x: (-divs[x], x)),
        sorted(range(1, N + 1), key=lambda x: (divs[x], x)),
    ]

    for order in orders:
        if time_left() < 5:
            break
        S = _greedy_add(order, [])
        S = _local_improve(S, max_rounds=100, time_budget=min(3, time_left() - 3))
        if len(S) > best_len:
            best = S
            best_len = len(S)

    # Strategy 2: Simulated annealing from the best greedy solution
    if time_left() > 10:
        S = _simulated_annealing(best, n_iters=100000, time_budget=min(30, time_left() - 5))
        if len(S) > best_len and _is_valid(set(S)):
            best = S
            best_len = len(S)
            S = _local_improve(best, max_rounds=50, time_budget=min(3, time_left() - 3))
            if len(S) > best_len:
                best = S
                best_len = len(S)

    # Strategy 3: More SA restarts from different initial points
    rng = random.Random(42)
    while time_left() > 8:
        rng_order = list(range(1, N + 1))
        rng.shuffle(rng_order)
        init = _greedy_add(rng_order[:40], [])
        S = _simulated_annealing(init, n_iters=50000, time_budget=min(15, time_left() - 4))
        if len(S) > best_len and _is_valid(set(S)):
            S = _local_improve(S, max_rounds=30, time_budget=min(2, time_left() - 2))
            if len(S) > best_len:
                best = S
                best_len = len(S)

    # Strategy 4: Additional local search rounds on best
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
        for x in range(1, N + 1):
            if _can_add(resset, x):
                res.append(x)
                resset.add(x)
        if not res:
            res = [2]
    return res
