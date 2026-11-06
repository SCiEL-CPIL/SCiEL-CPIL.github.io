"""Bitmask-accelerated local search for large integer sets avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

_TIME_LIMIT = 84.0
_DEADLINE = None

# Precompute gcd matrix for all pairs in [1, 128]
_N = 128
_gcd_matrix = [[0] * (_N + 1) for _ in range(_N + 1)]
for _i in range(1, _N + 1):
    for _j in range(_i, _N + 1):
        _g = gcd(_i, _j)
        _gcd_matrix[_i][_j] = _g
        _gcd_matrix[_j][_i] = _g

# Precompute prime factorization for each number
_prime_factors = [set() for _ in range(_N + 1)]
for _n in range(2, _N + 1):
    _tmp = _n
    _p = 2
    while _p * _p <= _tmp:
        if _tmp % _p == 0:
            _prime_factors[_n].add(_p)
            while _tmp % _p == 0:
                _tmp //= _p
        _p += 1
    if _tmp > 1:
        _prime_factors[_n].add(_tmp)

# Precompute conflict bitmasks: for each pair (i,j) with gcd g,
# which other elements k have gcd(i,k)==g AND gcd(j,k)==g
# This makes _can_add O(1) via bitmask intersection
_conflict_mask = [0] * (_N + 1)  # Not directly used this way; see below

# For faster checking: precompute for each x, the list of (g, mask) where
# mask has bit y set if gcd(x,y)==g
# Actually, let's precompute a different structure:
# For each x, for each possible gcd value g that appears, a bitmask of y such that gcd(x,y)==g
_gcd_groups = [dict() for _ in range(_N + 1)]
for _x in range(1, _N + 1):
    for _y in range(1, _N + 1):
        if _x == _y:
            continue
        _g = _gcd_matrix[_x][_y]
        if _g not in _gcd_groups[_x]:
            _gcd_groups[_x][_g] = 0
        _gcd_groups[_x][_g] |= (1 << (_y - 1))


def _can_add_mask(mask, x):
    """Check if adding x (1-indexed) to set represented by bitmask is valid."""
    idx = x - 1
    # For each element y in mask, compute gcd(x,y)
    # A triple (a,b,c) is forbidden if gcd(a,b)==gcd(a,c)==gcd(b,c)
    # When adding x, we check pairs (y1,y2) in mask: if gcd(x,y1)==gcd(x,y2)==gcd(y1,y2), conflict
    # Group elements in mask by their gcd with x
    # For each group g, if any two elements in that group have gcd == g, conflict
    # Precomputed: for x, _gcd_groups[x][g] gives bitmask of y with gcd(x,y)==g
    # We need: for some g, (mask & _gcd_groups[x][g]) contains two elements y1,y2 with gcd(y1,y2)==g
    # That's equivalent to: for some g, the submask (mask & _gcd_groups[x][g]) has an edge in the "gcd==g" graph
    # Simpler: iterate through elements in mask, track seen gcds, check conflicts
    # But that's O(n^2). Let's use a different approach:
    
    # For each pair of elements y1,y2 already in mask, check if adding x creates a bad triple
    # Bad triple: (x, y1, y2) where gcd(x,y1)==gcd(x,y2)==gcd(y1,y2)
    # So: for each y1 in mask, let g1 = gcd(x,y1)
    #     for each y2 in mask, y2 != y1, let g2 = gcd(x,y2)
    #     if g1 == g2 == gcd(y1,y2), conflict
    # 
    # Optimization: group by g1. For each group G_g of elements with gcd(x,y)==g,
    # check if any pair in G_g has gcd(y1,y2)==g.
    # Precompute for each (x,g): which pairs (y1,y2) in _gcd_groups[x][g] have gcd(y1,y2)==g
    # That's expensive to precompute. Instead, just do the O(n^2) check but with early exit.
    
    m = mask
    # Collect elements in mask
    elems = []
    while m:
        low = m & (-m)
        idx_y = low.bit_length() - 1
        elems.append(idx_y + 1)
        m ^= low
    
    n = len(elems)
    if n < 2:
        return True
    
    # Group by gcd with x
    groups = {}
    for y in elems:
        g = _gcd_matrix[x][y]
        if g not in groups:
            groups[g] = []
        groups[g].append(y)
    
    for g, group in groups.items():
        if len(group) < 2:
            continue
        # Check if any pair in group has gcd == g
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if _gcd_matrix[group[i]][group[j]] == g:
                    return False
    return True


def _can_add(S, x):
    """Check if adding x to S keeps the set triple-free using precomputed gcd matrix."""
    n = len(S)
    if n < 2:
        return True
    # Group S elements by gcd with x
    groups = {}
    for y in S:
        g = _gcd_matrix[x][y]
        if g not in groups:
            groups[g] = []
        groups[g].append(y)
    for g, group in groups.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if _gcd_matrix[group[i]][group[j]] == g:
                    return False
    return True


def _is_valid(S):
    """Full validation check."""
    S_list = list(S)
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            g_ab = _gcd_matrix[a][b]
            for k in range(j + 1, len(S_list)):
                c = S_list[k]
                if _gcd_matrix[a][c] == g_ab and _gcd_matrix[b][c] == g_ab:
                    return False
    return True


def _greedy(order, deadline):
    """Greedy: iterate order, add if valid."""
    S = []
    Sset = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in Sset:
            continue
        if _can_add(S, x):
            S.append(x)
            Sset.add(x)
    return S


def _add_all_possible(S, Sset, universe, deadline):
    """Add all elements that can be added, in order."""
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for x in universe:
            if time.time() > deadline:
                break
            if x in Sset:
                continue
            if _can_add(S, x):
                S.append(x)
                Sset.add(x)
                improved = True
                break
    return S, Sset


def _local_search(S, deadline, rng):
    """Enhanced local search with bitmask-accelerated 1-remove-k-add and 2-remove-k-add."""
    S = list(S)
    Sset = set(S)
    universe = list(range(1, 129))
    d = deadline - 3  # safety margin

    # Phase 1: add all possible
    S, Sset = _add_all_possible(S, Sset, universe, d)

    # Phase 2: Aggressive 1-remove-k-add with bitmask validity
    if time.time() < d - 8:
        for _iter in range(100):
            if time.time() > d - 8:
                break
            progress = False
            n = len(S)
            for idx in range(n):
                if time.time() > d - 8:
                    break
                S2 = S[:idx] + S[idx + 1:]
                S2set = set(S2)
                # Build bitmask of S2
                mask2 = 0
                for y in S2:
                    mask2 |= (1 << (y - 1))
                
                best_gain = 0
                best_x = None
                best_added = []
                for x in universe:
                    if x in S2set:
                        continue
                    if _can_add_mask(mask2, x):
                        # Try to greedily add more
                        S3 = S2 + [x]
                        S3set = S2set | {x}
                        mask3 = mask2 | (1 << (x - 1))
                        added = []
                        for y in universe:
                            if y in S3set:
                                continue
                            if _can_add_mask(mask3, y):
                                S3.append(y)
                                S3set.add(y)
                                mask3 |= (1 << (y - 1))
                                added.append(y)
                        gain = len(added)
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_added = added
                            if gain >= 4:
                                break
                if best_x is not None and best_gain > 0:
                    S = S2 + [best_x]
                    Sset = set(S)
                    for y in best_added:
                        S.append(y)
                        Sset.add(y)
                    progress = True
                    break
            if not progress:
                break

    # Phase 3: add all possible again
    S, Sset = _add_all_possible(S, Sset, universe, d)

    # Phase 4: 2-remove-k-add
    if time.time() < d - 5:
        for _iter in range(30):
            if time.time() > d - 5:
                break
            progress = False
            n = len(S)
            for i in range(n):
                if time.time() > d - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > d - 5:
                        break
                    S2 = S[:i] + S[i+1:j] + S[j+1:]
                    S2set = set(S2)
                    mask2 = 0
                    for y in S2:
                        mask2 |= (1 << (y - 1))
                    
                    best_gain = 0
                    best_x = None
                    best_added = []
                    for x in universe:
                        if time.time() > d - 5:
                            break
                        if x in S2set:
                            continue
                        if _can_add_mask(mask2, x):
                            S3 = S2 + [x]
                            S3set = S2set | {x}
                            mask3 = mask2 | (1 << (x - 1))
                            added = []
                            for y in universe:
                                if y in S3set:
                                    continue
                                if _can_add_mask(mask3, y):
                                    S3.append(y)
                                    S3set.add(y)
                                    mask3 |= (1 << (y - 1))
                                    added.append(y)
                                    if len(added) >= 3:
                                        break
                            gain = len(added)
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                                best_added = added
                                if gain >= 4:
                                    break
                    if best_x is not None and best_gain > 0:
                        S = S2 + [best_x]
                        Sset = set(S)
                        for y in best_added:
                            S.append(y)
                            Sset.add(y)
                        progress = True
                        break
                if progress:
                    break
            if not progress:
                break

    # Phase 5: final add pass
    S, Sset = _add_all_possible(S, Sset, universe, d)

    # Phase 6: polish - try replacing each element
    if time.time() < d - 2:
        polish_improved = True
        while polish_improved:
            if time.time() > d - 2:
                break
            polish_improved = False
            n = len(S)
            for idx in range(n):
                if time.time() > d - 2:
                    break
                S2 = S[:idx] + S[idx + 1:]
                S2set = set(S2)
                mask2 = 0
                for y in S2:
                    mask2 |= (1 << (y - 1))
                best_x = None
                for x in universe:
                    if x in S2set:
                        continue
                    if _can_add_mask(mask2, x):
                        best_x = x
                        break
                if best_x is not None and best_x != S[idx]:
                    S = S2 + [best_x]
                    Sset = set(S)
                    polish_improved = True
                    break

    # Phase 7: final add
    S, Sset = _add_all_possible(S, Sset, universe, d)

    return S


def _multi_greedy_local(deadline):
    rng = random.Random(42)
    universe = list(range(1, 129))
    best = []

    # Generate diverse orderings
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        [x for x in range(1, 129) if x % 2 == 1] + [x for x in range(1, 129) if x % 2 == 0],
        [x for x in range(1, 129) if x % 2 == 0] + [x for x in range(1, 129) if x % 2 == 1],
        sorted(range(1, 129), key=lambda x: -x.bit_count()),
        sorted(range(1, 129), key=lambda x: x.bit_count()),
        [x for x in range(1, 129) if x % 3 != 0] + [x for x in range(1, 129) if x % 3 == 0],
        [x for x in range(1, 129) if x % 3 == 0] + [x for x in range(1, 129) if x % 3 != 0],
        sorted(range(1, 129), key=lambda x: (x % 2, x)),
        sorted(range(1, 129), key=lambda x: (-x.bit_count(), x)),
        sorted(range(1, 129), key=lambda x: -len(_prime_factors[x])),
        sorted(range(1, 129), key=lambda x: max(_prime_factors[x]) if _prime_factors[x] else 0),
        sorted(range(1, 129), key=lambda x: len(_prime_factors[x])),
    ]
    for _ in range(16):
        o = universe[:]
        rng.shuffle(o)
        orders.append(o)

    # Run greedy on each ordering
    candidates = []
    for order in orders:
        if time.time() > deadline - 12:
            break
        cand = _greedy(order, deadline)
        candidates.append(cand)
        if len(cand) > len(best):
            best = cand

    # Local search on top candidates
    candidates.sort(key=len, reverse=True)
    for cand in candidates[:10]:
        if time.time() > deadline - 4:
            break
        improved = _local_search(cand, deadline, rng)
        if len(improved) > len(best):
            best = improved

    return best


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    try:
        result = _multi_greedy_local(_DEADLINE)
    except Exception:
        # Fallback: simple ascending greedy with precomputed matrix
        result = []
        for x in range(1, 129):
            ok = True
            for i in range(len(result)):
                a = result[i]
                ga = _gcd_matrix[a][x]
                for j in range(i + 1, len(result)):
                    b = result[j]
                    if ga == _gcd_matrix[b][x] == _gcd_matrix[a][b]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                result.append(x)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
