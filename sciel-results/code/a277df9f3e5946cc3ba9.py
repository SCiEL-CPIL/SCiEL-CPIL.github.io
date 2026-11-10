"""Cochromatic partition of a pinned 14-vertex graph; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
FULL = (1 << N) - 1

# --- build adjacency bitmask per vertex ---
_ADJ = [0] * N
for a, b in P['edges']:
    _ADJ[a] |= (1 << b)
    _ADJ[b] |= (1 << a)

# --- popcount for all masks ---
_POP = [0] * (1 << N)
for _i in range(1, 1 << N):
    _POP[_i] = _POP[_i >> 1] + (_i & 1)

# --- precompute is_clique / is_independent for all masks ---
_IS_CLQ = [False] * (1 << N)
_IS_IND = [False] * (1 << N)
_IS_CLQ[0] = True
_IS_IND[0] = True
for _m in range(1, 1 << N):
    _lsb = _m & -_m
    _v = _lsb.bit_length() - 1
    _rest = _m ^ _lsb
    _IS_CLQ[_m] = _IS_CLQ[_rest] and ((_ADJ[_v] & _rest) == _rest)
    _IS_IND[_m] = _IS_IND[_rest] and ((_ADJ[_v] & _rest) == 0)

_IS_VALID = [_IS_CLQ[_m] or _IS_IND[_m] for _m in range(1 << N)]

# --- precompute valid-placement bitmask per (part_mask, vertex) ---
_PLACE = [None] * (1 << N)
for _m in range(1 << N):
    _pm = [0] * N
    for _v in range(N):
        if _IS_VALID[_m | (1 << _v)]:
            _pm[_v] = 1 << _v
    _PLACE[_m] = _pm

# --- deterministic greedy lower bound on number of parts ---
def _greedy_lb():
    # clique lower bound: chromatic number of the graph
    best = N
    for _perm in range(1, 1 << N):
        if _POP[_perm] < best:
            if _IS_CLQ[_perm]:
                best = _POP[_perm]
    # independent-set lower bound: chromatic number of the complement
    best2 = N
    for _perm in range(1, 1 << N):
        if _POP[_perm] < best2:
            if _IS_IND[_perm]:
                best2 = _POP[_perm]
    return max(best, best2)

_LB = _greedy_lb()


def _mask_to_list(mask):
    out = []
    m = mask
    while m:
        lsb = m & -m
        out.append(lsb.bit_length() - 1)
        m ^= lsb
    return out


def construct():
    start = time.time()
    deadline = start + 82.0

    # ---------- exact feasibility test: can we partition FULL into <= k valid parts? ----------
    # parts stored as list of (mask, kind) where kind 0=clique,1=independent
    # We use a recursive backtracking that, given a set of open part "slots",
    # assigns the remaining vertices. To keep it fast we fix the number of parts = k
    # and use symmetry breaking.

    def feasible(k):
        # We'll build exactly k parts (some may end up empty? No - parts must be nonempty).
        # Instead: try to cover FULL with at most k valid parts. We do a search that
        # picks parts (valid subsets) covering all vertices.
        # This is a set-cover / exact-cover style search over valid subsets.
        # Precompute all valid nonempty subsets.
        # But there are many. Use a smarter recursive approach:
        # Pick the smallest unassigned vertex v. It must go into some part.
        # The part containing v is a valid subset S with v in S, S subset of unassigned.
        # Branch over all valid S containing v (subset of unassigned).
        # Symmetry: to avoid permutations, require parts to be processed in order of
        # their minimum vertex. We enforce that the next part's minimum vertex is the
        # smallest unassigned vertex. This guarantees each partition is generated once.

        # valid_subsets_by_min: for each v, list of valid masks with min vertex = v
        # Precompute once outside.
        return _feasible_inner(k, FULL)

    # Precompute valid masks grouped by their minimum vertex (for the search above)
    # min vertex of mask m = lowest set bit.
    _VALID_BY_MIN = [[] for _ in range(N)]
    for _m in range(1, 1 << N):
        if _IS_VALID[_m]:
            _minv = (_m & -_m).bit_length() - 1
            _VALID_BY_MIN[_minv].append(_m)
    # sort each group by size descending (bigger parts first -> fewer of them)
    for _v in range(N):
        _VALID_BY_MIN[_v].sort(key=lambda x: -_POP[x])

    def _feasible_inner(k, unassigned):
        # Returns True if unassigned can be covered by <= k valid parts,
        # with the constraint that we build parts in increasing-min-vertex order.
        if unassigned == 0:
            return True
        if k <= 0:
            return False
        # smallest unassigned vertex
        lsb = unassigned & -unassigned
        v = lsb.bit_length() - 1
        # the next part must contain v and be a valid subset of unassigned with min vertex v
        # (min vertex v is automatic since v is smallest in unassigned)
        for S in _VALID_BY_MIN[v]:
            if (S & unassigned) != S:
                continue  # S not subset of unassigned
            # choose S as the next part
            if _feasible_inner(k - 1, unassigned ^ S):
                return True
        return False

    # ---------- find the actual partition for a given k (reconstruct) ----------
    def find_partition(k):
        # returns list of masks (valid, nonempty, disjoint, union=FULL) of length <= k, or None
        chosen = []

        def rec(k_left, unassigned):
            if unassigned == 0:
                return True
            if k_left <= 0:
                return False
            lsb = unassigned & -unassigned
            v = lsb.bit_length() - 1
            for S in _VALID_BY_MIN[v]:
                if (S & unassigned) != S:
                    continue
                chosen.append(S)
                if rec(k_left - 1, unassigned ^ S):
                    return True
                chosen.pop()
            return False

        if rec(k, FULL):
            return chosen
        return None

    # ---------- greedy initialization (robust, always valid) ----------
    def greedy(order):
        parts = []
        for v in order:
            vb = 1 << v
            placed = False
            for i in range(len(parts)):
                if _IS_VALID[parts[i] | vb]:
                    parts[i] |= vb
                    placed = True
                    break
            if not placed:
                parts.append(vb)
        return parts

    degrees = [_POP[_ADJ[v]] for v in range(N)]
    orders = [
        list(range(N)),
        sorted(range(N), key=lambda v: -degrees[v]),
        sorted(range(N), key=lambda v: degrees[v]),
    ]
    rng = random.Random(12345)
    for _ in range(40):
        o = list(range(N))
        rng.shuffle(o)
        orders.append(o)
    best_greedy = None
    for o in orders:
        g = greedy(o)
        if best_greedy is None or len(g) < len(best_greedy):
            best_greedy = g
    best_masks = best_greedy[:]
    best_k = len(best_greedy)

    # ---------- exact search: try k from LB up to best_k-1 ----------
    for k in range(_LB, best_k):
        if time.time() > deadline:
            break
        part = find_partition(k)
        if part is not None:
            best_masks = part
            best_k = k
            break

    # ---------- final: always return a valid partition (the best found) ----------
    # Ensure best_masks is a valid partition (it always is, from greedy or exact).
    # Sort parts by their minimum vertex for determinism.
    best_masks.sort(key=lambda m: (m & -m).bit_length())
    result = []
    for m in best_masks:
        result.append([int(x) for x in _mask_to_list(m)])
    # convert to plain lists of ints
    return [list(p) for p in result]
# EVOLVE-BLOCK-END
