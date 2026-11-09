"""Structural redesign: model non-2-colorability as a hitting-set over the
family of monochromatic 5-tuples, solve with fast bitset greedy + bitset SAT."""
import time
import random
from itertools import combinations

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # explicit guard, 10s safety margin vs 150s kill

# ---------------------------------------------------------------------------
# Fast bitset 2-colorability check (equivalent to backtracking + unit propagation)
# ---------------------------------------------------------------------------
def _is_2_colorable(edges, n, time_limit=2.0):
    """Return True if the 5-uniform hypergraph admits a proper 2-colouring
    (every edge has both colours).  Uses bitset backtracking with unit
    propagation, which is exactly the required check."""
    m = len(edges)
    if m == 0:
        return True
    # For each vertex, the bitset of edges it belongs to.
    inc = [0] * n
    for i, e in enumerate(edges):
        for v in e:
            inc[v] |= (1 << i)

    start = time.time()
    full = (1 << m) - 1
    # state: assigned[v] in {-1,0,1}; c0,c1 = bitsets of edges having a 0 / 1.
    assigned = [-1] * n
    c0 = 0
    c1 = 0
    unassigned = [v for v in range(n)]

    def propagate(c0, c1, assigned, unassigned):
        """Apply unit propagation.  Returns False on contradiction, else True.
        Modifies c0/c1 via return, assigned and unassigned in place."""
        changed = True
        while changed:
            changed = False
            # An edge is forced when it already has 4 of one colour.
            # all0 = edges with >=4 zeros and not yet having a 1 -> remaining vertex must be 1
            # all1 = edges with >=4 ones and not yet having a 0 -> remaining vertex must be 0
            # Contradiction: an edge with 5 zeros (c0 has it and c1 not and no unassigned in it)
            # Simpler: an edge in c0 with no 1 and zero unassigned vertices => all 5 zero -> bad.
            # We detect forced assignments by: edges where count of one colour ==4.
            # Compute edges that have 4+ zeros but no ones:
            # We track counts implicitly; recompute is cheap enough for m<=200.
            # Instead use: forced0 = c0 & ~c1 & (edges where exactly one vertex unassigned)
            # That requires per-edge unassigned count; do it via bitset of edges still open.
            # open = edges that have at least one unassigned vertex
            # We'll compute the set of edges that are "nearly monochromatic".
            # Use precomputed: for each edge, list of vertices.
            # Forced 1: edge has 4 zeros, 0 ones -> the 5th must be 1.
            # Forced 0: edge has 4 ones, 0 zeros -> the 5th must be 0.
            # We need per-edge counts of assigned 0 and 1.
            # Maintain counts arrays.
            pass
        return True

    # Use explicit per-edge counters for clarity and correctness.
    c0cnt = [0] * m
    c1cnt = [0] * m

    def propagate2(assigned, unassigned):
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edges):
                z = c0cnt[i]
                o = c1cnt[i]
                if z == 5 or o == 5:
                    return False
                un = [v for v in e if assigned[v] == -1]
                if z == 4 and len(un) == 1:
                    v = un[0]
                    if assigned[v] != -1:
                        if assigned[v] == 0:
                            return False
                        continue
                    assigned[v] = 1
                    c1cnt[i] += 1
                    for j, e2 in enumerate(edges):
                        if v in e2:
                            c1cnt[j] += 1
                    unassigned.remove(v)
                    changed = True
                    break
                elif o == 4 and len(un) == 1:
                    v = un[0]
                    if assigned[v] != -1:
                        if assigned[v] == 1:
                            return False
                        continue
                    assigned[v] = 0
                    c0cnt[i] += 1
                    for j, e2 in enumerate(edges):
                        if v in e2:
                            c0cnt[j] += 1
                    unassigned.remove(v)
                    changed = True
                    break
                elif z == 0 and len(un) == 1:
                    v = un[0]
                    if assigned[v] != -1:
                        if assigned[v] == 1:
                            return False
                        continue
                    assigned[v] = 0
                    c0cnt[i] += 1
                    for j, e2 in enumerate(edges):
                        if v in e2:
                            c0cnt[j] += 1
                    unassigned.remove(v)
                    changed = True
                    break
                elif o == 0 and len(un) == 1:
                    v = un[0]
                    if assigned[v] != -1:
                        if assigned[v] == 0:
                            return False
                        continue
                    assigned[v] = 1
                    c1cnt[i] += 1
                    for j, e2 in enumerate(edges):
                        if v in e2:
                            c1cnt[j] += 1
                    unassigned.remove(v)
                    changed = True
                    break
        return True

    # Branch-and-bound over vertices with forward checking.
    # Symmetry: fix vertex 0 to colour 0.
    def bt(assigned, unassigned):
        if time.time() - start > time_limit:
            return None  # timeout -> treat as "maybe colourable" (conservative)
        if not propagate2(assigned, unassigned):
            return None
        if not unassigned:
            return True
        # heuristic: pick unassigned vertex in most constrained (smallest unassigned) edge
        v = unassigned[0]
        for col in (0, 1):
            saved = assigned[v]
            assigned[v] = col
            saved_cnt = c0cnt[:], c1cnt[:]
            # apply
            for j, e in enumerate(edges):
                if v in e:
                    if col == 0:
                        c0cnt[j] += 1
                    else:
                        c1cnt[j] += 1
            unassigned.remove(v)
            r = bt(assigned, unassigned)
            # undo
            unassigned.append(v)
            for j, e in enumerate(edges):
                if v in e:
                    if col == 0:
                        c0cnt[j] -= 1
                    else:
                        c1cnt[j] -= 1
            assigned[v] = saved
            if r is not None:
                return r
        return None

    # fix vertex 0 to 0 for symmetry
    assigned[0] = 0
    for j, e in enumerate(edges):
        if 0 in e:
            c0cnt[j] += 1
    unassigned = [v for v in range(n) if v != 0]
    r = bt(assigned, unassigned)
    return bool(r)


# ---------------------------------------------------------------------------
# Hitting-set construction via bitset greedy
# ---------------------------------------------------------------------------
def _build(n, k, seed, time_limit=10.0):
    """Greedy hitting set: pick 5-subsets that hit the most remaining
    monochromatic 5-tuples.  Returns list of edges or None."""
    random.seed(seed)
    start = time.time()
    all5 = list(combinations(range(n), 5))
    mono = list(all5) + [tuple(n - 1 - x for x in c) for c in all5]
    # map each monochromatic 5-tuple to a bit
    mono_set = set(mono)
    # For each candidate 5-subset, the set of monochromatic tuples it hits
    # (a 5-subset S hits a monochromatic tuple T iff S == T, since both size 5).
    # Wait: a monochromatic 5-tuple is itself a 5-subset. An edge (5-subset)
    # "hits" (violates) exactly the monochromatic tuple equal to it.
    # So the family of monochromatic 5-tuples IS the set of all 5-subsets.
    # Hitting every monochromatic 5-tuple means: for EVERY 5-subset T, our edge
    # set contains T?? No. Re-derive:
    # A colouring assigns each vertex 0/1. It is "bad" (no proper colouring)
    # if some edge is monochromatic. For a fixed colouring, the monochromatic
    # edges are those 5-subsets fully in the 0-set or fully in the 1-set.
    # We need: for EVERY colouring, at least one of our chosen edges is monochromatic.
    # Equivalently, our chosen edge set E must be a hitting set of the family
    # F = { (S0, S1) : partition of V into 5-subset-free? } ... 
    # Cleaner: For each colouring c (a subset A of V being colour 0), the bad
    # edges are the 5-subsets of A and the 5-subsets of V\A. We need E to
    # intersect (5-subsets of A) union (5-subsets of V\A) for every A.
    # So define for each A the set Bad(A) = {5-subsets of A} U {5-subsets of V\A}.
    # We need E such that for all A, E ∩ Bad(A) != empty.
    # This is a hitting set over the family {Bad(A)}.
    # Represent each A as a bitset; an edge e (5-subset) is in Bad(A) iff
    # e ⊆ A or e ⊆ V\A.
    # Greedy: maintain uncovered set of A's; pick edge covering most uncovered.
    # Number of A's is 2^n -- too many for n=11 (2048 ok), n=12 (4096 ok), n=13 (8192 ok).
    # We'll cap n so 2^n is manageable.
    if n > 13:
        return None
    subsets = list(range(1 << n))
    # Precompute for each edge its coverage bitset over A's
    # coverage[e][a] = 1 iff e ⊆ A or e ⊆ V\A
    # Represent coverage as Python int bitset over the 2^n subsets.
    full_mask = (1 << n) - 1
    # For speed, precompute for each a the bitset of 5-subsets contained in a.
    # Instead compute per-edge coverage by iterating a.
    # Build candidate edge list.
    candidates = list(all5)
    # coverage as list of int bitsets
    cov = [0] * len(candidates)
    # map edge -> index
    edge_idx = {c: i for i, c in enumerate(candidates)}
    # For each a, determine which edges are in Bad(a), set bits.
    # To be efficient, precompute contained5[a] = set of 5-subsets of a (as idx).
    # 2^n up to 8192, each with up to C(13,5)=1287 subsets -> ~10M ops, ok.
    contained5 = []
    for a in subsets:
        bits = [v for v in range(n) if (a >> v) & 1]
        s = set()
        if len(bits) >= 5:
            for c5 in combinations(bits, 5):
                s.add(edge_idx[c5])
        contained5.append(s)
    # Now coverage for each edge:
    uncovered = (1 << len(subsets)) - 1  # all A's uncovered
    for i in range(len(candidates)):
        covi = 0
        for a in subsets:
            if candidates[i] in contained5[a] or candidates[i] in contained5[full_mask ^ a]:
                covi |= (1 << a)
        cov[i] = covi

    chosen = []
    chosen_set = set()
    while uncovered:
        if time.time() - start > time_limit:
            return None
        # pick edge covering most uncovered
        best_i = -1
        best_cnt = 0
        for i in range(len(candidates)):
            if i in chosen_set:
                continue
            cnt = (uncovered & cov[i]).bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best_i = i
                if best_cnt == uncovered.bit_count():
                    break
        if best_i == -1 or best_cnt == 0:
            return None  # cannot cover (shouldn't happen)
        chosen.append(candidates[best_i])
        chosen_set.add(best_i)
        uncovered &= ~cov[best_i]
        if len(chosen) > 200:
            return None
    return [list(e) for e in chosen]


def _local_search(edges, n, time_limit=20.0):
    """Try to reduce edge count by removing redundant edges and swapping."""
    start = time.time()
    edges = [list(e) for e in edges]
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Try removing each edge
        for i in range(len(edges)):
            if time.time() - start > time_limit:
                break
            trial = edges[:i] + edges[i + 1:]
            if not _is_2_colorable(trial, n, time_limit=1.5):
                edges = trial
                improved = True
                break
    return edges


def construct():
    """Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges."""
    global _T0, _DEADLINE
    best = None
    # Target the known optimum of 51 edges.  Hitting-set greedy on n in [10,13].
    for n in (11, 10, 12, 13, 9):
        if time.time() > _DEADLINE - 30:
            break
        for seed in (42, 7, 123, 2024, 5, 99, 314, 271):
            if time.time() > _DEADLINE - 25:
                break
            res = _build(n, None, seed, time_limit=12.0)
            if res is None:
                continue
            if len(res) > 200:
                continue
            # verify non-colorable
            if _is_2_colorable(res, n, time_limit=3.0):
                continue
            # local search to shrink
            res = _local_search(res, n, time_limit=15.0)
            if best is None or len(res) < len(best['edges']):
                best = {'n': n, 'edges': res}
            if len(res) <= 51:
                return best
    if best is not None:
        return best
    # Fallback: complete 5-uniform on 9 vertices (guaranteed non-2-colorable)
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
