"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _subset_sums_distinct(S):
    """Check all 2^|S| subset sums of reciprocals are distinct (as fractions)."""
    from fractions import Fraction
    seen = set()
    n = len(S)
    for mask in range(1 << n):
        s = Fraction(0, 1)
        for i in range(n):
            if mask & (1 << i):
                s += Fraction(1, S[i])
        if s in seen:
            return False
        seen.add(s)
    return True

def _is_dissociated(S):
    """Dissociated: no two disjoint subsets have equal reciprocal sum."""
    from fractions import Fraction
    n = len(S)
    sums = [Fraction(1, x) for x in S]
    seen = {}
    for mask in range(1 << n):
        s = Fraction(0, 1)
        for i in range(n):
            if mask & (1 << i):
                s += sums[i]
        if s in seen:
            return False
        seen[s] = mask
    return True

def _validate(S, strict_dissociated=True):
    if not S:
        return False
    if strict_dissociated:
        return _is_dissociated(S)
    return _subset_sums_distinct(S)

def _mis_branch_and_bound(vertices, adj, deadline, best_set, best_size):
    """Branch and bound MIS. adj[v] is a set of neighbors. Returns (size, set)."""
    n = len(vertices)
    # index mapping
    idx = {v: i for i, v in enumerate(vertices)}
    adj_bits = [0] * n
    for v in vertices:
        i = idx[v]
        bits = 0
        for u in adj[v]:
            if u in idx:
                bits |= (1 << idx[u])
        adj_bits[i] = bits

    def bound(cand_bits, current_size, remaining):
        # upper bound: current + popcount of remaining
        return current_size + bin(remaining).count('1')

    best = [best_size, list(best_set)]
    # order vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda i: bin(adj_bits[i]).count('1'), reverse=True)
    order_pos = {i: p for p, i in enumerate(order)}

    def rec(cand_bits, chosen):
        nonlocal best
        if time.time() > deadline:
            return
        if cand_bits == 0:
            if len(chosen) > best[0]:
                best[0] = len(chosen)
                best[1] = list(chosen)
            return
        # bound
        if len(chosen) + bin(cand_bits).count('1') <= best[0]:
            return
        # pick a vertex with max degree in cand
        # iterate candidates
        # choose pivot
        cands = [i for i in range(n) if cand_bits & (1 << i)]
        if not cands:
            return
        # pick vertex with max degree within cand
        pivot = max(cands, key=lambda i: bin(adj_bits[i] & cand_bits).count('1'))
        # Branch 1: include pivot
        new_cand = cand_bits & ~(1 << pivot) & ~adj_bits[pivot]
        chosen.append(vertices[pivot])
        rec(new_cand, chosen)
        chosen.pop()
        # Branch 2: exclude pivot
        new_cand2 = cand_bits & ~(1 << pivot)
        rec(new_cand2, chosen)

    rec((1 << n) - 1, [])
    return best[0], best[1]

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin before 90s
    universe = list(range(1, 23))

    # Build conflict graph: edge between i,j if {i,j} is NOT dissociated.
    # A pair is always dissociated (2 elements, 4 signed sums: 0, 1/i, 1/j, 1/i+1/j all distinct since positive).
    # Actually for the "all subset sums distinct" interpretation, any set works.
    # For dissociated (signed) interpretation: need no nontrivial signed sum = 0.
    # For pairs: 1/i = 1/j only if i=j, so pairs are fine.
    # Conflicts arise for larger sets.
    # 
    # Strategy: greedily build a large candidate, then use local search + B&B to maximize.
    # 
    # First, find a good initial set via greedy.
    # Use the dissociated check (signed sums distinct) as the real constraint.

    # Greedy construction with multiple orderings
    def greedy(order):
        S = []
        for v in order:
            trial = S + [v]
            if _is_dissociated(trial):
                S = trial
        return S

    best_set = []
    best_size = 0

    # Try several orderings
    import random
    random.seed(42)
    orderings = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
        sorted(range(1, 23), key=lambda x: 1/x, reverse=True),
        sorted(range(1, 23), key=lambda x: 1/x),
    ]
    for _ in range(30):
        o = list(range(1, 23))
        random.shuffle(o)
        orderings.append(o)

    for o in orderings:
        S = greedy(o)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
        if time.time() > deadline - 30:
            break

    # Local search: try to add elements
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for v in range(1, 23):
            if v not in best_set:
                trial = best_set + [v]
                if _is_dissociated(trial):
                    best_set = trial
                    best_size += 1
                    improved = True
                    break

    # Try to swap: remove one, add two
    if time.time() < deadline - 15:
        for i in range(len(best_set)):
            for v1 in range(1, 23):
                if v1 in best_set:
                    continue
                for v2 in range(1, 23):
                    if v2 in best_set or v2 == v1:
                        continue
                    trial = [x for j, x in enumerate(best_set) if j != i] + [v1, v2]
                    if len(trial) > len(best_set) and _is_dissociated(trial):
                        best_set = trial
                        best_size = len(trial)
                        break
                if time.time() > deadline - 15:
                    break
            if time.time() > deadline - 15:
                break

    # Build conflict graph for B&B: edge if pair not dissociated (shouldn't happen for pairs)
    # Instead, use the full set constraint. B&B on subset selection.
    # Vertices = 1..22, edge between i,j if they cannot coexist in a dissociated set.
    # But dissociation is not pairwise — it's a global property.
    # So B&B on the full constraint is expensive. Use the local search result.

    # Final validation
    if not _is_dissociated(best_set):
        # Fallback to seed
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        best_size = 9

    return best_set
# EVOLVE-BLOCK-END
