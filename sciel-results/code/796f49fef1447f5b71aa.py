import time
import math
import random
import sys
import heapq
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """All (a,b,c) with distinct a,b,c in [1,N] and 1/a = 1/b + 1/c."""
    triples = []
    seen = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    key = frozenset((a, b, c))
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c))
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    N = 180

    triples = _precompute_triples(N)
    T = len(triples)

    # element (1..N) -> list of triple indices
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)

    def greedy_seed():
        """Greedy: repeatedly add the element that creates the fewest new conflicts.
        Return (set, count array) where count[t] = number of chosen elements in triple t."""
        S = set()
        counts = [0] * T
        in_S = [False] * (N + 1)
        # For each element, precompute its triple list
        while True:
            if time.time() > deadline - 2.0:
                break
            best_x = -1
            best_cost = None
            for x in range(1, N + 1):
                if in_S[x]:
                    continue
                cost = 0
                for t_idx in elem_triples[x]:
                    if counts[t_idx] >= 2:
                        cost = 10**9
                        break
                    else:
                        cost += 1
                if best_cost is None or cost < best_cost or (cost == best_cost and x < best_x):
                    best_cost = cost
                    best_x = x
            if best_x == -1:
                break
            S.add(best_x)
            in_S[best_x] = True
            for t_idx in elem_triples[best_x]:
                counts[t_idx] += 1
        return S, counts

    best_S, best_counts = greedy_seed()
    best_size = len(best_S)

    # Precompute a fractional (LP-relaxation) upper bound on the independent-set size.
    # Constraint: for each triple t, sum_{x in t} y_x <= 1, y in [0,1].
    # This is a 3-uniform set packing LP; the fractional optimum >= integer optimum.
    def fractional_bound():
        y = [1.0] * (N + 1)  # 1-indexed
        # Simple multiplicative update (like a relaxation of the dual / subgradient)
        # We want a feasible y in [0,1] with sum y as large as possible subject to
        # for each triple: y_a+y_b+y_c <= 1.
        # Use iterative proportional fitting (Sinkhorn-like) to satisfy constraints.
        for it in range(200):
            changed = False
            for (a, b, c) in triples:
                s = y[a] + y[b] + y[c]
                if s > 1.0:
                    scale = 1.0 / s
                    y[a] *= scale
                    y[b] *= scale
                    y[c] *= scale
                    changed = True
            if not changed:
                break
        return sum(y[1:])

    ub_frac = fractional_bound()

    def local_add(S, counts, random_seed, max_rounds=5):
        """Try to add back any element that doesn't violate a triple."""
        random.seed(random_seed)
        S = set(S)
        counts = list(counts)
        removed = [x for x in range(1, N + 1) if x not in S]
        for _ in range(max_rounds):
            if time.time() > deadline - 2.0:
                break
            random.shuffle(removed)
            added_any = False
            for x in removed:
                if x in S:
                    continue
                ok = True
                for t_idx in elem_triples[x]:
                    if counts[t_idx] >= 2:
                        ok = False
                        break
                if ok:
                    S.add(x)
                    removed.remove(x)
                    for t_idx in elem_triples[x]:
                        counts[t_idx] += 1
                    added_any = True
            if not added_any:
                break
        return S, counts

    def two_opt(S, counts, random_seed, max_attempts=40):
        """Try removing one element and then greedily adding many removed elements."""
        random.seed(random_seed)
        S = set(S)
        counts = list(counts)
        removed = [x for x in range(1, N + 1) if x not in S]
        if not removed or not S:
            return S, counts
        best_S = set(S)
        best_counts = list(counts)
        best_size = len(S)
        for _ in range(max_attempts):
            if time.time() > deadline - 2.0:
                break
            x_remove = random.choice(list(S))
            S.discard(x_remove)
            for t_idx in elem_triples[x_remove]:
                counts[t_idx] -= 1
            random.shuffle(removed)
            for y in removed:
                if y in S:
                    continue
                ok = True
                for t_idx in elem_triples[y]:
                    if counts[t_idx] >= 2:
                        ok = False
                        break
                if ok:
                    S.add(y)
                    for t_idx in elem_triples[y]:
                        counts[t_idx] += 1
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
                best_counts = list(counts)
            else:
                S = set(best_S)
                counts = list(best_counts)
        removed = [x for x in range(1, N + 1) if x not in best_S]
        return best_S, best_counts

    def branch_and_bound(S, counts, remaining, depth):
        """Branch and bound. remaining = list of elements not decided yet (in some order).
        S = chosen, counts = current conflict counts.
        Returns best independent set (as set) found in this subtree."""
        nonlocal best_size, best_S, best_counts
        if time.time() > deadline - 2.0:
            return None
        # Upper bound: |S| + |remaining|
        if len(S) + len(remaining) <= best_size:
            return None
        # Also use fractional bound scaled: ub for remaining elements given constraints
        # (approx: each remaining element contributes at most 1, but triple constraints limit)
        # Simple: count how many remaining elements are "free" (no active conflict)
        # For a tighter bound, use the fractional LP on the subproblem, but that's expensive.
        # Use a cheap bound: for each remaining element, it can be added only if all its
        # triples have counts < 2 considering only S (not other remaining). Overestimate.
        # We'll rely on |S|+|remaining| and branch order.

        if not remaining:
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
                best_counts = list(counts)
            return None

        x = remaining[0]
        rest = remaining[1:]

        # Option 1: include x (if feasible)
        feasible = True
        for t_idx in elem_triples[x]:
            if counts[t_idx] >= 2:
                feasible = False
                break
        if feasible:
            counts2 = list(counts)
            for t_idx in elem_triples[x]:
                counts2[t_idx] += 1
            res = branch_and_bound(S | {x}, counts2, rest, depth + 1)
            if res is not None and time.time() > deadline - 2.0:
                return res

        # Option 2: exclude x
        res = branch_and_bound(S, counts, rest, depth + 1)
        if res is not None:
            return res
        return None

    # Use branch-and-bound on a reduced problem: fix the greedy seed, only branch on
    # a small set of "candidate" elements (those not in S that could potentially be added
    # if we remove some S elements). To keep it tractable, branch on a limited frontier.
    # We'll do a limited B&B: pick up to K elements from the removed set to branch on,
    # ordered by their conflict degree.
    removed = [x for x in range(1, N + 1) if x not in best_S]
    # Order removed by number of active conflicts (higher = more constrained, branch first)
    def conflict_degree(x):
        d = 0
        for t_idx in elem_triples[x]:
            if best_counts[t_idx] >= 2:
                d += 1
        return d
    removed.sort(key=lambda x: (-conflict_degree(x), x))

    # Limit B&B to a manageable subset
    K = 12
    branch_set = removed[:K]

    # We'll do a smarter B&B: branch on whether to include each element of branch_set,
    # and for the rest, greedily add if feasible.
    def bb_with_greedy(S, counts, branch_elems):
        nonlocal best_size, best_S, best_counts
        if time.time() > deadline - 2.0:
            return
        if not branch_elems:
            # Greedily add all remaining feasible elements
            S2 = set(S)
            counts2 = list(counts)
            for x in range(1, N + 1):
                if x in S2:
                    continue
                ok = True
                for t_idx in elem_triples[x]:
                    if counts2[t_idx] >= 2:
                        ok = False
                        break
                if ok:
                    S2.add(x)
                    for t_idx in elem_triples[x]:
                        counts2[t_idx] += 1
            if len(S2) > best_size:
                best_size = len(S2)
                best_S = set(S2)
                best_counts = list(counts2)
            return

        x = branch_elems[0]
        rest = branch_elems[1:]

        # Include x if feasible
        feasible = True
        for t_idx in elem_triples[x]:
            if counts[t_idx] >= 2:
                feasible = False
                break
        if feasible:
            counts2 = list(counts)
            for t_idx in elem_triples[x]:
                counts2[t_idx] += 1
            bb_with_greedy(S | {x}, counts2, rest)
            if time.time() > deadline - 2.0:
                return

        # Exclude x
        bb_with_greedy(S, counts, rest)

    # Run B&B on the branch set
    bb_with_greedy(best_S, best_counts, branch_set)

    # Now run local search and 2-opt on the best found
    for seed in range(30):
        if time.time() > deadline - 2.0:
            break
        S_l, c_l = local_add(best_S, best_counts, seed)
        if len(S_l) > best_size:
            best_size = len(S_l)
            best_S = set(S_l)
            best_counts = list(c_l)
        S_2, c_2 = two_opt(best_S, best_counts, seed + 100, max_attempts=25)
        if len(S_2) > best_size:
            best_size = len(S_2)
            best_S = set(S_2)
            best_counts = list(c_2)

    # Final verification: remove elements to break any remaining triples
    changed = True
    while changed:
        changed = False
        for (a, b, c) in triples:
            if a in best_S and b in best_S and c in best_S:
                best_S.discard(c)
                changed = True
                break

    if not best_S:
        best_S = set(range(90, 181))

    return list(best_S)

# EVOLVE-BLOCK-END
