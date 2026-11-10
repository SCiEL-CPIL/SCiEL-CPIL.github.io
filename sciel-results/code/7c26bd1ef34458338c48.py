"""Construct 20 points with no four collinear, minimizing the largest no-three-collinear subset."""
import time
import math
import random

DEADLINE = 140.0  # safety margin before the 150s kill


def _slope_key(p, q):
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    g = math.gcd(abs(dx), abs(dy))
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return (dx, dy)


def _build_triples(points):
    """Return list of frozenset of 3 collinear point indices (no 4 collinear assumed)."""
    n = len(points)
    triples = []
    seen = set()
    for i in range(n):
        by_slope = {}
        for j in range(n):
            if j == i:
                continue
            k = _slope_key(points[i], points[j])
            by_slope.setdefault(k, []).append(j)
        for k, lst in by_slope.items():
            L = len(lst)
            if L >= 3:
                for a in range(L):
                    for b in range(a + 1, L):
                        t = frozenset((i, lst[a], lst[b]))
                        if t not in seen:
                            seen.add(t)
                            triples.append(t)
    return triples


def _no_four(points):
    n = len(points)
    for i in range(n):
        by_slope = {}
        for j in range(n):
            if j == i:
                continue
            k = _slope_key(points[i], points[j])
            by_slope[k] = by_slope.get(k, 0) + 1
        for c in by_slope.values():
            if c >= 3:
                return False
    return True


def _independence_bound(triples, rem):
    """Upper bound on independence number of rem: each triple forces <=2, so <= 2*|rem|/3 rounded up-ish.
    Use a simple LP-free bound: for each triple, at most 2 chosen. Greedy set-cover style bound."""
    # Simple bound: |rem| is trivial. Stronger: if a triple is fully in rem, at most 2 of its 3.
    # Use fractional: bound = min over ... keep simple: 2 * ceil(|rem| / 3) is NOT valid generally.
    # Valid simple bound: for each maximal clique of the 3-uniform hypergraph... use greedy.
    # We'll use: bound = |rem| and improve by counting disjoint triples.
    rem_set = set(rem)
    used = set()
    disjoint = 0
    for t in triples:
        if t <= rem_set and t.isdisjoint(used):
            used |= t
            disjoint += 1
    # Each disjoint triple contributes at most 2, rest at most 1
    bound = 2 * disjoint + (len(rem) - 3 * disjoint)
    return max(0, bound)


def _max_independent(triples, n, time_limit):
    """Exact branch and bound for max subset with no triple fully included."""
    # Precompute for each point the triples it belongs to
    point_triples = [[] for _ in range(n)]
    for t in triples:
        for p in t:
            point_triples[p].append(t)

    best = [0]
    start = time.time()

    def rec(rem, chosen, current):
        if time.time() - start > time_limit:
            return
        if not rem:
            if current > best[0]:
                best[0] = current
            return
        # Upper bound
        ub = _independence_bound(triples, rem) + current
        if ub <= best[0]:
            return
        # Pick a point with most remaining triples (branch on it)
        rem_list = list(rem)
        # order by descending number of triples still active
        def score(p):
            c = 0
            for t in point_triples[p]:
                if t <= rem:
                    c += 1
            return c
        p = max(rem_list, key=score)
        # Option 1: exclude p
        rec(rem - {p}, chosen, current)
        # Option 2: include p, but then any triple containing p must lose another member
        # new rem: remove p, and for each triple containing p, we must exclude one of the other two
        # This is complex; instead: include p means we cannot include both other members of any triple with p.
        # Simpler: include p, remove p from rem, and record that p is chosen.
        # We need to ensure no triple fully in chosen. Since we only add p, check triples with p:
        # if a triple {p,a,b} has both a,b in chosen -> invalid. We handle by constraint.
        # To keep exact and simple, do: include p, then for each triple containing p,
        # at least one of the other two must be excluded. We branch on that.
        new_rem = rem - {p}
        # Gather constraints: for each triple with p, the other two
        constraints = []
        for t in point_triples[p]:
            others = [x for x in t if x != p]
            if len(others) == 2 and others[0] in new_rem and others[1] in new_rem:
                constraints.append((others[0], others[1]))
        if not constraints:
            rec(new_rem, chosen | {p}, current + 1)
        else:
            # For each constraint (a,b), exclude a or exclude b. Branch.
            # To avoid explosion, pick the first constraint and branch, threading the rest.
            def rec_constrained(rem_c, chosen_c, cur_c, cons_idx):
                if time.time() - start > time_limit:
                    return
                if cons_idx == len(constraints):
                    rec(rem_c, chosen_c, cur_c)
                    return
                a, b = constraints[cons_idx]
                if a not in rem_c or b not in rem_c:
                    rec_constrained(rem_c, chosen_c, cur_c, cons_idx + 1)
                    return
                # exclude a
                rec_constrained(rem_c - {a}, chosen_c, cur_c, cons_idx + 1)
                # exclude b
                rec_constrained(rem_c - {b}, chosen_c, cur_c, cons_idx + 1)
            rec_constrained(new_rem, chosen | {p}, current + 1, 0)

    rec(frozenset(range(n)), frozenset(), 0)
    return best[0]


def _try_grid():
    # 4 x 5 grid, no four collinear, many triples
    pts = []
    for i in range(4):
        for j in range(5):
            pts.append([i * 2, j * 2])
    return pts


def construct():
    start = time.time()
    deadline = start + DEADLINE

    points = _try_grid()
    if not _no_four(points):
        # fallback to parabola
        return [[x, x * x] for x in range(-10, 10)]

    triples = _build_triples(points)
    n = len(points)

    # Compute independence with a time budget
    budget = max(1.0, deadline - time.time() - 2.0)
    score = _max_independent(triples, n, budget)

    return points
