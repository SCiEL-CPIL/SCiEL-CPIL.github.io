import time
import random
import sys
from collections import deque
from functools import lru_cache
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=3: algebraic voltage-lift of the Petersen graph with a
    bit-encoded backtracking solver (constraint propagation + MRV),
    then an incremental girth-aware 2-switch polish.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 130.0

    if k == 3:
        return _construct_k3(g, deadline, start_time)
    return _construct_luw(k, g)


def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _petersen_data():
    pet = [(0,1),(1,2),(2,3),(3,4),(4,0),
           (5,7),(7,9),(9,6),(6,8),(8,5),
           (0,5),(1,6),(2,7),(3,8),(4,9)]
    ei = {}
    for i,(u,v) in enumerate(pet):
        ei[(u,v)] = i
        ei[(v,u)] = i
    return pet, ei


def _build_walk_constraints(g):
    """Return list of (sign, frozenset of (edge_idx, dir) terms) for
    each non-backtracking closed walk of length in [4, g-1].
    A walk sums to 0 iff the voltage assignment gives 0 in Z_m."""
    pet, ei = _petersen_data()
    nb = [[[] for _ in range(10)] for __ in range(10)]
    for (u,v), i in ei.items():
        nb[u][v].append(i)
        nb[v][u].append(i)
    Lmin, Lmax = 4, g - 1
    constraints = []
    seen = set()
    for s in range(10):
        # state: (v, prev, path_set)
        for (a, b) in nb[s]:
            path = {(a, 0)}
            stack = [(b, a, 1, path)]
            while stack:
                v, pv, l, pathf = stack.pop()
                if l >= Lmax:
                    continue
                if v == s and Lmin <= l <= Lmax:
                    t = frozenset(pathf)
                    if t not in seen:
                        seen.add(t)
                        constraints.append(t)
                    continue
                for (x, d) in nb[v]:
                    if x == pv:
                        continue
                    key = (x, d)
                    if key in pathf:
                        continue
                    stack.append((x, v, l + 1, pathf | {key}))
    return constraints


def _solve_voltages(constraints, m, time_budget, seed):
    """Backtracking solver. Variables = 15 edges. Find voltages in [1,m-1]
    such that no constraint sum is 0 mod m. Returns list or None."""
    rng = random.Random(seed)
    # precompute term index and sign per constraint
    C = []
    for t in constraints:
        terms = []
        for (e, d) in t:
            terms.append((e, 1 if d == 0 else -1))
        C.append(terms)
    nvars = 15
    nC = len(C)
    # domain sizes
    domsize = max(1, m - 1)
    # order constraints by number of distinct variables (most constrained first)
    order = sorted(range(nC), key=lambda c: len({t[0] for t in C[c]}))
    # map variable -> list of constraint ids containing it (for propagation)
    var_to_c = [[] for _ in range(nvars)]
    for c in range(nC):
        vs = {t[0] for t in C[c]}
        for v in vs:
            var_to_c[v].append(c)
    # per-constraint per-variable sign (net) and remaining sum bookkeeping
    # For each constraint c: net_sum = sum of assigned terms; we need to ensure
    # net_sum + (remaining terms) != 0 is achievable.
    # Maintain: assigned contribution a[c] (int, can be large but we mod later).
    # For feasibility of unassigned vars in c: the set of achievable remaining
    # sums is union over each unassigned var of (its sign)*D plus 0 for none.
    # We'll do a light forward check: if c has exactly one unassigned var v with
    # net sign s, then forbidden value = -a[c] * s mod m.
    a = [0] * nC            # assigned contribution (raw int)
    assigned = [None] * nvars
    uncount = [len({t[0] for t in C[c]}) for c in range(nC)]
    # for each c, count how many unassigned vars remain
    start = time.time()
    # MRV: choose variable with fewest feasible values among unassigned
    def feasible_values_for(var, exclude):
        vals = set(range(1, m))
        if exclude is not None:
            vals.discard(exclude)
        if not vals:
            return None
        for c in var_to_c[var]:
            if uncount[c] <= 1:
                # this var is the last (or only) unassigned in c
                # forbidden = -a[c] * s  where s is net sign of var in c
                s = 0
                for (e, sg) in C[c]:
                    if e == var:
                        s += sg
                if s != 0:
                    forbid = (-a[c] * s) % m
                    if forbid in vals:
                        vals.discard(forbid)
                if not vals:
                    return None
        return vals

    # order variables dynamically via MRV; start with a heuristic fixed order
    base_order = list(range(nvars))

    def backtrack(depth):
        if time.time() - start > time_budget:
            return False
        if depth == nvars:
            return True
        # pick unassigned var with fewest feasible values
        best_var = None
        best_vals = None
        for var in base_order:
            if assigned[var] is not None:
                continue
            fv = feasible_values_for(var, None)
            if fv is None:
                return False
            if best_vals is None or len(fv) < len(best_vals):
                best_var = var
                best_vals = fv
                if len(best_vals) == 1:
                    break
        if best_var is None:
            return True
        for val in list(best_vals):
            assigned[best_var] = val
            ok = True
            changed = []
            for c in var_to_c[best_var]:
                s = 0
                for (e, sg) in C[c]:
                    if e == best_var:
                        s += sg
                a[c] += s * val
                uncount[c] -= 1
                changed.append(c)
                if uncount[c] == 0:
                    # all assigned, must not be 0 mod m
                    if a[c] % m == 0:
                        ok = False
                        break
            if not ok:
                # undo
                for c in changed:
                    s = 0
                    for (e, sg) in C[c]:
                        if e == best_var:
                            s += sg
                    a[c] -= s * val
                    uncount[c] += 1
                assigned[best_var] = None
                continue
            # forward check on constraints that now have <=1 unassigned
            feasible = True
            for c in changed:
                if uncount[c] <= 1:
                    # check the last unassigned var (if any) has a value
                    last = None
                    for (e, sg) in C[c]:
                        if assigned[e] is None:
                            last = e
                            break
                    if last is not None:
                        fv = feasible_values_for(last, None)
                        if fv is None:
                            feasible = False
                            break
            if feasible and backtrack(depth + 1):
                return True
            # undo
            for c in changed:
                s = 0
                for (e, sg) in C[c]:
                    if e == best_var:
                        s += sg
                a[c] -= s * val
                uncount[c] += 1
            assigned[best_var] = None
        return False

    if backtrack(0):
        return [assigned[i] for i in range(nvars)]
    return None


def _lift_from_voltages(pet, voltages, m):
    n = 10 * m
    ei = {}
    for i,(u,v) in enumerate(pet):
        ei[(u,v)] = i
        ei[(v,u)] = i
    es = set()
    for (u,v), i in ei.items():
        if u == v:
            continue
        a, b = pet[i]
        vol = voltages[i]
        sgn = 1 if (u == a and v == b) else -1
        for layer in range(m):
            uu = layer * 10 + u
            tl = (layer + sgn * vol) % m
            vv = tl * 10 + v
            es.add((min(uu,vv), max(uu,vv)))
    return n, list(es)


def _has_girth_at_least(n, edges, g):
    adj = [[] for _ in range(n)]
    for u,v in edges:
        adj[u].append(v)
        adj[v].append(u)
    max_len = g - 1
    for start in range(n):
        q = deque()
        for nb in adj[start]:
            q.append((nb, start, 1))
        while q:
            u, prev, l = q.popleft()
            if u == start and l > 0:
                return False
            if l >= max_len:
                continue
            for v in adj[u]:
                if v != prev:
                    q.append((v, u, l + 1))
    return True


def _construct_k3(g, deadline, start_time):
    pet, ei = _petersen_data()
    constraints = _build_walk_constraints(g)
    rng = random.Random(12345)
    best = None
    m = 2
    seed = 0
    while True:
        if time.time() > deadline - 25.0:
            break
        n_cand = 10 * m
        if best is not None and n_cand >= best[0]:
            break
        # try a few seeds
        sol = None
        for s in range(4):
            if time.time() > deadline - 25.0:
                break
            sol = _solve_voltages(constraints, m, time_budget=min(6.0, max(0.5, deadline - 25.0 - time.time())), seed=seed + s)
            if sol is not None:
                break
        if sol is not None:
            n, edges = _lift_from_voltages(pet, sol, m)
            if _has_girth_at_least(n, edges, g):
                best = (n, edges)
                break  # smallest m that works is best; stop
        # next candidate modulus
        seed += 1
        m += 1
        if m % 2 == 0 and m != 2:
            continue
        if not _is_prime(m):
            continue
        if m > 500:
            break
    if best is None:
        # fallback: LUW
        return _construct_luw(3, g)
    n, edges = best
    # Optional polish: try to reduce via 2-switches (incremental)
    remaining = deadline - time.time()
    if remaining > 20.0 and n > 60:
        n2, edges2 = _polish_2switch(n, edges, g, deadline, start_time)
        if edges2 is not None:
            return n2, edges2
    return n, edges


def _polish_2switch(n, edges, g, deadline, start_time):
    """Try 2-switches that eliminate the shortest cycle. Keep only if still
    girth >= g and n unchanged (2-switch preserves n). Goal: not to shrink n
    but to ensure validity; here it's a safety polish. Return (n, edges)."""
    adj = [set() for _ in range(n)]
    es = set()
    for u,v in edges:
        adj[u].add(v); adj[v].add(u)
        es.add((min(u,v), max(u,v)))
    best_edges = list(es)
    best_len = g  # we only accept if girth stays >= g
    start = time.time()
    attempts = 0
    while time.time() < deadline - 10.0 and attempts < 2000:
        attempts += 1
        u, v = random.choice(list(es))
        forbidden = adj[u] | adj[v]
        forbidden.add(u); forbidden.add(v)
        # sample x not forbidden
        x = random.randrange(n)
        ok = False
        for _ in range(8):
            x = random.randrange(n)
            if x not in forbidden:
                ok = True
                break
        if not ok:
            continue
        if not adj[x]:
            continue
        y = random.choice(list(adj[x]))
        if y in (u, v):
            continue
        # try (u,x),(v,y) or (u,y),(v,x)
        newes = None
        if (min(u,x),max(u,x)) not in es and (min(v,y),max(v,y)) not in es:
            newes = [(u,x),(v,y)]
        elif (min(u,y),max(u,y)) not in es and (min(v,x),max(v,x)) not in es:
            newes = [(u,y),(v,x)]
        else:
            continue
        # apply
        adj[u].remove(v); adj[v].remove(u); es.discard((min(u,v),max(u,v)))
        adj[x].remove(y); adj[y].remove(x); es.discard((min(x,y),max(x,y)))
        for a,b in newes:
            adj[a].add(b); adj[b].add(a); es.add((min(a,b),max(a,b)))
        # girth check
        if _has_girth_at_least(n, list(es), g):
            best_edges = list(es)
        else:
            # revert
            adj[u].add(v); adj[v].add(u); es.add((min(u,v),max(u,v)))
            adj[x].add(y); adj[y].add(x); es.add((min(x,y),max(x,y)))
            for a,b in newes:
                adj[a].remove(b); adj[b].remove(a); es.discard((min(a,b),max(a,b)))
    return n, best_edges


def _construct_luw(k, g, d=None, variant=0):
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    q = k
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x ^ y
        sub = lambda x,y: x ^ y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x + y) % q
        sub = lambda x,y: (x - y) % q
        mul = lambda x,y: x * y % q
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
