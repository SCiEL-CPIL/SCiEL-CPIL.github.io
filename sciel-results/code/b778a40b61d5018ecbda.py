import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _check_valid(coloring):
    """Exact validity: adjacent differ, and every two-color induced subgraph is a forest."""
    n = P['n']
    edges = P['edges']
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    colors_used = sorted(set(coloring))
    for i in range(len(colors_used)):
        c1 = colors_used[i]
        for j in range(i + 1, len(colors_used)):
            c2 = colors_used[j]
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[rx] = ry
                return False

            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if union(u, v):
                        return False
    return True


def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _structural_coloring(adj, n):
    """Deterministic fallback: greedy degeneracy ordering with a fixed color order.
    Produces a valid bicolored-forest coloring (each color class is a path)."""
    deg = [len(adj[i]) for i in range(n)]
    remaining = set(range(n))
    order = []
    degc = deg[:]
    for _ in range(n):
        mv = min(remaining, key=lambda v: (degc[v], v))
        order.append(mv)
        remaining.discard(mv)
        for nb in adj[mv]:
            if nb in remaining:
                degc[nb] -= 1
    coloring = [-1] * n
    for v in order:
        used = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    return coloring


def _sat_coloring(k, adj, n, deadline):
    """CDCL SAT search for a k-coloring whose every two-color induced subgraph is a forest.
    Uses edge-elimination: a color-pair subgraph is a forest iff we can drop edges to hit
    every simple cycle. Each edge has an 'active' literal; a cycle is forbidden by a clause
    that requires at least one of its edges to be dropped (cycle-hitting set)."""
    try:
        from pysat.solvers import Glucose42
    except Exception:
        return None

    edges = P['edges']
    m = len(edges)
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i

    # literal mapping
    # color var: (v, c) -> 1-based int
    def col_lit(v, c):
        return (v * k + c) * 2 + 1
    def act_lit(i):
        return (n * k + i) * 2 + 1

    clauses = []
    # each vertex gets exactly one color (at least one)
    for v in range(n):
        clauses.append([col_lit(v, c) for c in range(k)])
    # each vertex at most one color
    for v in range(n):
        for a in range(k):
            for b in range(a + 1, k):
                clauses.append([-col_lit(v, a), -col_lit(v, b)])
    # adjacent vertices differ
    for u, v in edges:
        for c in range(k):
            clauses.append([-col_lit(u, c), -col_lit(v, c)])

    # cycle-hitting clauses per color pair
    # For a fixed pair (c1,c2), an edge is "active in pair" iff both endpoints carry that
    # pair's colors. We build, per pair, a SAT subproblem over the pair's edges whose
    # 'active' status is implied by endpoint colors. To keep it as hard unit/clauses:
    # For each simple cycle in the *full* graph, add a clause: not(all edges of cycle are
    # active in pair). An edge e=(u,v) is active in pair (c1,c2) iff
    #   (col(u) in {c1,c2}) AND (col(v) in {c1,c2}).
    # We encode "edge e active in pair" as a fresh literal A_e^{pair} with:
    #   A_e^{pair} -> col(u) in pair  and  A_e^{pair} -> col(v) in pair
    #   (col(u) in pair) and (col(v) in pair) -> A_e^{pair}   (to make cycle clauses tight)
    # Simpler & sound: only need soundness (no spurious solutions). So we can use the
    # implication (col(u) in pair) AND (col(v) in pair) -> A_e^{pair} is NOT needed for
    # soundness; we need A_e^{pair} -> endpoints-in-pair so that a cycle clause
    # "not all active" is satisfied whenever the coloring has a cycle (because if the
    # coloring makes all endpoints in-pair, then to satisfy A->endpoints, if we set A true
    # we must have endpoints in pair; the cycle clause forces some A false; but a false A
    # is fine). Actually we want: if the coloring induces a cycle in the pair, then the
    # coloring is INVALID. We want the SAT to reject it. So we need: (coloring induces
    # cycle) => UNSAT. Equivalently add clauses so that any assignment where the pair
    # subgraph has a cycle is blocked.
    # Cleanest exact encoding: introduce A_e^{pair} with A_e^{pair} <-> (u in pair) AND (v in pair).
    # Two-way:
    #   A -> u_in  : [-A, u_in]
    #   A -> v_in  : [-A, v_in]
    #   u_in AND v_in -> A : [-u_in, -v_in, A]
    # where u_in = (col(u)==c1 or col(u)==c2)  -- but that's a disjunction, not a single lit.
    # Replace u_in by two literals: we need A -> (u==c1 or u==c2): [-A, col(u,c1), col(u,c2)].
    # And (u==c1 or u==c2) AND (v==c1 or v==c2) -> A:
    #   For each cu in {c1,c2}, cv in {c1,c2}: [-col(u,cu), -col(v,cv), A].
    # Then cycle clause for cycle edges E_cyc: not(all A active) = [ -A_e for e in E_cyc ]? No:
    # "not all active" = at least one inactive = [ -A_{e1}, -A_{e2}, ... ]? That clause is
    # satisfied unless ALL A_e are true. We want it UNSAT when all edges of a cycle are
    # active, i.e. when all A_e true. Clause [ -A_e for e in cycle ] is violated exactly
    # when every A_e is true. 
    # So: for each cycle, clause = [ -A_e for e in cycle ]. 

    # Enumerate simple cycles up to a length cap (girth-based). Use a bounded DFS cycle finder.
    # To bound work, find cycles via a simple approach: for each edge, find paths between its
    # endpoints of length up to L-1 not using that edge.
    L = 8  # max cycle length to encode
    cycles = set()
    # adjacency with edge ids
    adj_e = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj_e[u].append((v, i))
        adj_e[v].append((u, i))

    def find_cycles_from_edge(eid):
        u, v = edges[eid]
        # BFS/DFS from u to v avoiding edge eid, up to length L-1
        found = []
        # state: (node, path_edges)
        stack = [(u, [])]
        while stack:
            node, path = stack.pop()
            if len(path) >= L - 1:
                continue
            for nb, ei in adj_e[node]:
                if ei == eid:
                    continue
                if nb == v:
                    cyc = tuple(sorted(path + [ei]))
                    found.append(cyc)
                    continue
                if nb in path:
                    continue
                stack.append((nb, path + [ei]))
        return found

    # Only enumerate cycles on a subset to keep it tractable; cap total cycles.
    max_cycles = 400
    seen_cycles = set()
    for eid in range(m):
        if len(seen_cycles) >= max_cycles:
            break
        for cyc in find_cycles_from_edge(eid):
            if cyc not in seen_cycles:
                seen_cycles.add(cyc)

    cycles = list(seen_cycles)

    # Build A literals: A[(c1,c2,eid)]
    A_lit = {}
    next_lit = 2 * (n * k + m) + 1
    for (c1, c2) in [(a, b) for a in range(k) for b in range(a + 1, k)]:
        for eid in range(m):
            A_lit[(c1, c2, eid)] = next_lit
            next_lit += 1

    for (c1, c2) in [(a, b) for a in range(k) for b in range(a + 1, k)]:
        for eid in range(m):
            u, v = edges[eid]
            A = A_lit[(c1, c2, eid)]
            # A -> (u==c1 or u==c2)
            clauses.append([-A, col_lit(u, c1), col_lit(u, c2)])
            # A -> (v==c1 or v==c2)
            clauses.append([-A, col_lit(v, c1), col_lit(v, c2)])
            # (u in pair) AND (v in pair) -> A
            for cu in (c1, c2):
                for cv in (c1, c2):
                    clauses.append([-col_lit(u, cu), -col_lit(v, cv), A])
        # cycle clauses
        for cyc in cycles:
            clause = [-A_lit[(c1, c2, eid)] for eid in cyc]
            clauses.append(clause)

    try:
        s = Glucose42(bootstrap_with=clauses)
    except Exception:
        return None

    result = s.solve()
    if result:
        model = s.get_model()
        coloring = [-1] * n
        for v in range(n):
            for c in range(k):
                if model[col_lit(v, c) - 1] == 1:
                    coloring[v] = c
                    break
        if all(coloring[v] != -1 for v in range(n)):
            if _check_valid(coloring):
                return coloring
    s.delete()
    return None


def construct():
    deadline = time.time() + 85
    n = P['n']
    adj = _build_adj()

    # Get a valid fallback immediately.
    fallback = _structural_coloring(adj, n)
    if not _check_valid(fallback):
        fallback = [int(i) for i in range(n)]

    best = fallback
    best_k = len(set(best))

    # Try to beat the fallback with a SAT solver for small k.
    for k in range(2, min(best_k, 9) + 1):
        if time.time() > deadline - 5:
            break
        coloring = _sat_coloring(k, adj, n, deadline)
        if coloring is not None:
            if _check_valid(coloring):
                kk = len(set(coloring))
                if kk < best_k:
                    best = coloring
                    best_k = kk
                if kk <= 2:
                    break

    return [int(c) for c in best]


if __name__ == '__main__':
    result = construct()
    print(result)
    print(len(set(result)))
