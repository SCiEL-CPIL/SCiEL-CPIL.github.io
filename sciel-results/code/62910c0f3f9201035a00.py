"""Rewritten program: maximize edge-disjoint Hamiltonian cycles on the 21-vertex host."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _find_hamiltonian_cycle(n, edges, adj):
    """Return a Hamiltonian cycle (list of vertices) using only given edges, or None."""
    if not edges:
        return None
    m = len(edges)
    # edge index map
    eidx = {}
    for i, (a, b) in enumerate(edges):
        eidx[(a, b)] = i
        eidx[(b, a)] = i
    # variables: x_e in {0,1} for each edge; plus flow vars y_{u,v} for directed arcs (u->v) if edge exists
    # We'll use MTZ subtour elimination: order vars w_v in [1, n], w_root = 1.
    # Arcs: for each undirected edge {u,v}, two directed arcs u->v and v->u.
    arc_list = []
    for (a, b) in edges:
        arc_list.append((a, b))
        arc_list.append((b, a))
    A = len(arc_list)
    # total vars: m (x) + A (y) + n (w)
    nvar = m + A + n
    # objective: maximize sum x_e  -> minimize -sum x_e
    c = np.zeros(nvar)
    for i in range(m):
        c[i] = -1.0
    # integrality: x and y binary, w integer
    integrality = np.zeros(nvar)
    integrality[:m] = 1
    integrality[m:m+A] = 1
    integrality[m+A:m+A+n] = 1
    lb = np.zeros(nvar)
    ub = np.ones(nvar)
    ub[m+A:m+A+n] = n  # w in [1,n]
    # root = 0, w[0] = 1
    # Build constraints
    cons = []
    # degree: for each v, sum of x over incident edges = 2
    for v in range(n):
        row = np.zeros(nvar)
        for i, (a, b) in enumerate(edges):
            if a == v or b == v:
                row[i] = 1.0
        cons.append(LinearConstraint(row, 2, 2))
    # flow conservation: for v != root, sum_in y - sum_out y = 0
    arc_idx = {}
    for k, (u, v) in enumerate(arc_list):
        arc_idx[(u, v)] = m + k
    for v in range(1, n):
        row = np.zeros(nvar)
        for k, (u, w) in enumerate(arc_list):
            if w == v:
                row[m+k] = 1.0
            if u == v:
                row[m+k] = -1.0
        cons.append(LinearConstraint(row, 0, 0))
    # link y to x: y_{u,v} + y_{v,u} <= x_{uv}  (for each undirected edge)
    for i, (a, b) in enumerate(edges):
        row = np.zeros(nvar)
        row[i] = 1.0
        row[arc_idx[(a, b)]] = 1.0
        row[arc_idx[(b, a)]] = 1.0
        cons.append(LinearConstraint(row, 0, 1))
    # MTZ: for each arc (u,v), w_u + 1 - w_v <= (n-1)*(1 - y_{u,v})
    # i.e., w_u - w_v + 1 <= (n-1) - (n-1)*y
    for k, (u, v) in enumerate(arc_list):
        row = np.zeros(nvar)
        row[m+A+u] = 1.0
        row[m+A+v] = -1.0
        row[m+k] = -(n-1)
        cons.append(LinearConstraint(row, -np.inf, n-2))
    # root fixed
    row = np.zeros(nvar)
    row[m+A+0] = 1.0
    cons.append(LinearConstraint(row, 1, 1))
    # also require root degree 2 already covered; ensure cycle includes root via flow (root has no conservation, that's fine for a cycle)
    res = milp(c=c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub))
    if not res.success:
        return None
    x = res.x[:m]
    chosen = [i for i in range(m) if x[i] > 0.5]
    if len(chosen) != n:
        return None
    # reconstruct cycle from chosen edges
    deg = [0]*n
    adjc = [set() for _ in range(n)]
    for i in chosen:
        a, b = edges[i]
        adjc[a].add(b)
        adjc[b].add(a)
        deg[a] += 1
        deg[b] += 1
    if any(d != 2 for d in deg):
        return None
    # walk
    start = 0
    cyc = [start]
    prev = -1
    cur = start
    for _ in range(n):
        nxts = [u for u in adjc[cur] if u != prev]
        if not nxts:
            return None
        prev, cur = cur, nxts[0]
        if cur == start:
            break
        cyc.append(cur)
    if len(cyc) != n:
        return None
    if cyc[-1] != start:
        cyc.append(start)
        cyc.pop()  # keep distinct list without closing dup
    # verify all consecutive + closing are edges
    for i in range(len(cyc)):
        a, b = cyc[i], cyc[(i+1) % len(cyc)]
        if (a, b) not in eidx and (b, a) not in eidx:
            return None
    return cyc

def construct():
    n = P['n']
    edges = [tuple(e) for e in P['edges']]
    adj = _build_adj(n, edges)
    used = set()
    cycles = []
    for _ in range(10):
        remaining = [e for e in edges if frozenset(e) not in used]
        if len(remaining) < n:
            break
        cyc = _find_hamiltonian_cycle(n, remaining, adj)
        if cyc is None:
            break
        cycles.append(list(cyc))
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i+1) % len(cyc)]
            used.add(frozenset((a, b)))
    if not cycles:
        # fallback: the known Hamiltonian cycle
        cycles = [list(range(n))]
    # ensure types
    out = [tuple(int(v) for v in c) for c in cycles]
    return out
# EVOLVE-BLOCK-END
