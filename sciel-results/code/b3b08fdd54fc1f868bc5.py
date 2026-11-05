# EVOLVE-BLOCK-START
import time
from random import Random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_DEADLINE = 140.0  # safety margin under the 150s kill

def _build_graph(n, p, seed):
    """Reproduce random_graph(n, p, seed) edge list exactly as the seed program."""
    rng = Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    return edges

def _cliques_at_least_2(n, edges):
    """Return all maximal cliques of size >= 2 via Bron-Kerbosch with pivot."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    cliques = []

    def bron_kerbosch(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(tuple(sorted(R)))
            return
        # choose pivot
        candidates = list(P | X)
        if candidates:
            max_deg = -1
            pivot = candidates[0]
            for v in candidates:
                d = len(adj[v] & P)
                if d > max_deg:
                    max_deg = d
                    pivot = v
            order = [v for v in P if pivot not in adj[v]]
        else:
            order = list(P)
        for v in order:
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    bron_kerbosch(set(), set(range(n)), set())
    return cliques

def _greedy_solve(n, cliques, vertex_hits, deadline):
    """Greedy set cover as fallback/warm-start."""
    uncovered = set(range(len(cliques)))
    chosen = set()
    while uncovered and time.time() < deadline:
        best_v = None
        best_count = -1
        for v, hits in vertex_hits.items():
            if v in chosen:
                continue
            c = len(hits & uncovered)
            if c > best_count:
                best_count = c
                best_v = v
        if best_v is None or best_count <= 0:
            break
        chosen.add(best_v)
        uncovered -= vertex_hits[best_v]
    return chosen

def _local_search_refine(S, cliques, vertex_hits, n, deadline, max_iters=500):
    """Local search: try removing vertices, swapping, and 2-for-1 replacements."""
    all_cliques = set(range(len(cliques)))
    
    def coverage_of(S):
        cov = set()
        for v in S:
            cov |= vertex_hits[v]
        return cov
    
    # Pass 1: remove redundant vertices
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for v in list(S):
            if time.time() > deadline:
                break
            S.discard(v)
            cov = coverage_of(S)
            if cov == all_cliques:
                changed = True
            else:
                S.add(v)
    
    # Pass 2: try single-vertex swaps (remove v, add w if it covers all lost)
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        cov = coverage_of(S)
        for v in list(S):
            S.discard(v)
            cov2 = coverage_of(S)
            lost = cov - cov2
            if not lost:
                improved = True
                continue
            # try to find a vertex w not in S that covers all of lost
            found = False
            for w, hits in vertex_hits.items():
                if w in S:
                    continue
                if lost <= hits:
                    S.add(w)
                    improved = True
                    found = True
                    break
            if not found:
                S.add(v)
        if not improved:
            break
    
    # Pass 3: try 2-for-1 replacements (remove v, add two vertices w1,w2 if together they cover lost)
    for _ in range(max_iters // 2):
        if time.time() > deadline:
            break
        improved = False
        cov = coverage_of(S)
        for v in list(S):
            if time.time() > deadline:
                break
            S.discard(v)
            cov2 = coverage_of(S)
            lost = cov - cov2
            if not lost:
                improved = True
                continue
            # try pairs of vertices not in S
            candidates = [w for w in vertex_hits if w not in S]
            found = False
            for i, w1 in enumerate(candidates):
                if time.time() > deadline:
                    break
                h1 = vertex_hits[w1]
                if not (lost & h1):
                    continue  # w1 doesn't help with lost
                remaining = lost - h1
                if not remaining:
                    S.add(w1)
                    improved = True
                    found = True
                    break
                for w2 in candidates[i+1:]:
                    h2 = vertex_hits[w2]
                    if remaining <= (h1 | h2):
                        S.add(w1)
                        S.add(w2)
                        improved = True
                        found = True
                        break
                if found:
                    break
            if not found:
                S.add(v)
        if not improved:
            break
    
    return S

def _ilp_solve(n, cliques, vertex_hits, deadline):
    """Solve minimum hitting set as ILP using scipy.optimize.milp."""
    m = len(cliques)
    if m == 0:
        return set()
    
    # Only consider vertices that appear in at least one clique
    relevant_vertices = sorted(vertex_hits.keys())
    v_to_idx = {v: i for i, v in enumerate(relevant_vertices)}
    k = len(relevant_vertices)
    
    if k == 0:
        return set()
    
    # Objective: minimize sum of x_v
    c = np.ones(k)
    
    # Constraints: for each clique, sum of x_v for v in clique >= 1
    # Build constraint matrix A such that A @ x >= 1 for all cliques
    A = np.zeros((m, k))
    for idx, cl in enumerate(cliques):
        for v in cl:
            if v in v_to_idx:
                A[idx, v_to_idx[v]] = 1
    
    # Lower bound: 1 for each clique constraint, upper bound: inf
    lb = np.ones(m)
    ub = np.full(m, np.inf)
    
    # Variable bounds: 0 <= x_v <= 1, integer
    var_bounds = Bounds(lb=np.zeros(k), ub=np.ones(k))
    integrality = np.ones(k)  # all variables are binary
    
    constraint = LinearConstraint(A, lb=lb, ub=ub)
    
    try:
        result = milp(c=c, constraints=constraint, integrality=integrality, bounds=var_bounds)
        if result.status == 0 and result.x is not None:
            selected = set()
            for i, v in enumerate(relevant_vertices):
                if result.x[i] > 0.5:
                    selected.add(v)
            # Verify coverage
            cov = set()
            for v in selected:
                cov |= vertex_hits[v]
            if cov == set(range(m)):
                return selected
    except Exception:
        pass
    
    return None  # ILP failed

def _solve_instance(n, edges, deadline):
    cliques = _cliques_at_least_2(n, edges)
    if not cliques:
        return []

    # Map each vertex to the set of clique-indices it hits.
    vertex_hits = {}
    for idx, cl in enumerate(cliques):
        for v in cl:
            vertex_hits.setdefault(v, set()).add(idx)

    # Try ILP first
    ilp_solution = _ilp_solve(n, cliques, vertex_hits, deadline)
    if ilp_solution is not None:
        # Refine with local search
        S = set(ilp_solution)
        S = _local_search_refine(S, cliques, vertex_hits, n, deadline)
        return sorted(S)
    
    # Fallback to greedy + local search
    S = _greedy_solve(n, cliques, vertex_hits, deadline)
    S = _local_search_refine(S, cliques, vertex_hits, n, deadline)
    return sorted(S)

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            # not enough time; fall back to a valid (large) solution
            edges = _build_graph(n, p, seed)
            out.append(sorted({x for e in edges for x in e}))
            continue
        edges = _build_graph(n, p, seed)
        out.append(_solve_instance(n, edges, deadline))
    return out
# EVOLVE-BLOCK-END
