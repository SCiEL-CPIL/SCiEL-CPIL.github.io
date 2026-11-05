# EVOLVE-BLOCK-START
import time
from random import Random

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

def _solve_instance(n, edges, deadline):
    cliques = _cliques_at_least_2(n, edges)
    if not cliques:
        return []

    # Map each vertex to the set of clique-indices it hits.
    # Only keep vertices that hit at least one clique.
    vertex_hits = {}
    for idx, cl in enumerate(cliques):
        for v in cl:
            vertex_hits.setdefault(v, set()).add(idx)

    # Greedy set cover: repeatedly pick the vertex hitting the most uncovered cliques.
    uncovered = set(range(len(cliques)))
    chosen = []
    chosen_set = set()
    while uncovered:
        if time.time() > deadline:
            break
        best_v = None
        best_count = -1
        for v, hits in vertex_hits.items():
            if v in chosen_set:
                continue
            c = len(hits & uncovered)
            if c > best_count:
                best_count = c
                best_v = v
        if best_v is None or best_count <= 0:
            break
        chosen.append(best_v)
        chosen_set.add(best_v)
        uncovered -= vertex_hits[best_v]

    # Local search pass 1: drop any vertex that is redundant.
    def coverage_of(S):
        cov = set()
        for v in S:
            cov |= vertex_hits[v]
        return cov

    S = set(chosen)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for v in list(S):
            if time.time() > deadline:
                break
            S.discard(v)
            cov = coverage_of(S)
            if cov == set(range(len(cliques))):
                changed = True
            else:
                S.add(v)

    # Local search pass 2: try to replace a vertex with a cheaper combination
    # (single-vertex swap: remove v, add w only if it keeps coverage and reduces size is impossible
    #  with one-for-one; instead try remove v and add a set of w that together cover v's loss).
    # Simpler: try removing v and, if coverage lost, see if a single other vertex covers the loss.
    for _ in range(200):
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
