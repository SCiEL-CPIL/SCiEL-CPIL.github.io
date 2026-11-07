"""Rewritten: hypergraph-independence via Lagrangian relaxation + subgradient greedy + polish."""
import time
import random
import numpy as np

N = 128
DEADLINE_MARGIN = 5.0  # seconds to leave before 60s limit

def _build_edges():
    """Build the set of forbidden 4-sets: {a,b,c,s1,s2,s3} where s_i are pair sums, all in 1..N."""
    edges = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            for c in range(b+1, N+1):
                s1 = a+b
                s2 = a+c
                s3 = b+c
                if s1 > N or s2 > N or s3 > N:
                    continue
                # need all four to be distinct? The task says "three distinct members a,b,c"
                # and "a+b, a+c, b+c are also selected". The sums need not be distinct from each
                # other or from a,b,c. But a,b,c are distinct. For the constraint to bind,
                # all six values a,b,c,s1,s2,s3 must be in the set. If some sum equals one of
                # a,b,c, it's still "also selected" trivially. So the forbidden config is:
                # a,b,c in S AND s1 in S AND s2 in S AND s3 in S.
                # The "edge" is the set of vertices that must NOT all be present.
                edge = frozenset([a,b,c,s1,s2,s3])
                if len(edge) < 4:
                    # Degenerate: some sums coincide with a,b,c. Still forbidden if all present.
                    pass
                edges.append(edge)
    return edges

def _edge_contains_all(edge, selected):
    """Check if all vertices in edge are in selected (a set of ints)."""
    for v in edge:
        if v not in selected:
            return False
    return True

def _count_violations(selected, edges):
    cnt = 0
    sel = selected
    for e in edges:
        ok = True
        for v in e:
            if v not in sel:
                ok = False
                break
        if ok:
            cnt += 1
    return cnt

def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN

    # Build edges once
    edges = _build_edges()
    # Map vertex -> list of edge indices containing it
    vert_to_edges = {v: [] for v in range(1, N+1)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)

    # Lagrangian relaxation: maximize |S| subject to for each edge e, not(all v in e in S)
    # Equivalently: for each edge e, sum_{v in e} x_v <= |e| - 1
    # Lagrangian: L(x, lambda) = sum x_v - sum_e lambda_e * max(0, sum_{v in e} x_v - (|e|-1))
    # Subgradient method on lambda, greedy primal on x given lambda.

    random.seed(42)
    best_set = set(range(N//2+1, N+1))  # seed solution
    best_size = len(best_set)

    # Greedy primal: given vertex weights w, greedily add vertices in order of w,
    # skipping if it would create a violation.
    def _greedy_primal(weights, max_time=None):
        """weights: dict or array indexed 1..N. Returns a set."""
        sel = set()
        # Track for each edge how many of its vertices are selected
        edge_count = np.zeros(len(edges), dtype=np.int8)
        # Process vertices in descending weight order
        order = sorted(range(1, N+1), key=lambda v: -weights[v])
        for v in order:
            if max_time is not None and time.time() > max_time:
                break
            # Check if adding v creates any violation
            # v is in edges vert_to_edges[v]
            conflict = False
            for ei in vert_to_edges[v]:
                if edge_count[ei] == len(edges[ei]) - 1:
                    # Adding v would make all vertices of this edge selected
                    conflict = True
                    break
            if not conflict:
                sel.add(v)
                for ei in vert_to_edges[v]:
                    edge_count[ei] += 1
        return sel

    # Subgradient iteration
    n_edges = len(edges)
    lam = np.zeros(n_edges)
    step_sizes = [10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1]
    
    for it in range(200):
        if time.time() > deadline:
            break
        # Compute vertex weights: w_v = 1 - sum_{e: v in e} lambda_e
        weights = np.ones(N+1)
        for ei in range(n_edges):
            if lam[ei] > 0:
                for v in edges[ei]:
                    weights[v] -= lam[ei]
        # Greedy primal
        sel = _greedy_primal(weights, max_time=deadline)
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel.copy()
        
        # Subgradient update: gradient of L w.r.t. lam_e = max(0, sum_{v in e} x_v - (|e|-1))
        x = np.zeros(N+1)
        for v in sel:
            x[v] = 1.0
        grad = np.zeros(n_edges)
        for ei, e in enumerate(edges):
            s = sum(x[v] for v in e)
            cap = len(e) - 1
            if s > cap:
                grad[ei] = s - cap
        # Step size
        ss = step_sizes[min(it // 30, len(step_sizes)-1)]
        lam = np.maximum(0, lam + ss * grad)

    # Final polish: try to add any missing vertex that doesn't create a violation
    if time.time() < deadline:
        sel = set(best_set)
        edge_count = np.zeros(n_edges, dtype=np.int8)
        for ei, e in enumerate(edges):
            for v in e:
                if v in sel:
                    edge_count[ei] += 1
        # Try adding each non-selected vertex
        for v in range(1, N+1):
            if v in sel:
                continue
            if time.time() > deadline:
                break
            conflict = False
            for ei in vert_to_edges[v]:
                if edge_count[ei] == len(edges[ei]) - 1:
                    conflict = True
                    break
            if not conflict:
                sel.add(v)
                for ei in vert_to_edges[v]:
                    edge_count[ei] += 1
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel.copy()

    # Convert to list of Python ints
    result = [int(v) for v in sorted(best_set)]
    return result
