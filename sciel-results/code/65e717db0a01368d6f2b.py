"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 140.0  # safety margin before 150s kill
_START = None

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _deadline_hit():
    return (_now() - _START) > _DEADLINE

def instance_graph(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _edge_set(edges):
    return {tuple(sorted(e)) for e in edges}


def _adjacency(edge_set):
    adj = {}
    for (u, v) in edge_set:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _find_long_cycle(adj, n, max_iter=200):
    """Find a reasonably long cycle via randomized DFS/walk."""
    best_cycle = None
    best_len = 0
    nodes = list(adj.keys())
    rng = random.Random()
    for _ in range(max_iter):
        start = rng.choice(nodes)
        # DFS to find a cycle back to start
        visited = {start}
        stack = [(start, [start])]
        found = None
        while stack:
            node, path = stack.pop()
            for nb in list(adj.get(node, [])):
                if nb == start and len(path) >= 3:
                    found = path
                    break
                if nb not in visited:
                    visited.add(nb)
                    stack.append((nb, path + [nb]))
            if found:
                break
        if found and len(found) > best_len:
            best_len = len(found)
            best_cycle = found
        if best_len >= n:
            break
    return best_cycle


def _cycle_edges(cycle):
    """Return set of edges for a cycle (list of vertices, closing edge implied)."""
    edges = set()
    k = len(cycle)
    for i in range(k):
        u, v = cycle[i], cycle[(i + 1) % k]
        edges.add(tuple(sorted((u, v))))
    return edges


def _pack_cycles(edge_set, n):
    """Greedy: repeatedly find a long cycle, remove its edges."""
    pieces = []
    remaining = set(edge_set)
    while remaining:
        if _deadline_hit():
            break
        adj = _adjacency(remaining)
        cycle = _find_long_cycle(adj, n)
        if cycle is None:
            break
        ce = _cycle_edges(cycle)
        if not ce.issubset(remaining):
            # safety
            break
        pieces.append(list(cycle))
        remaining -= ce
    return pieces, remaining


def _local_search(pieces, remaining_edges, n, max_rounds=30):
    """Try to improve: insert single edges into cycles, or form new cycles from singles."""
    remaining = list(remaining_edges)
    cycles = [list(p) for p in pieces]
    for _round in range(max_rounds):
        if _deadline_hit():
            break
        improved = False
        # Build adjacency from remaining single edges
        rem_set = {tuple(sorted(e)) for e in remaining}
        # Try to extend a cycle by inserting a single edge
        for i, cyc in enumerate(cycles):
            if _deadline_hit():
                break
            # For each edge (a,b) in cycle, try to replace with a path through a single edge
            # Simpler: try to find a single edge (u,v) such that u and v are in cycle and
            # inserting creates a longer valid cycle.
            cyc_set = set(cyc)
            for (u, v) in rem_set:
                if u in cyc_set and v in cyc_set:
                    # Check: can we form a longer cycle by going u->v (single) instead of along cycle?
                    # Actually we need edge-disjointness: the single edge (u,v) is not in the cycle.
                    # New cycle: take cycle, find positions of u and v, replace the arc between them with (u,v).
                    idx_u = cyc.index(u)
                    idx_v = cyc.index(v)
                    if idx_u == idx_v:
                        continue
                    # Two arcs between u and v; we can replace one arc with the single edge (u,v).
                    # The single edge must not already be in the cycle.
                    if (min(u, v), max(u, v)) in _cycle_edges(cyc):
                        continue
                    # New cycle length = len(cyc) + 1 (we add one vertex? no, we add one edge but same vertices)
                    # Actually replacing an arc of length L with 1 edge: new cycle length = len(cyc) - L + 1.
                    # This is SHORTER, not longer. So this doesn't help.
                    # Better: find a new cycle that includes the single edge and some cycle edges.
                    pass
            # Try: form a new cycle from remaining singles + some cycle edges
        # Try to form a new cycle entirely from remaining edges
        if len(rem_set) >= 3:
            adj = _adjacency(rem_set)
            cyc = _find_long_cycle(adj, n, max_iter=50)
            if cyc:
                ce = _cycle_edges(cyc)
                if ce.issubset(rem_set):
                    cycles.append(list(cyc))
                    rem_set -= ce
                    remaining = [list(e) for e in rem_set]
                    improved = True
        if not improved:
            break
    return cycles, [list(e) for e in remaining]


def _solve_one(n, p, seed):
    edges = instance_graph(n, p, seed)
    edge_set = _edge_set(edges)
    pieces, remaining = _pack_cycles(edge_set, n)
    pieces, remaining = _local_search(pieces, remaining, n)
    # Add remaining as single edges
    for e in remaining:
        pieces.append(list(e))
    return pieces


def construct():
    global _START
    _START = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        if _deadline_hit():
            # Fallback: all single edges
            edges = instance_graph(*spec)
            results.append([list(e) for e in edges])
            continue
        pieces = _solve_one(*spec)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
