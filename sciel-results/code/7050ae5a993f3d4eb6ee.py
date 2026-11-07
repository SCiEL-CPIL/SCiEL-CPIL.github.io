"""Improved program: greedy distinct-degree construction + edge addition."""
# EVOLVE-BLOCK-START
import time

def _is_valid(edges, n):
    """Check no 3-edge path s-u-v-t (s!=t) has deg(s)==deg(t)."""
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    for (u, v) in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u:
                    continue
                if s == t:
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 78.0
    n = 15

    # Base: distinct degrees 0..14 via a "ladder" construction.
    # Vertex i has degree i (i in 0..14). This satisfies the constraint.
    base = []
    # Build edges so that deg(i) = i for i in 0..14.
    # Use a simple construction: vertex 14 connects to 0..13 (deg 14).
    # Then we need vertex 13 to have deg 13, etc.
    # A clean construction: for i in 0..14, connect i to vertices i+1..min(14, 14-(14-i))...
    # Simpler: use the known distinct-degree sequence 0,1,...,14 (sum=105, even? 105 is odd -> not graphical!)
    # Sum 0+1+...+14 = 105 (odd) -> NOT a valid degree sequence. Need even sum.
    # Use degrees: 0,1,2,...,13,14 -> sum 105 odd. Adjust: make one degree even-shift.
    # Try degrees: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 -> 105 odd.
    # Use: 0,1,2,3,4,5,6,7,8,9,10,11,12,14,14 -> not distinct.
    # Distinct degrees with even sum: e.g. 0,1,2,...,13,15? max deg is 14.
    # Max sum of 15 distinct values in [0,14] is 0+1+...+14=105 (odd).
    # So we cannot have all 15 distinct with even sum in [0,14].
    # We can have 14 distinct + one repeat. E.g. degrees 0,1,...,13,13 -> sum=104 even, but 13 appears twice.
    # Or 1,2,...,14,14 -> sum=120 even, 14 twice.
    # Strategy: aim for as many distinct degrees as possible; the constraint only fails when
    # two equal-degree vertices are endpoints of a 3-edge path. If duplicates are few and
    # not co-connected via a length-3 path, it's fine.

    # Build base with degrees 1..14 and one 14 (i.e., 1,2,...,13,14,14): sum=1+2+...+14+14=105+14=119 odd.
    # Try 0,1,2,...,12,14,14: sum=0+..+12=78, +14+14=106 even. Distinct except 14.
    # Let's just greedily build a valid dense graph directly.

    # Greedy: start empty, add edges in order of "most promising" while valid.
    # Order: try to add edges that don't create equal-degree endpoint conflicts.
    # To be fast, iterate over candidate edges and add if valid.
    # Use a heuristic: prioritize edges incident to high-index vertices to build distinct degrees.

    edges = []
    edge_set = set()
    # Candidate edges in a good order: sort by (u+v, u) to encourage spread.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            candidates.append((u, v))
    # Heuristic order: edges that connect far-apart vertices first (to spread degrees).
    candidates.sort(key=lambda e: (-(e[0] + e[1]), e[0]))

    # Greedy add
    for (u, v) in candidates:
        if time.time() > deadline:
            break
        if (u, v) in edge_set:
            continue
        new_edges = edges + [[u, v]]
        if _is_valid(new_edges, n):
            edges.append([u, v])
            edge_set.add((u, v))

    # Local search: try swapping or adding more edges within remaining budget.
    # Try to add any remaining candidate edge.
    for (u, v) in candidates:
        if time.time() > deadline:
            break
        if (u, v) in edge_set:
            continue
        new_edges = edges + [[u, v]]
        if _is_valid(new_edges, n):
            edges.append([u, v])
            edge_set.add((u, v))

    # Ensure return type: list of [int, int]
    result = [[int(u), int(v)] for (u, v) in edges]
    return result
# EVOLVE-BLOCK-END
