"""
Structural change: model the edge decomposition as a minimum-path-cover problem
on a directed line graph, solved via maximum-weight bipartite matching (Hungarian).
Each undirected edge becomes two directed arcs. A perfect matching on the line graph
gives a set of edge-disjoint directed paths covering all arcs. Cycles in the
undirected sense correspond to directed cycles or pairs of opposite directed paths.
We then merge opposite directed paths into undirected cycles where possible.

This is a structural change from greedy cycle peeling to an exact optimization
formulation using the Hungarian algorithm for minimum-weight perfect matching.
"""
import time
import sys
from random import Random
from collections import defaultdict

_DEADLINE = 140.0
_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _hungarian(cost_matrix):
    """
    Solve the assignment problem (minimum cost perfect matching) using the
    Hungarian algorithm. cost_matrix is an n x n matrix.
    Returns (row_assign, col_assign) where row_assign[i] = col assigned to row i.
    """
    n = len(cost_matrix)
    if n == 0:
        return [], []
    if n == 1:
        return [0], [0]

    # Hungarian algorithm (Jonker-Volgenant style) for min cost
    INF = float('inf')
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)  # p[j] = row assigned to column j
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_assign = [0] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            row_assign[p[j] - 1] = j - 1
    col_assign = [0] * n
    for i in range(n):
        col_assign[row_assign[i]] = i
    return row_assign, col_assign


def _decompose_matching(edges):
    """
    Decompose edges into cycles and single edges using matching on the line graph.
    
    Strategy:
    1. Create directed arcs for each undirected edge (both directions).
    2. Build a bipartite graph: left = tails, right = heads, edge exists if
       there's an arc from tail to head.
    3. Find maximum cardinality matching (minimize number of paths).
    4. The matching gives a set of directed paths covering all arcs.
    5. Merge opposite directed paths into undirected cycles.
    6. Remaining paths become single edges or longer paths (treated as edges).
    """
    m = len(edges)
    if m == 0:
        return []
    if m == 1:
        return [[edges[0][0], edges[0][1]]]

    # Create arcs: each undirected edge (u,v) with u<v gives arcs (u,v) and (v,u)
    # We label arcs by index
    arcs = []  # list of (u, v)
    arc_id = {}  # (u,v) -> index
    for (u, v) in edges:
        idx = len(arcs)
        arcs.append((u, v))
        arc_id[(u, v)] = idx
        arcs.append((v, u))
        arc_id[(v, u)] = idx + 1

    n_arcs = len(arcs)  # 2*m

    # Build bipartite graph for matching
    # Left side: arc i (as tail), Right side: arc j (as head)
    # Edge from left i to right j if arcs[i][1] == arcs[j][0] and i != j
    # Cost: 0 if we match (continue path), INF if not possible
    # We want maximum cardinality matching, so minimize number of unmatched
    
    # For each left node i, find compatible right nodes
    # Compatible: arcs[i][1] == arcs[j][0] and i != j
    # Also, we should avoid matching arc i to its reverse arc j (i.e., if arcs[j] is reverse of arcs[i])
    # Actually for path cover, we can match any arc to any other arc sharing the vertex
    
    # Build cost matrix (n_arcs x n_arcs)
    # cost[i][j] = 0 if arcs[i][1] == arcs[j][0] and i != j, else INF
    # We want to maximize the number of matches = minimize unmatched
    # So we use cost 0 for valid matches, large cost for invalid
    
    INF_COST = 1000000
    cost = [[INF_COST] * n_arcs for _ in range(n_arcs)]
    
    # For efficiency, precompute adjacency
    # left i connects to right j if arcs[i][1] == arcs[j][0] and i != j
    # Group arcs by their head/tail
    arcs_by_head = defaultdict(list)  # vertex -> list of arc indices starting at that vertex
    arcs_by_tail = defaultdict(list)  # vertex -> list of arc indices ending at that vertex
    
    for idx, (u, v) in enumerate(arcs):
        arcs_by_head[v].append(idx)  # arcs starting at v (v is tail)
        arcs_by_tail[u].append(idx)  # arcs ending at u (u is head)
    
    # Actually: arc (u,v) has tail=u, head=v
    # left i (arc i as tail): we can match to right j (arc j as head) if head of arc i = tail of arc j
    # i.e., arcs[i][1] == arcs[j][0] and i != j
    
    for i in range(n_arcs):
        head_i = arcs[i][1]
        for j in arcs_by_head[head_i]:
            if j != i:
                cost[i][j] = 0
    
    # Now solve the assignment problem
    # But n_arcs can be up to ~2*30*30/2 = 900, so n_arcs x n_arcs = 810000 entries
    # Hungarian is O(n^3) which for n=900 is ~729M operations - too slow in Python
    
    # Let's check the size
    if n_arcs > 200:
        # Too large for Hungarian in pure Python, fall back to greedy
        return _decompose_greedy_fallback(edges)
    
    row_assign, col_assign = _hungarian(cost)
    
    # Build the path cover from the matching
    # matched[i] = j means arc i's head connects to arc j's tail
    # So in the path: ... -> arc i -> arc j -> ...
    
    # Find which arcs are "starts" (not matched as right side) and "ends" (not matched as left side)
    matched_as_right = set()
    matched_as_left = set()
    for i in range(n_arcs):
        if cost[i][row_assign[i]] < INF_COST:
            matched_as_left.add(i)
            matched_as_right.add(row_assign[i])
    
    # Starts: arcs not in matched_as_right (no predecessor)
    # Ends: arcs not in matched_as_left (no successor)
    starts = [i for i in range(n_arcs) if i not in matched_as_right]
    
    # Trace paths
    next_arc = {}
    for i in range(n_arcs):
        if cost[i][row_assign[i]] < INF_COST:
            next_arc[i] = row_assign[i]
    
    paths = []
    visited = set()
    for s in starts:
        if s in visited:
            continue
        path = []
        cur = s
        while cur is not None and cur not in visited:
            visited.add(cur)
            path.append(cur)
            cur = next_arc.get(cur)
        if path:
            paths.append(path)
    
    # Now we have directed paths. Each path is a sequence of directed arcs.
    # Convert to vertex sequences
    vertex_paths = []
    for path in paths:
        verts = [arcs[path[0]][0]]
        for idx in path:
            verts.append(arcs[idx][1])
        vertex_paths.append(verts)
    
    # Now try to merge paths into cycles
    # Two directed paths can form a cycle if:
    #   path1: a -> b -> ... -> c
    #   path2: c -> d -> ... -> a
    # i.e., end of path1 = start of path2 and end of path2 = start of path1
    # Also, a single path is a cycle if start == end
    
    pieces = []
    used_paths = [False] * len(vertex_paths)
    
    # First, check for self-cycles (start == end)
    for i, vp in enumerate(vertex_paths):
        if len(vp) >= 2 and vp[0] == vp[-1]:
            # It's a cycle
            # Extract unique vertices (the cycle)
            cycle = vp[:-1]  # remove last vertex (same as first)
            if len(cycle) >= 3:
                pieces.append(cycle)
                used_paths[i] = True
            else:
                # 2-cycle in directed sense = just an edge
                pieces.append([vp[0], vp[1]])
                used_paths[i] = True
    
    # Now try to merge pairs of paths into cycles
    # Build index: vertex -> list of path indices ending at that vertex
    ends_at = defaultdict(list)
    starts_at = defaultdict(list)
    for i, vp in enumerate(vertex_paths):
        if not used_paths[i]:
            ends_at[vp[-1]].append(i)
            starts_at[vp[0]].append(i)
    
    merged = set()
    for v in list(ends_at.keys()):
        if v in starts_at:
            for i in ends_at[v]:
                if used_paths[i] or i in merged:
                    continue
                for j in starts_at[v]:
                    if i == j or used_paths[j] or j in merged:
                        continue
                    # Check if path j ends at the start of path i
                    if vertex_paths[j][-1] == vertex_paths[i][0]:
                        # Merge: path i + path j forms a cycle
                        cycle = vertex_paths[i] + vertex_paths[j][1:]
                        # cycle should start and end at same vertex
                        if cycle[0] == cycle[-1] and len(cycle) >= 4:
                            # Valid cycle
                            cycle_verts = cycle[:-1]
                            if len(set(cycle_verts)) == len(cycle_verts):
                                pieces.append(cycle_verts)
                                used_paths[i] = True
                                used_paths[j] = True
                                merged.add(i)
                                merged.add(j)
                                break
                if i in merged:
                    break
    
    # Remaining paths: convert to pieces
    # A path of length 1 (single arc) is a single edge
    # A path of length > 1 is a sequence of edges - we need to represent it
    # Since pieces must be either single edges or cycles, a non-cyclic path
    # of multiple edges must be broken into single edges
    for i, vp in enumerate(vertex_paths):
        if not used_paths[i]:
            if len(vp) == 2:
                # Single edge
                pieces.append([vp[0], vp[1]])
            else:
                # Multiple edges in a path - break into single edges
                for k in range(len(vp) - 1):
                    pieces.append([vp[k], vp[k + 1]])
    
    return pieces


def _decompose_greedy_fallback(edges):
    """Greedy cycle peeling as fallback for large graphs."""
    edges_set = set(edges)
    pieces = []
    remaining = set(edges_set)
    
    deadline = _START + _DEADLINE
    max_iter = 200
    
    for _ in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        if not remaining:
            break
        
        # Build adjacency
        adj = defaultdict(set)
        for (u, v) in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find a cycle using DFS
        best_cycle = None
        for v in sorted(adj.keys()):
            if time.time() > deadline - 5.0:
                break
            # DFS to find a cycle
            stack = [(v, [v], {v})]
            found = 0
            while stack:
                node, path, on_path = stack.pop()
                found += 1
                if found > 50000:
                    break
                for nb in adj.get(node, ()):
                    if nb == path[0] and len(path) >= 3:
                        cyc = list(path)
                        if best_cycle is None or len(cyc) > len(best_cycle):
                            best_cycle = cyc
                        break
                    if nb in on_path:
                        continue
                    stack.append((nb, path + [nb], on_path | {nb}))
                if best_cycle and len(best_cycle) >= 8:
                    break
            if best_cycle and len(best_cycle) >= 8:
                break
        
        if best_cycle is None:
            break
        
        # Remove cycle edges
        valid = True
        for i in range(len(best_cycle)):
            u = best_cycle[i]
            v = best_cycle[(i + 1) % len(best_cycle)]
            e = (min(u, v), max(u, v))
            if e in remaining:
                remaining.discard(e)
            else:
                valid = False
                break
        
        if valid:
            pieces.append(best_cycle)
    
    for e in remaining:
        pieces.append([e[0], e[1]])
    
    return pieces


def construct():
    global _START
    _START = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        if time.time() > _START + _DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        pieces = _decompose_matching(edges)
        results.append(pieces)
    return results
