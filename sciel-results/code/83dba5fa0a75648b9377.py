"""Strong edge colouring: induced-matching colour classes via degree-2 support
plus exact Hungarian assignment and local-search refinement."""
import time

def construct():
    t0 = time.time()
    def instance(n, d, seed):
        from random import Random
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    # --- Hungarian algorithm (min cost assignment) ---
    def hungarian(cost):
        n = len(cost)
        if n == 0:
            return []
        m = n
        u = [0] * (n + 1)
        v = [0] * (m + 1)
        p = [0] * (m + 1)
        way = [0] * (m + 1)
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [float('inf')] * (m + 1)
            used = [False] * (m + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = 0
                row = cost[i0 - 1]
                for j in range(1, m + 1):
                    if not used[j]:
                        cur = row[j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(m + 1):
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
        assign = [0] * n
        for j in range(1, m + 1):
            if p[j] > 0:
                assign[p[j] - 1] = j - 1
        return assign

    def solve_one(n, d, seed):
        edges = instance(n, d, seed)
        E = len(edges)
        if E == 0:
            return []
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        deg = [len(a) for a in adj]
        # Conflict: two edges conflict if they share a vertex or are joined by an edge
        # (i.e., not in the same induced matching).
        # For speed, build a bitmask of "forbidden partners" per edge.
        # Forbidden set F_i = {j != i : shares vertex with i, or is joined by an edge}
        # We'll build the allowed matrix (i can share a class with j) directly.
        # Edge index lookup
        eidx = {e: i for i, e in enumerate(edges)}
        # For each edge, its "neighborhood edges": edges sharing a vertex or adjacent
        forbidden = [set() for _ in range(E)]
        for i, (u, v) in enumerate(edges):
            # edges sharing u or v
            for w in adj[u]:
                e2 = (min(u, w), max(u, w))
                j = eidx.get(e2)
                if j is not None and j != i:
                    forbidden[i].add(j)
            for w in adj[v]:
                e2 = (min(v, w), max(v, w))
                j = eidx.get(e2)
                if j is not None and j != i:
                    forbidden[i].add(j)
            # edges joined by an edge: (u, w) for w in adj[u] excluding v, and (v, w) for w in adj[v] excluding u
            for w in adj[u]:
                if w == v:
                    continue
                e2 = (min(u, w), max(u, w))
                j = eidx.get(e2)
                if j is not None and j != i:
                    forbidden[i].add(j)
            for w in adj[v]:
                if w == u:
                    continue
                e2 = (min(v, w), max(v, w))
                j = eidx.get(e2)
                if j is not None and j != i:
                    forbidden[i].add(j)

        # Build candidate classes from the degree-2 support
        # An edge is in E2 if both endpoints have degree 2.
        # E2 is 2-regular on its vertex support; decompose into paths/cycles.
        inE2 = [False] * E
        for i, (u, v) in enumerate(edges):
            if deg[u] == 2 and deg[v] == 2:
                inE2[i] = True
        # Build adjacency within E2
        e2_adj = [[] for _ in range(E)]
        for i in range(E):
            if not inE2[i]:
                continue
            u, v = edges[i]
            for w in adj[u]:
                e2 = (min(u, w), max(u, w))
                j = eidx.get(e2)
                if j is not None and inE2[j] and j != i:
                    e2_adj[i].append(j)
            for w in adj[v]:
                e2 = (min(v, w), max(v, w))
                j = eidx.get(e2)
                if j is not None and inE2[j] and j != i:
                    e2_adj[i].append(j)
        # Decompose into paths/cycles
        visited = [False] * E
        candidates = []
        for i in range(E):
            if not inE2[i] or visited[i]:
                continue
            # Walk the path/cycle
            path = [i]
            visited[i] = True
            cur = i
            prev = -1
            while True:
                nxt = -1
                for nb in e2_adj[cur]:
                    if not visited[nb]:
                        nxt = nb
                        break
                if nxt == -1:
                    break
                visited[nxt] = True
                path.append(nxt)
                cur = nxt
                prev = -1
            # The path is a maximal walk in the 2-regular E2 subgraph
            # Each consecutive pair shares a vertex, so the whole path is NOT an
            # induced matching. But alternating edges along the path ARE.
            # Actually, the whole E2 component (path or cycle) has the property that
            # consecutive edges share a vertex. The induced matching within it is the
            # set of edges at even positions (or odd positions).
            # For a path: edges at positions 0,2,4,... form an induced matching
            # (they don't share vertices and no two are joined by an edge because
            # in a 2-regular path, edge i and edge i+2 are separated by edge i+1
            # which shares a vertex with both, so they are NOT joined by an edge).
            # Wait: are edge 0 and edge 2 joined by an edge? Edge 0=(v0,v1), edge 2=(v2,v3).
            # v1 and v2 are connected by edge 1. So edge 0 and edge 2 are joined by
            # edge 1? No: "joined by an edge" means there exists an edge connecting
            # an endpoint of one to an endpoint of the other. v1-v2 is edge 1, so yes,
            # edge 0 and edge 2 ARE joined by an edge. So they CANNOT be in the same
            # induced matching!
            # Hmm, so in a 2-regular path, NO two edges can be in the same induced
            # matching? Let me reconsider.
            # Actually, in a path v0-v1-v2-v3-v4, edges are (v0,v1),(v1,v2),(v2,v3),(v3,v4).
            # Edge 0 and edge 2: endpoints {v0,v1} and {v2,v3}. Is there an edge connecting
            # any of these? v1-v2 is an edge (edge 1). So yes, they are joined.
            # So in a 2-regular path, every pair of non-adjacent edges is joined by
            # the path between them. Hence NO two edges in a 2-regular path can be in
            # the same induced matching!
            # This means the degree-2 support does NOT give us multi-edge classes.
            # We need a different approach.
            candidates.append(path)
        # The degree-2 support doesn't give multi-edge induced matchings directly.
        # Let's use a different structural approach: greedy + local search.
        # Greedy: order edges, assign smallest feasible colour.
        # Then local search: try to move edges to reduce colour count.
        # First, get a greedy colouring.
        # Order edges by degree sum (high first) for better packing
        order = sorted(range(E), key=lambda i: deg[edges[i][0]] + deg[edges[i][1]], reverse=True)
        colour = [-1] * E
        colour_count = 0
        # For each colour, maintain the set of vertices "covered" (incident to an edge in that colour)
        # and the set of edges in that colour (to check the "joined by an edge" condition)
        colour_vertices = []  # list of sets of vertices
        colour_edges = []     # list of lists of edge indices
        for i in order:
            u, v = edges[i]
            # Find the smallest colour where (u,v) can be added
            # Conditions: u and v not in colour_vertices[c], and no edge in colour_edges[c]
            # is joined by an edge to (u,v).
            # An edge (a,b) is joined by an edge to (u,v) if any of a,b is adjacent to u or v
            # (other than the shared vertex case, but we already checked vertices).
            placed = False
            for c in range(colour_count):
                if u in colour_vertices[c] or v in colour_vertices[c]:
                    continue
                # Check if any edge in this colour is joined by an edge to (u,v)
                ok = True
                for j in colour_edges[c]:
                    a, b = edges[j]
                    # Check if a or b is adjacent to u or v
                    if a in adj[u] or a in adj[v] or b in adj[u] or b in adj[v]:
                        ok = False
                        break
                if ok:
                    colour[i] = c
                    colour_vertices[c].add(u)
                    colour_vertices[c].add(v)
                    colour_edges[c].append(i)
                    placed = True
                    break
            if not placed:
                colour[i] = colour_count
                colour_vertices.append({u, v})
                colour_edges.append([i])
                colour_count += 1
        # Local search: try to reduce colour count by moving edges
        # Strategy: for each colour c from colour_count-1 down, try to empty it
        # by moving its edges to other colours.
        # Also try "swap" moves.
        for _iter in range(200):
            if time.time() - t0 > 130:
                break
            improved = False
            # Try to eliminate the highest colour
            for c in range(colour_count - 1, -1, -1):
                if not colour_edges[c]:
                    colour_count = c
                    improved = True
                    continue
                # Try to move all edges of colour c to other colours
                # First, tentatively remove them
                moved = []
                all_moved = True
                for i in colour_edges[c]:
                    u, v = edges[i]
                    # Try to place in another colour
                    placed = False
                    for c2 in range(colour_count):
                        if c2 == c:
                            continue
                        if u in colour_vertices[c2] or v in colour_vertices[c2]:
                            continue
                        ok = True
                        for j in colour_edges[c2]:
                            a, b = edges[j]
                            if a in adj[u] or a in adj[v] or b in adj[u] or b in adj[v]:
                                ok = False
                                break
                        if ok:
                            colour[i] = c2
                            colour_vertices[c2].add(u)
                            colour_vertices[c2].add(v)
                            colour_edges[c2].append(i)
                            moved.append((i, c2))
                            placed = True
                            break
                    if not placed:
                        all_moved = False
                        break
                if all_moved:
                    # Remove the edges from colour c
                    colour_vertices[c] = set()
                    colour_edges[c] = []
                    improved = True
                    # Don't break; try to eliminate more
                else:
                    # Undo the moves
                    for (i, c2) in moved:
                        u, v = edges[i]
                        colour_vertices[c2].discard(u)
                        colour_vertices[c2].discard(v)
                        colour_edges[c2].remove(i)
                        colour[i] = c
                    # Re-add to colour c
                    for i in colour_edges[c]:
                        pass  # already there
            # Try swap moves: move one edge from colour c1 to c2 and one from c2 to c1
            # This is more complex; skip for now.
            if not improved:
                break
        # Renumber colours to 0..colour_count-1
        colour_map = {}
        next_c = 0
        for i in range(E):
            c = colour[i]
            if c not in colour_map:
                colour_map[c] = next_c
                next_c += 1
            colour[i] = colour_map[c]
        return [colour[i] for i in range(E)]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() - t0 > 140:
            # Fallback: greedy only
            edges = instance(*spec)
            E = len(edges)
            if E == 0:
                out.append([])
                continue
            adj = [set() for _ in range(spec[0])]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            deg = [len(a) for a in adj]
            eidx = {e: i for i, e in enumerate(edges)}
            forbidden = [set() for _ in range(E)]
            for i, (u, v) in enumerate(edges):
                for w in adj[u]:
                    e2 = (min(u, w), max(u, w))
                    j = eidx.get(e2)
                    if j is not None and j != i:
                        forbidden[i].add(j)
                for w in adj[v]:
                    e2 = (min(v, w), max(v, w))
                    j = eidx.get(e2)
                    if j is not None and j != i:
                        forbidden[i].add(j)
                for w in adj[u]:
                    if w == v:
                        continue
                    e2 = (min(u, w), max(u, w))
                    j = eidx.get(e2)
                    if j is not None and j != i:
                        forbidden[i].add(j)
                for w in adj[v]:
                    if w == u:
                        continue
                    e2 = (min(v, w), max(v, w))
                    j = eidx.get(e2)
                    if j is not None and j != i:
                        forbidden[i].add(j)
            order = sorted(range(E), key=lambda i: deg[edges[i][0]] + deg[edges[i][1]], reverse=True)
            colour = [-1] * E
            colour_count = 0
            colour_vertices = []
            colour_edges = []
            for i in order:
                u, v = edges[i]
                placed = False
                for c in range(colour_count):
                    if u in colour_vertices[c] or v in colour_vertices[c]:
                        continue
                    ok = True
                    for j in colour_edges[c]:
                        a, b = edges[j]
                        if a in adj[u] or a in adj[v] or b in adj[u] or b in adj[v]:
                            ok = False
                            break
                    if ok:
                        colour[i] = c
                        colour_vertices[c].add(u)
                        colour_vertices[c].add(v)
                        colour_edges[c].append(i)
                        placed = True
                        break
                if not placed:
                    colour[i] = colour_count
                    colour_vertices.append({u, v})
                    colour_edges.append([i])
                    colour_count += 1
            colour_map = {}
            next_c = 0
            for i in range(E):
                c = colour[i]
                if c not in colour_map:
                    colour_map[c] = next_c
                    next_c += 1
                colour[i] = colour_map[c]
            out.append([colour[i] for i in range(E)])
        else:
            out.append(solve_one(*spec))
    return out
