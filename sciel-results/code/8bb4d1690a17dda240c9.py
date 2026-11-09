"""Dense triangle-free 4-chromatic graph on 16 vertices."""
import time
import random

N = 16
DEADLINE_MARGIN = 10.0  # seconds before the 150s wall-clock limit

def _groetzsch_edges():
    """Grötzsch graph (Mycielski of C5) on vertices 0..10."""
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    return edges

def _build_adj(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def _is_triangle_free(adj, u, v):
    """Check if adding edge (u,v) creates a triangle."""
    if u == v:
        return False
    if (adj[u] >> v) & 1:
        return False  # already an edge
    common = adj[u] & adj[v]
    return common == 0

def _is_3_colorable(adj):
    """Fast backtracking 3-coloring check using bitsets."""
    # Order vertices by degree descending for better pruning
    order = sorted(range(N), key=lambda x: -bin(adj[x]).count('1'))
    # color[v] = 0,1,2 or -1
    color = [-1] * N
    # For each vertex, track which colors are forbidden by colored neighbors
    # Use a recursive approach with MRV heuristic

    def backtrack(pos):
        if pos == N:
            return True
        v = order[pos]
        # Find available colors
        forbidden = 0
        for u in range(N):
            if color[u] >= 0 and (adj[v] >> u) & 1:
                forbidden |= 1 << color[u]
        avail = 7 ^ forbidden
        if avail == 0:
            return False
        # Try colors in order of least constraining (simple: try all)
        # For speed, try the color that forbids the fewest options for uncolored neighbors
        best_color = -1
        best_score = -1
        c = 0
        while c < 3:
            if (avail >> c) & 1:
                # Count how many uncolored neighbors would lose color c
                score = 0
                for u in range(pos + 1, N):
                    w = order[u]
                    if (adj[v] >> w) & 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_color = c
            c += 1
        # Actually, for 3 colors, just try all available
        c = 0
        while c < 3:
            if (avail >> c) & 1:
                color[v] = c
                if backtrack(pos + 1):
                    return True
                color[v] = -1
            c += 1
        return False

    return backtrack(0)

def _try_add_edges(adj, edges, time_limit):
    """Greedy edge addition: try to add edges that preserve triangle-free and 4-chromatic."""
    # Generate all non-edges
    non_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if not ((adj[u] >> v) & 1):
                non_edges.append((u, v))

    # Try to add edges one by one, preferring those that don't break 4-chromaticity
    # Use multiple passes since adding one edge may enable/disable others
    improved = True
    while improved:
        if time.time() > time_limit:
            break
        improved = False
        # Shuffle for diversity
        random.shuffle(non_edges)
        for u, v in non_edges[:]:
            if time.time() > time_limit:
                break
            if (adj[u] >> v) & 1:
                non_edges.remove((u, v))
                continue
            if not _is_triangle_free(adj, u, v):
                continue
            # Try adding the edge
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            if _is_3_colorable(adj):
                # Revert: graph became 3-colorable, don't add
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Keep the edge
                edges.append([u, v])
                non_edges.remove((u, v))
                improved = True

    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    # Start with Grötzsch graph
    edges = _groetzsch_edges()
    adj = _build_adj(edges)

    # Verify initial graph is 4-chromatic (should be)
    # It's triangle-free and not 3-colorable by construction

    # Greedy edge addition
    edges = _try_add_edges(adj, edges, deadline)

    # Try multiple restarts with different random seeds for diversity
    best_edges = edges
    best_count = len(edges)

    remaining = deadline - time.time()
    if remaining > 5.0:
        # Try a few more passes with different orderings
        for attempt in range(5):
            if time.time() > deadline:
                break
            # Start fresh from Grötzsch with different random seed
            random.seed(12345 + attempt)
            edges2 = _groetzsch_edges()
            adj2 = _build_adj(edges2)
            edges2 = _try_add_edges(adj2, edges2, deadline)
            if len(edges2) > best_count:
                best_edges = edges2
                best_count = len(edges2)

    return best_edges
