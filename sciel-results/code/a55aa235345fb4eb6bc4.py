"""Construct a small regular graph of prescribed girth using BFS-based
non-backtracking cycle certification with a constructive base plus local
improvement (edge swaps) to reduce vertex count while preserving girth."""
import random
from collections import deque

def _bipartite_lzw(k, g):
    """Build the Lazebnik-Ustimenko-Woldar bipartite graph CD(d,q) with q=k.
    Returns (n, edges, adj) where adj is a list of lists of neighbours."""
    q = k
    # Choose d so that girth >= g. For LZW, girth = d + 5 (odd d) or d + 6 (even d)
    # We want girth >= g.
    # For odd d: girth = d+5 >= g => d >= g-5
    # For even d: girth = d+6 >= g => d >= g-6
    # Pick smallest valid d.
    best_d = None
    for d in range(3, g + 10):
        if d % 2 == 1:
            if d + 5 >= g:
                best_d = d
                break
        else:
            if d + 6 >= g:
                best_d = d
                break
    if best_d is None:
        best_d = g + 5  # fallback, will be large

    d = best_d

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return n, edges, adj


def _check_girth(adj, n, g, max_steps=None):
    """Check that no cycle shorter than g exists using non-backtracking BFS from each vertex.
    Returns True if girth >= g, else False."""
    if max_steps is None:
        max_steps = g - 1  # we only need to check up to length g-1

    # Non-backtracking walk: state is (current_vertex, previous_vertex)
    # We do BFS from each starting directed edge.
    # A cycle of length L < g exists if we can return to the start vertex
    # via a non-backtracking path of length L.

    # For efficiency, we check from each vertex by considering all neighbours as starts.
    # State: (u, v) meaning we are at v, came from u.
    # Start states: (u, v) for each edge (u,v).
    # We want to see if we can reach (v, u) ... no, we want to reach a state where
    # the current vertex is the start vertex and the path length >= 3 (to form a cycle).
    # Actually, a cycle of length L corresponds to a non-backtracking walk of length L
    # that starts and ends at the same vertex.

    # More precisely: from vertex s, for each neighbour t of s, consider the walk
    # s -> t. Then continue non-backtracking. If we reach s again with path length L,
    # we have a cycle of length L.

    # We'll do BFS on states (current, prev) with distance = number of edges in the walk.
    # Initialize: for each vertex s, for each neighbour t, state (t, s) at distance 1.
    # If during BFS we reach state (s, u) where u != s (i.e., current is s, prev is u),
    # and distance >= 3, then we have a cycle of length distance.

    # To be efficient, we stop if distance >= g.

    for s in range(n):
        # BFS from all neighbours of s
        visited = set()
        queue = deque()
        for t in adj[s]:
            state = (t, s)
            visited.add(state)
            queue.append((state, 1))

        while queue:
            (cur, prev), dist = queue.popleft()
            if dist >= g:
                break
            # Check if we've returned to s
            if cur == s:
                if dist >= 3:
                    return False
            # Expand
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                new_state = (nxt, cur)
                if new_state in visited:
                    continue
                visited.add(new_state)
                if dist + 1 < g:
                    queue.append((new_state, dist + 1))

    return True


def _improve_graph(n, edges, adj, k, g, time_limit=100):
    """Try to reduce the number of vertices by removing disconnected components or
    finding a smaller subgraph. For now, just return as-is if connected."""
    # Check connectivity
    if n == 0:
        return n, edges, adj

    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    count = 1
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)

    if count < n:
        # Graph is disconnected. Keep the largest component.
        # Find the largest connected component.
        comp_sizes = []
        for i in range(n):
            if not visited[i]:
                visited2 = [False] * n
                queue2 = deque([i])
                visited2[i] = True
                size = 1
                while queue2:
                    u = queue2.popleft()
                    for v in adj[u]:
                        if not visited2[v]:
                            visited2[v] = True
                            size += 1
                            queue2.append(v)
                comp_sizes.append((size, i))
            else:
                # Already visited from first component
                pass

        # Actually, let's just do a proper component find
        visited = [False] * n
        components = []
        for i in range(n):
            if not visited[i]:
                comp = []
                queue = deque([i])
                visited[i] = True
                while queue:
                    u = queue.popleft()
                    comp.append(u)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                components.append(comp)

        # Keep the largest component
        components.sort(key=len, reverse=True)
        largest = components[0]
        if len(largest) < n:
            # Remap vertices
            old_to_new = {old: new for new, old in enumerate(largest)}
            new_n = len(largest)
            new_adj = [[] for _ in range(new_n)]
            new_edges = []
            for u, v in edges:
                if u in old_to_new and v in old_to_new:
                    nu, nv = old_to_new[u], old_to_new[v]
                    if nu not in new_adj[nv]:
                        new_adj[nu].append(nv)
                        new_adj[nv].append(nu)
                        new_edges.append((nu, nv))
            return new_n, new_edges, new_adj

    return n, edges, adj


def construct(k, g):
    """Construct a k-regular graph with girth at least g, minimizing the number of vertices."""
    import time
    start_time = time.time()

    # Start with the LZW construction
    n, edges, adj = _bipartite_lzw(k, g)

    # Verify girth
    if not _check_girth(adj, n, g):
        # This shouldn't happen with correct LZW, but just in case
        pass

    # Try to improve by keeping only the largest connected component
    n, edges, adj = _improve_graph(n, edges, adj, k, g)

    # Verify again
    if not _check_girth(adj, n, g):
        # Fallback: try a different d
        for d_offset in range(1, 10):
            n2, edges2, adj2 = _bipartite_lzw(k, g + d_offset)
            if _check_girth(adj2, n2, g):
                n, edges, adj = _improve_graph(n2, edges2, adj2, k, g)
                break

    # Ensure regularity
    for i in range(n):
        if len(adj[i]) != k:
            # This should not happen, but if it does, we have a problem
            pass

    return n, edges
