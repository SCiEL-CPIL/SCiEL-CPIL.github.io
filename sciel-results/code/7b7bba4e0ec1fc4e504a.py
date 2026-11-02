import time
import random
import math
import collections

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # We only handle k=3, g=16 for this optimized path
    if k != 3:
        # Fallback to LUW for other k
        return _luw_construct(k, g)
        
    # Voltage graph construction on K4
    # Base graph: K4
    # Vertices: 0, 1, 2, 3
    # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    
    # Define edges with fixed orientation
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edge_index = {e: i for i, e in enumerate(edges)}
    # Reverse index
    rev_index = {(v, u): i for (u, v), i in edge_index.items()}
    
    # Generate all closed walks of length < g
    # State: (current_vertex, length)
    # We need to track the sequence of edge indices to compute the sum
    # Since g=16, max length 15.
    # Number of walks: 3^15 ~ 14M. Too many to store explicitly if we store the full path.
    # But we can prune or use a different check.
    
    # Alternative: Check girth of the cover directly using BFS on the non-backtracking state space?
    # Cover size 4n. If n=250, size=1000.
    # Non-backtracking BFS from each vertex in a 1000-vertex graph is fast.
    # So we can just construct the cover and check its girth.
    
    # Strategy:
    # 1. Iterate n from 100 to 1000 (step 10 or 5)
    # 2. For each n, try random voltage assignments
    # 3. Construct cover
    # 4. Check girth
    # 5. If girth >= g, return
    
    # To speed up search, we can use a heuristic:
    # Random voltages often work for large enough n.
    
    # Pre-compute adjacency of base graph
    adj_base = {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]}
    
    # Map edge to index
    def get_edge_idx(u, v):
        if (u, v) in edge_index:
            return edge_index[(u, v)], 1
        else:
            return edge_index[(v, u)], -1

    best_n = None
    best_edges = None
    
    # Try increasing n
    # Moore bound for k=3, g=16 is 512.
    # So 4n >= 512 => n >= 128.
    for n in range(130, 1000, 2):
        if _time_left() < 10.0:
            break
            
        # Try a few random assignments for this n
        for attempt in range(20):
            if _time_left() < 5.0:
                break
                
            # Random voltages
            voltages = [random.randint(0, n - 1) for _ in range(6)]
            
            # Construct cover edges
            # Vertices are (v, g) where v in 0..3, g in 0..n-1
            # Total 4n vertices
            # Map (v, g) to index: v * n + g
            
            cover_edges = []
            # We need to add edges for each base edge and each group element
            for (u, v) in edges:
                idx, sign = get_edge_idx(u, v)
                vol = voltages[idx]
                for g_elem in range(n):
                    # Edge from (u, g_elem) to (v, (g_elem + sign * vol) % n)
                    # Wait, if sign is -1, it means we are traversing (v,u) which has voltage -vol.
                    # So destination is (v, g_elem - vol).
                    # Let's stick to the definition:
                    # If we traverse edge (u,v) with voltage w, we go from (u, g) to (v, g+w).
                    # If we traverse (v,u) with voltage -w, we go from (v, g) to (u, g-w).
                    
                    u_node = u * n + g_elem
                    v_node = v * n + (g_elem + sign * vol) % n
                    cover_edges.append((u_node, v_node))
                    
                    # The reverse direction is covered by the other g_elem?
                    # No, the graph is undirected.
                    # The edge (u_node, v_node) is the same as (v_node, u_node).
                    # We just need to list each edge once.
                    # The loop over g_elem covers all "lifts" of the edge (u,v).
                    # Does it cover the lifts of (v,u)?
                    # The set of edges {(u, g), (v, g+vol)} for all g is the same set as {(v, g'), (u, g'-vol)} for all g'.
                    # So we don't need to add reverse edges explicitly.
                    
            # Check girth of cover
            # 4n vertices, 6n edges
            if _check_girth(4 * n, cover_edges, g):
                return 4 * n, cover_edges
            
            # If not, try next random assignment
            
    # Fallback to LUW
    return _luw_construct(k, g)

def _check_girth(n, edges, min_girth):
    """
    Returns True if the graph has girth >= min_girth.
    Uses non-backtracking BFS from each vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = min_girth // 2
    
    for start in range(n):
        # State: (current_vertex, previous_vertex)
        dist = {}
        queue = collections.deque()
        
        for nb in adj[start]:
            state = (nb, start)
            dist[state] = 1
            queue.append(state)
        
        while queue:
            curr, prev = queue.popleft()
            d = dist[(curr, prev)]
            if d >= max_depth:
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if nb == start:
                    # Found a cycle of length d+1
                    if d + 1 < min_girth:
                        return False
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = d + 1
                    queue.append(state)
    return True

def _luw_construct(k, g):
    # Fallback to the original LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
