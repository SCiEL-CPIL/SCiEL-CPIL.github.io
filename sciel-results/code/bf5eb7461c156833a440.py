"""
Construct a k-regular graph with girth at least g.
Uses voltage lift of Petersen graph over Z_m with random edge voltages.
"""
import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 118  # safety margin
    
    def check_deadline():
        return time.time() > deadline
    
    if k == 3 and g == 16:
        # Try voltage lift of Petersen graph over Z_m
        # Petersen has 10 vertices, 15 edges, girth 5, 3-regular
        # Lift over Z_m gives 10m vertices
        # Moore bound for (3,16): 2*(2^8-1) = 510, so m >= 52 gives n=520
        best_result = None
        
        # Try m values starting from smallest (52) upward
        # More trials for small m significantly help
        for m in range(52, 80):
            if check_deadline():
                break
            
            # Number of trials: more for smaller m
            if m <= 60:
                max_trials = 500
            elif m <= 70:
                max_trials = 200
            else:
                max_trials = 80
            
            for trial in range(max_trials):
                if check_deadline():
                    break
                
                # Random voltage assignment to Petersen edges
                n, edges, voltages = _petersen_lift(m, trial)
                
                # Verify girth using non-backtracking walks
                if _verify_girth_nb(n, edges, g):
                    if best_result is None or n < best_result[0]:
                        best_result = (n, edges)
                        # Found valid graph, can try smaller m or return
                        return n, edges
        
        # If voltage lift didn't find anything, fall back to LUW
        if best_result is not None:
            return best_result
        
        try:
            n_luw, edges_luw = _luw_construct_with_d(3, 16, 7)
            if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                return n_luw, edges_luw
        except Exception:
            pass
        
        return _luw_construct(k, g)
    
    # General case fallback
    return _luw_construct(k, g)


def _petersen_lift(m, seed):
    """
    Construct a voltage lift of the Petersen graph over Z_m.
    Petersen graph: 10 vertices, 15 edges, 3-regular, girth 5.
    Returns (n, edges, voltages) where n=10*m.
    Vertex (i, j) maps to index i*m + j where i in 0..9, j in 0..m-1.
    """
    rng = random.Random(seed * 1000003 + 7)
    
    # Petersen graph edges (standard labeling)
    # Vertices: 0-4 (outer pentagon), 5-9 (inner star)
    # Outer: 0-1-2-3-4-0
    # Inner: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    
    # Assign random voltages to each edge
    voltages = {}
    for (u, v) in petersen_edges:
        voltages[(u, v)] = rng.randint(1, m - 1)
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    n = 10 * m
    edges = []
    
    for (u, v) in petersen_edges:
        volt = voltages[(u, v)]
        for j in range(m):
            a = u * m + j
            b = v * m + (j + volt) % m
            if a < b:
                edges.append((a, b))
            else:
                edges.append((b, a))
    
    return n, edges, voltages


def _verify_girth_nb(n, edges, g):
    """
    Verify girth >= g using non-backtracking walks from every vertex.
    A non-backtracking walk of length L < g that returns to start = cycle of length L.
    We track (current_vertex, previous_vertex) states.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    
    # For each start vertex, do non-backtracking BFS up to depth g-1
    # State: (current, prev) where prev is the vertex we came from
    # A cycle of length L exists if we can return to start via non-backtracking walk of length L
    
    for src in range(n):
        # Non-backtracking walk states: (current, prev)
        # Start: for each neighbor of src, state (neighbor, src) at depth 1
        # We want to find if src is reachable from (src, prev) for any prev != src
        # Actually: a cycle through src of length L means there's a non-backtracking
        # walk src -> v1 -> v2 -> ... -> v_{L-1} -> src where v_{L-1} != v1 (non-backtracking)
        # Equivalently: from state (v1, src) at depth 1, can we reach (src, v_{L-1}) at depth L-1?
        # Or: from state (src, v) for some neighbor v, can we reach (v, src)? No that's length 2.
        
        # Better approach: BFS on state space (u, prev) for u != prev
        # Initial states: (v, src) for each neighbor v of src, depth 1
        # We check if any state (src, v) is reached at depth L-1 for L < g, i.e., depth < g-1
        # Wait: walk src->v1->...->v_{L-1}->src has L edges, L-1 intermediate steps
        # State after first edge: (v1, src), depth 1
        # State after last edge before returning to src: (v_{L-1}, v_{L-2}), depth L-1
        # Then from (v_{L-1}, v_{L-2}) we go to src. This is valid if src != v_{L-2} (non-backtracking).
        # So we need: is there a state (u, prev) at depth d where u is adjacent to src and prev != src,
        # and d+1 < g? That gives cycle of length d+1 < g.
        # Actually: from state (u, prev) at depth d, if u is adjacent to src and src != prev,
        # then we have a cycle of length d+1.
        
        # So: BFS from states (v, src) for neighbors v of src.
        # At each state (u, prev) at depth d, check if src is a neighbor of u and src != prev.
        # If d + 1 < g, we have a short cycle.
        
        visited = set()
        # Queue: (u, prev, depth)
        queue = deque()
        for v in adj[src]:
            visited.add((v, src))
            queue.append((v, src, 1))
        
        while queue:
            u, prev, depth = queue.popleft()
            
            # Check if we can close a cycle through src
            if depth + 1 < g:
                for w in adj[u]:
                    if w == src and src != prev:
                        return False  # Found cycle of length depth+1 < g
            
            if depth + 1 >= g:
                continue  # No need to expand further
            
            for w in adj[u]:
                if w == prev:
                    continue  # Non-backtracking: can't go back to where we came from
                state = (w, u)
                if state not in visited:
                    visited.add(state)
                    queue.append((w, u, depth + 1))
    
    return True


# (circulant search helpers removed - using voltage lift approach)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # make it odd
    
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k
    
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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    
    # BFS from each vertex to find shortest cycle
    for src in range(n):
        if _bfs_short_cycle(src, adj, n, g):
            return False
    
    return True


def _bfs_short_cycle(src, adj, n, g):
    """
    BFS from src to find if there's a cycle shorter than g.
    Returns True if a short cycle is found.
    """
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])
    
    max_depth = g // 2
    
    while queue:
        u = queue.popleft()
        depth = dist[u]
        
        if depth >= max_depth:
            # Still check edges from u for back edges to shallow nodes
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue
        
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                # Back edge - check for cycle
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True
    
    return False

# EVOLVE-BLOCK-END
