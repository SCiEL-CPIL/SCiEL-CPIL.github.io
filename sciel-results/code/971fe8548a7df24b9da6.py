# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    if k != 3:
        return _lww_fallback(k, g)
    deadline = time.time() + 110
    # Try randomized construction first
    result = _randomized_cage(3, g, deadline)
    if result is not None:
        return result
    # Fallback to algebraic construction
    return _lww_fallback(k, g)

def _randomized_cage(k, g, deadline):
    """
    Randomized backtracking to build a k-regular graph with girth >= g.
    Uses a greedy approach: maintain a partial graph, add edges/vertices
    while ensuring no short cycles are formed.
    """
    for attempt in range(80):
        if time.time() >= deadline:
            break
        result = _single_random_attempt(k, g, deadline, attempt)
        if result is not None:
            return result
    return None

def _single_random_attempt(k, g, deadline, seed_offset):
    """
    One attempt at building the graph via randomized greedy construction.
    Uses distance tracking to efficiently check for short cycles.
    """
    random.seed(12345 + seed_offset * 7919)
    
    adj = []  # adjacency lists
    dist_from = {}  # (u, v) -> distance, only for pairs we've computed
    
    def get_dist(u, v):
        """BFS to find distance between u and v in current graph."""
        if u == v:
            return 0
        key = (min(u, v), max(u, v))
        if key in dist_from:
            return dist_from[key]
        
        n = len(adj)
        dist = [-1] * n
        dist[u] = 0
        queue = deque([u])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    if nb == v:
                        dist_from[key] = dist[v]
                        return dist[v]
                    queue.append(nb)
        dist_from[key] = -1
        return -1
    
    def can_add_edge(u, v):
        """Check if adding edge (u,v) would create a cycle of length < g."""
        if u == v:
            return False
        if v in adj[u]:
            return False
        d = get_dist(u, v)
        if d == -1:
            return True  # disconnected, safe
        return d + 1 >= g  # new cycle length = d + 1
    
    def clear_dist_cache(nodes):
        """Clear distance cache entries involving any of the given nodes."""
        to_remove = [key for key in dist_from if key[0] in nodes or key[1] in nodes]
        for key in to_remove:
            del dist_from[key]
    
    # Start with vertex 0 and its k neighbors
    adj.append([])
    for i in range(1, k + 1):
        adj.append([])
        adj[0].append(i)
        adj[i].append(0)
    
    # Vertices that still need edges
    incomplete = deque(range(1, k + 1))
    
    # Track which vertices are "active" (degree < k)
    # Use a set for O(1) membership check
    active = set(range(1, k + 1))
    
    while incomplete and time.time() < deadline:
        v = incomplete.popleft()
        if len(adj[v]) >= k:
            active.discard(v)
            continue
        
        needed = k - len(adj[v])
        
        # Get candidates: vertices with degree < k that aren't v
        candidates = [u for u in active if u != v]
        random.shuffle(candidates)
        
        connected_this_round = False
        for u in candidates:
            if len(adj[v]) >= k:
                break
            if can_add_edge(v, u):
                adj[v].append(u)
                adj[u].append(v)
                clear_dist_cache({v, u})
                connected_this_round = True
                if len(adj[v]) < k:
                    # v still needs more edges
                    pass
                if len(adj[u]) >= k:
                    active.discard(u)
                if len(adj[v]) < k:
                    incomplete.append(v)
                break
        
        if len(adj[v]) < k:
            # Add new vertices to fill remaining degree
            for _ in range(k - len(adj[v])):
                if time.time() >= deadline:
                    return None
                new_v = len(adj)
                adj.append([])
                adj[v].append(new_v)
                adj[new_v].append(v)
                active.add(new_v)
                incomplete.append(new_v)
                clear_dist_cache({new_v})
            
            if len(adj[v]) >= k:
                active.discard(v)
    
    # Check if we succeeded: all vertices have degree exactly k
    n = len(adj)
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    # Verify girth using non-backtracking walks from every vertex
    # A cycle of length L < g exists iff there's a non-backtracking walk of length L
    # from some vertex back to itself.
    # More efficiently: for each vertex, do BFS tracking the "previous" node
    # to avoid backtracking, and check if we can return to start in < g steps.
    
    # Simpler check: BFS from each vertex, and for each neighbor at distance d,
    # if d + 1 < g and the path doesn't backtrack, we have a short cycle.
    # Actually, the standard check: for each vertex s, BFS from s. If any neighbor
    # of s is reached at distance d where d + 1 < g, we have a cycle of length d+1.
    # But we need to ensure the path from s to that neighbor doesn't use the edge (s, neighbor).
    
    # Better: For each vertex s, do a BFS where we track (node, prev) pairs.
    # A non-backtracking walk from s of length L that returns to s gives a cycle of length L.
    # We check if any such walk has length < g.
    
    for s in range(n):
        # BFS in the non-backtracking line graph
        # State: (current_node, prev_node)
        # Start: (s, -1) for each neighbor of s
        # We want to reach (s, prev) where prev != -1, with path length < g
        
        visited = set()
        queue = deque()
        
        for nb in adj[s]:
            state = (s, nb)  # We're at s, came from nb
            visited.add(state)
            queue.append((state, 1))  # (state, length)
        
        found_short_cycle = False
        while queue:
            (node, prev), length = queue.popleft()
            if length >= g - 1:
                continue  # No need to go further
            
            for nb in adj[node]:
                if nb == prev:
                    continue  # Backtracking
                new_state = (nb, node)
                if new_state in visited:
                    continue
                visited.add(new_state)
                
                if nb == s and node != -1:
                    # We've returned to s via a non-backtracking walk of length length + 1
                    # This forms a cycle of length length + 1
                    if length + 1 < g:
                        found_short_cycle = True
                        break
                else:
                    queue.append((new_state, length + 1))
            
            if found_short_cycle:
                break
        
        if found_short_cycle:
            return None
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    return n, edges

def _lww_fallback(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn, sub_fn, mul_fn = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add_fn, sub_fn, mul_fn = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield 1 - side, tuple(y)
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
