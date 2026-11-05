import time
import random
import numpy as np
from collections import deque

_DEADLINE = 115.0

def _build_edge_index(adj, n):
    """Build index for directed edges and next-edge lists for fast BFS."""
    # Map (u, v) to a unique integer ID
    # There are 2 * sum(deg) = 2 * 4 * n = 8n directed edges
    edge_id = {}
    next_edges = []
    edge_u = []
    edge_v = []
    
    # Assign IDs
    # We iterate u, then v in adj[u]
    # To make it fast, we can use a dict or a 2D array
    # n is small (320), so 2D array of size n x n is fine (320x320 = 100k)
    # But adj is sparse.
    
    # Let's just use a dict for simplicity first, then optimize if needed.
    # Actually, for n=320, dict is fast enough.
    
    current_id = 0
    for u in range(n):
        for v in adj[u]:
            edge_id[(u, v)] = current_id
            edge_u.append(u)
            edge_v.append(v)
            current_id += 1
            
    total_edges = current_id
    
    # Precompute next_edges
    # For edge (u, v) with id i, next edges are (v, w) for w in adj[v], w != u
    next_edges = [[] for _ in range(total_edges)]
    for i in range(total_edges):
        u = edge_u[i]
        v = edge_v[i]
        for w in adj[v]:
            if w != u:
                next_id = edge_id[(v, w)]
                next_edges[i].append(next_id)
                
    return edge_id, next_edges, edge_u, edge_v, total_edges

def _girth_check_fast(adj, n, g):
    """
    Fast girth check using precomputed edge indices and numpy-like arrays (lists).
    Returns the girth (min cycle length) or g if no cycle < g.
    """
    edge_id, next_edges, edge_u, edge_v, total_edges = _build_edge_index(adj, n)
    
    min_len = g
    
    # We will do BFS from each directed edge
    # To speed up, we can use a 'visited' array that we reset for each BFS.
    # Resetting an array of size 8n is cheap.
    visited = np.zeros(total_edges, dtype=np.int16)
    
    for start_id in range(total_edges):
        u_start = edge_u[start_id]
        
        # BFS
        # queue of (edge_id, depth)
        # We use a list as a queue, head pointer
        queue = [start_id]
        depths = [1]
        visited[start_id] = 1
        
        head = 0
        found_cycle = False
        
        while head < len(queue):
            curr_id = queue[head]
            d = depths[head]
            head += 1
            
            # Check if we closed a cycle
            # A cycle closes if we reach an edge (x, u_start) where x is the start vertex?
            # No, we start with edge (u_start, v).
            # We are at edge (curr_u, curr_v).
            # If curr_u == u_start, we have returned to the start vertex.
            # The cycle length is d.
            # Wait, the start edge is (u_start, v).
            # If we are at edge (x, u_start), the next step would be (u_start, y).
            # The cycle is closed when we traverse an edge INTO u_start?
            # No, the cycle is a sequence of edges.
            # Start: u_start -> v.
            # Path: u_start -> v -> ... -> x -> u_start.
            # The last edge is (x, u_start).
            # So if edge_u[curr_id] == u_start, we have a cycle of length d.
            
            if edge_u[curr_id] == u_start:
                if d < min_len:
                    min_len = d
                found_cycle = True
                break # BFS guarantees shortest path for this start
            
            # Expand
            for next_id in next_edges[curr_id]:
                if visited[next_id] == 0:
                    visited[next_id] = 1
                    queue.append(next_id)
                    depths.append(d + 1)
                    
        # Reset visited for this start_id
        # Instead of zeroing the whole array, we can just mark the visited ones as 0
        for q_id in queue:
            visited[q_id] = 0
            
        if min_len == 3: # Can't get better
            break

    return min_len

def _random_4reg(n, rng):
    """Generate a random 4-regular graph on n vertices."""
    half_edges = []
    for i in range(n):
        for j in range(4):
            half_edges.append(i)
    rng.shuffle(half_edges)
    
    adj = [set() for _ in range(n)]
    valid = True
    for i in range(0, len(half_edges), 2):
        u = half_edges[i]
        v = half_edges[i + 1]
        if u == v:
            valid = False
            break
        if v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
        
    if not valid:
        return None
        
    return [sorted(adj[i]) for i in range(n)]

def _two_switch(adj, n, edge1, edge2):
    a, b = edge1
    c, d = edge2
    
    if b not in adj[a] or d not in adj[c]:
        return None
        
    # Option 1: (a,c) and (b,d)
    if a != c and c not in adj[a] and b != d and d not in adj[b]:
        new_adj = [list(x) for x in adj]
        new_adj[a].remove(b)
        new_adj[b].remove(a)
        new_adj[c].remove(d)
        new_adj[d].remove(c)
        new_adj[a].append(c)
        new_adj[c].append(a)
        new_adj[b].append(d)
        new_adj[d].append(b)
        return new_adj
        
    # Option 2: (a,d) and (b,c)
    if a != d and d not in adj[a] and b != c and c not in adj[b]:
        new_adj = [list(x) for x in adj]
        new_adj[a].remove(b)
        new_adj[b].remove(a)
        new_adj[c].remove(d)
        new_adj[d].remove(c)
        new_adj[a].append(d)
        new_adj[d].append(a)
        new_adj[b].append(c)
        new_adj[c].append(b)
        return new_adj
        
    return None

def _luw_fallback(k, g):
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5
    
    if k != 4:
        return _luw_fallback(k, g)
        
    rng = random.Random(42)
    best_n = None
    best_edges = None
    
    # Try to find a graph with n=320, 322, etc.
    # The known smallest is 320.
    for n in [320, 322, 324, 326, 328, 330]:
        if time.time() > deadline:
            break
        if n % 2 != 0:
            continue
            
        # Try multiple random restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            adj = _random_4reg(n, rng)
            if adj is None:
                continue
                
            # Check girth
            girth = _girth_check_fast(adj, n, g)
            if girth >= g:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_n = n
                best_edges = edges
                break # Found for this n, try smaller n? No, we iterate n in increasing order.
                
            # If not valid, try local search
            # Only do local search if we have time
            if time.time() < deadline - 10:
                adj = _local_search_fast(adj, n, g, rng, deadline)
                if adj is not None:
                    girth = _girth_check_fast(adj, n, g)
                    if girth >= g:
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        best_n = n
                        best_edges = edges
                        break

    if best_n is not None:
        return best_n, best_edges
        
    return _luw_fallback(4, g)

def _local_search_fast(adj, n, g, rng, deadline):
    """Optimized local search."""
    adj = [list(x) for x in adj]
    current_girth = _girth_check_fast(adj, n, g)
    
    if current_girth >= g:
        return adj
        
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
                
    for iteration in range(200): # Reduced iterations due to fast check
        if time.time() > deadline:
            break
            
        # Pick random edge to switch
        e1 = rng.choice(edges)
        e2 = rng.choice(edges)
        if e1 == e2:
            continue
            
        new_adj = _two_switch(adj, n, e1, e2)
        if new_adj is None:
            continue
            
        new_girth = _girth_check_fast(new_adj, n, g)
        if new_girth > current_girth:
            adj = new_adj
            current_girth = new_girth
            if current_girth >= g:
                return adj
                
    return adj
