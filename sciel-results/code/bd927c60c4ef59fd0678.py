# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 5.0

def _validate_graph(n, edges, k, g, max_vertices=None):
    """
    Validate simple k-regular graph with girth >= g.
    Returns True if valid, False otherwise.
    """
    if max_vertices is not None and n > max_vertices:
        return False
        
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check girth using non-backtracking BFS from each vertex
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges).
    """
    q = k
    d = g - 5
    if d < 3:
        d = 3
        
    if q == 4:
        table = [[0, 0, 0, 0], 
                 [0, 1, 2, 3], 
                 [0, 2, 3, 1], 
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

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

def _has_short_path(adj, start, end, max_len):
    """
    Check if there is a path from start to end of length <= max_len.
    Returns True if such a path exists.
    """
    if start == end:
        return 0 <= max_len
        
    dq = deque()
    dist = {start: 0}
    dq.append(start)
    
    while dq:
        curr = dq.popleft()
        d = dist[curr]
        if d >= max_len:
            continue
        for nb in adj[curr]:
            if nb not in dist:
                dist[nb] = d + 1
                if nb == end:
                    return True
                dq.append(nb)
    return False

def _random_4_regular(n, seed):
    """
    Generate a random 4-regular simple graph on n vertices.
    """
    rng = random.Random(seed)
    
    for attempt in range(100):
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
        rng.shuffle(stubs)
        
        edges = set()
        valid = True
        adj_count = [0] * n
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
            adj_count[u] += 1
            adj_count[v] += 1
            if adj_count[u] > 4 or adj_count[v] > 4:
                valid = False
                break
        
        if valid and all(c == 4 for c in adj_count):
            return list(edges)
            
    return None

def _improve_girth(n, edges, g, time_limit, rng):
    """
    Use simulated annealing (or greedy hill climbing) to increase girth.
    We accept a move if it does not create any NEW short cycles.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    best_edges = list(edges)
    
    t_start = time.time()
    
    # Pre-compute list of edges for fast random access
    edge_list = list(edges)
    
    while time.time() < t_start + time_limit:
        # Pick two random edges
        e1 = rng.choice(edge_list)
        e2 = rng.choice(edge_list)
        if e1 == e2:
            continue
            
        u1, v1 = e1
        u2, v2 = e2
        
        # Try swap: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2)
        # Check validity: no self-loops, no multi-edges
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
            
        # Check if new edges already exist
        if u2 in adj[u1] or v2 in adj[v1]:
            continue
            
        # Check if this swap creates any short cycles
        # A new edge (u, v) creates a short cycle if there is a path of length < g-1 between u and v.
        # We must check for both new edges.
        
        # Temporarily remove old edges to allow path finding?
        # No, the path finding should be on the graph *after* removing old edges and *before* adding new ones?
        # Actually, the cycle would use the new edge.
        # Cycle: u1 - u2 (new) - ... - u1.
        # The path ... must not use the new edge (u1, u2).
        # It might use the old edges?
        # If the path uses an old edge that is being removed, it's not a valid cycle in the new graph.
        # So we should check for paths in the graph with old edges removed.
        
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        # Check for short cycles involving new edge (u1, u2)
        # Need path u1 -> u2 of length < g-1 in current adj (without e1, e2)
        # But wait, the path could be length 0? No, u1 != u2.
        # Length 1? That would mean u2 in adj[u1], which we already checked (and it's not there).
        # So we check for path length 2 to g-2.
        
        creates_cycle_1 = _has_short_path(adj, u1, u2, g - 2)
        
        if not creates_cycle_1:
            # Check for short cycles involving new edge (v1, v2)
            creates_cycle_2 = _has_short_path(adj, v1, v2, g - 2)
            
            if not creates_cycle_2:
                # Safe to apply
                adj[u1].add(u2)
                adj[u2].add(u1)
                adj[v1].add(v2)
                adj[v2].add(v1)
                
                edge_list.remove(e1)
                edge_list.remove(e2)
                new_e1 = (min(u1, u2), max(u1, u2))
                new_e2 = (min(v1, v2), max(v1, v2))
                edge_list.append(new_e1)
                edge_list.append(new_e2)
                
                best_edges = list(edge_list)
            else:
                # Revert
                adj[u1].add(v1)
                adj[v1].add(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
        else:
            # Revert
            adj[u1].add(v1)
            adj[v1].add(u1)
            adj[u2].add(v2)
            adj[v2].add(u2)
            
    return best_edges

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # 1. Try LUW Graph (guaranteed valid, potentially large)
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. Try to find a smaller graph
    # Calculate Moore Bound
    if g % 2 == 0:
        r = g // 2
    else:
        r = (g + 1) // 2
        
    moore_bound = 2
    if r > 1:
        moore_bound += k * sum([(k-1)**i for i in range(r-1)])
        
    # Search range: from Moore bound to half of LUW size (to save time)
    max_search_n = max(moore_bound, n_luw // 2)
    
    rng = random.Random(42)
    
    # Try a few sizes
    current_n = moore_bound
    # Ensure current_n has correct parity for regular graph?
    # Sum of degrees = n*k must be even.
    # If k is even, n can be anything.
    # If k is odd, n must be even.
    if k % 2 == 1 and current_n % 2 != 0:
        current_n += 1
        
    while current_n <= max_search_n and time.time() < deadline:
        # Try to find a valid graph of size current_n
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            edges = _random_4_regular(current_n, seed=1000 + current_n + attempt)
            if edges is None:
                continue
                
            # Quick check: if random graph is already valid (unlikely)
            if _validate_graph(current_n, edges, k, g):
                best_n = current_n
                best_edges = edges
                found = True
                break
                
            # Try to improve girth
            time_left = deadline - time.time()
            if time_left < 5:
                break
                
            # Allocate some time for this size
            time_for_improve = min(10.0, time_left - 2)
            
            improved_edges = _improve_girth(current_n, edges, g, time_for_improve, rng)
            
            if _validate_graph(current_n, improved_edges, k, g):
                best_n = current_n
                best_edges = improved_edges
                found = True
                break
                
        if found:
            # If we found a smaller one, we can try to go even smaller?
            # No, we are iterating upwards. If we found one at current_n, 
            # it's the smallest we've found so far in this loop.
            # We should break if we found one at the Moore bound?
            # If we found one, it's a valid upper bound.
            # We could try to go lower? No, we started at Moore bound.
            # So if we found one, it's the best possible (or close to it).
            break
            
        current_n += 2 if (k % 2 == 1) else 1
        
    return best_n, best_edges
# EVOLVE-BLOCK-END
