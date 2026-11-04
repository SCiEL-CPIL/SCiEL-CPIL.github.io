"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

def _count_short_cycles(adj, n, g):
    """
    Count cycles of length < g.
    """
    count = 0
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
                    count += 1
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return count

def _random_4_regular(n, seed):
    """
    Generate a random 4-regular simple graph on n vertices.
    Uses a robust configuration model with retries to ensure simplicity and regularity.
    """
    rng = random.Random(seed)
    
    # We need 2*n edges. n must be even for 4-regular? 
    # Sum of degrees = 4n. 4n must be even, so n can be anything.
    # But we are pairing stubs.
    
    for attempt in range(50):
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
    Use simulated annealing to increase girth.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    current_cost = _count_short_cycles(adj, n, g)
    if current_cost == 0:
        return edges, 0
        
    best_edges = list(edges)
    best_cost = current_cost
    
    t_start = time.time()
    temp = 50.0
    cooling = 0.999
    
    while time.time() < t_start + time_limit:
        # Pick two random edges
        e1 = rng.choice(edges)
        e2 = rng.choice(edges)
        if e1 == e2:
            continue
            
        u1, v1 = e1
        u2, v2 = e2
        
        # Try swap: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2)
        # Check validity
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
            
        # Check if new edges already exist
        if u2 in adj[u1] or v2 in adj[v1]:
            continue
            
        # Apply swap
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        new_e1 = (min(u1, u2), max(u1, u2))
        new_e2 = (min(v1, v2), max(v1, v2))
        
        adj[u1].add(u2)
        adj[u2].add(u1)
        adj[v1].add(v2)
        adj[v2].add(v1)
        
        edges.remove(e1)
        edges.remove(e2)
        edges.append(new_e1)
        edges.append(new_e2)
        
        new_cost = _count_short_cycles(adj, n, g)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_edges = list(edges)
            
        if new_cost <= current_cost or rng.random() < pow(2.718, -(new_cost - current_cost) / temp):
            current_cost = new_cost
        else:
            # Revert
            adj[u1].remove(u2)
            adj[u2].remove(u1)
            adj[v1].remove(v2)
            adj[v2].remove(v1)
            
            edges.remove(new_e1)
            edges.remove(new_e2)
            edges.append(e1)
            edges.append(e2)
            
        temp *= cooling
        
    return best_edges, best_cost

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # 1. Try LUW Graph
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. If k=4, g=10, try to find a smaller graph (Foster graph, n=320)
    if k == 4 and g == 10:
        # The Moore bound for (4,10) is 243.
        # We know 320 is the best known.
        # Trying to find a graph with n < 320 is extremely hard.
        # However, we can try to see if we can find a valid graph at n=320
        # to potentially beat the LUW size if LUW is larger.
        
        # LUW for (4,10) with d=5, q=4:
        # n = 2 * 4^5 = 2048.
        # So LUW is 2048. We can definitely do better.
        
        # Let's try to construct a graph of size 320.
        # Or even smaller?
        # The task says "as few vertices as possible".
        # If we can't find < 320, returning 320 is a huge improvement over 2048.
        
        target_n = 320
        rng = random.Random(42)
        
        # Try a few random starts
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            edges = _random_4_regular(target_n, seed=100 + attempt)
            if edges is None:
                continue
                
            # Check if it's already valid (unlikely)
            if _validate_graph(target_n, edges, k, g):
                best_n = target_n
                best_edges = edges
                break
                
            # Try to improve girth
            time_left = deadline - time.time()
            if time_left < 5:
                break
                
            improved_edges, cost = _improve_girth(target_n, edges, g, time_left - 2, rng)
            
            if cost == 0:
                if _validate_graph(target_n, improved_edges, k, g):
                    best_n = target_n
                    best_edges = improved_edges
                    break
                    
    return best_n, best_edges

# EVOLVE-BLOCK-END
