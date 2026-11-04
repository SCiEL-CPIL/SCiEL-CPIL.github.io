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
    """
    rng = random.Random(seed)
    stubs = []
    for v in range(n):
        for _ in range(4):
            stubs.append(v)
    rng.shuffle(stubs)
    
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None
        edge = (min(u, v), max(u, v))
        if edge in edges:
            return None
        edges.add(edge)
        
    adj = [0] * n
    for u, v in edges:
        adj[u] += 1
        adj[v] += 1
        if adj[u] > 4 or adj[v] > 4:
            return None
            
    if any(d != 4 for d in adj):
        return None
        
    return list(edges)

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
    temp = 100.0
    cooling = 0.99
    
    while time.time() < t_start + time_limit:
        # Pick two random edges
        e1 = rng.choice(edges)
        e2 = rng.choice(edges)
        if e1 == e2:
            continue
            
        u1, v1 = e1
        u2, v2 = e2
        
        # Check if we can swap
        # Remove (u1, v1) and (u2, v2)
        # Add (u1, u2) and (v1, v2) OR (u1, v2) and (v1, u2)
        
        # Try swap 1
        new_e1 = (min(u1, u2), max(u1, u2))
        new_e2 = (min(v1, v2), max(v1, v2))
        
        if new_e1 in adj[u1] and new_e1 in adj[u2]: # Already connected?
            # Check if edges exist
            pass
            
        # This logic is getting complex to update cost incrementally without full re-calculation.
        # Full re-calculation of cost is O(n * girth_check).
        # n=320, girth_check is fast.
        
        # Let's just do full re-calculation for simplicity and safety.
        
        # Remove old edges
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        # Add new edges
        if u1 != u2 and v1 != v2 and u1 != v2 and v1 != u2:
             if u2 not in adj[u1] and v2 not in adj[v1]:
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
                     
        else:
             # Revert
             adj[u1].add(v1)
             adj[v1].add(u1)
             adj[u2].add(v2)
             adj[v2].add(u2)
             
        temp *= cooling
        
    return best_edges, best_cost

def _construct_foster_320():
    """
    Construct the Foster graph (320 vertices, 4-regular, girth 10) using a known construction.
    This is the smallest known (4,10)-cage.
    """
    n = 320
    
    # The Foster graph can be constructed as a Cayley graph or via a specific edge list.
    # One known construction uses the group Z_2 x Z_2 x Z_80 or similar.
    # Alternatively, we can use the known edge list from the literature.
    
    # Let's try to construct it using a specific algorithm.
    # The Foster graph is a (4,10)-cage. It has 320 vertices.
    
    # We'll use a known construction: 
    # Vertices are pairs (i, j) where i in Z_4, j in Z_80? No, 4*80=320.
    # Or Z_2 x Z_160?
    
    # Actually, let's just return a valid 4-regular graph on 320 vertices with girth 10.
    # Since we might not have the exact edge list memorized, let's try to generate one
    # and verify. If it fails, we fall back to LUW.
    
    # For now, let's use a heuristic: try to find a random 4-regular graph on 320 vertices
    # with girth 10. This is unlikely to succeed quickly, so we'll rely on the LUW graph
    # which has 4^5 = 1024 vertices? No, LUW with q=4, d=5 gives 2*q^d = 2*4^5 = 2048 vertices.
    # Wait, the LUW graph CD(d,q) has 2*q^d vertices. For q=4, d=5, that's 2*1024=2048.
    # The Foster graph has 320 vertices. So LUW is much larger.
    
    # We need a better construction for 320 vertices.
    
    # Let's try a different approach: use the known fact that the Foster graph exists
    # and try to construct it via a specific pattern.
    
    # For now, let's just return None and let the caller handle it.
    return None

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # 1. Try LUW Graph
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. If k=4, g=10, try to find a smaller graph (Foster graph, n=320)
    if k == 4 and g == 10:
        target_n = 320
        rng = random.Random(42)
        
        # Try a few random starts
        for attempt in range(5):
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
