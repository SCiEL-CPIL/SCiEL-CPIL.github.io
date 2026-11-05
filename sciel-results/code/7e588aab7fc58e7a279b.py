"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque, defaultdict

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
    # Optimization: We only need to check up to depth g-1
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

def _find_shortest_cycle(adj, g_limit):
    """
    Finds the shortest cycle in the graph.
    Returns (length, path) or (inf, None) if no cycle < g_limit found.
    We just need the length to check validity, but path helps for switching.
    """
    n = len(adj)
    min_len = float('inf')
    min_path = None
    
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s, [s, nb]))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev, path = dq.popleft()
            depth = len(path) - 1
            
            if curr == s:
                if depth + 1 < min_len:
                    min_len = depth + 1
                    min_path = path
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append((nb, curr, path + [nb]))
                    
    return min_len, min_path

def _random_4_regular(n, rng):
    """
    Generates a random 4-regular simple graph on n vertices using Configuration Model.
    Returns list of edges or None if failed.
    """
    if n < 4 or (n * 4) % 2 != 0:
        return None
        
    stubs = []
    for v in range(n):
        for _ in range(4):
            stubs.append(v)
            
    edges = set()
    valid = True
    
    # Shuffle
    rng.shuffle(stubs)
    
    for i in range(0, n * 4, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            valid = False
            break
        edge = (min(u, v), max(u, v))
        if edge in edges:
            valid = False
            break
        edges.add(edge)
        
    if not valid:
        return None
        
    return list(edges)

def _switch_edge(adj, edges_set, u1, v1, u2, v2, rng):
    """
    Attempts to switch edges (u1,v1) and (u2,v2) to (u1,v2) and (u2,v1).
    Returns True if successful.
    """
    if u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1:
        return False
        
    # Check if new edges already exist
    if (min(u1, v2), max(u1, v2)) in edges_set:
        return False
    if (min(u2, v1), max(u2, v1)) in edges_set:
        return False
        
    # Remove old edges
    edges_set.discard((min(u1, v1), max(u1, v1)))
    edges_set.discard((min(u2, v2), max(u2, v2)))
    
    # Add new edges
    edges_set.add((min(u1, v2), max(u1, v2)))
    edges_set.add((min(u2, v1), max(u2, v1)))
    
    # Update adjacency
    adj[u1].discard(v1)
    adj[u1].add(v2)
    adj[v1].discard(u1)
    adj[v1].add(u2)
    
    adj[u2].discard(v2)
    adj[u2].add(v1)
    adj[v2].discard(u2)
    adj[v2].add(u1)
    
    return True

def _optimize_girth(n, k, g, time_limit, rng):
    """
    Starts with a random k-regular graph and tries to increase girth to >= g
    using edge switching.
    """
    start_time = time.time()
    
    # Initial random graph
    edges_list = None
    for _ in range(1000):
        edges_list = _random_4_regular(n, rng)
        if edges_list:
            break
            
    if not edges_list:
        return None, None
        
    edges_set = set(edges_list)
    adj = [set() for _ in range(n)]
    for u, v in edges_list:
        adj[u].add(v)
        adj[v].add(u)
        
    best_girth = 0
    best_edges = None
    
    for _ in range(5000): # Iterations
        if time.time() - start_time > time_limit:
            break
            
        # Check current girth
        # Fast check: just see if any cycle < g exists
        is_valid = True
        shortest_len, shortest_path = _find_shortest_cycle(adj, g)
        
        if shortest_len >= g:
            return n, list(edges_set)
            
        best_girth = max(best_girth, shortest_len)
        
        # If we have a short cycle, try to break it
        if shortest_path:
            # Pick a random edge in the cycle to switch
            # Cycle is a list of vertices [v0, v1, ..., v0]
            # Edges are (v0,v1), (v1,v2), etc.
            # We need another edge (u,v) not in the cycle to switch with.
            # For simplicity, pick a random edge from the whole graph and try to switch with a random edge in the cycle.
            
            cycle_edges = []
            for i in range(len(shortest_path) - 1):
                u = shortest_path[i]
                v = shortest_path[i+1]
                cycle_edges.append((u, v))
                
            # Pick a random edge in the cycle
            u1, v1 = rng.choice(cycle_edges)
            
            # Pick a random edge from the entire graph (not in cycle ideally, but any)
            # To ensure we don't just swap within the cycle, pick from all edges
            all_edges = list(edges_set)
            if not all_edges:
                continue
            u2, v2 = rng.choice(all_edges)
            
            # Ensure (u2,v2) is not the same as (u1,v1)
            if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                continue
                
            # Try the switch
            if _switch_edge(adj, edges_set, u1, v1, u2, v2, rng):
                # Continue loop to re-check girth
                pass
            else:
                # Switch failed, do nothing
                pass
                
    # If we didn't reach g, return the best found (or None)
    # For this task, we only return valid graphs
    return None, None

def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges).
    """
    q = k
    # We need girth >= g. LUW CD(d,q) has girth >= d+5.
    # So d+5 >= g => d >= g-5.
    # Also d must be odd for the standard construction?
    # The inspiration says "girth is at least d + 5 for odd d".
    # If g-5 is even, we might need d+1?
    # Let's use d = g - 5 if (g-5) % 2 == 1 else g - 4
    # Wait, if g=10, g-5=5 (odd). d=5. Girth >= 10.
    d = g - 5
    if d % 2 == 0:
        d += 1
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

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # Fallback: LUW graph
    n_luw, edges_luw = _luw_graph(k, g)
    
    # If k=4 and g=10, try to find a smaller graph (320 vertices)
    if k == 4 and g == 10:
        # Try random search with optimization for N=320
        # Time budget for search: 30 seconds
        search_time = 30.0
        rng = random.Random(42)
        
        # We can try a few different N values if we have time, but 320 is the target.
        # Moore bound for (4,10) is 1 + 4 + 4*3 + 4*3^2 + 4*3^3 = 1+4+12+36+108 = 161? 
        # Wait, Moore bound formula: 1 + k + k(k-1) + ... + k(k-1)^{d-1} for girth 2d+1?
        # For girth 10 (2d=10, d=5):
        # 1 + 4 + 4*3 + 4*3^2 + 4*3^3 + 4*3^4? No.
        # Bipartite Moore bound: 2 * (1 + (k-1) + ... + (k-1)^{g/2 - 1})
        # g=10, k=4.
        # 2 * (1 + 3 + 9 + 27) = 2 * 40 = 80?
        # Wait, the lower bound is 243.
        # Formula: n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        # i=0: 1
        # i=1: 3
        # i=2: 9
        # i=3: 27
        # i=4: 81
        # Sum = 121. 2*121 = 242.
        # So n >= 243 (since it must be even? 4-regular implies even n? No, 4n is even, n can be odd? No, sum of degrees 4n is even. If n is odd, 4n is even. But in a bipartite graph, partitions are equal size. So n must be even. 242 is even. So lower bound is 242?
        # Actually, the Moore bound is a lower bound.
        
        # Let's try N=320 first as it's the known best.
        n_target = 320
        time_left = deadline - time.time()
        
        if time_left > search_time:
            n_opt, edges_opt = _optimize_girth(n_target, k, g, time_left, rng)
            if n_opt and edges_opt:
                # Final validation
                if _validate_graph(n_opt, edges_opt, k, g):
                    return n_opt, edges_opt
        
        # If optimization failed, try a quick random check for N=320 (maybe 1000 tries)
        # But _optimize_girth already does this.
        
        # Fallback to LUW
        return n_luw, edges_luw
    else:
        return n_luw, edges_luw

# EVOLVE-BLOCK-END
