# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 8.0

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
    Generate a random 4-regular simple graph on n vertices using configuration model
    with rejection sampling for loops and multi-edges.
    """
    rng = random.Random(seed)
    
    for attempt in range(200):
        # Create stubs
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
    Use greedy hill climbing to increase girth.
    We accept a move if it does not create any NEW short cycles.
    Uses a dictionary-based edge storage for O(1) removal.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Use a set for O(1) membership, and a list for random access
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    edge_list = list(edge_set)
    
    t_start = time.time()
    deadline = t_start + time_limit
    
    # Precompute neighbors for faster path checking
    # We'll use BFS with early termination
    
    while time.time() < deadline:
        if len(edge_list) < 2:
            break
            
        # Pick two random edges by index for O(1) access
        i1 = rng.randrange(len(edge_list))
        i2 = rng.randrange(len(edge_list))
        if i1 == i2:
            continue
            
        e1 = edge_list[i1]
        e2 = edge_list[i2]
        
        u1, v1 = e1
        u2, v2 = e2
        
        # Try swap: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2)
        # Check validity: no self-loops, no multi-edges
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
            
        # Check if new edges already exist
        if u2 in adj[u1] or v2 in adj[v1]:
            continue
            
        # Temporarily remove old edges
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        # Check for short cycles involving new edge (u1, u2)
        creates_cycle_1 = _has_short_path(adj, u1, u2, g - 2)
        
        if not creates_cycle_1:
            # Check for short cycles involving new edge (v1, v2)
            creates_cycle_2 = _has_short_path(adj, v1, v2, g - 2)
            
            if not creates_cycle_2:
                # Safe to apply - update data structures
                adj[u1].add(u2)
                adj[u2].add(u1)
                adj[v1].add(v2)
                adj[v2].add(v1)
                
                # Update edge storage: remove old, add new
                new_e1 = (min(u1, u2), max(u1, u2))
                new_e2 = (min(v1, v2), max(v1, v2))
                
                # Remove old edges from set and list
                edge_set.discard(e1)
                edge_set.discard(e2)
                edge_list[i1] = new_e1
                edge_list[i2] = new_e2
                edge_set.add(new_e1)
                edge_set.add(new_e2)
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
            
    return list(edge_set)

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
    
    # For k=4, g=10: moore_bound = 2 + 4*(1+3+9+27+81) = 2 + 4*121 = 486
    # But the known best is 320, so we should search around that range
    # The LUW graph for k=4,g=10 gives n=320 (q=4, d=5, so 2*4^5 = 2048? Let me recalculate)
    # Actually LUW CD(d,q) has n = 2*q^d vertices
    # For k=4, g=10: d = g-5 = 5, q = k = 4, so n = 2*4^5 = 2048
    # That's much larger than 320. We need a better search strategy.
    
    # Better approach: search in a range around known good values
    # Known best for (4,10) is 320
    # Search from moore_bound up to min(n_luw, 400) to keep it manageable
    
    max_search_n = min(n_luw, max(moore_bound + 50, 400))
    
    rng = random.Random(42)
    
    # Try sizes from moore_bound upward
    current_n = moore_bound
    # Ensure current_n has correct parity for regular graph?
    # Sum of degrees = n*k must be even.
    # If k is even, n can be anything.
    # If k is odd, n must be even.
    if k % 2 == 1 and current_n % 2 != 0:
        current_n += 1
    
    # Allocate time budget: spend more time on promising sizes
    total_time_budget = deadline - time.time()
    if total_time_budget <= 0:
        return best_n, best_edges
    
    # Try a limited number of sizes, spending more time per size
    num_sizes_to_try = min(20, max(3, int(total_time_budget / 5)))
    
    for size_idx in range(num_sizes_to_try):
        if time.time() >= deadline:
            break
            
        # Try current_n, then increment
        n_try = current_n + size_idx * (2 if k % 2 == 1 else 1)
        
        if n_try > max_search_n or n_try >= best_n:
            break
            
        # Time remaining for this size
        time_left = deadline - time.time()
        if time_left < 3:
            break
            
        # Allocate time: more time for earlier (smaller) sizes
        time_for_this_size = min(15.0, time_left * 0.6)
        
        found = False
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() >= deadline - 2:
                break
                
            edges = _random_4_regular(n_try, seed=1000 + n_try + attempt * 7)
            if edges is None:
                continue
                
            # Quick check: if random graph is already valid (unlikely)
            if _validate_graph(n_try, edges, k, g):
                best_n = n_try
                best_edges = edges
                found = True
                break
                
            # Try to improve girth
            time_left_improve = deadline - time.time() - 2
            if time_left_improve < 2:
                break
                
            improved_edges = _improve_girth(n_try, edges, g, time_for_this_size / 3.0, rng)
            
            if _validate_graph(n_try, improved_edges, k, g):
                best_n = n_try
                best_edges = improved_edges
                found = True
                break
                
        if found:
            # Found a valid graph at this size - this is our best so far
            # We could try smaller sizes, but we started at moore_bound
            # So this is the smallest possible (or close to it)
            break
        
    return best_n, best_edges
# EVOLVE-BLOCK-END
