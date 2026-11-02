import time
import random
import heapq
from collections import deque

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _compute_girth(adj, n, max_depth):
    """Compute the girth of the graph using BFS from each vertex."""
    girth = max_depth + 1
    for start in range(n):
        # BFS from start, tracking distances
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif dist[v] > dist[u] + 1:
                    # Found a shorter path, update
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif dist[v] == dist[u] + 1:
                    # Found a cycle of length dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth:
                        girth = cycle_len
    return girth

def _try_switch(adj, n, u, v, w, x, g):
    """Try to switch edges (u,v) and (w,x) to (u,w) and (v,x) or (u,x) and (v,w).
    Returns True if switch was applied and girth is maintained or improved."""
    # Check if switch is valid (no parallel edges, no loops)
    if u == w or u == x or v == w or v == x:
        return False
    if w in adj[u] or x in adj[u] or w in adj[v] or x in adj[v]:
        return False
    
    # Compute current girth
    current_girth = _compute_girth(adj, n, g - 1)
    
    # Try both switch options
    for option in [(u, w, v, x), (u, x, v, w)]:
        a, b, c, d = option
        # Apply switch
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        
        adj[a].add(b)
        adj[b].add(a)
        adj[c].add(d)
        adj[d].add(c)
        
        # Wait, I made an error. Let me redo this.
        # Original edges: (u,v) and (w,x)
        # Option 1: replace with (u,w) and (v,x)
        # Option 2: replace with (u,x) and (v,w)
        
        # Undo the switch
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        
        adj[a].add(b)
        adj[b].add(a)
        adj[c].add(d)
        adj[d].add(c)
        
        # Now apply correctly
        if option == (u, w, v, x):
            # Remove (u,v) and (w,x), add (u,w) and (v,x)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            adj[u].add(w)
            adj[w].add(u)
            adj[v].add(x)
            adj[x].add(v)
        else:
            # Remove (u,v) and (w,x), add (u,x) and (v,w)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(w)
            adj[w].add(v)
        
        new_girth = _compute_girth(adj, n, g - 1)
        
        if new_girth >= g:
            return True
        else:
            # Revert
            if option == (u, w, v, x):
                adj[u].remove(w)
                adj[w].remove(u)
                adj[v].remove(x)
                adj[x].remove(v)
                
                adj[u].add(v)
                adj[v].add(u)
                adj[w].add(x)
                adj[x].add(w)
            else:
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(w)
                adj[w].remove(v)
                
                adj[u].add(v)
                adj[v].add(u)
                adj[w].add(x)
                adj[x].add(w)
    
    return False

def _greedy_construct(k, g, n, time_limit=10.0):
    """Construct a k-regular graph of girth >= g using edge switching."""
    start_time = time.time()
    
    # Generate a random k-regular graph using configuration model
    # Each vertex has k stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    # Randomly pair stubs to form edges
    random.shuffle(stubs)
    adj = [set() for _ in range(n)]
    valid = True
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
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
    
    # Verify regularity
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    # Now try to improve girth using edge switches
    # We'll do local search: find short cycles and try to break them
    
    max_iterations = 10000
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Compute current girth
        current_girth = _compute_girth(adj, n, g - 1)
        if current_girth >= g:
            # We have a valid graph
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
        
        # Find a short cycle and try to break it
        # For simplicity, pick random edges and try to switch
        all_edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    all_edges.append((i, j))
        
        random.shuffle(all_edges)
        
        # Try a few random switches
        for _ in range(10):
            if time.time() - start_time > time_limit:
                break
            
            # Pick two random edges
            if len(all_edges) < 2:
                continue
            e1, e2 = random.sample(all_edges, 2)
            u, v = e1
            w, x = e2
            
            if _try_switch(adj, n, u, v, w, x, g):
                # Switch was successful, update all_edges
                all_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            all_edges.append((i, j))
                break
    
    # Final check
    final_girth = _compute_girth(adj, n, g - 1)
    if final_girth >= g:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return None

def construct(k, g):
    # For the specific case k=5, g=9, try greedy
    if k == 5 and g == 9:
        # Try a few sizes
        # The best known is 1116.
        # We try slightly larger to increase probability of success.
        target_ns = [1116, 1120, 1150, 1200, 1500]
        
        # Time budget: 100s for greedy, 10s for fallback
        time_per_attempt = 10.0 
        
        for n in target_ns:
            res = _greedy_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
                
    # Fallback to LUW
    return _luw_construct(k, g)
