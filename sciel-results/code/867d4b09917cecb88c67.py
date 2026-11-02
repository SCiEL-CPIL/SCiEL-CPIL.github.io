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

def _generate_random_regular(k, n):
    """Generates a random simple k-regular graph on n vertices."""
    while True:
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in edges:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
        
        if valid:
            # Convert to adjacency list
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            return adj

def _is_girth_at_least(adj, g):
    """Checks if graph has girth at least g."""
    n = len(adj)
    max_depth = g - 2  # We check cycles up to length g-1
    
    for v in range(n):
        visited = [False] * n
        dist = [-1] * n
        parent = [-1] * n
        queue = deque()
        queue.append(v)
        visited[v] = True
        dist[v] = 0
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for w in adj[u]:
                if not visited[w]:
                    visited[w] = True
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    queue.append(w)
                else:
                    if w != parent[u]:
                        # Cycle found
                        # Length is dist[u] + dist[w] + 1
                        # This assumes w is an ancestor, which is true for undirected BFS 
                        # if we don't mark "in queue" separately, but standard BFS 
                        # on undirected graphs: if w is visited and not parent, 
                        # it's a back edge to an ancestor.
                        cycle_len = dist[u] + dist[w] + 1
                        if cycle_len < g:
                            return False
    return True

def _switching_construct(k, g, n, time_limit=10.0):
    start_time = time.time()
    
    # 1. Generate initial random graph
    adj = _generate_random_regular(k, n)
    
    # 2. Check initial girth
    current_girth_ok = _is_girth_at_least(adj, g)
    if current_girth_ok:
        # Convert to edges list
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges

    # 3. Optimization loop
    # We accept a switch if it results in a graph with girth >= g
    # Or if it increases the "potential" (e.g. removes short cycles)
    # Simple hill climbing: Accept if new graph has girth >= g.
    # Since we start with girth < g, we need to find a sequence of switches 
    # that leads to girth >= g.
    # This is a hard constraint satisfaction problem.
    # Random switching might get stuck.
    # We can try to accept switches that *increase* the girth.
    # But calculating exact girth is expensive.
    # We use _is_girth_at_least which is fast.
    
    # Strategy: 
    # Try random switches. 
    # If the new graph has girth >= g, accept.
    # If not, reject.
    # This is a very strict filter. 
    # It might be too strict to ever succeed from a random start.
    
    # Better: 
    # Track the number of "bad" cycles (cycles < g).
    # Minimize this count.
    # But counting is expensive.
    
    # Alternative: 
    # Use the fact that we just need *a* valid graph.
    # We can try to "repair" short cycles.
    # If a short cycle is found, try to switch an edge in it to break it.
    
    # For now, let's try the strict filter with a large number of attempts.
    # If it fails, we return None.
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        
        # Pick a random edge (u, v)
        # We need a list of edges for random access
        # Rebuilding edge list every step is O(n). 
        # We can maintain a list of edges.
        # But switching changes edges.
        # Let's just pick random u, then random v in adj[u].
        
        u = random.randint(0, n - 1)
        if not adj[u]:
            continue
        v = random.choice(list(adj[u]))
        
        # Pick a random edge (x, y) not incident to u, v
        # To avoid bias, pick random x, then random y in adj[x]
        # Ensure x, y not in {u, v} and (x,y) != (u,v)
        
        found_switch = False
        for _ in range(10): # Try a few times to find a valid partner edge
            x = random.randint(0, n - 1)
            if x in (u, v):
                continue
            if not adj[x]:
                continue
            y = random.choice(list(adj[x]))
            if y in (u, v):
                continue
            if x == y:
                continue
            # Ensure (x,y) is not (u,v)
            if (x == u and y == v) or (x == v and y == u):
                continue
            
            # Potential switch: (u,v) and (x,y) -> (u,x) and (v,y)
            # Or (u,y) and (v,x)
            # Check if new edges exist
            if x in adj[u] or y in adj[v]:
                continue
            
            # Try switch 1: (u,x) and (v,y)
            # Remove (u,v) and (x,y)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            # Add (u,x) and (v,y)
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            # Check girth
            if _is_girth_at_least(adj, g):
                found_switch = True
                break
            else:
                # Revert
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                adj[u].add(v)
                adj[v].add(u)
                adj[x].add(y)
                adj[y].add(x)
                
        if found_switch:
            # Success
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
    return None

def construct(k, g):
    # For the specific case k=5, g=9
    if k == 5 and g == 9:
        # Try switching for a few sizes
        # Time budget: 100s total
        # We try 1116 first.
        target_ns = [1116, 1120, 1150]
        
        time_per_attempt = 30.0
        
        for n in target_ns:
            res = _switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
                
    # Fallback to LUW
    return _luw_construct(k, g)
