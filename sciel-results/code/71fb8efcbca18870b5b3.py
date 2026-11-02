import time
import random
import heapq
import sys

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

def _girth_check(adj, n, g):
    """Check if graph has girth >= g using non-backtracking walks (edge removal BFS)."""
    # Efficient check: For each edge (u,v), remove it and check if dist(u,v) < g-1.
    # If yes, cycle of length < g.
    
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    for u, v in edges:
        # BFS from u to v, not using edge (u,v)
        # If dist < g-1, then cycle length = dist + 1 < g
        visited = {u: 0}
        queue = [(u, 0)]
        found = False
        while queue:
            node, dist = queue.pop(0)
            if dist >= g - 1:
                continue
            for neighbor in adj[node]:
                # Skip the specific edge (u,v) in both directions
                if (node == u and neighbor == v) or (node == v and neighbor == u):
                    continue
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    if neighbor == v:
                        found = True
                        break
                    queue.append((neighbor, dist + 1))
            if found:
                break
        if found:
            return False
    
    return True

def _construct_via_switching(k, g, n, time_limit=10.0):
    start_time = time.time()
    
    # 1. Generate initial random k-regular graph
    adj = _generate_random_k_regular(n, k)
    if adj is None:
        return None
    
    # 2. Check initial girth
    if _girth_check(adj, n, g):
        edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
        return n, edges
    
    # 3. Switching loop
    # Strategy: Find a short cycle, pick two edges in it, and try to switch them with random other edges to break the cycle.
    # Or just random switches and check periodically.
    
    num_switches = 0
    last_check = time.time()
    
    # Precompute edge list for faster random selection?
    # adj is list of sets.
    
    while time.time() - start_time < time_limit:
        # Pick random edge (u,v)
        u = random.randint(0, n - 1)
        if not adj[u]:
            continue
        v = random.choice(list(adj[u]))
        
        # Pick random edge (w,x) not sharing u or v
        w = random.randint(0, n - 1)
        if w in (u, v) or not adj[w]:
            continue
        x = random.choice(list(adj[w]))
        if x in (u, v) or x == w:
            continue
        
        # Check if u-x and v-w exist (to avoid multi-edges)
        if x in adj[u] or w in adj[v]:
            continue
        
        # Perform switch
        adj[u].remove(v)
        adj[v].remove(u)
        adj[w].remove(x)
        adj[x].remove(w)
        
        adj[u].add(x)
        adj[x].add(u)
        adj[v].add(w)
        adj[w].add(v)
        
        num_switches += 1
        
        # Check girth periodically
        if num_switches % 20 == 0:
            if _girth_check(adj, n, g):
                edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
                return n, edges
    
    # Final check
    if _girth_check(adj, n, g):
        edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
        return n, edges
    
    return None

def _generate_random_k_regular(n, k):
    """Generate a random k-regular simple graph using configuration model + retries."""
    if k % 2 != 0 or (n * k) % 2 != 0:
        return None
    
    vertices = list(range(n))
    attempts = 500
    for _ in range(attempts):
        half_edges_list = []
        for v in vertices:
            half_edges_list.extend([v] * k)
        random.shuffle(half_edges_list)
        
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(half_edges_list), 2):
            u, v = half_edges_list[i], half_edges_list[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if valid:
            all_reg = True
            for i in range(n):
                if len(adj[i]) != k:
                    all_reg = False
                    break
            if all_reg:
                return adj
    
    return None

def construct(k, g):
    # Global deadline
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin for 150s limit
    
    # For the specific case k=5, g=9, try switching
    if k == 5 and g == 9:
        # Try a few sizes
        # The best known is 1116.
        # We try slightly larger to increase probability of success.
        target_ns = [1116, 1120, 1150, 1200, 1500]
        
        for n in target_ns:
            # Calculate remaining time
            remaining = deadline - time.time()
            if remaining < 5.0:
                break
            
            # Allocate a portion of remaining time, but not too much for one attempt
            # We want to try multiple n's if possible, but prioritize smaller n
            time_per_attempt = min(15.0, remaining * 0.6)
            
            res = _construct_via_switching(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
            
            # Check if we are running out of time
            if time.time() > deadline - 5.0:
                break
                
    # Fallback to LUW
    return _luw_construct(k, g)
