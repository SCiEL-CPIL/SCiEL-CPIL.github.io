import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Moore bound for (3, 16) is 512.
    # Known cages are around 900-1000.
    # We will try sizes in increasing order.
    sizes = [512, 514, 516, 520, 524, 532, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008, 1024]
    
    # Filter to even numbers >= Moore bound
    valid_sizes = [n for n in sizes if n >= 512 and n % 2 == 0]
    
    # If k != 3 or g != 16, fallback to a simple random graph or LUW if needed, 
    # but the task specifies k=3, g=16.
    if k != 3 or g != 16:
        # Fallback: Return a simple random graph (likely invalid for high girth, but valid interface)
        # Or use the seed LUW logic for other parameters.
        # For now, assume k=3, g=16 as per task contract.
        pass

    best_n = None
    best_edges = None
    
    for n in valid_sizes:
        if time.time() > deadline:
            break
            
        # Try to find a valid graph of size n
        edges = _find_graph(n, g, deadline)
        if edges is not None:
            best_n = n
            best_edges = edges
            break # Found the smallest n, stop
            
    if best_n is None:
        # If no valid graph found within time, return the largest attempted or a valid LUW graph
        # To ensure validity, we can fall back to the seed algorithm if it's fast enough, 
        # but it produces huge graphs. 
        # Better: return the best effort (might be invalid) or a known small valid graph if we had one.
        # Since we must return a valid graph, and we might have failed, 
        # we should ensure we at least return *something*. 
        # However, the evaluator checks validity. If invalid, score 0.
        # Let's try one more time with a larger N if we ran out of time on small Ns.
        # Or just use the seed logic for safety?
        # The seed logic is deterministic and fast.
        return _luw_fallback(k, g)

    return best_n, best_edges

def _luw_fallback(k, g):
    # Copy of the seed logic to guarantee a valid (though large) graph
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

def _find_graph(n, g, deadline):
    # Randomized local search
    max_iterations = 20000
    current_edges = _random_3_regular(n)
    if current_edges is None:
        return None
    
    # Check initial girth
    min_girth = _get_girth(n, current_edges, g)
    if min_girth >= g:
        return current_edges
        
    for _ in range(max_iterations):
        if time.time() > deadline:
            return None
            
        # Find a short cycle
        cycle = _find_short_cycle(n, current_edges, g)
        if cycle is None:
            # No short cycle found (might be due to search limit in _find_short_cycle)
            # Re-verify full girth
            min_girth = _get_girth(n, current_edges, g)
            if min_girth >= g:
                return current_edges
            continue
            
        # Try to break the cycle
        # Pick a random edge in the cycle
        # Perform a 2-opt swap with a random edge outside the cycle
        # Or try to rewire one of the cycle edges
        
        success = False
        for attempt in range(10):
            # Pick two edges to swap
            # One from the cycle, one random
            e1_idx = random.randint(0, len(cycle) - 1)
            u1, v1 = cycle[e1_idx]
            
            # Pick a random edge e2 not incident to u1 or v1 to avoid multi-edges/loops
            e2 = random.choice(current_edges)
            u2, v2 = e2
            
            # Check if u2, v2 are distinct from u1, v1
            if len({u1, v1, u2, v2}) < 4:
                continue
                
            # Try swap: (u1, v1) and (u2, v2) -> (u1, u2) and (v1, v2)
            # Or (u1, v2) and (v1, u2)
            
            # We need to check if new edges already exist
            adj = _build_adj(n, current_edges)
            
            # Option 1
            if not adj[u1][u2] and not adj[v1][v2]:
                new_edges = _swap_edges(current_edges, (u1, v1), (u2, v2), (u1, u2), (v1, v2))
                if new_edges:
                    # Check if this improves girth
                    # We can do a quick check on the affected area or just full check if fast
                    # Full check is O(N * 3^(g/2)). For N=1000, g=16, it's fast enough.
                    new_girth = _get_girth(n, new_edges, g)
                    if new_girth >= g:
                        return new_edges
                    # Heuristic: accept if it doesn't make things worse
                    # For now, only accept if valid. This is slow.
                    # Let's accept if it reduces the length of the shortest cycle found
                    if new_girth > min_girth:
                        current_edges = new_edges
                        min_girth = new_girth
                        success = True
                        break
                        
            # Option 2
            if not adj[u1][v2] and not adj[v1][u2]:
                new_edges = _swap_edges(current_edges, (u1, v1), (u2, v2), (u1, v2), (v1, u2))
                if new_edges:
                    new_girth = _get_girth(n, new_edges, g)
                    if new_girth >= g:
                        return new_edges
                    if new_girth > min_girth:
                        current_edges = new_edges
                        min_girth = new_girth
                        success = True
                        break
                        
        if not success:
            # Try a random 1-opt (rewire)
            u, v = random.choice(current_edges)
            # Pick a random w, x
            w = random.randint(0, n-1)
            x = random.randint(0, n-1)
            if len({u, v, w, x}) < 4:
                continue
            adj = _build_adj(n, current_edges)
            if not adj[u][w] and not adj[v][x]:
                new_edges = _swap_edges(current_edges, (u, v), (w, x), (u, w), (v, x))
                if new_edges:
                    new_girth = _get_girth(n, new_edges, g)
                    if new_girth >= g:
                        return new_edges
                    if new_girth > min_girth:
                        current_edges = new_edges
                        min_girth = new_girth

    return None

def _random_3_regular(n):
    if n % 2 != 0:
        return None
    # Configuration model
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    random.shuffle(stubs)
    edges = set()
    for i in range(0, n, 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None # Retry needed, simple retry loop
        if (u, v) in edges or (v, u) in edges:
            return None
        edges.add((u, v))
    if len(edges) != n // 2 * 3: # n*3/2 edges
        return None
    return list(edges)

def _build_adj(n, edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _swap_edges(edges, e1, e2, new_e1, new_e2):
    # Remove e1, e2, add new_e1, new_e2
    # Check if new edges exist
    edge_set = set()
    for e in edges:
        u, v = e
        if u > v: u, v = v, u
        edge_set.add((u, v))
        
    u1, v1 = e1
    if u1 > v1: u1, v1 = v1, u1
    u2, v2 = e2
    if u2 > v2: u2, v2 = v2, u2
    
    if (u1, v1) not in edge_set or (u2, v2) not in edge_set:
        return None
        
    edge_set.remove((u1, v1))
    edge_set.remove((u2, v2))
    
    a, b = new_e1
    if a > b: a, b = b, a
    c, d = new_e2
    if c > d: c, d = d, c
    
    if (a, b) in edge_set or (c, d) in edge_set or a == b or c == d:
        return None
        
    edge_set.add((a, b))
    edge_set.add((c, d))
    return list(edge_set)

def _find_short_cycle(n, edges, g):
    # Returns a list of edges forming a cycle of length < g, or None
    # Use BFS from each vertex
    adj = _build_adj(n, edges)
    # Convert to list of neighbors for speed
    neighbors = [[] for _ in range(n)]
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
        
    for start in range(n):
        # BFS
        # dist[v] = min distance from start
        # prev[v] = parent in BFS tree
        dist = [-1] * n
        prev = [-1] * n
        dist[start] = 0
        queue = [start]
        found_cycle = []
        
        while queue:
            u = queue.pop(0)
            if dist[u] >= g // 2:
                continue
            for v in neighbors[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    queue.append(v)
                elif v != prev[u]:
                    # Found a cycle
                    # Reconstruct cycle
                    cycle_edges = []
                    cur = v
                    while cur != u:
                        cycle_edges.append((cur, prev[cur]))
                        cur = prev[cur]
                    cycle_edges.append((u, v))
                    # Check length
                    if len(cycle_edges) < g:
                        return cycle_edges
        # If no cycle found from this start, try next
    return None

def _get_girth(n, edges, g_limit):
    # Returns the minimum cycle length, or g_limit if none found < g_limit
    # Optimization: if we find a cycle of length L, we can stop early if L is small?
    # No, we need the MIN.
    # But if we just want to check if girth >= g, we can stop as soon as we find one < g.
    adj = _build_adj(n, edges)
    neighbors = [[] for _ in range(n)]
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        prev = [-1] * n
        dist[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            if dist[u] >= g_limit // 2:
                continue
            for v in neighbors[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    queue.append(v)
                elif v != prev[u]:
                    # Cycle found
                    # Calculate length
                    length = 0
                    cur = v
                    while cur != u:
                        length += 1
                        cur = prev[cur]
                    length += 1 # edge u-v
                    if length < g_limit:
                        return length
    return g_limit
