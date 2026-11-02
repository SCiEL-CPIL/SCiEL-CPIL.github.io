import time
import random
import heapq
from collections import deque

def _compute_girth_fast(adjacency, n, target_girth):
    """
    Compute the girth of the graph.
    Returns the length of the shortest cycle.
    If the shortest cycle is >= target_girth, returns target_girth (or larger).
    We only care if girth >= target_girth.
    """
    # To check if girth >= g, we need to ensure no cycle of length < g.
    # This is equivalent to checking that the graph has no cycles of length 3, ..., g-1.
    # We can do a BFS from each vertex up to depth (g-1)//2.
    # If we find a cycle, we return its length.
    
    min_girth = target_girth + 1
    
    # For each vertex, do a BFS to find the shortest cycle involving that vertex.
    # Actually, finding the global girth requires checking all vertices.
    # Optimization: If we find a cycle of length L, we can stop if L >= target_girth?
    # No, we need to know if there is a SHORTER cycle.
    # So we must find the MINIMUM cycle length.
    
    # Efficient BFS for girth:
    # For each start vertex s:
    #   BFS from s, keeping track of the parent.
    #   If we encounter a visited vertex v (not parent), we found a cycle.
    #   Cycle length = dist[u] + dist[v] + 1.
    #   Update min_girth.
    #   If min_girth >= target_girth, we can stop early? 
    #   No, a later start vertex might have a shorter cycle.
    #   However, if we are searching for a graph with girth >= g, we just need to verify it.
    #   For the switching heuristic, we need to know if the girth DECREASED.
    #   So we need the exact girth (or at least if it's < g).
    
    # Let's just compute the exact girth, but stop if it drops below some threshold?
    # No, we need to know if it's >= g.
    
    # We can prune the BFS if the current distance exceeds min_girth/2?
    # If dist[u] + dist[v] + 1 >= min_girth, it won't improve.
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            
            # If we've already found a cycle of length L, 
            # any new cycle found from this start vertex with length >= L is not useful.
            # But we need to check if there's a shorter one.
            
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    # Found a cycle
                    cycle_len = du + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        # If we found a very short cycle, we can break early?
                        # If min_girth < target_girth, we know the graph is bad.
                        # But we need the exact value for the switching decision?
                        # The switching decision is: accept if new_girth >= current_girth.
                        # So we need to compare.
                        
        # Early termination for this start vertex if min_girth is already small?
        # No.
        
        # Global early termination:
        # If min_girth == 3, it's the minimum possible. Stop.
        if min_girth == 3:
            return 3
            
    return min_girth

def _random_5_regular(n):
    """Generate a random 5-regular graph on n vertices."""
    if n * 5 % 2 != 0:
        return None
    
    stubs = []
    for i in range(n):
        stubs.extend([i] * 5)
    random.shuffle(stubs)
    
    adjacency = [set() for _ in range(n)]
    i = 0
    while i < len(stubs):
        u, v = stubs[i], stubs[i + 1]
        i += 2
        if u == v or v in adjacency[u]:
            return None
        adjacency[u].add(v)
        adjacency[v].add(u)
        
    return adjacency

def _switch_improves_girth(adjacency, n, u1, v1, u2, v2, current_girth):
    """Check if switching edges (u1,v1) and (u2,v2) to (u1,u2) and (v1,v2) 
    maintains or improves girth."""
    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
        return False
        
    # Apply switch
    adjacency[u1].remove(v1)
    adjacency[v1].remove(u1)
    adjacency[u2].remove(v2)
    adjacency[v2].remove(u2)
    
    adjacency[u1].add(u2)
    adjacency[u2].add(u1)
    adjacency[v1].add(v2)
    adjacency[v2].add(v1)
    
    # Compute new girth
    new_girth = _compute_girth_fast(adjacency, n, current_girth)
    
    # Revert
    adjacency[u1].remove(u2)
    adjacency[u2].remove(u1)
    adjacency[v1].remove(v2)
    adjacency[v2].remove(v1)
    
    adjacency[u1].add(v1)
    adjacency[v1].add(u1)
    adjacency[u2].add(v2)
    adjacency[v2].add(u2)
    
    return new_girth >= current_girth

def _edge_switching_optimized(k, g, n, time_limit=60.0):
    start_time = time.time()
    adjacency = None
    
    # Try to get a good initial graph
    for _ in range(100):
        if time.time() - start_time > time_limit:
            return None
        adjacency = _random_5_regular(n)
        if adjacency is not None:
            break
            
    if adjacency is None:
        return None
        
    current_girth = _compute_girth_fast(adjacency, n, g)
    
    if current_girth >= g:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
        
    # Edge switching
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() - start_time < time_limit:
        # Pick two random edges
        # Efficient edge list generation?
        # For n=1116, edges = 2790.
        # Generating list every time is O(E).
        
        edge_list = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edge_list.append((i, j))
                    
        if len(edge_list) < 2:
            break
            
        idx1 = random.randint(0, len(edge_list) - 1)
        idx2 = random.randint(0, len(edge_list) - 1)
        
        if idx1 == idx2:
            continue
            
        u1, v1 = edge_list[idx1]
        u2, v2 = edge_list[idx2]
        
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
            
        if _switch_improves_girth(adjacency, n, u1, v1, u2, v2, current_girth):
            # Apply switch
            adjacency[u1].remove(v1)
            adjacency[v1].remove(u1)
            adjacency[u2].remove(v2)
            adjacency[v2].remove(u2)
            
            adjacency[u1].add(u2)
            adjacency[u2].add(u1)
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
            
            no_improve = 0
            current_girth = _compute_girth_fast(adjacency, n, g)
            
            if current_girth >= g:
                edges = []
                for i in range(n):
                    for j in adjacency[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Restart
                adjacency = None
                for _ in range(10):
                    if time.time() - start_time > time_limit:
                        return None
                    adjacency = _random_5_regular(n)
                    if adjacency is not None:
                        break
                if adjacency is None:
                    return None
                current_girth = _compute_girth_fast(adjacency, n, g)
                if current_girth >= g:
                    edges = []
                    for i in range(n):
                        for j in adjacency[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
                no_improve = 0
                max_no_improve = 500
                
    return None

def _luw_construct(k, g):
    # Fallback to LUW if switching fails
    # This is slow and produces large graphs, so we use it only as last resort
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
    if k == 5 and g == 9:
        # Try to find a 1116-vertex graph
        target_n = 1116
        res = _edge_switching_optimized(k, g, target_n, time_limit=100.0)
        if res is not None:
            return res
            
        # If it fails, try a slightly larger n to ensure success?
        # No, we want to minimize n.
        # If 1116 fails, we return the LUW graph (which is larger).
        
    return _luw_construct(k, g)
