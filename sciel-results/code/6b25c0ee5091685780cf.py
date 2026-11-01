import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_n = None
    best_edges = None
    
    # Target sizes: Start near Moore bound, go up.
    # Moore bound 766.
    # Try 800, 900, 936, 1000, 1200, 1500
    target_sizes = [800, 900, 936, 1000, 1200, 1500]
    
    for n in target_sizes:
        if time.time() > deadline - 5:
            break
            
        # Try to find a valid graph of size n
        found = False
        # Limit time per n to 20s
        n_deadline = min(deadline - 5, time.time() + 20.0)
        
        attempts = 0
        while time.time() < n_deadline and attempts < 10:
            attempts += 1
            edges = generate_random_3_reg(n)
            if edges is None:
                continue
                
            repaired = repair_girth(edges, n, g, n_deadline)
            if repaired is not None:
                # Final verification
                if compute_girth(repaired, n, g) >= g:
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = repaired
                    found = True
                    break
        
        if found:
            break # Found minimal n in our list
            
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback: LUW Graph
    return luw_construct(k, g)

def generate_random_3_reg(n):
    if n < 4 or n % 2 != 0: # 3n must be even -> n must be even? No, 3n even -> n even.
        return None
        
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    edges = []
    adj = [set() for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop
        if v in adj[u]:
            return None # Multi-edge
        edges.append((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
    return edges

def compute_girth(edges, n, limit):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    min_girth = limit + 1
    
    for start in range(n):
        # BFS
        visited = {start: (None, 0)}
        queue = [(start, None, 0)]
        
        while queue:
            node, parent, depth = queue.pop(0)
            if depth >= limit // 2: # Girth 2*depth
                continue
                
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    # Cycle found
                    # Reconstruct? We just need length.
                    # Length is depth + visited[neighbor][1] + 1?
                    # No, BFS tree.
                    # If we hit a visited node, the cycle length is depth + 1 + visited[neighbor][1]?
                    # Not exactly.
                    # We need to find the LCA.
                    # Easier: just return a specific cycle found by DFS/BFS parent tracking.
                    pass
                else:
                    visited[neighbor] = (node, depth + 1)
                    queue.append((neighbor, node, depth + 1))
                    
    # This BFS girth check is tricky to implement correctly for "any" cycle length quickly.
    # Let's use a simpler "find one short cycle" function for repair.
    return limit + 1 # Placeholder, we use find_short_cycle for repair

def find_short_cycle(adj, n, max_len):
    # Returns a list of nodes in a cycle of length < max_len, or None
    for start in range(n):
        # BFS
        parent = {start: None}
        depth = {start: 0}
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            if depth[node] >= max_len // 2:
                continue
                
            for neighbor in adj[node]:
                if neighbor == parent[node]:
                    continue
                if neighbor in parent:
                    # Cycle found
                    # Reconstruct
                    cycle = []
                    curr = node
                    while curr != neighbor:
                        cycle.append(curr)
                        curr = parent[curr]
                    cycle.append(neighbor)
                    cycle.reverse() # Not strictly necessary for length
                    if len(cycle) < max_len:
                        return cycle
                else:
                    parent[neighbor] = node
                    depth[neighbor] = depth[node] + 1
                    queue.append(neighbor)
    return None

def repair_girth(edges, n, g, deadline):
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
        
    # We need to keep edges as a list for output, but use adj for speed.
    # Modify adj and edge_set.
    
    while time.time() < deadline:
        cycle = find_short_cycle(adj, n, g)
        if cycle is None:
            # No short cycle found
            # Reconstruct edges from adj
            final_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        final_edges.append((u, v))
            return final_edges
            
        # Pick 2 edges on cycle
        L = len(cycle)
        # Pick i, j
        # Try a few random pairs
        for _ in range(20):
            i = random.randint(0, L-1)
            j = random.randint(0, L-1)
            if i == j or abs(i-j) == 1 or abs(i-j) == L-1:
                continue
                
            u1 = cycle[i]
            v1 = cycle[(i+1)%L]
            u2 = cycle[j]
            v2 = cycle[(j+1)%L]
            
            # Swap (u1, v1) and (u2, v2) with (u1, u2) and (v1, v2)
            # Check validity
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
            e_new1 = (min(u1, u2), max(u1, u2))
            e_new2 = (min(v1, v2), max(v1, v2))
            
            if e_new1 in edge_set or e_new2 in edge_set:
                continue
                
            # Apply swap
            edge_set.remove((min(u1, v1), max(u1, v1)))
            edge_set.remove((min(u2, v2), max(u2, v2)))
            edge_set.add(e_new1)
            edge_set.add(e_new2)
            
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].add(u2)
            adj[u2].add(u1)
            adj[v1].add(v2)
            adj[v2].add(v1)
            
            break # Restart cycle search
            
    return None # Timeout

def luw_construct(k, g):
    # Copy from seed
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
