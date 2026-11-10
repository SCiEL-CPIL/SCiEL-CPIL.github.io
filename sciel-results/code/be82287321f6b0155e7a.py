import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # 31

    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append((i, j))

    # Use bitmasks for adjacency
    # adj[i] is an integer where bit j is 1 if (i,j) is an edge
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        
    # Function to count 4-cycles
    def count_4cycles(adj_list):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = adj_list[i] & adj_list[j]
                c = common.bit_count()
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2

    base_count = count_4cycles(adj)
    
    # Add the best 91st edge
    # We need to find the non-edge that minimizes new C4s
    best_edge = None
    best_cost = float('inf')
    
    # Precompute neighbors as lists for iteration in greedy step
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in edge_set:
                continue
            cost = 0
            # For each neighbor x of i, count how many neighbors of x are in neighbors[j]
            for x in neighbors[i]:
                common = neighbors[x] & neighbors[j]
                cost += len(common)
                
            if cost < best_cost:
                best_cost = cost
                best_edge = (i, j)

    if best_edge is not None:
        u, v = best_edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        neighbors[u].add(v)
        neighbors[v].add(u)
        edge_set.add((u, v))
        edges.append((u, v))

    current_count = count_4cycles(adj)

    # Local Search
    # Swap edge (u,v) with non-edge (x,y)
    # Delta = C4(new) - C4(old)
    
    def delta_swap(u, v, x, y):
        # We remove (u,v) and add (x,y)
        
        # Save state
        old_u = adj[u]
        old_v = adj[v]
        old_x = adj[x]
        old_y = adj[y]
        
        # Apply change
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj[x] |= (1 << y)
        adj[y] |= (1 << x)
        
        c_new = count_4cycles(adj)
        
        # Restore
        adj[u] = old_u
        adj[v] = old_v
        adj[x] = old_x
        adj[y] = old_y
        
        return c_new - current_count

    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        best_delta = 0
        best_swap = None
        
        edge_list = list(edge_set)
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    non_edges.append((i, j))
        
        for e1 in edge_list:
            if time.time() > deadline - 10:
                break
            u, v = e1
            for e2 in non_edges:
                x, y = e2
                d = delta_swap(u, v, x, y)
                if d < best_delta:
                    best_delta = d
                    best_swap = (e1, e2)
                    
        if best_swap:
            e1, e2 = best_swap
            u, v = e1
            x, y = e2
            
            # Apply swap
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            
            neighbors[u].discard(v)
            neighbors[v].discard(u)
            neighbors[x].add(y)
            neighbors[y].add(x)
            
            edge_set.discard(e1)
            edge_set.add(e2)
            
            edges = [list(e) for e in edge_set]
            current_count += best_delta
            improved = True
        else:
            break

    final_edges = [list(e) for e in edge_set]
    return final_edges[:91]
