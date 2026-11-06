import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n = 20
    
    # Initialize with C_20(1, 2, 3)
    adj = [0] * n
    for i in range(n):
        for k in [1, 2, 3]:
            j1 = (i + k) % n
            j2 = (i - k) % n
            adj[i] |= (1 << j1)
            adj[i] |= (1 << j2)
            adj[j1] |= (1 << i)
            adj[j2] |= (1 << i)
            
    for i in range(n):
        adj[i] &= ~(1 << i)

    def count_edges(adj_list):
        return sum(bin(x).count('1') for x in adj_list) // 2

    def count_violations(adj_list):
        violations = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_list[u] & adj_list[v]
                if bin(common).count('1') >= 3:
                    violations += 1
        return violations

    current_violations = count_violations(adj)
    current_edges = count_edges(adj)
    current_score = -100 * current_violations + current_edges
    
    best_valid_adj = [0] * n
    best_valid_edges = 0
    best_valid_score = -10**9
    
    # If initial is valid, update best
    if current_violations == 0:
        best_valid_adj = adj[:]
        best_valid_edges = current_edges
        best_valid_score = current_score

    T = 10.0
    cooling = 0.9995
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    while time.time() - start_time < deadline:
        # Generate move
        if random.random() < 0.8:
            u, v = random.choice(all_edges)
            if adj[u] & (1 << v):
                # Remove
                new_adj = adj[:]
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
            else:
                # Add
                new_adj = adj[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
        else:
            e1, e2 = random.sample(all_edges, 2)
            u, v = e1
            x, y = e2
            if len({u, v, x, y}) < 4:
                continue
            if not (adj[u] & (1 << v)) or not (adj[x] & (1 << y)):
                continue
            if (adj[u] & (1 << x)) or (adj[v] & (1 << y)):
                continue
            new_adj = adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            new_adj[x] &= ~(1 << y)
            new_adj[y] &= ~(1 << x)
            new_adj[u] |= (1 << x)
            new_adj[x] |= (1 << u)
            new_adj[v] |= (1 << y)
            new_adj[y] |= (1 << v)

        new_violations = count_violations(new_adj)
        new_edges = count_edges(new_adj)
        new_score = -100 * new_violations + new_edges
        delta = new_score - current_score
        
        if delta > 0 or random.random() < pow(2.71828, delta / T):
            adj = new_adj
            current_score = new_score
            current_violations = new_violations
            current_edges = new_edges
            
            if current_violations == 0:
                if current_edges > best_valid_edges:
                    best_valid_edges = current_edges
                    best_valid_adj = adj[:]
                    best_valid_score = current_score
                    
        T *= cooling
        if T < 0.5:
            T = 0.5

    # If we didn't find any valid graph, we need a fallback.
    # A simple valid graph: Empty graph (0 edges).
    # Or a sparse random graph.
    # Let's try to construct a valid graph by adding edges greedily to an empty graph.
    if best_valid_edges == 0:
        # Greedy construction
        G = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                # Check if adding (u,v) creates a violation
                # A violation is a pair (a,b) with >=3 common neighbors.
                # Adding (u,v) affects common neighbors of pairs involving u or v.
                # Specifically, for any w, common(u, w) increases by 1 if v is in N(w).
                # common(v, w) increases by 1 if u is in N(w).
                # Also common(u, v) is the number of common neighbors of u and v.
                
                # Check all pairs (a,b) that could be violated.
                # Only pairs involving u or v can change their common neighbor count.
                # Pairs: (u, v), (u, w), (v, w) for all w != u, v.
                
                valid = True
                # Check (u, v)
                common_uv = 0
                for w in range(n):
                    if w != u and w != v:
                        if (G[u] & (1 << w)) and (G[v] & (1 << w)):
                            common_uv += 1
                if common_uv >= 3:
                    valid = False
                
                if valid:
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        # Check pair (u, w)
                        # New common neighbors of u and w:
                        # Old common + (1 if v in N(w) else 0)
                        old_common_uw = 0
                        for z in range(n):
                            if z != u and z != w:
                                if (G[u] & (1 << z)) and (G[w] & (1 << z)):
                                    old_common_uw += 1
                        add_v = 1 if (G[v] & (1 << w)) else 0
                        if old_common_uw + add_v >= 3:
                            valid = False
                            break
                        
                        if not valid:
                            break
                        
                        # Check pair (v, w)
                        old_common_vw = 0
                        for z in range(n):
                            if z != v and z != w:
                                if (G[v] & (1 << z)) and (G[w] & (1 << z)):
                                    old_common_vw += 1
                        add_u = 1 if (G[u] & (1 << w)) else 0
                        if old_common_vw + add_u >= 3:
                            valid = False
                            break
                
                if valid:
                    G[u].add(v)
                    G[v].add(u)
        
        # Convert G to edge list
        edges = []
        for u in range(n):
            for v in G[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # Convert best_valid_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_valid_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
