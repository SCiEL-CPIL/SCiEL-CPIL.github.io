import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n = 31
    q = 5
    
    # Build polarity graph of PG(2,5)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    edge_set = set()
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edge_set.add((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    # Add the best edge
    # cost(i, j) = number of paths of length 3 between i and j
    # cost(i, j) = sum_{a in N(i)} |N(a) \cap N(j)|
    
    best_edge = None
    min_cost = float('inf')
    
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in edge_set:
                continue
            cost = 0
            for a in adj[i]:
                cost += len(adj[a] & adj[j])
            if cost < min_cost:
                min_cost = cost
                best_edge = (i, j)
    
    if best_edge:
        u, v = best_edge
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute common neighbor counts C[i][j]
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            c = len(adj[i] & adj[j])
            C[i][j] = c
            C[j][i] = c
    
    def count_c4_from_C(C, n):
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                if C[i][j] >= 2:
                    total += C[i][j] * (C[i][j] - 1) // 2
        return total // 2
    
    current_c4 = count_c4_from_C(C, n)
    best_c4 = current_c4
    best_edge_set = set(edge_set)
    best_adj = [set(adj[i]) for i in range(n)]
    best_C = [row[:] for row in C]
    
    # Local search with incremental updates
    # Swap edge (u,v) with non-edge (x,y)
    
    # Lists for iteration
    edges = list(edge_set)
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
    
    random.seed(42)
    no_improve = 0
    max_no_improve = 500
    
    while no_improve < max_no_improve:
        if time.time() > deadline - 5:
            break
            
        # Random swap
        if not edges or not non_edges:
            break
            
        idx_e = random.randint(0, len(edges)-1)
        idx_n = random.randint(0, len(non_edges)-1)
        
        u, v = edges[idx_e]
        x, y = non_edges[idx_n]
        
        # Ensure u,v,x,y are distinct? 
        # If they share vertices, the update is more complex.
        # For simplicity, only consider disjoint swaps initially?
        # Or handle general case.
        # Let's handle general case by recalculating C for affected pairs.
        
        # Affected pairs: any pair (a,b) where a or b is in {u,v,x,y}
        # And the pair (u,v) itself and (x,y) itself.
        
        # Save old C values for affected pairs
        affected = set()
        for a in (u, v, x, y):
            for b in range(n):
                if a != b:
                    affected.add((min(a,b), max(a,b)))
        
        old_C_vals = {}
        for a, b in affected:
            old_C_vals[(a,b)] = C[a][b]
            
        # Apply swap to adj
        adj[u].discard(v)
        adj[v].discard(u)
        adj[x].add(y)
        adj[y].add(x)
        
        # Update C for affected pairs
        for a, b in affected:
            C[a][b] = len(adj[a] & adj[b])
            C[b][a] = C[a][b]
            
        new_c4 = count_c4_from_C(C, n)
        
        if new_c4 < current_c4:
            # Accept
            current_c4 = new_c4
            edges[idx_e] = (x, y)
            non_edges[idx_n] = (u, v)
            # Update edge_set
            edge_set.discard((u, v))
            edge_set.add((x, y))
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edge_set = set(edge_set)
                best_adj = [set(adj[i]) for i in range(n)]
                best_C = [row[:] for row in C]
                
            no_improve = 0
        else:
            # Revert
            adj[u].add(v)
            adj[v].add(u)
            adj[x].discard(y)
            adj[y].discard(x)
            
            for a, b in affected:
                C[a][b] = old_C_vals[(a,b)]
                C[b][a] = C[a][b]
                
            no_improve += 1
            
            if no_improve > 100:
                # Random restart or change strategy?
                # Let's just continue
                pass

    # Return best found
    final_edges = list(best_edge_set)
    # Ensure format
    edges_out = [[u, v] for u, v in final_edges]
    
    return edges_out
