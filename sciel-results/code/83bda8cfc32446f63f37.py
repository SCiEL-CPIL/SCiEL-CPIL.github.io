import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n = 40
    target_edges = 401
    
    # 1. Structural Initialization: Replicate Inspiration 1 (Score 2.22)
    # This structure likely has a book size of 7 or 8.
    A = list(range(9))
    B = list(range(9, 40))
    
    edges = set()
    
    # 1. Add all edges between A and B
    for a in A:
        for b in B:
            edges.add((a, b))
            
    # 2. Add 122 edges within B
    B1 = list(range(9, 24))
    B2 = list(range(24, 40))
    
    needed = 122
    added = 0
    
    deg_b1 = {v: 0 for v in B1}
    deg_b2 = {v: 0 for v in B2}
    
    # Greedy addition of intra-B edges
    # We want to minimize the max book size.
    # We will maintain adjacency sets for fast calculation.
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def get_book_size(u, v):
        if v not in adj[u]:
            return 0
        return len(adj[u].intersection(adj[v]))
        
    # Pre-compute all possible pairs in B1 x B2
    pairs = [(u, v) for u in B1 for v in B2]
    
    # We will greedily pick the pair that minimizes the "cost"
    # Cost is the maximum book size that would result from adding this edge.
    # As derived: Adding (u,v) increases book size of (u,z) for z in N(v) and (v,z) for z in N(u).
    # So new max book size for this "local neighborhood" is:
    # max( max(Book(u,z) for z in N(v)), max(Book(v,z) for z in N(u)), Book(u,v) )
    # Note: Book(u,v) is |N(u) cap N(v)|.
    
    for _ in range(needed):
        if time.time() > deadline:
            break
        best_pair = None
        min_cost = float('inf')
        
        # To speed up, we can filter candidates by degree constraints if we still care,
        # but the greedy cost metric is more important for balance.
        # However, keeping degrees balanced helps avoid local clusters.
        # Let's just use the cost metric.
        
        for u, v in pairs:
            if (u, v) in edges:
                continue
            
            # Calculate cost
            # Book(u, v)
            b_uv = len(adj[u].intersection(adj[v]))
            
            # Max book size of neighbors of v connected to u
            # These are edges (u, z) where z in N(v).
            # Note: u is not in N(v) yet.
            max_b_u = 0
            for z in adj[v]:
                if z == u: continue
                if z in adj[u]:
                    # Edge (u, z) exists
                    b_uz = len(adj[u].intersection(adj[z]))
                    if b_uz + 1 > max_b_u: # It will increase by 1
                        max_b_u = b_uz + 1
            
            # Max book size of neighbors of u connected to v
            max_b_v = 0
            for z in adj[u]:
                if z == v: continue
                if z in adj[v]:
                    # Edge (v, z) exists
                    b_vz = len(adj[v].intersection(adj[z]))
                    if b_vz + 1 > max_b_v:
                        max_b_v = b_vz + 1
            
            cost = max(b_uv, max_b_u, max_b_v)
            
            if cost < min_cost:
                min_cost = cost
                best_pair = (u, v)
                
        if best_pair is None:
            break
            
        u, v = best_pair
        edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        added += 1

    # Fallback if greedy didn't fill enough (shouldn't happen)
    if added < needed:
        for u, v in pairs:
            if added == needed:
                break
            if (u, v) not in edges:
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                added += 1

    def get_max_book_and_edges():
        max_b = 0
        max_edges = []
        for u, v in edges:
            b = get_book_size(u, v)
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
        return max_b, max_edges

    current_max_book, current_max_edges = get_max_book_and_edges()
    
    # 2. Local Search Refinement
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if current_max_book <= 7:
            break
        
        # Pick a random edge from the max-book set to remove
        u, v = random.choice(current_max_edges)
        
        # Find the best non-edge to add using the same cost metric
        min_cost = float('inf')
        best_x, best_y = -1, -1
        
        # Sample non-edges to save time? N=40 is small, 780 pairs.
        # We can iterate all non-edges.
        for x in range(n):
            for y in range(x+1, n):
                if (x, y) in edges or (y, x) in edges:
                    continue
                
                # Calculate cost of adding (x, y)
                b_xy = len(adj[x].intersection(adj[y]))
                
                max_b_x = 0
                for z in adj[y]:
                    if z == x: continue
                    if z in adj[x]:
                        b_xz = len(adj[x].intersection(adj[z]))
                        if b_xz + 1 > max_b_x:
                            max_b_x = b_xz + 1
                
                max_b_y = 0
                for z in adj[x]:
                    if z == y: continue
                    if z in adj[y]:
                        b_yz = len(adj[y].intersection(adj[z]))
                        if b_yz + 1 > max_b_y:
                            max_b_y = b_yz + 1
                
                cost = max(b_xy, max_b_x, max_b_y)
                
                if cost < min_cost:
                    min_cost = cost
                    best_x, best_y = x, y
        
        if best_x == -1:
            break
            
        # Perform swap
        old_max = current_max_book
        old_max_count = len(current_max_edges)
        
        # Remove (u, v)
        edges.remove((u, v))
        adj[u].remove(v)
        adj[v].remove(u)
        
        # Add (best_x, best_y)
        edges.add((best_x, best_y))
        adj[best_x].add(best_y)
        adj[best_y].add(best_x)
        
        new_max, new_max_edges = get_max_book_and_edges()
        
        accepted = False
        if new_max < old_max:
            accepted = True
            no_improve_count = 0
        elif new_max == old_max:
            if len(new_max_edges) < old_max_count:
                accepted = True
                no_improve_count = 0
            else:
                if random.random() < 0.05:
                    accepted = True
        else:
            accepted = False

        if accepted:
            current_max_book = new_max
            current_max_edges = new_max_edges
        else:
            # Rollback
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            
            edges.remove((best_x, best_y))
            adj[best_x].remove(best_y)
            adj[best_y].remove(best_x)
            
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation
                for _ in range(10):
                    e1 = random.choice(list(edges))
                    u1, v1 = e1
                    while True:
                        x = random.randint(0, n-1)
                        y = random.randint(0, n-1)
                        if x == y: continue
                        if x > y: x, y = y, x
                        if (x, y) not in edges: break
                    
                    edges.remove(e1)
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    
                    edges.add((x, y))
                    adj[x].add(y)
                    adj[y].add(x)
                    
                    new_max, new_max_edges = get_max_book_and_edges()
                    if new_max <= current_max_book:
                        current_max_book = new_max
                        current_max_edges = new_max_edges
                        no_improve_count = 0
                        break
                    else:
                        edges.remove((x, y))
                        adj[x].remove(y)
                        adj[y].remove(x)
                        edges.add(e1)
                        adj[u1].add(v1)
                        adj[v1].add(u1)
                
                no_improve_count = 0
                max_no_improve += 100

    result = [[u, v] for u, v in edges]
    return result
