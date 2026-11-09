import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n = 20
    # M[i] is a bitmask of neighbors of i
    # M[i] & (1 << j) is 1 if i is connected to j
    M = [0] * n
    
    # Helper to check if adding edge (u, v) is valid
    def can_add(u, v):
        if u == v:
            return False
        if (M[u] >> v) & 1:
            return False # already exists
        
        # The new masks
        new_M_u = M[u] | (1 << v)
        new_M_v = M[v] | (1 << u)
        
        # Check all triples involving u or v
        # Triples {u, x, y} where x != u, y != u, x < y
        # We need to check if popcount(N(u) & N(x) & N(y)) > 2
        # With new edge, N(u) changes to new_M_u.
        
        # 1. Triples of form {u, v, w}
        for w in range(n):
            if w == u or w == v:
                continue
            # N(u) is new_M_u, N(v) is new_M_v, N(w) is M[w]
            # Intersection: new_M_u & new_M_v & M[w]
            if (new_M_u & new_M_v & M[w]).bit_count() > 2:
                return False
        
        # 2. Triples of form {u, x, y} where x, y != u, v
        # N(u) is new_M_u. N(x) is M[x], N(y) is M[y].
        # We need to check if new_M_u & M[x] & M[y] has > 2 bits.
        # Note: M[x] and M[y] are unchanged.
        # The only change is in N(u).
        # So we check if adding v to N(u) creates a 3rd common neighbor with x and y.
        # Basically, if M[x] & M[y] & (1 << v) is non-zero, then v is a common neighbor.
        # If M[x] & M[y] & M[u] already had 2 bits, and v is also in M[x] & M[y], then it becomes 3.
        
        # Efficient check:
        # For each pair (x, y) not involving u or v:
        # Let common_xy = M[x] & M[y]
        # If (common_xy & (1 << v)) != 0:
        #    Then v is a common neighbor of x and y.
        #    The new common neighbors of u, x, y are (new_M_u & common_xy).
        #    new_M_u = M[u] | (1 << v).
        #    So new intersection = (M[u] & common_xy) | (1 << v if v in common_xy else 0)
        #    We need popcount(new intersection) <= 2.
        
        # We can iterate x from 0 to n-1, y from x+1 to n-1
        # Skip if x==u or y==u or x==v or y==v
        
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                common_xy = M[x] & M[y]
                # If v is in common_xy, it adds to the count
                # current count for u,x,y is (M[u] & common_xy).bit_count()
                # new count is (new_M_u & common_xy).bit_count()
                # This is efficient enough for n=20.
                if (new_M_u & common_xy).bit_count() > 2:
                    return False
        
        # 3. Triples of form {v, x, y} where x, y != u, v
        # Symmetric to case 2
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                common_xy = M[x] & M[y]
                if (new_M_v & common_xy).bit_count() > 2:
                    return False
        
        return True

    def add_edge(u, v):
        M[u] |= (1 << v)
        M[v] |= (1 << u)

    def remove_edge(u, v):
        M[u] &= ~(1 << v)
        M[v] &= ~(1 << u)

    def count_edges():
        return sum(M[i].bit_count() for i in range(n)) // 2

    # Initialize with a random graph
    # Start with a fairly dense random graph and prune
    # Or start empty and add.
    # Let's start with a random graph with p=0.6 and prune invalid edges.
    
    best_edges = 0
    best_M = [0] * n
    
    # Greedy construction
    random.seed(42)
    
    # Try multiple random restarts
    while time.time() < deadline:
        # Random initialization
        M = [0] * n
        # Add random edges
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    # Try to add
                    if can_add(u, v):
                        add_edge(u, v)
        
        current_edges = count_edges()
        
        # Local search / Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any missing edge
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    if not ((M[u] >> v) & 1):
                        if can_add(u, v):
                            add_edge(u, v)
                            improved = True
                            current_edges += 1
                            break # restart inner loop to pick up new state? 
                            # Actually, just continue, can_add is fast.
            if not improved:
                # Try to swap: remove an edge, then add two others?
                # Or just restart with perturbation
                # Simple perturbation: remove a random edge
                if current_edges > 0:
                    # Pick a random existing edge
                    # Find one
                    edges_list = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (M[u] >> v) & 1:
                                edges_list.append((u, v))
                    if edges_list:
                        u, v = random.choice(edges_list)
                        remove_edge(u, v)
                        current_edges -= 1
                        improved = True # Force another pass
                        # After removal, try to add more
                        # The outer loop will handle adding
    
        if current_edges > best_edges:
            best_edges = current_edges
            best_M = M[:]
            
        # If we found a very good graph, we might want to refine it more
        # But simple hill climbing on n=20 should be fast.
        
    # Return the best found
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_M[u] >> v) & 1:
                edges.append([u, v])
    return edges
