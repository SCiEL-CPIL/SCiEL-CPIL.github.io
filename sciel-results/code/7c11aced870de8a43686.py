import random
import sys

def construct():
    # Vertices 0..15
    n = 16
    # Edge representation: set of (u, v) with u < v
    edges = set()
    
    # Helper: Check if graph is triangle-free
    def is_triangle_free(e_set):
        # For each edge (u, v), check if any w is connected to both
        # Adjacency list for speed
        adj = {i: set() for i in range(n)}
        for u, v in e_set:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Check common neighbors
                    common = adj[u].intersection(adj[v])
                    if common:
                        return False
        return True

    # Helper: Check if graph is 3-colorable
    def is_3_colorable(e_set):
        adj = {i: set() for i in range(n)}
        for u, v in e_set:
            adj[u].add(v)
            adj[v].add(u)
        
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                valid = True
                for u in adj[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        # Map original indices to new order for backtracking? 
        # Easier: just use the sorted order in the recursion
        def backtrack_order(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(3):
                valid = True
                for u in adj[v]:
                    # Only check colors of vertices already colored (earlier in order)
                    # Actually, simple check against all neighbors is fine for n=16
                    if colors[u] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack_order(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack_order(0)

    # Initialize with Grötzsch graph (vertices 0..10)
    # Cycle 0-1-2-3-4-0
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 0), (6, 2),
        (7, 1), (7, 3),
        (8, 2), (8, 4),
        (9, 3), (9, 0),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    for u, v in groetzsch_edges:
        edges.add((u, v))
    
    # Verify initial state
    if not is_triangle_free(edges):
        raise ValueError("Initial graph not triangle-free")
    if is_3_colorable(edges):
        raise ValueError("Initial graph is 3-colorable")

    best_edges = set(edges)
    best_count = len(edges)
    
    # Simulated Annealing / Local Search
    # We try to add edges. If we can't add, we try swaps.
    # Since adding edges preserves non-3-colorability, we only need to check triangle-freeness for additions.
    # However, if we remove edges, we might make it 3-colorable.
    # Strategy: Only add edges. If we get stuck, we might be in a local optimum.
    # To escape, we can try to remove a low-degree edge and add a high-degree one?
    # But removing edges might make it 3-colorable.
    
    # Let's try a simple greedy addition first, then a randomized restart.
    
    def greedy_add(e_set):
        e_set = set(e_set)
        changed = True
        while changed:
            changed = False
            # Generate all potential edges
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in e_set:
                        # Check if adding (u, v) creates a triangle
                        new_e = e_set | {(u, v)}
                        if is_triangle_free(new_e):
                            e_set = new_e
                            changed = True
        return e_set

    # Run greedy from the seed
    current = greedy_add(edges)
    if len(current) > best_count:
        best_edges = current
        best_count = len(current)

    # Try randomized local search
    # We will try to improve by swapping edges.
    # A move is valid if:
    # 1. New graph is triangle-free.
    # 2. New graph is NOT 3-colorable.
    
    import time
    start_time = time.time()
    
    for restart in range(100): # 100 restarts
        if time.time() - start_time > 100: # 100s limit
            break
            
        # Start from best found or random perturbation
        if restart == 0:
            current = set(best_edges)
        else:
            # Perturb best
            current = set(best_edges)
            # Remove 5 random edges
            to_remove = random.sample(list(current), min(5, len(current)))
            for e in to_remove:
                current.discard(e)
            # Add 5 random valid edges
            added = 0
            attempts = 0
            while added < 5 and attempts < 50:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if u > v: u, v = v, u
                    if (u, v) not in current:
                        new_e = current | {(u, v)}
                        if is_triangle_free(new_e):
                            current = new_e
                            added += 1
                attempts += 1
        
        # Local Search Phase
        # Try to add any valid edge
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in current:
                        new_e = current | {(u, v)}
                        if is_triangle_free(new_e):
                            # Check 3-colorability? 
                            # Adding edges to a non-3-colorable graph keeps it non-3-colorable.
                            # So we don't need to check 3-colorability for pure additions!
                            current = new_e
                            improved = True
                            
        # Now try swaps: Remove 1, Add 1 (or Remove 1, Add 2?)
        # Swap: Remove edge e1, Add edge e2.
        # Must maintain triangle-free and non-3-colorable.
        
        for _ in range(1000): # Iterations per restart
            if time.time() - start_time > 110:
                break
                
            # Try a random swap
            if len(current) == 0: break
            e1 = random.choice(list(current))
            u1, v1 = e1
            
            # Pick a random edge to add
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2: continue
            if u2 > v2: u2, v2 = v2, u2
            
            if (u2, v2) in current: continue
            if (u2, v2) == e1: continue
            
            new_current = set(current)
            new_current.discard(e1)
            new_current.add((u2, v2))
            
            if is_triangle_free(new_current):
                if not is_3_colorable(new_current):
                    current = new_current
                    if len(current) > best_count:
                        best_edges = current
                        best_count = len(current)
                        
    # Final check
    if not is_triangle_free(best_edges):
        # Fallback to seed
        return [[u, v] for u, v in groetzsch_edges]
    if is_3_colorable(best_edges):
        # Fallback to seed
        return [[u, v] for u, v in groetzsch_edges]
        
    return [[u, v] for u, v in best_edges]
