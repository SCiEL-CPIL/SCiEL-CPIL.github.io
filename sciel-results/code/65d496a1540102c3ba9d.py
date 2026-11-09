import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 40
    target_edges = 401
    
    # 1. Greedy Construction
    # We want to add edges to minimize the growth of the max book size.
    adj_masks = [0] * n
    edges = set()
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_pairs)
    
    # To make it deterministic-ish but varied, we can just shuffle once.
    # Greedy step:
    # For each step, we could check all remaining non-edges and pick the best one.
    # That is O(m * n^2 * cost_book). 
    # cost_book is O(1) with bitmasks.
    # 401 * 780 * 1 = ~312k ops. Very fast.
    
    def get_book_size(u, v):
        return (adj_masks[u] & adj_masks[v]).bit_count()
    
    def get_max_book():
        max_b = 0
        for u, v in edges:
            b = get_book_size(u, v)
            if b > max_b:
                max_b = b
        return max_b

    # Instead of pure greedy (which is slow if we re-evaluate all), 
    # let's just use a random initialization and then strong local search.
    # The greedy might get stuck in a bad local structure if it's too greedy early on.
    
    # Let's try a "Quasi-Random" initialization:
    # Start with a random graph with 401 edges.
    edges = set()
    random.shuffle(all_pairs)
    for i in range(target_edges):
        edges.add(all_pairs[i])
        
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    current_max_book = get_max_book()
    
    # 2. Local Search
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Random restart: perturb the graph
            # Remove 10 random edges, add 10 random non-edges
            for _ in range(10):
                if not edges: break
                u, v = random.choice(list(edges))
                edges.remove((u, v))
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
            for _ in range(10):
                # Find a non-edge
                found = False
                while not found:
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v: continue
                    if (u, v) in edges or (v, u) in edges: continue
                    # Check if it's a non-edge
                    if not ((1 << v) & adj_masks[u]):
                        edges.add((u, v))
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        found = True
            no_improve = 0
            current_max_book = get_max_book()
            continue

        # Pick a random edge from those with high book size
        # To be efficient, we can just pick a random edge and check if it's "bad"
        # Or maintain a list of bad edges.
        
        # For simplicity and speed, let's pick a random edge.
        # If it's not the max, we might not improve, but we might find a move.
        # Better: Pick from edges with book size >= current_max_book - 1
        
        u, v = random.choice(list(edges))
        
        # Find best non-edge to swap with
        # We want to minimize the book size of the new edge, and ensure it doesn't increase the max book size of other edges.
        
        best_x, best_y = -1, -1
        best_score = float('inf')
        
        # Sample a subset of non-edges to speed up? 
        # With n=40, 780 pairs is small. Let's check all.
        for x in range(n):
            for y in range(x + 1, n):
                if (1 << y) & adj_masks[x]:
                    continue
                
                # Estimate impact
                # New edge (x,y) has book size b_new
                b_new = (adj_masks[x] & adj_masks[y]).bit_count()
                
                # If we add (x,y), it increases book size of all edges (x, z) and (y, z) where z is common neighbor.
                # This is complex to compute exactly for all affected edges.
                # Heuristic: Just minimize b_new.
                
                if b_new < best_score:
                    best_score = b_new
                    best_x, best_y = x, y
                    if b_new == 0:
                        break
            if best_score == 0:
                break
        
        if best_x == -1:
            continue
            
        # Perform swap
        edges.remove((u, v))
        adj_masks[u] &= ~(1 << v)
        adj_masks[v] &= ~(1 << u)
        
        edges.add((best_x, best_y))
        adj_masks[best_x] |= (1 << best_y)
        adj_masks[best_y] |= (1 << best_x)
        
        new_max_book = get_max_book()
        
        if new_max_book < current_max_book:
            current_max_book = new_max_book
            no_improve = 0
        elif new_max_book == current_max_book:
            # Accept with probability 0.1 to explore
            if random.random() < 0.1:
                no_improve = 0
            else:
                # Rollback
                edges.add((u, v))
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
                edges.remove((best_x, best_y))
                adj_masks[best_x] &= ~(1 << best_y)
                adj_masks[best_y] &= ~(1 << best_x)
                
                no_improve += 1
        else:
            # Rollback
            edges.add((u, v))
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
            edges.remove((best_x, best_y))
            adj_masks[best_x] &= ~(1 << best_y)
            adj_masks[best_y] &= ~(1 << best_x)
            
            no_improve += 1

    result = [[u, v] for u, v in edges]
    return result
