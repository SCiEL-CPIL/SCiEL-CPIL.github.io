import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 150s limit
    
    n = 40
    target_edges = 401
    
    # 1. Structural Initialization: Replicate the high-performing structure from Inspiration 1
    # This structure has a lower inherent book size than random or K_{20,20}+1.
    A = list(range(9))
    B = list(range(9, 40))
    
    edges = set()
    
    # 1. Add all edges between A and B (9 * 31 = 279 edges)
    for a in A:
        for b in B:
            edges.add((a, b))
            
    # 2. Add 122 edges within B to reach 401 total
    # Split B into B1 (15 vertices: 9..23) and B2 (16 vertices: 24..39)
    B1 = list(range(9, 24))
    B2 = list(range(24, 40))
    
    needed = 122
    added = 0
    
    # We want to distribute edges between B1 and B2 to minimize common neighbors.
    # A balanced distribution is good.
    deg_b1 = {v: 0 for v in B1}
    deg_b2 = {v: 0 for v in B2}
    
    # Generate all possible pairs between B1 and B2
    pairs = [(u, v) for u in B1 for v in B2]
    random.shuffle(pairs)
    
    # Add edges with a degree constraint to keep the structure "sparse" within B
    # Max degree 9 was used in inspiration. 15*9=135 > 122, so it's feasible.
    for u, v in pairs:
        if added == needed:
            break
        if deg_b1[u] < 9 and deg_b2[v] < 9:
            edges.add((u, v))
            deg_b1[u] += 1
            deg_b2[v] += 1
            added += 1
            
    if added < needed:
        # Fallback: fill remaining ignoring constraints if necessary
        for u, v in pairs:
            if added == needed:
                break
            if (u, v) not in edges:
                edges.add((u, v))
                added += 1

    # 2. Efficient Data Structures for Local Search
    # Use bitmasks for adjacency to speed up common neighbor calculation
    adj_masks = [0] * n
    
    def update_adj(u, v, add):
        if add:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        else:
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            
    for u, v in edges:
        update_adj(u, v, True)
        
    # Precompute bit counts for 16-bit chunks to speed up popcount? 
    # Python's int.bit_count() is very fast in modern versions.
    
    def get_book_size(u, v):
        if u == v:
            return 0
        # Common neighbors are bits set in both adj_masks[u] and adj_masks[v]
        return (adj_masks[u] & adj_masks[v]).bit_count()
        
    def get_max_book_info():
        max_b = -1
        max_edges = []
        # Iterate over edges to find max book
        # Optimization: We can just iterate over all pairs if we maintain a list of edges
        for u, v in edges:
            b = get_book_size(u, v)
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
        return max_b, max_edges

    current_max_book, current_max_edges = get_max_book_info()
    
    # 3. Local Search Refinement
    # Goal: Reduce current_max_book
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if current_max_book <= 7:
            # We hit the theoretical lower bound (integer part of 40/6 is 6, but theorem says >= 40/6. 
            # For m=401, n=40, the bound is tight around 7. If we reach 7, we likely can't go lower 
            # without violating the edge count or structure constraints significantly.
            # However, let's allow a few more iterations to ensure it's stable or if 6 is possible.
            # Actually, 40/6 = 6.66. So 7 is the integer ceiling. 6 might be impossible.
            # Let's keep searching if we have time, but prioritize if we are stuck at 7.
            if current_max_book == 7 and no_improve_count > 200:
                break
            elif current_max_book <= 6:
                break

        # Pick a random edge from the max-book set to remove
        if not current_max_edges:
            break
            
        u, v = random.choice(current_max_edges)
        
        # Find the best non-edge to add
        # We want to minimize the book size of the new edge, and ideally not increase others.
        # Heuristic: Find non-edge (x,y) with min common neighbors.
        min_book_new = float('inf')
        best_x, best_y = -1, -1
        
        # Iterate over all possible non-edges
        # N=40, so 780 pairs. This is fast.
        for x in range(n):
            for y in range(x + 1, n):
                # Check if (x,y) is an edge
                # Using bitmask check is faster than set lookup? 
                # (1 << y) & adj_masks[x]
                if (1 << y) & adj_masks[x]:
                    continue
                
                # Calculate book size for potential new edge
                b = (adj_masks[x] & adj_masks[y]).bit_count()
                if b < min_book_new:
                    min_book_new = b
                    best_x, best_y = x, y
                    if b == 0:
                        # Can't get better than 0
                        break
            if min_book_new == 0:
                break
        
        if best_x == -1:
            # No non-edges? Impossible with 401 edges out of 780.
            break
            
        # Perform the swap: remove (u,v), add (best_x, best_y)
        
        # Save state for rollback
        old_max = current_max_book
        old_max_count = len(current_max_edges)
        
        # Remove (u,v)
        edges.remove((u, v))
        update_adj(u, v, False)
        
        # Add (best_x, best_y)
        edges.add((best_x, best_y))
        update_adj(best_x, best_y, True)
        
        # Recalculate max book
        new_max, new_max_edges = get_max_book_info()
        
        # Acceptance Criteria
        accepted = False
        if new_max < old_max:
            accepted = True
            no_improve_count = 0
        elif new_max == old_max:
            # Prefer moves that reduce the number of max-book edges
            if len(new_max_edges) < old_max_count:
                accepted = True
                no_improve_count = 0
            else:
                # Small probability to accept to escape local minima
                if random.random() < 0.02:
                    accepted = True
                else:
                    accepted = False
        else:
            accepted = False

        if accepted:
            current_max_book = new_max
            current_max_edges = new_max_edges
        else:
            # Rollback
            edges.add((u, v))
            update_adj(u, v, True)
            
            edges.remove((best_x, best_y))
            update_adj(best_x, best_y, False)
            
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart of the search phase? 
                # Or just perturb the graph slightly?
                # Let's try a random valid swap that doesn't increase max book too much
                # Actually, if we are stuck, the structure might be optimal for this init.
                # Let's break to save time if we've been stuck for a long time.
                if time.time() > deadline - 20:
                    break

    result = [[u, v] for u, v in edges]
    return result
