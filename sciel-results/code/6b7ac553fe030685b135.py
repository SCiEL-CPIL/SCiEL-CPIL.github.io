import time
import random
import math

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115

    # Bitmasks for adjacency
    # adj[i] is a bitmask of neighbors of i
    
    def get_initial_books(adj, edges):
        books = {}
        max_b = 0
        sum_b = 0
        for u, v in edges:
            common = adj[u] & adj[v]
            b = common.bit_count()
            books[(u, v)] = b
            if b > max_b:
                max_b = b
            sum_b += b
        return max_b, sum_b, books

    def solve_single_seed(seed):
        random.seed(seed)
        adj = [0] * n
        edges = set()
        
        # Initialization: Random
        # We can also try Bipartite + Random perturbation
        # But Random is often better for SA to explore the space.
        # Let's stick to Random but maybe bias towards regular degrees?
        # For now, simple random.
        
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                u, v = min(u, v), max(u, v)
                if (u, v) not in edges:
                    edges.add((u, v))
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        max_b, sum_b, books = get_initial_books(adj, edges)
        
        best_max = max_b
        best_sum = sum_b
        best_edges = edges.copy()
        
        T = 10.0
        cooling = 0.9995
        min_T = 0.05
        no_improve_counter = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 10 and T > min_T:
            # 1. Remove random edge
            # To make it more targeted, we could pick from top 50% book edges
            # But random is robust. Let's stick to random for simplicity and speed.
            e_rem = random.choice(tuple(edges))
            u1, v1 = e_rem
            
            # Save state
            adj_backup = adj[:]
            edges_backup = edges.copy()
            books_backup = books.copy() # Expensive? 401 entries. Maybe ok.
            
            # Apply removal
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            edges.discard(e_rem)
            
            # Incremental update of books for removal
            # Book of (u1, v1) is removed from dict
            del books[(u1, v1)]
            
            # For all neighbors w of u1, if (u1, w) is an edge, and v1 was in N(w),
            # then book(u1, w) decreases by 1.
            # Condition: v1 in N(w) is equivalent to (w, v1) in edges.
            # Since we just removed (u1, v1), v1 is still in N(w) if (w, v1) exists.
            
            # Iterate neighbors of u1
            neighbors_u1 = adj[u1]
            # We need to check edges (u1, w).
            # It's faster to iterate over current edges involving u1?
            # Or iterate over bits in neighbors_u1.
            
            # Let's just recalculate books for edges incident to u1 and v1.
            # There are at most 2 * 40 = 80 such edges.
            # This is much faster than O(m) full scan if m is large, but here m=401.
            # Full scan is 401 ops. Local scan is ~40 ops.
            
            # Actually, let's just do a full scan for simplicity and correctness, 
            # as 401 ops is very small in Python if optimized.
            # But incremental is better.
            
            # Let's try incremental:
            # 1. Remove (u1, v1) from books.
            # 2. For each w in N(u1):
            #    if (u1, w) in edges:
            #       if (w, v1) in edges: # v1 is common neighbor
            #           books[(u1, w)] -= 1
            # 3. For each w in N(v1):
            #    if (v1, w) in edges:
            #       if (w, u1) in edges: # u1 is common neighbor
            #           books[(v1, w)] -= 1
            # Note: (u1, w) and (w, u1) are the same edge key (min, max).
            
            # Implementation of incremental removal:
            # We already did del books[(u1, v1)]
            
            # Get neighbors of u1
            # adj[u1] is a bitmask.
            # Iterate through set bits?
            # Or just iterate 0..n-1?
            
            # Let's iterate 0..n-1 for simplicity, it's only 40.
            for w in range(n):
                if w == u1 or w == v1: continue
                # Check if (u1, w) is an edge
                if adj[u1] & (1 << w):
                    key = (min(u1, w), max(u1, w))
                    if key in books:
                        # Check if v1 is a common neighbor of u1 and w
                        # i.e., is v1 in N(w)?
                        if adj[v1] & (1 << w):
                            books[key] -= 1
                            
                # Check if (v1, w) is an edge
                if adj[v1] & (1 << w):
                    key = (min(v1, w), max(v1, w))
                    if key in books:
                        # Check if u1 is a common neighbor of v1 and w
                        # i.e., is u1 in N(w)?
                        if adj[u1] & (1 << w):
                            books[key] -= 1

            # 2. Find best add
            # Sample 50 random non-edges
            best_add = None
            best_potential = 999
            for _ in range(50):
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2: continue
                u2, v2 = min(u2, v2), max(u2, v2)
                if (u2, v2) in edges: continue
                potential = (adj[u2] & adj[v2]).bit_count()
                if potential < best_potential:
                    best_potential = potential
                    best_add = (u2, v2)
                    
            if best_add is None:
                # Revert
                adj = adj_backup
                edges = edges_backup
                books = books_backup
                continue
                
            # Apply addition
            u2, v2 = best_add
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            edges.add(best_add)
            
            # Incremental update of books for addition
            # 1. Add (u2, v2) to books with value potential
            books[best_add] = potential
            
            # 2. For each w in N(u2):
            #    if (u2, w) in edges:
            #       if (w, v2) in edges:
            #           books[(u2, w)] += 1
            # 3. For each w in N(v2):
            #    if (v2, w) in edges:
            #       if (w, u2) in edges:
            #           books[(v2, w)] += 1
            
            for w in range(n):
                if w == u2 or w == v2: continue
                if adj[u2] & (1 << w):
                    key = (min(u2, w), max(u2, w))
                    if key in books:
                        if adj[v2] & (1 << w):
                            books[key] += 1
                            
                if adj[v2] & (1 << w):
                    key = (min(v2, w), max(v2, w))
                    if key in books:
                        if adj[u2] & (1 << w):
                            books[key] += 1

            # 3. Evaluate
            # We need new_max_b and new_sum_b.
            # We can update these incrementally too, but finding max is O(m).
            # However, we can just scan the books dict.
            new_max_b = 0
            new_sum_b = 0
            for b in books.values():
                if b > new_max_b:
                    new_max_b = b
                new_sum_b += b
            
            # 4. Accept/Reject
            accept = False
            if new_max_b < max_b:
                accept = True
                no_improve_counter = 0
            elif new_max_b == max_b and new_sum_b < sum_b:
                accept = True
                no_improve_counter = 0
            else:
                delta = (new_max_b - max_b) * 10 + (new_sum_b - sum_b)
                if delta < 0:
                    accept = True
                    no_improve_counter = 0
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        accept = True
                    else:
                        no_improve_counter += 1
                        if no_improve_counter > max_no_improve:
                            # Restart or aggressive cooling
                            T = max(T * 0.5, min_T)
                            no_improve_counter = 0
                        
            if accept:
                max_b = new_max_b
                sum_b = new_sum_b
                if max_b < best_max or (max_b == best_max and sum_b < best_sum):
                    best_max = max_b
                    best_sum = sum_b
                    best_edges = edges.copy()
            else:
                # Revert
                adj = adj_backup
                edges = edges_backup
                books = books_backup
                
            T *= cooling
            
            if best_max <= 7:
                break
                
        return best_max, best_sum, best_edges

    global_best_max = 999
    global_best_sum = 999
    global_best_edges = None
    
    # Run multiple restarts
    for seed in range(100):
        if time.time() > deadline - 5:
            break
            
        max_b, sum_b, edges = solve_single_seed(seed)
        
        if max_b < global_best_max or (max_b == global_best_max and sum_b < global_best_sum):
            global_best_max = max_b
            global_best_sum = sum_b
            global_best_edges = edges
            
        if global_best_max <= 7:
            break
            
    if global_best_edges is None:
        # Fallback
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return [[u, v] for (u, v) in sorted(global_best_edges)]
