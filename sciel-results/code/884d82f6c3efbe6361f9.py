"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 40
    m = 401
    
    # Initialize with K_20,20 + (0,1)
    # Vertices 0-19 in A, 20-39 in B
    adj = np.zeros((n, n), dtype=bool)
    for i in range(20):
        for j in range(20, 40):
            adj[i, j] = True
            adj[j, i] = True
    adj[0, 1] = True
    adj[1, 0] = True
    
    # Helper to calculate book sizes for all edges
    def calc_books(adj):
        books = {}
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    b = np.logical_and(adj[i], adj[j]).sum()
                    books[(i, j)] = int(b)
        return books

    # Initial state
    books = calc_books(adj)
    current_max = max(books.values()) if books else 0
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    # List of current edges
    current_edges = set(books.keys())
    
    # Non-edges
    non_edges = [e for e in all_edges if e not in current_edges]
    
    # Helper to swap edges
    def swap(u, v, x, y, adj, books):
        # Normalize keys
        if u > v: u, v = v, u
        if x > y: x, y = y, x
        
        # Check if (u,v) is in books
        if (u, v) not in books:
            return False
            
        # Find common neighbors of u,v
        common_uv = np.where(np.logical_and(adj[u], adj[v]))[0]
        
        # Update books for edges (u,w) and (v,w)
        for w in common_uv:
            if w == u or w == v: continue
            # Edge (u,w)
            if u < w:
                key_uw = (u, w)
            else:
                key_uw = (w, u)
            if key_uw in books:
                books[key_uw] -= 1
                
            # Edge (v,w)
            if v < w:
                key_vw = (v, w)
            else:
                key_vw = (w, v)
            if key_vw in books:
                books[key_vw] -= 1
        
        # Remove (u,v) from books
        del books[(u, v)]
        
        # Update adj
        adj[u, v] = False
        adj[v, u] = False
        
        # Add (x,y)
        # Find common neighbors of x,y
        common_xy = np.where(np.logical_and(adj[x], adj[y]))[0]
        
        # Book size for (x,y) is len(common_xy)
        new_book_size = len(common_xy)
        
        key_xy = (x, y)
        
        books[key_xy] = new_book_size
        
        # Update books for edges (x,z) and (y,z)
        for z in common_xy:
            if z == x or z == y: continue
            # Edge (x,z)
            if x < z:
                key_xz = (x, z)
            else:
                key_xz = (z, x)
            if key_xz in books:
                books[key_xz] += 1
            else:
                # This should not happen if graph is consistent
                pass
            
            # Edge (y,z)
            if y < z:
                key_yz = (y, z)
            else:
                key_yz = (z, y)
            if key_yz in books:
                books[key_yz] += 1
        
        # Update adj
        adj[x, y] = True
        adj[y, x] = True
        
        return True

    # Main Loop
    best_adj = adj.copy()
    best_max = current_max
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if not books:
            break
            
        current_max = max(books.values())
        
        if current_max <= 7:
            break
            
        # Find an edge with max book
        max_edges = [e for e, b in books.items() if b == current_max]
        if not max_edges:
            break
        e_out = random.choice(max_edges)
        
        # Sample candidates for e_in
        # We want low common neighbors
        samples = random.sample(non_edges, min(20, len(non_edges)))
        
        found_improvement = False
        best_swap_in = None
        best_new_max = current_max
        best_new_sum_sq = sum(b*b for b in books.values())
        
        for e_in in samples:
            # Simulate swap
            swap(e_out[0], e_out[1], e_in[0], e_in[1], adj, books)
            
            new_max = max(books.values()) if books else 0
            new_sum_sq = sum(b*b for b in books.values())
            
            # Revert swap
            swap(e_in[0], e_in[1], e_out[0], e_out[1], adj, books)
            
            if new_max < best_new_max or (new_max == best_new_max and new_sum_sq < best_new_sum_sq):
                best_new_max = new_max
                best_new_sum_sq = new_sum_sq
                best_swap_in = e_in
                
                if new_max < current_max:
                    # Accept this swap immediately
                    swap(e_out[0], e_out[1], e_in[0], e_in[1], adj, books)
                    non_edges.remove(e_in)
                    non_edges.append(e_out)
                    
                    best_adj = adj.copy()
                    best_max = new_max
                    no_improve_count = 0
                    found_improvement = True
                    break
        
        if not found_improvement:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random kick
                for _ in range(10):
                    if not books: break
                    e_rand_out = random.choice(list(books.keys()))
                    e_rand_in = random.choice(non_edges)
                    swap(e_rand_out[0], e_rand_out[1], e_rand_in[0], e_rand_in[1], adj, books)
                    non_edges.remove(e_rand_in)
                    non_edges.append(e_rand_out)
                no_improve_count = 0
                cur_max = max(books.values()) if books else 0
                if cur_max < best_max:
                    best_adj = adj.copy()
                    best_max = cur_max

    # Return edges from best_adj
    final_adj = best_adj
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if final_adj[i, j]:
                edges.append([i, j])
                
    return edges
# EVOLVE-BLOCK-END
