import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140  # 140s safety margin for 150s limit
    
    n = 40
    target_edges = 401
    
    def compute_books(adj_matrix):
        """Compute book size for each edge. Returns dict of (u,v) -> count."""
        books = {}
        for u in range(n):
            for v in range(u+1, n):
                if adj_matrix[u][v]:
                    # Count common neighbors
                    common = np.sum(np.logical_and(adj_matrix[u], adj_matrix[v]))
                    books[(u, v)] = int(common)
        return books
    
    def max_book(books):
        if not books:
            return 0
        return max(books.values())
    
    def energy(books):
        """Energy function: primarily max book, with penalty for many high books."""
        mb = max_book(books)
        # Add penalty for edges with book > 7
        excess = sum(max(0, b - 7) for b in books.values())
        return mb * 100 + excess
    
    # Initial graph: start with a random graph with 401 edges
    # Better: start with a structured graph that's more balanced
    # Use a random regular-ish graph approach
    
    best_adj = None
    best_score = float('inf')
    best_books = None
    
    # Try multiple random starts
    rng = np.random.default_rng(42)
    
    for start_iter in range(5):
        if time.time() > deadline - 20:
            break
        
        # Create initial adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        edge_count = 0
        # Random edges
        edges_to_add = target_edges
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for u, v in all_possible[:edges_to_add]:
            adj[u][v] = 1
            adj[v][u] = 1
            edge_count += 1
        
        if edge_count != target_edges:
            continue
        
        books = compute_books(adj)
        cur_energy = energy(books)
        
        if cur_energy < best_score:
            best_score = cur_energy
            best_adj = adj.copy()
            best_books = books
        
        # Simulated annealing
        T = 50.0
        T_min = 0.1
        cooling = 0.9995
        cur_adj = adj.copy()
        cur_books = books
        cur_e = cur_energy
        cur_mb = max_book(cur_books)
        
        step = 0
        while T > T_min:
            if time.time() > deadline - 10:
                break
            
            # Propose move: remove one edge, add another
            # Find a random existing edge
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if cur_adj[i][j]]
            if not existing_edges:
                break
            
            u1, v1 = random.choice(existing_edges)
            
            # Find a random non-edge
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not cur_adj[i][j]]
            if not non_edges:
                break
            
            u2, v2 = random.choice(non_edges)
            
            # Apply move
            cur_adj[u1][v1] = 0
            cur_adj[v1][u1] = 0
            cur_adj[u2][v2] = 1
            cur_adj[v2][u2] = 1
            
            # Recompute books (expensive, but necessary)
            new_books = compute_books(cur_adj)
            new_e = energy(new_books)
            new_mb = max_book(new_books)
            
            # Accept or reject
            if new_e < cur_e or random.random() < np.exp(-(new_e - cur_e) / T):
                cur_e = new_e
                cur_books = new_books
                cur_mb = new_mb
                
                if cur_e < best_score:
                    best_score = cur_e
                    best_adj = cur_adj.copy()
                    best_books = cur_books
            else:
                # Revert
                cur_adj[u1][v1] = 1
                cur_adj[v1][u1] = 1
                cur_adj[u2][v2] = 0
                cur_adj[v2][u2] = 0
            
            T *= cooling
            step += 1
        
        if time.time() > deadline - 15:
            break
    
    # Convert best_adj to edge list
    if best_adj is None:
        # Fallback: K20,20 + 1 edge
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    # Ensure exactly 401 edges
    if len(edges) != target_edges:
        # Fallback
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return edges
