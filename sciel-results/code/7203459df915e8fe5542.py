"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book.
    Approach: Spectral optimization via local search.
    1. Initialize with a random graph with 401 edges.
    2. Use local edge swaps to minimize the maximum book size (max common neighbors).
    3. Efficiently evaluate swaps by tracking book changes.
    """
    n = 40
    target_edges = 401
    random.seed(42)

    # Step 1: Initialize with a random graph
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    edges_set = set(all_edges[:target_edges])
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int32)
    for u, v in edges_set:
        adj[u, v] = 1
        adj[v, u] = 1
        
    # Helper to compute max book from adjacency matrix
    def get_max_book(adj):
        A2 = adj @ adj
        max_b = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    if A2[u, v] > max_b:
                        max_b = A2[u, v]
        return max_b

    current_max = get_max_book(adj)
    best_adj = adj.copy()
    best_max = current_max
    
    # Local search
    max_iterations = 2000
    for iteration in range(max_iterations):
        if current_max <= 7: # Theoretical lower bound is ~6.66, so 7 is a great target
            # Try a few more iterations to see if we can hit 6, but 7 is likely optimal
            if current_max == 6:
                break
            
        # Compute A2 and books
        A2 = adj @ adj
        # Find edges with max book
        high_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] and A2[u, v] == current_max:
                    high_edges.append((u, v))
        
        if not high_edges:
            break
            
        random.shuffle(high_edges)
        improved = False
        
        for (u, v) in high_edges[:3]: # Try a few high edges
            # Temporarily remove (u,v)
            adj[u, v] = 0
            adj[v, u] = 0
            
            # Compute new A2
            A2_temp = adj @ adj
            
            # Find non-edges
            non_edges = []
            for w in range(n):
                for x in range(w+1, n):
                    if not adj[w, x]:
                        # Estimate book size of new edge (w,x)
                        # It is A2_temp[w, x]
                        non_edges.append((w, x, A2_temp[w, x]))
            
            # Sort non-edges by their potential book size (ascending)
            non_edges.sort(key=lambda t: t[2])
            
            best_candidate = None
            best_candidate_max = current_max + 100
            
            # Try top 50 candidates
            for w, x, new_book_size in non_edges[:50]:
                # Add (w,x)
                adj[w, x] = 1
                adj[x, w] = 1
                
                # Calculate new max book
                # We can just recompute A2 for speed since N=40 is small
                A2_check = adj @ adj
                new_max = 0
                for a in range(n):
                    for b in range(a+1, n):
                        if adj[a, b]:
                            if A2_check[a, b] > new_max:
                                new_max = A2_check[a, b]
                
                # Remove (w,x)
                adj[w, x] = 0
                adj[x, w] = 0
                
                if new_max < best_candidate_max:
                    best_candidate_max = new_max
                    best_candidate = (w, x)
            
            # Restore (u,v)
            adj[u, v] = 1
            adj[v, u] = 1
            
            if best_candidate and best_candidate_max < current_max:
                # Apply swap
                adj[u, v] = 0
                adj[v, u] = 0
                w, x = best_candidate
                adj[w, x] = 1
                adj[x, w] = 1
                current_max = best_candidate_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_adj = adj.copy()
                break # Restart loop with new state
        
        if not improved:
            # Random perturbation
            if iteration % 50 == 49:
                edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
                u, v = random.choice(edge_list)
                adj[u, v] = 0
                adj[v, u] = 0
                
                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
                w, x = random.choice(non_edges)
                adj[w, x] = 1
                adj[x, w] = 1
                
                cur_max = get_max_book(adj)
                if cur_max < best_max:
                    best_max = cur_max
                    best_adj = adj.copy()
    
    adj = best_adj
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                result.append([u, v])
                
    if len(result) != target_edges:
        # Fallback
        result = all_edges[:target_edges]
        return [[a, b] for a, b in result]
        
    return result
# EVOLVE-BLOCK-END
