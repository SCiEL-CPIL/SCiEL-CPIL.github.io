import random
import time
import numpy as np

def construct():
    n = 40
    m = 401
    start_time = time.time()
    time_limit = 145  # Leave 5s margin
    
    best_graph = None
    best_score = float('inf')
    
    # Helper to compute max book size from adjacency matrix
    def get_max_book(A):
        # A is 0/1 matrix. Codegrees are A^2.
        # For edge (i,j), book size is A[i,j] * (A^2)[i,j]
        C = A @ A
        # Only consider edges where A[i,j] == 1
        # We can mask it
        max_book = 0
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 1:
                    b = C[i,j]
                    if b > max_book:
                        max_book = b
        return max_book

    # Helper to get list of edges and non-edges
    def get_edges_nonedges(A):
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 1:
                    edges.append((i,j))
                else:
                    non_edges.append((i,j))
        return edges, non_edges

    for restart in range(50): # Try multiple restarts if time permits
        # Initialize random graph
        A = np.zeros((n, n), dtype=int)
        edges = []
        # Generate m random edges
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs[:m]:
            A[u, v] = 1
            A[v, u] = 1
            edges.append((u, v))
            
        current_score = get_max_book(A)
        if current_score < best_score:
            best_score = current_score
            best_graph = A.copy()
            
        if time.time() - start_time > time_limit:
            break
            
        # Local search
        no_improvement_count = 0
        while time.time() - start_time < time_limit:
            if no_improvement_count > 1000:
                break # Restart local search or new restart
            
            edges, non_edges = get_edges_nonedges(A)
            
            # Find the edge with max book size
            C = A @ A
            max_b = -1
            worst_edge = None
            for u, v in edges:
                b = C[u, v]
                if b > max_b:
                    max_b = b
                    worst_edge = (u, v)
            
            if worst_edge is None:
                break
                
            # Try to swap worst_edge with a random non-edge
            # We want to pick a non-edge (x,y) that results in low book sizes
            # Ideally, x,y have few common neighbors.
            # Let's just pick a random non-edge and check if it improves or maintains score.
            # To be more effective, pick a few random non-edges and choose the best one.
            
            best_new_score = current_score + 1
            best_swap = None
            
            # Sample a subset of non-edges to save time, or all if small
            # n=40, non-edges ~ 379. Iterating all is fine.
            candidates = []
            for x, y in non_edges:
                # Don't add edges that are already present (checked by non_edges)
                # Don't create multi-edges (checked)
                # Check if swapping creates a valid graph (it does, 1 remove 1 add)
                
                # Calculate book size for new edge (x,y) in current graph
                # Note: Removing (u,v) might change common neighbors of (x,y) if they are involved.
                # But since n is small, we can just simulate the swap on a copy? 
                # No, simulation is expensive. 
                # Approximation: book size of (x,y) is C[x,y] in current A.
                # Removing (u,v) reduces C[x,y] by 1 if u or v is a common neighbor of x,y.
                # Adding (x,y) increases C[u,v] (removed) and others.
                
                # Let's just do a full re-evaluation for the top few candidates to be safe and simple.
                # Or just accept if C[x,y] < max_b.
                if C[x, y] < max_b:
                    candidates.append((x, y))
            
            if not candidates:
                no_improvement_count += 1
                continue
            
            # Pick a random candidate
            x, y = random.choice(candidates)
            u, v = worst_edge
            
            # Perform swap
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            
            new_score = get_max_book(A)
            
            if new_score < current_score:
                current_score = new_score
                no_improvement_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_graph = A.copy()
            else:
                # Revert if not better? Or accept with some probability (Annealing)?
                # For simplicity and determinism in improvement, revert if not better.
                # But this might get stuck. Let's accept if new_score == current_score with prob 0.5?
                # Or just revert to ensure we don't degrade.
                # Given the flatness of max, let's try to minimize sum of squares as secondary.
                # For now, simple hill climbing on max.
                A[u, v] = 1
                A[v, u] = 1
                A[x, y] = 0
                A[y, x] = 0
                no_improvement_count += 1

    # Convert best_graph to list of edges
    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_graph[i, j] == 1:
                result_edges.append([i, j])
                
    return result_edges
