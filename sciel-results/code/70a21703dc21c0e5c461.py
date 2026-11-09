import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Initial graph: 10 disjoint triangles
    n = 30
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    
    # Adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best_n = n
    best_edges = edges[:]
    
    # Helper: Check C4 constraint for adding vertex v with neighbors N
    def is_c4_free(adj, n, new_v, new_neighbors):
        # For any two neighbors u, w of new_v, N(u) and N(w) must be disjoint (excluding new_v)
        # Since new_v is not in adj yet, we just check N(u) ∩ N(w) == empty
        for i in range(len(new_neighbors)):
            for j in range(i+1, len(new_neighbors)):
                u = new_neighbors[i]
                w = new_neighbors[j]
                if adj[u] & adj[w]:
                    return False
        return True

    # Helper: Check if graph has independent set of size k
    def has_independent_set(adj, n, k):
        # Backtracking search
        # Order vertices by degree (ascending) to prune faster
        # But n is small, so simple backtracking might work
        # We want to find an independent set of size k
        # If we can find one, return True
        # If we exhaust possibilities, return False
        
        # Use a greedy heuristic to order vertices
        # Actually, let's just use a simple recursive search with pruning
        # State: current independent set, remaining vertices
        # Pruning: if len(current) + len(remaining) < k, return False
        
        # To speed up, we can use bitmasks for n <= 64
        # n <= 43, so bitmasks are perfect.
        
        # Convert adj to bitmask list
        adj_mask = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_mask[i] |= (1 << j)
        
        # We want to find a set of k vertices such that no two are adjacent.
        # This is equivalent to finding a clique of size k in the complement graph.
        
        # Let's use a simple backtracking on the complement graph.
        # Complement adj: i and j are adjacent if i not in adj[j]
        
        # Function to find clique of size k in complement
        def find_clique_size_k(adj_mask, k, n):
            # Use Bron-Kerbosch or simple backtracking
            # Simple backtracking:
            # Try to add vertices one by one
            
            # Sort vertices by degree in complement (descending) to find large cliques faster
            # Degree in complement = n - 1 - degree in original
            degs = [n - 1 - len(adj[i]) for i in range(n)]
            order = sorted(range(n), key=lambda x: degs[x], reverse=True)
            
            # Recursive search
            def backtrack(current_clique, candidates):
                if len(current_clique) == k:
                    return True
                if len(candidates) < k - len(current_clique):
                    return False
                
                # Try adding each candidate
                for i in range(len(candidates)):
                    v = candidates[i]
                    # Check if v is compatible with all in current_clique
                    compatible = True
                    for u in current_clique:
                        if adj_mask[u] & (1 << v):
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    new_clique = current_clique + [v]
                    new_candidates = [w for w in candidates[i+1:] if not (adj_mask[v] & (1 << w))]
                    if backtrack(new_clique, new_candidates):
                        return True
                return False
            
            return backtrack([], order)
        
        return find_clique_size_k(adj_mask, k, n)

    # Try to add vertices
    for target_n in range(31, 44):
        if time.time() > deadline:
            break
        
        added = False
        attempts = 0
        while not added and attempts < 1000 and time.time() < deadline:
            attempts += 1
            
            # Choose a random neighbor for each triangle
            new_neighbors = []
            for t in range(10):
                # Pick a random vertex from triangle t
                v = random.choice([3*t, 3*t+1, 3*t+2])
                new_neighbors.append(v)
            
            # Check C4
            if not is_c4_free(adj, n, target_n, new_neighbors):
                continue
            
            # Check independence number
            # Temporarily add the vertex
            new_adj = [set(s) for s in adj]
            new_adj.append(set())
            for u in new_neighbors:
                new_adj[u].add(target_n)
                new_adj[target_n].add(u)
            
            if has_independent_set(new_adj, target_n, 11):
                # Invalid
                continue
            else:
                # Valid!
                # Update adj and edges
                for u in new_neighbors:
                    adj[u].add(target_n)
                    adj[target_n].add(u)
                    edges.append([u, target_n])
                adj.append(set(new_neighbors))
                n = target_n
                best_n = n
                best_edges = edges[:]
                added = True
                break
        
        if not added:
            # If we can't add a vertex, we might need to stop or try a different strategy
            # For now, just break
            break
    
    return {'n': best_n, 'edges': best_edges}
