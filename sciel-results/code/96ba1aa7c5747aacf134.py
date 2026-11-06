import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    n = 16
    m = 65
    
    # Helper to create the seed graph: K_{8,8} + edge (0,1)
    def get_seed_edges():
        edges = set()
        # Bipartite edges between 0-7 and 8-15
        for u in range(8):
            for v in range(8, 16):
                edges.add((min(u, v), max(u, v)))
        # Add one internal edge to reach 65 edges (64 + 1)
        edges.add((0, 1))
        return list(edges)

    def edges_to_adjacency(edges):
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj

    def compute_degrees(adj):
        return [sum(adj[i]) for i in range(n)]

    def find_triangles(adj):
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            triangles.append((i, j, k))
        return triangles

    def compute_score(adj, triangles):
        if not triangles:
            return 0
        degs = compute_degrees(adj)
        max_sum = 0
        for u, v, w in triangles:
            s = degs[u] + degs[v] + degs[w]
            if s > max_sum:
                max_sum = s
        return max_sum

    # Initial seed
    seed_edges = get_seed_edges()
    seed_adj = edges_to_adjacency(seed_edges)
    seed_triangles = find_triangles(seed_adj)
    best_score = compute_score(seed_adj, seed_triangles)
    best_edges = seed_edges.copy()

    # List of initial states to explore
    initial_states = [seed_edges.copy()]
    
    # Generate some diverse initial states by random swaps
    random.seed(42)
    for _ in range(10):
        current_edges = seed_edges.copy()
        # Perform a few random valid swaps
        for _ in range(5):
            e1 = random.choice(current_edges)
            # Find a non-edge to swap with
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            non_edges = [p for p in all_pairs if p not in current_edges]
            if not non_edges:
                break
            e2 = random.choice(non_edges)
            # Swap: remove e1, add e2
            current_edges.remove(e1)
            current_edges.append(e2)
        initial_states.append(current_edges)

    def local_search(initial_edges):
        current_edges = initial_edges.copy()
        current_adj = edges_to_adjacency(current_edges)
        current_triangles = find_triangles(current_adj)
        current_score = compute_score(current_adj, current_triangles)
        
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 5:
                return current_score, current_edges
            
            improved = False
            best_swap_score = current_score
            best_swap = None
            
            # Get current edges set for fast lookup
            edge_set = set(current_edges)
            
            # Iterate over all possible swaps: remove (u,v), add (x,y)
            # To be efficient, we only consider swaps that maintain simple graph property.
            # There are ~65 edges and ~121 non-edges. 65*121 ~ 7865 swaps.
            # For each swap, we need to update triangles and score.
            
            # Optimization: Pre-compute neighbors for faster triangle updates?
            # Given n=16, full recalculation is fast enough.
            
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in edge_set:
                        non_edges.append((i, j))
            
            for e_remove in current_edges:
                u, v = e_remove
                for e_add in non_edges:
                    x, y = e_add
                    # Check if swap is valid (doesn't create duplicate, etc - already ensured by sets)
                    
                    # Apply swap
                    current_adj[u][v] = False
                    current_adj[v][u] = False
                    current_adj[x][y] = True
                    current_adj[y][x] = True
                    
                    # Recompute triangles and score
                    # This is O(n^3) = 4096 ops, very fast.
                    new_triangles = find_triangles(current_adj)
                    new_score = compute_score(current_adj, new_triangles)
                    
                    # Revert swap
                    current_adj[u][v] = True
                    current_adj[v][u] = True
                    current_adj[x][y] = False
                    current_adj[y][x] = False
                    
                    if new_score < best_swap_score:
                        best_swap_score = new_score
                        best_swap = (e_remove, e_add)
                        
            if best_swap:
                e_remove, e_add = best_swap
                current_edges.remove(e_remove)
                current_edges.append(e_add)
                current_score = best_swap_score
                improved = True
                
        return current_score, current_edges

    # Run local search from each initial state
    for init_edges in initial_states:
        if time.time() - start_time > time_limit - 10:
            break
            
        score, edges = local_search(init_edges)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            # If we hit the theoretical lower bound (23), we can stop early
            if best_score <= 23:
                break

    # Return the best graph found
    return {'n': 16, 'edges': best_edges}
