"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    n = 16
    target_edges = 65  # floor(16^2/4) + 1 = 64 + 1

    # Start with K_{8,8} plus one edge inside a part (the seed construction)
    # Part A: vertices 0-7, Part B: vertices 8-15
    # K_{8,8} has 64 edges, add one edge within Part A
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((min(u,v), max(u,v)))
    edges.add((0, 1))  # one edge within Part A
    edges = list(edges)
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    def compute_degrees(adj):
        return adj.sum(axis=1).astype(int)
    
    def find_triangles(adj):
        """Find all triangles in the graph."""
        triangles = []
        n = adj.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    # Find common neighbors
                    common = np.where((adj[i] & adj[j]) == 1)[0]
                    for k in common:
                        if k > j:
                            triangles.append((i, j, k))
        return triangles
    
    def max_triangle_degree_sum(adj, degrees):
        """Compute the maximum degree sum over all triangles."""
        triangles = find_triangles(adj)
        if not triangles:
            return 0
        max_sum = 0
        worst_tri = None
        for (i, j, k) in triangles:
            s = degrees[i] + degrees[j] + degrees[k]
            if s > max_sum:
                max_sum = s
                worst_tri = (i, j, k)
        return max_sum, worst_tri
    
    def score_graph(adj):
        degrees = compute_degrees(adj)
        max_sum, worst_tri = max_triangle_degree_sum(adj, degrees)
        return max_sum, worst_tri, degrees
    
    # Initial score
    current_score, current_worst, current_degrees = score_graph(adj)
    best_score = current_score
    best_adj = adj.copy()
    
    # Edge replacement heuristic
    # Strategy: find edges in worst triangles and try to replace them with non-edges
    # that reduce the degree sum of the worst triangle
    
    # Precompute all possible edges
    all_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_edges.add((i, j))
    
    def get_edge_set(adj):
        s = set()
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    s.add((i, j))
        return s
    
    def apply_swap(adj, remove_edge, add_edge):
        """Apply an edge swap: remove one edge, add another."""
        new_adj = adj.copy()
        u, v = remove_edge
        new_adj[u, v] = 0
        new_adj[v, u] = 0
        a, b = add_edge
        new_adj[a, b] = 1
        new_adj[b, a] = 1
        return new_adj
    
    # Local search with edge swaps
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        current_score, current_worst, current_degrees = score_graph(adj)
        
        if current_score <= best_score:
            best_score = current_score
            best_adj = adj.copy()
        
        # Get current edge set
        current_edges = get_edge_set(adj)
        non_edges = all_edges - current_edges
        
        # Find edges that are in the worst triangle(s)
        # Try swapping each edge in the worst triangle with a non-edge
        # that doesn't create a worse triangle
        
        # Collect all edges involved in high-degree-sum triangles
        triangles = find_triangles(adj)
        if not triangles:
            break
        
        # Sort triangles by degree sum descending
        tri_sums = [(degrees[i]+degrees[j]+degrees[k], (i,j,k)) for (i,j,k) in triangles]
        tri_sums.sort(key=lambda x: -x[0])
        
        # Focus on triangles with sum >= current_score - 1
        candidate_edges = set()
        for s, tri in tri_sums:
            if s >= current_score - 1:
                i, j, k = tri
                candidate_edges.add((min(i,j), max(i,j)))
                candidate_edges.add((min(i,k), max(i,k)))
                candidate_edges.add((min(j,k), max(j,k)))
        
        # For each candidate edge, try swapping with non-edges
        best_swap = None
        best_swap_score = current_score
        
        for rem_edge in sorted(candidate_edges):
            if rem_edge not in current_edges:
                continue
            # Try adding various non-edges
            for add_edge in non_edges:
                # Skip if it would create a triangle with the removed edge's endpoints
                # that is worse
                new_adj = apply_swap(adj, rem_edge, add_edge)
                new_score, _, _ = score_graph(new_adj)
                if new_score < best_swap_score:
                    best_swap_score = new_score
                    best_swap = (rem_edge, add_edge, new_adj)
                if time.time() > deadline - 10.0:
                    break
            if time.time() > deadline - 10.0:
                break
        
        if best_swap is not None:
            rem_edge, add_edge, new_adj = best_swap
            adj = new_adj
            improved = True
    
    # Final polish: try more extensive local search
    if time.time() < deadline - 15.0:
        # Try random restarts with perturbation
        for restart in range(50):
            if time.time() > deadline - 10.0:
                break
            
            # Start from best known
            adj = best_adj.copy()
            
            # Perturb: remove 2-3 edges and add 2-3 random edges
            current_edges = get_edge_set(adj)
            non_edges = all_edges - current_edges
            
            num_perturb = random.randint(1, 3)
            for _ in range(num_perturb):
                if not current_edges or not non_edges:
                    break
                rem = random.choice(list(current_edges))
                add = random.choice(list(non_edges))
                adj = apply_swap(adj, rem, add)
                current_edges = get_edge_set(adj)
                non_edges = all_edges - current_edges
            
            # Local search from perturbed state
            for iteration in range(200):
                if time.time() > deadline - 10.0:
                    break
                
                current_score, current_worst, current_degrees = score_graph(adj)
                if current_score <= best_score:
                    best_score = current_score
                    best_adj = adj.copy()
                
                triangles = find_triangles(adj)
                if not triangles:
                    break
                
                tri_sums = [(degrees[i]+degrees[j]+degrees[k], (i,j,k)) for (i,j,k) in triangles]
                tri_sums.sort(key=lambda x: -x[0])
                
                candidate_edges = set()
                for s, tri in tri_sums:
                    if s >= current_score - 1:
                        i, j, k = tri
                        candidate_edges.add((min(i,j), max(i,j)))
                        candidate_edges.add((min(i,k), max(i,k)))
                        candidate_edges.add((min(j,k), max(j,k)))
                
                current_edges = get_edge_set(adj)
                non_edges = all_edges - current_edges
                
                best_swap = None
                best_swap_score = current_score
                
                for rem_edge in sorted(candidate_edges):
                    if rem_edge not in current_edges:
                        continue
                    for add_edge in non_edges:
                        new_adj = apply_swap(adj, rem_edge, add_edge)
                        new_score, _, _ = score_graph(new_adj)
                        if new_score < best_swap_score:
                            best_swap_score = new_score
                            best_swap = (rem_edge, add_edge, new_adj)
                    if time.time() > deadline - 10.0:
                        break
                    if time.time() > deadline - 10.0:
                        break
                
                if best_swap is not None:
                    rem_edge, add_edge, new_adj = best_swap
                    adj = new_adj
                else:
                    break
    
    # Return the best graph found
    final_adj = best_adj
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if final_adj[i, j]:
                final_edges.append([i, j])
    
    # Verify exactly 65 edges
    if len(final_edges) != target_edges:
        # Fallback to seed construction
        final_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    
    return {'n': n, 'edges': final_edges}
# EVOLVE-BLOCK-END
