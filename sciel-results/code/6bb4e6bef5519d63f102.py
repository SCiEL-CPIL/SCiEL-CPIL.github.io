"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def _build_adjacency(n, edges):
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj

def _compute_score(n, adj):
    """Compute the max triangle degree sum."""
    deg = adj.sum(axis=1)
    max_score = 0
    # Find all triangles
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                common = np.where(adj[i] & adj[j])[0]
                for k in common:
                    if k > j:
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_score:
                            max_score = s
    return max_score

def _get_all_edges(n):
    return list(combinations(range(n), 2))

def _get_non_edges(n, adj):
    return [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]

def _local_search(n, edges, max_time=100):
    start_time = time.time()
    deadline = start_time + max_time
    
    adj = _build_adjacency(n, edges)
    deg = adj.sum(axis=1)
    current_score = _compute_score(n, adj)
    best_score = current_score
    best_edges = list(edges)
    
    all_edges = _get_all_edges(n)
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # Precompute for speed
    n2 = n * n
    
    while time.time() < deadline:
        non_edges = _get_non_edges(n, adj)
        if not non_edges:
            break
        
        improved = False
        # Try edge swaps: remove one edge, add another
        # Focus on swaps that might reduce max triangle degree sum
        for e_remove in list(edge_set):
            if time.time() >= deadline:
                break
            u1, v1 = e_remove
            # Try adding each non-edge
            for e_add in non_edges:
                u2, v2 = e_add
                if (u2, v2) in edge_set or (v2, u2) in edge_set:
                    continue
                # Skip if this would create parallel structure issues
                # Check if swap is valid (simple graph)
                if u2 == u1 and v2 == v1:
                    continue
                
                # Tentative swap
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                
                new_deg = adj.sum(axis=1)
                new_score = _compute_score(n, adj)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = [list(e) for e in edge_set | {tuple(sorted(e_add))} - {tuple(sorted(e_remove))}]
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
                    continue
                
                # If improved, update edge_set
                edge_set.remove(tuple(sorted(e_remove)))
                edge_set.add(tuple(sorted(e_add)))
                break
        
        if not improved:
            # Try single edge moves if no swaps help
            found_move = False
            for e_remove in list(edge_set):
                if time.time() >= deadline:
                    break
                u1, v1 = e_remove
                for e_add in non_edges:
                    u2, v2 = e_add
                    if (u2, v2) in edge_set or (v2, u2) in edge_set:
                        continue
                    if u2 == u1 and v2 == v1:
                        continue
                    
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    
                    new_score = _compute_score(n, adj)
                    
                    if new_score < best_score:
                        best_score = new_score
                        edge_set.remove(tuple(sorted(e_remove)))
                        edge_set.add(tuple(sorted(e_add)))
                        best_edges = [list(e) for e in edge_set]
                        found_move = True
                        break
                    else:
                        adj[u1, v1] = 1
                        adj[v1, u1] = 1
                        adj[u2, v2] = 0
                        adj[v2, u2] = 0
                
                if found_move:
                    break
            
            if not found_move:
                break
    
    return best_edges, best_score

def construct():
    n = 16
    target_edges = 65
    
    # Start from K_{8,8} + one internal edge (balanced construction)
    initial_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    
    # Run local search
    best_edges, best_score = _local_search(n, initial_edges, max_time=110)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
