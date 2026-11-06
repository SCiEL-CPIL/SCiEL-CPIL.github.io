import numpy as np
import time
import random

def count_triangles(adj):
    """Count triangles in a graph given by boolean adjacency matrix."""
    a3 = (adj.astype(np.int32) @ adj.astype(np.int32)) @ adj.astype(np.int32)
    return np.trace(a3) // 6

def _paley_37_edges():
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

def _build_adjacency(n, edges):
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    return adj

def _has_k5(adj):
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j]:
                continue
            for k in range(j + 1, n):
                if not adj[i, k] or not adj[j, k]:
                    continue
                for l in range(k + 1, n):
                    if not adj[i, l] or not adj[j, l] or not adj[k, l]:
                        continue
                    for m in range(l + 1, n):
                        if adj[i, m] and adj[j, m] and adj[k, m] and adj[l, m]:
                            return True
    return False

def _has_ind5(adj):
    n = adj.shape[0]
    # complement adjacency
    comp = ~adj
    np.fill_diagonal(comp, False)
    for i in range(n):
        for j in range(i + 1, n):
            if not comp[i, j]:
                continue
            for k in range(j + 1, n):
                if not comp[i, k] or not comp[j, k]:
                    continue
                for l in range(k + 1, n):
                    if not comp[i, l] or not comp[j, l] or not comp[k, l]:
                        continue
                    for m in range(l + 1, n):
                        if comp[i, m] and comp[j, m] and comp[k, m] and comp[l, m]:
                            return True
    return False

def _is_valid(adj):
    return not _has_k5(adj) and not _has_ind5(adj)

def construct():
    # Seed graph: Paley 37
    seed_edges = _paley_37_edges()
    best_n = 37
    best_edges = seed_edges
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290
    
    # Try to expand Paley 37 to 42 vertices
    # Strategy: Start with Paley 37, then try to add vertices one by one.
    # When adding a vertex, we need to choose its adjacency to existing vertices
    # such that no K5 or I5 is formed.
    
    # We can use a simple heuristic:
    # For each new vertex, try all 2^37 possible adjacency vectors? No, too many.
    # Instead, we can use a randomized local search for the new vertex's adjacency.
    
    # Alternatively, we can start with a random graph on 42 vertices and use SA,
    # but initialize it close to the Paley graph structure where possible.
    
    # Let's stick to the SA approach but improve initialization and speed.
    # The previous SA was too slow due to O(n^5) init and update costs.
    
    # Let's try a different approach:
    # 1. Start with Paley 37.
    # 2. Add a new vertex. Its adjacency to the 37 vertices is a 37-bit vector.
    # 3. Check if adding this vertex creates a K5 or I5.
    #    - K5: The new vertex + 4 existing vertices forming a K4 in the existing graph
    #      where the new vertex is connected to all 4.
    #    - I5: The new vertex + 4 existing vertices forming an I4 in the existing graph
    #      where the new vertex is NOT connected to any of the 4.
    
    # So, for a new vertex v, let S be the set of vertices it connects to.
    # We need:
    # 1. No K4 in G[S] (because v + K4 in S = K5)
    # 2. No I4 in G[~S] (because v + I4 in ~S = I5)
    
    # This is a constraint on S. We can search for a valid S.
    
    # Let's implement this.
    
    def add_vertex(adj, n):
        # adj is n x n, we want to add vertex n
        # S is a subset of {0, ..., n-1}
        # We want to find S such that:
        # - No K4 in adj[S, S]
        # - No I4 in (~adj)[~S, ~S]
        
        # We can use a randomized search.
        # Or we can use a greedy approach.
        
        # Let's try a few random S values.
        for _ in range(10000):
            if time.time() > deadline:
                return None
            
            # Random S
            s = np.random.rand(n) < 0.5
            
            # Check K4 in G[S]
            # Get indices of S
            s_idx = np.where(s)[0]
            if len(s_idx) < 4:
                # No K4 possible
                pass
            else:
                sub_adj = adj[np.ix_(s_idx, s_idx)]
                # Check for K4
                found_k4 = False
                for i in range(len(s_idx)):
                    for j in range(i + 1, len(s_idx)):
                        if not sub_adj[i, j]:
                            continue
                        for k in range(j + 1, len(s_idx)):
                            if not sub_adj[i, k] or not sub_adj[j, k]:
                                continue
                            for l in range(k + 1, len(s_idx)):
                                if sub_adj[i, l] and sub_adj[j, l] and sub_adj[k, l]:
                                    found_k4 = True
                                    break
                            if found_k4:
                                break
                        if found_k4:
                            break
                    if found_k4:
                        break
                if found_k4:
                    continue
            
            # Check I4 in G[~S]
            t_idx = np.where(~s)[0]
            if len(t_idx) < 4:
                pass
            else:
                sub_adj_t = ~adj[np.ix_(t_idx, t_idx)]
                np.fill_diagonal(sub_adj_t, False)
                # Check for K4 in sub_adj_t (which is I4 in G[~S])
                found_k4 = False
                for i in range(len(t_idx)):
                    for j in range(i + 1, len(t_idx)):
                        if not sub_adj_t[i, j]:
                            continue
                        for k in range(j + 1, len(t_idx)):
                            if not sub_adj_t[i, k] or not sub_adj_t[j, k]:
                                continue
                            for l in range(k + 1, len(t_idx)):
                                if sub_adj_t[i, l] and sub_adj_t[j, l] and sub_adj_t[k, l]:
                                    found_k4 = True
                                    break
                            if found_k4:
                                break
                        if found_k4:
                            break
                    if found_k4:
                        break
                if found_k4:
                    continue
            
            # Valid S found
            # Create new adjacency matrix
            new_adj = np.zeros((n + 1, n + 1), dtype=bool)
            new_adj[:n, :n] = adj
            new_adj[n, :n] = s
            new_adj[:n, n] = s
            return new_adj
        
        return None
    
    # Start with Paley 37
    adj = _build_adjacency(37, seed_edges)
    
    # Try to expand to 42
    for _ in range(5):
        if time.time() > deadline:
            break
        
        new_adj = add_vertex(adj, adj.shape[0])
        if new_adj is None:
            break
        
        adj = new_adj
    
    # Check if we found a valid graph
    if adj.shape[0] > best_n:
        if _is_valid(adj):
            n = adj.shape[0]
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            best_n = n
            best_edges = edges
    
    return best_n, best_edges
