import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    edge_map = {}
    
    # Generate all edges
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_map[(u, v)] = len(edges) - 1
    
    n_edges = len(edges)
    
    # Precompute for each edge: list of C4s it belongs to
    edge_to_c4s = [[] for _ in range(n_edges)]
    all_c4s = []
    
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            idx1 = edge_map[e1]
            idx2 = edge_map[e2]
            idx3 = edge_map[e3]
            
            c4 = frozenset([idx, idx1, idx2, idx3])
            all_c4s.append(c4)
            edge_to_c4s[idx].append(len(all_c4s) - 1)
    
    n_c4s = len(all_c4s)
    
    # Build: for each C4, list of its 4 edge indices
    c4_edges = [list(c4) for c4 in all_c4s]
    
    # Convert to numpy arrays for faster operations
    c4_edges_np = np.array(c4_edges, dtype=np.int32)  # shape (n_c4s, 4)
    
    # For each edge, precompute which C4s contain it (as numpy array)
    edge_c4_indices = [np.array(edge_to_c4s[e], dtype=np.int32) for e in range(n_edges)]
    
    def greedy_hitting_set(deleted_np, rng, top_k=3):
        """
        Greedy hitting set using numpy.
        deleted_np: boolean array of shape (n_edges,)
        Returns updated deleted_np
        """
        # deleted_count per C4: how many of its 4 edges are deleted
        deleted_count = np.zeros(n_c4s, dtype=np.int32)
        uncovered = np.ones(n_c4s, dtype=bool)
        
        while np.any(uncovered):
            # For uncovered C4s, count how many non-deleted edges each edge covers
            # An edge covers a C4 if the C4 is uncovered and the edge is not deleted
            # and the C4 contains the edge
            
            # Build edge coverage: for each edge, count uncovered C4s it belongs to
            edge_coverage = np.zeros(n_edges, dtype=np.int32)
            
            # Get indices of uncovered C4s
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            
            # For each uncovered C4, its 4 edges
            unc_c4_edges = c4_edges_np[unc_idx]  # shape (len(unc_idx), 4)
            
            # Count coverage for each edge
            for e in unc_c4_edges:
                edge_coverage[e] += 1
            
            # Only consider edges not already deleted
            edge_coverage[deleted_np] = 0
            
            max_count = np.max(edge_coverage)
            if max_count == 0:
                break
            
            # Get candidates with max coverage
            max_candidates = np.where(edge_coverage == max_count)[0]
            
            # Randomized: pick from top-k
            if len(max_candidates) > top_k and rng.random() < 0.5:
                # Pick from top-k
                top_k_candidates = np.sort(np.where(edge_coverage >= max_count - 1)[0])
                if len(top_k_candidates) > top_k:
                    chosen = rng.choice(top_k_candidates[:top_k])
                else:
                    chosen = rng.choice(top_k_candidates)
            else:
                chosen = rng.choice(max_candidates)
            
            deleted_np[chosen] = True
            
            # Update uncovered: C4s containing chosen edge become covered
            # A C4 is covered if any of its edges is deleted
            # Mark C4s containing chosen as covered
            c4s_with_chosen = edge_c4_indices[chosen]
            uncovered[c4s_with_chosen] = False
        
        return deleted_np
    
    def prune_deletions(deleted_np, rng, max_iter=50):
        """
        Try to remove edges from the hitting set if the C4s they cover are still hit by others.
        """
        # deleted_count per C4
        deleted_count = np.zeros(n_c4s, dtype=np.int32)
        for e in range(n_edges):
            if deleted_np[e]:
                deleted_count[edge_c4_indices[e]] += 1
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            
            # Try removing each deleted edge
            for e in range(n_edges):
                if not deleted_np[e]:
                    continue
                
                # Check: is every C4 that e covers also covered by another deleted edge?
                safe = True
                for c4_idx in edge_c4_indices[e]:
                    if deleted_count[c4_idx] <= 1:
                        safe = False
                        break
                
                if safe:
                    deleted_np[e] = False
                    deleted_count[edge_c4_indices[e]] -= 1
                    improved = True
        
        return deleted_np
    
    def local_search_swap(deleted_np, rng, max_iter=100):
        """
        Try to swap: remove one deleted edge and add one non-deleted edge if it reduces total deletions.
        Or try to remove a deleted edge if all its C4s are still covered.
        """
        # deleted_count per C4
        deleted_count = np.zeros(n_c4s, dtype=np.int32)
        for e in range(n_edges):
            if deleted_np[e]:
                deleted_count[edge_c4_indices[e]] += 1
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            
            # Try removing each deleted edge
            for e in range(n_edges):
                if not deleted_np[e]:
                    continue
                
                # Check if safe to remove
                safe = True
                for c4_idx in edge_c4_indices[e]:
                    if deleted_count[c4_idx] <= 1:
                        safe = False
                        break
                
                if safe:
                    deleted_np[e] = False
                    deleted_count[edge_c4_indices[e]] -= 1
                    improved = True
        
        return deleted_np
    
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    best_deleted = np.zeros(n_edges, dtype=bool)
    best_size = 0
    
    # Multiple restarts with different randomizations
    n_restarts = 200
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        deleted = np.zeros(n_edges, dtype=bool)
        
        # Randomized greedy hitting set
        # Vary top_k based on restart
        if restart % 3 == 0:
            top_k = 1  # Pure greedy
        elif restart % 3 == 1:
            top_k = 2
        else:
            top_k = 3
        
        deleted = greedy_hitting_set(deleted, rng, top_k=top_k)
        
        # Prune
        deleted = prune_deletions(deleted, rng, max_iter=30)
        
        # Local search swap
        deleted = local_search_swap(deleted, rng, max_iter=50)
        
        size = n_edges - np.sum(deleted)
        if size > best_size:
            best_size = size
            best_deleted = deleted.copy()
        
        # Periodic deep prune on best
        if restart % 10 == 0 and best_size > 0:
            best_deleted = prune_deletions(best_deleted, rng, max_iter=100)
            best_size = n_edges - np.sum(best_deleted)
    
    # Final deep prune
    if time.time() - start_time < time_limit - 5:
        best_deleted = prune_deletions(best_deleted, rng, max_iter=200)
        best_size = n_edges - np.sum(best_deleted)
    
    # Convert to list of [u, v]
    result = []
    for idx in range(n_edges):
        if not best_deleted[idx]:
            result.append(list(edges[idx]))
    
    return result
