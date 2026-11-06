import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Helper to check if a graph is (5,5)-free
    # Graph represented as list of sets or bitmasks
    # n up to 42, so 64-bit integers work for adjacency masks
    
    def count_k5_and_indep5(adj_masks, n):
        """
        adj_masks: list of integers, adj_masks[i] has bit j set if edge (i,j) exists.
        Returns (num_k5, num_indep5)
        """
        num_k5 = 0
        num_indep5 = 0
        
        for i in range(n-4):
            for j in range(i+1, n-3):
                for k in range(j+1, n-2):
                    for l in range(k+1, n-1):
                        for m in range(l+1, n):
                            # Check K5
                            if ((adj_masks[i] >> j) & 1) and \
                               ((adj_masks[i] >> k) & 1) and \
                               ((adj_masks[i] >> l) & 1) and \
                               ((adj_masks[i] >> m) & 1) and \
                               ((adj_masks[j] >> k) & 1) and \
                               ((adj_masks[j] >> l) & 1) and \
                               ((adj_masks[j] >> m) & 1) and \
                               ((adj_masks[k] >> l) & 1) and \
                               ((adj_masks[k] >> m) & 1) and \
                               ((adj_masks[l] >> m) & 1):
                                num_k5 += 1
                                
                            # Indep5 check
                            if not ((adj_masks[i] >> j) & 1) and \
                               not ((adj_masks[i] >> k) & 1) and \
                               not ((adj_masks[i] >> l) & 1) and \
                               not ((adj_masks[i] >> m) & 1) and \
                               not ((adj_masks[j] >> k) & 1) and \
                               not ((adj_masks[j] >> l) & 1) and \
                               not ((adj_masks[j] >> m) & 1) and \
                               not ((adj_masks[k] >> l) & 1) and \
                               not ((adj_masks[k] >> m) & 1) and \
                               not ((adj_masks[l] >> m) & 1):
                                num_indep5 += 1
                                
        return num_k5, num_indep5

    def is_valid(adj_masks, n):
        k5, i5 = count_k5_and_indep5(adj_masks, n)
        return k5 == 0 and i5 == 0

    def get_edges(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append((i, j))
        return edges

    # Precompute neighbor and non-neighbor lists for fast affected-count computation
    def build_neighbor_lists(adj_masks, n):
        """
        Returns (neighbors, non_neighbors) where:
        neighbors[i] = list of j > i such that (i,j) is an edge
        non_neighbors[i] = list of j > i such that (i,j) is NOT an edge
        """
        neighbors = [[] for _ in range(n)]
        non_neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if (adj_masks[i] >> j) & 1:
                    neighbors[i].append(j)
                else:
                    non_neighbors[i].append(j)
        return neighbors, non_neighbors

    def count_affected_fast(adj_masks, n, u, v, neighbors, non_neighbors):
        """
        Counts K5 and Indep5 that include the pair (u, v) using precomputed neighbor lists.
        Much faster than the naive approach.
        """
        # Ensure u < v for consistent indexing
        if u > v:
            u, v = v, u
        
        num_k5 = 0
        num_indep5 = 0
        
        # For K5: {u,v,a,b,c} must be a clique
        # This means:
        # - (u,v) is an edge
        # - a, b, c must be in N(u) ∩ N(v)
        # - {a,b,c} must form a triangle in the induced subgraph
        
        # For Indep5: {u,v,a,b,c} must be independent
        # This means:
        # - (u,v) is NOT an edge
        # - a, b, c must be in nonN(u) ∩ nonN(v)
        # - {a,b,c} must form an independent set in the induced subgraph
        
        uv_is_edge = (adj_masks[u] >> v) & 1
        
        if uv_is_edge:
            # Count K5 containing (u,v)
            # Find common neighbors of u and v (excluding u and v themselves)
            # Common neighbors: vertices w such that w is neighbor of both u and v
            
            # Get neighbors of u that are > v (to avoid double counting with v's neighbors < u)
            # Actually, let's just get all common neighbors
            common_neighbors = []
            for w in neighbors[u]:
                if w != v and (adj_masks[v] >> w) & 1:
                    common_neighbors.append(w)
            
            # Now count triangles in the induced subgraph on common_neighbors
            # A triangle {a,b,c} in common_neighbors forms a K5 with u,v
            cn_set = set(common_neighbors)
            cn_list = common_neighbors
            
            for idx_a in range(len(cn_list)):
                a = cn_list[idx_a]
                for idx_b in range(idx_a + 1, len(cn_list)):
                    b = cn_list[idx_b]
                    if (adj_masks[a] >> b) & 1:
                        # a-b is an edge, check if there's a c such that c is neighbor of both a and b
                        # and c is in common_neighbors
                        # Count c in common_neighbors that is neighbor of both a and b
                        for idx_c in range(idx_b + 1, len(cn_list)):
                            c = cn_list[idx_c]
                            if (adj_masks[a] >> c) & 1 and (adj_masks[b] >> c) & 1:
                                num_k5 += 1
        
        else:
            # Count Indep5 containing (u,v)
            # Find common non-neighbors of u and v
            common_non_neighbors = []
            for w in non_neighbors[u]:
                if w != v and w not in neighbors[v]:
                    common_non_neighbors.append(w)
            
            # Now count independent triples in the induced subgraph on common_non_neighbors
            cn_list = common_non_neighbors
            
            for idx_a in range(len(cn_list)):
                a = cn_list[idx_a]
                for idx_b in range(idx_a + 1, len(cn_list)):
                    b = cn_list[idx_b]
                    if (adj_masks[a] >> b) & 1 == 0:
                        # a-b is NOT an edge, check if there's a c such that c is non-neighbor of both a and b
                        for idx_c in range(idx_b + 1, len(cn_list)):
                            c = cn_list[idx_c]
                            if (adj_masks[a] >> c) & 1 == 0 and (adj_masks[b] >> c) & 1 == 0:
                                num_indep5 += 1
        
        return num_k5, num_indep5

    # 1. Try Paley Graph on 41 vertices
    n = 41
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    
    adj_masks = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % q in residues:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
                
    if is_valid(adj_masks, n):
        return n, get_edges(adj_masks, n)
        
    # Build neighbor lists for fast affected counting
    neighbors, non_neighbors = build_neighbor_lists(adj_masks, n)
    
    # Initialize global counts
    current_k5, current_i5 = count_k5_and_indep5(adj_masks, n)
    
    best_cost = current_k5 + current_i5
    best_adj = adj_masks[:]
    
    # Search loop with fast affected counting
    while time.time() < deadline:
        # Pick a random edge to flip
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Calculate old counts for affected 5-sets using fast method
        old_k5_aff, old_i5_aff = count_affected_fast(adj_masks, n, u, v, neighbors, non_neighbors)
        
        # Flip edge
        mask_u = adj_masks[u]
        mask_v = adj_masks[v]
        
        # Flip
        adj_masks[u] = mask_u ^ (1 << v)
        adj_masks[v] = mask_v ^ (1 << u)
        
        # Update neighbor lists
        if (mask_u >> v) & 1:
            # Was an edge, now non-edge
            neighbors[u].remove(v)
            non_neighbors[u].append(v)
        else:
            # Was non-edge, now edge
            non_neighbors[u].remove(v)
            neighbors[u].append(v)
        
        # Calculate new counts for affected 5-sets using fast method
        new_k5_aff, new_i5_aff = count_affected_fast(adj_masks, n, u, v, neighbors, non_neighbors)
        
        # Update global counts
        current_k5 = current_k5 - old_k5_aff + new_k5_aff
        current_i5 = current_i5 - old_i5_aff + new_i5_aff
        
        new_cost = current_k5 + current_i5
        
        # Acceptance criterion
        if new_cost < best_cost:
            best_cost = new_cost
            best_adj = adj_masks[:]
            
            if best_cost == 0:
                # Found valid graph!
                return n, get_edges(adj_masks, n)
                
        else:
            # Revert move
            adj_masks[u] = mask_u
            adj_masks[v] = mask_v
            
            # Revert neighbor lists
            if (mask_u >> v) & 1:
                # Was edge, reverted to non-edge
                non_neighbors[u].remove(v)
                neighbors[u].append(v)
            else:
                # Was non-edge, reverted to edge
                neighbors[u].remove(v)
                non_neighbors[u].append(v)
            
            current_k5 = current_k5 - new_k5_aff + old_k5_aff
            current_i5 = current_i5 - new_i5_aff + old_i5_aff
            
    # If we didn't find a valid graph on 41, try 40?
    if is_valid(best_adj, n):
        return n, get_edges(best_adj, n)
        
    # Fallback: Try n=40 with random init + local search?
    if time.time() < deadline - 50:
         n2 = 40
         adj2 = [0] * n2
         for i in range(n2):
             for j in range(i+1, n2):
                 if random.random() < 0.5:
                     adj2[i] |= (1 << j)
                     adj2[j] |= (1 << i)
                     
         neighbors2, non_neighbors2 = build_neighbor_lists(adj2, n2)
         k5_2, i5_2 = count_k5_and_indep5(adj2, n2)
         best_cost_2 = k5_2 + i5_2
         best_adj_2 = adj2[:]
         
         while time.time() < deadline - 10:
             u = random.randint(0, n2-1)
             v = random.randint(0, n2-1)
             if u == v: continue
             if u > v: u, v = v, u
             
             old_k5_aff, old_i5_aff = count_affected_fast(adj2, n2, u, v, neighbors2, non_neighbors2)
             
             mask_u = adj2[u]
             mask_v = adj2[v]
             adj2[u] = mask_u ^ (1 << v)
             adj2[v] = mask_v ^ (1 << u)
             
             # Update neighbor lists
             if (mask_u >> v) & 1:
                 neighbors2[u].remove(v)
                 non_neighbors2[u].append(v)
             else:
                 non_neighbors2[u].remove(v)
                 neighbors2[u].append(v)
             
             new_k5_aff, new_i5_aff = count_affected_fast(adj2, n2, u, v, neighbors2, non_neighbors2)
             
             k5_2 = k5_2 - old_k5_aff + new_k5_aff
             i5_2 = i5_2 - old_i5_aff + new_i5_aff
             
             new_cost_2 = k5_2 + i5_2
             
             if new_cost_2 < best_cost_2:
                 best_cost_2 = new_cost_2
                 best_adj_2 = adj2[:]
                 if best_cost_2 == 0:
                     return n2, get_edges(adj2, n2)
             else:
                 adj2[u] = mask_u
                 adj2[v] = mask_v
                 
                 # Revert neighbor lists
                 if (mask_u >> v) & 1:
                     non_neighbors2[u].remove(v)
                     neighbors2[u].append(v)
                 else:
                     neighbors2[u].remove(v)
                     non_neighbors2[u].append(v)
                 
                 k5_2 = k5_2 - new_k5_aff + old_k5_aff
                 i5_2 = i5_2 - new_i5_aff + old_i5_aff
                 
         if is_valid(best_adj_2, n2):
             return n2, get_edges(best_adj_2, n2)
             
    # Final fallback: Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj37 = [0] * 37
    for i in range(37):
        for j in range(i+1, 37):
            if (j - i) % q in residues:
                adj37[i] |= (1 << j)
                adj37[j] |= (1 << i)
                
    return 37, get_edges(adj37, 37)
