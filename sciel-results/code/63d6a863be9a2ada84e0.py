import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency as list of sets for easy intersection, or bitmasks
    # Bitmasks are faster for intersection and subset checks
    adj_masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
    
    def has_clique_of_size_k(adj, k):
        """
        Check if graph with adjacency masks 'adj' contains a clique of size k.
        Optimized for small k (4 or 5).
        """
        n = len(adj)
        if k == 1: return n > 0
        if k == 2:
            for i in range(n):
                if adj[i] != 0: return True
            return False
        if k == 3:
            for i in range(n):
                # Find neighbors of i
                neighbors = adj[i]
                # For each pair of neighbors, check if they are connected
                # Iterate through set bits of neighbors
                # Optimization: pick the smallest degree neighbor to iterate?
                # For k=3, just check intersections
                # Let's iterate pairs in neighbors
                # Get list of neighbors
                neigh_list = []
                temp = neighbors
                while temp:
                    lsb = temp & -temp
                    idx = (lsb.bit_length() - 1)
                    neigh_list.append(idx)
                    temp ^= lsb
                for u in neigh_list:
                    # Check if u is connected to any other in neigh_list
                    # adj[u] & neighbors (excluding u)
                    if adj[u] & (neighbors ^ (1 << u)):
                        return True
            return False
        if k == 4:
            # Check for K4
            # For each edge (u, v), check if N(u) \cap N(v) contains a K2 (edge)
            # Or simpler: For each vertex u, check if N(u) contains a K3
            for u in range(n):
                if has_clique_of_size_k_in_induced(adj, n, u, 3):
                    return True
            return False
        if k == 5:
            # For each vertex u, check if N(u) contains a K4
            for u in range(n):
                if has_clique_of_size_k_in_induced(adj, n, u, 4):
                    return True
            return False
        return False

    def has_clique_of_size_k_in_induced(adj, n, center, k):
        """
        Check if the induced subgraph on N(center) has a clique of size k.
        """
        if k <= 1: return True
        # Get neighbors of center
        neighbors_mask = adj[center]
        if neighbors_mask == 0: return False
        
        # Extract list of neighbors
        neigh_list = []
        temp = neighbors_mask
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            neigh_list.append(idx)
            temp ^= lsb
        
        # Recursively check for K_k in this induced subgraph
        # But we need to be careful: the adjacency within this subgraph is determined by 'adj'
        # We can map these vertices to a smaller index space or just use the masks directly.
        # For k=3 or 4, the list is small enough.
        
        if k == 2:
            # Check if any two neighbors are connected
            for i in range(len(neigh_list)):
                for j in range(i+1, len(neigh_list)):
                    if adj[neigh_list[i]] & (1 << neigh_list[j]):
                        return True
            return False
        if k == 3:
            # Check for triangle in neighbors
            for i in range(len(neigh_list)):
                u = neigh_list[i]
                # Neighbors of u within the neighbor set
                # adj[u] & neighbors_mask (excluding u)
                common = adj[u] & neighbors_mask & ~(1 << u)
                if common == 0: continue
                # Check if common contains an edge
                # Pick one vertex v from common, check if it connects to another in common
                # Or just check if any pair in common is connected
                # Iterate through common
                c_list = []
                temp_c = common
                while temp_c:
                    lsb = temp_c & -temp_c
                    idx = (lsb.bit_length() - 1)
                    c_list.append(idx)
                    temp_c ^= lsb
                for a in c_list:
                    if adj[a] & common & ~(1 << a):
                        return True
            return False
        if k == 4:
            # Check for K4 in neighbors
            # For each u in neighbors, check if N(u) \cap N(center) contains a K3
            for u in neigh_list:
                # Intersection of N(u) and N(center)
                common = adj[u] & neighbors_mask & ~(1 << u)
                if common == 0: continue
                # Check if common contains a K3
                # This is getting deep. Let's simplify.
                # Since n is small, we can just extract the subgraph and check.
                # But let's try to be smart.
                # A K4 in N(center) implies a K3 in N(u) \cap N(center) for any u in the K4.
                # So we just need to check if any induced subgraph on (N(u) \cap N(center)) has a K3.
                # We can reuse the logic for K3 on the set 'common'.
                
                # To avoid code duplication, let's write a helper that checks K_k on a given set of vertices
                # defined by a mask, using the global adj.
                pass # Fallback to slower method if needed, but for now, let's just do direct check for K4 in N(center)
            
            # Direct check for K4 in N(center)
            # For each pair (u, v) in N(center) that are connected,
            # check if (N(u) \cap N(v) \cap N(center)) contains a K2.
            for i in range(len(neigh_list)):
                u = neigh_list[i]
                for j in range(i+1, len(neigh_list)):
                    v = neigh_list[j]
                    if adj[u] & (1 << v):
                        # u and v are connected
                        # Common neighbors within N(center)
                        common = adj[u] & adj[v] & neighbors_mask & ~(1 << u) & ~(1 << v)
                        if common == 0: continue
                        # Check if common contains an edge
                        c_list = []
                        temp_c = common
                        while temp_c:
                            lsb = temp_c & -temp_c
                            idx = (lsb.bit_length() - 1)
                            c_list.append(idx)
                            temp_c ^= lsb
                        for a in c_list:
                            if adj[a] & common & ~(1 << a):
                                return True
            return False
        return False

    def has_independent_set_of_size_k(adj, k):
        """
        Check if graph has an independent set of size k.
        Equivalent to checking for clique of size k in complement.
        """
        n = len(adj)
        # Build complement masks
        comp_adj = [0] * n
        for i in range(n):
            # All bits set for n vertices
            all_bits = (1 << n) - 1
            # Remove i
            # Remove neighbors
            comp_adj[i] = all_bits & ~(adj[i] | (1 << i))
        
        return has_clique_of_size_k(comp_adj, k)

    def is_valid(adj, n):
        if has_clique_of_size_k(adj, 5):
            return False
        if has_independent_set_of_size_k(adj, 5):
            return False
        return True

    # Verify seed
    if not is_valid(adj_masks, n):
        # Fallback to empty edges if seed is broken (shouldn't happen)
        return n, []

    best_n = n
    best_adj = adj_masks[:]
    best_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                best_edges.append((i, j))

    # 2. Try to extend
    current_n = n
    current_adj = adj_masks
    
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline:
            break
            
        found = False
        max_attempts = 5000 # Increase attempts
        
        # Heuristic: The neighbor set size should be around half
        # For Paley(37), degree is 18.
        # Let's try sizes around current_n / 2
        
        while not found:
            if time.time() > deadline:
                break
            max_attempts -= 1
            if max_attempts < 0:
                break
                
            # Random neighbor set
            # Bias towards sizes that work
            # In Ramsey graphs, degrees are often balanced
            size = random.randint(current_n // 3, (2 * current_n) // 3)
            neighbors = set(random.sample(range(current_n), size))
            
            # Check if adding vertex connected to 'neighbors' is valid
            # Condition 1: G[neighbors] has no K4
            # Condition 2: G[non-neighbors] has no I4 (i.e., K4 in complement)
            
            # Check K4 in neighbors
            valid = True
            # Extract induced subgraph on neighbors
            # Map neighbors to 0..size-1? No, just use masks.
            
            # Check K4 in G[neighbors]
            # We can reuse has_clique_of_size_k but it expects a full list of adj masks.
            # Let's write a specific check for a subset.
            
            def has_k_in_subset(adj, subset_mask, k):
                if k == 4:
                    # Check for K4 in subset
                    # Iterate pairs in subset
                    # Get list
                    s_list = []
                    temp = subset_mask
                    while temp:
                        lsb = temp & -temp
                        idx = (lsb.bit_length() - 1)
                        s_list.append(idx)
                        temp ^= lsb
                    for i in range(len(s_list)):
                        u = s_list[i]
                        for j in range(i+1, len(s_list)):
                            v = s_list[j]
                            if adj[u] & (1 << v):
                                common = adj[u] & adj[v] & subset_mask & ~(1 << u) & ~(1 << v)
                                if common == 0: continue
                                # Check edge in common
                                c_list = []
                                temp_c = common
                                while temp_c:
                                    lsb = temp_c & -temp_c
                                    idx = (lsb.bit_length() - 1)
                                    c_list.append(idx)
                                    temp_c ^= lsb
                                for a in c_list:
                                    if adj[a] & common & ~(1 << a):
                                        return True
                    return False
                return False # Only implementing k=4 for now

            if has_k_in_subset(current_adj, 0, 4): # Placeholder, need mask of neighbors
                pass # Logic error in above, need to pass neighbors mask
                
            # Let's do the check properly
            neighbors_mask = 0
            for v in neighbors:
                neighbors_mask |= (1 << v)
                
            if has_k_in_subset(current_adj, neighbors_mask, 4):
                valid = False
            
            if valid:
                # Check I4 in non-neighbors
                # Non-neighbors mask
                all_mask = (1 << current_n) - 1
                non_neighbors_mask = all_mask & ~neighbors_mask
                
                # Check K4 in complement of G[non_neighbors]
                # This is equivalent to checking if G[non_neighbors] has an independent set of size 4.
                # Which is K4 in the complement graph induced by non_neighbors.
                
                # We can just check if there is a K4 in the complement.
                # Let's build a temporary check.
                # Or simpler: Check if there exist 4 vertices in non_neighbors with no edges between them.
                
                # Use the same has_k_in_subset but on the complement?
                # Let's write a helper for independent set in subset.
                
                def has_independent_set_in_subset(adj, subset_mask, k):
                    if k == 4:
                        s_list = []
                        temp = subset_mask
                        while temp:
                            lsb = temp & -temp
                            idx = (lsb.bit_length() - 1)
                            s_list.append(idx)
                            temp ^= lsb
                        # Check for I4
                        # For each pair (u,v) in s_list that are NOT connected,
                        # check if (non-neighbors of u in s_list) \cap (non-neighbors of v in s_list) contains an I2 (non-edge).
                        for i in range(len(s_list)):
                            u = s_list[i]
                            for j in range(i+1, len(s_list)):
                                v = s_list[j]
                                if not (adj[u] & (1 << v)):
                                    # u and v are not connected
                                    # Find common non-neighbors in s_list
                                    # Non-neighbors of u in s_list: s_list & ~adj[u] & ~(1<<u)
                                    # Non-neighbors of v in s_list: s_list & ~adj[v] & ~(1<<v)
                                    
                                    # Let's just get the list of non-neighbors of u in s_list
                                    # And check if any pair in that list is not connected.
                                    
                                    # Actually, simpler:
                                    # We need 4 vertices a,b,c,d in s_list such that no edges.
                                    # If u,v are non-adjacent, we need 2 more vertices x,y in s_list
                                    # such that x,y are non-adjacent to u,v and to each other.
                                    
                                    # Candidates for x,y:
                                    # Must be in s_list
                                    # Must not be adjacent to u
                                    # Must not be adjacent to v
                                    
                                    # Let's build the mask of candidates
                                    # cand_mask = subset_mask & ~adj[u] & ~adj[v] & ~(1<<u) & ~(1<<v)
                                    cand_mask = subset_mask & ~(adj[u] | adj[v] | (1 << u) | (1 << v))
                                    if cand_mask == 0: continue
                                    
                                    # Check if cand_mask contains a non-edge
                                    c_list = []
                                    temp_c = cand_mask
                                    while temp_c:
                                        lsb = temp_c & -temp_c
                                        idx = (lsb.bit_length() - 1)
                                        c_list.append(idx)
                                        temp_c ^= lsb
                                    
                                    for a in c_list:
                                        for b in c_list:
                                            if a == b: continue
                                            if not (adj[a] & (1 << b)):
                                                return True
                        return False
                    return False

                if has_independent_set_in_subset(current_adj, non_neighbors_mask, 4):
                    valid = False
            
            if valid:
                # Add vertex
                found = True
                new_n = current_n + 1
                new_adj = current_adj + [0]
                for v in neighbors:
                    new_adj[v] |= (1 << new_n)
                    new_adj[new_n] |= (1 << v)
                
                current_n = new_n
                current_adj = new_adj
                best_n = current_n
                best_adj = current_adj[:]
                # Update best_edges
                # We can reconstruct edges from best_adj at the end
                break

        if not found:
            # If we can't add this vertex, try to restart with a different random seed?
            # Or just stop.
            # For now, stop extending.
            break

    # Reconstruct edges from best_adj
    final_edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if best_adj[i] & (1 << j):
                final_edges.append((i, j))
                
    return best_n, final_edges
# EVOLVE-BLOCK-END
