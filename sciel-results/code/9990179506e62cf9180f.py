import itertools
import random
import time

def construct():
    n = 18
    # H is the complement of G.
    # We want to minimize max_triangle_free_size(G).
    # This is equivalent to minimizing max_subset_with_no_ind_set_3(H).
    # Constraint: G is K4-free <=> alpha(H) <= 3.
    
    # Initialize H as complete graph (G is empty)
    # adj_H[i] is bitmask of neighbors in H

    # Let's use 0..n-1.
    adj_H = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                adj_H[i] |= (1 << j)
                
    def has_ind_set_3(H_adj):
        # Check if H has an independent set of size 3
        # Independent set in H means no edges in H.
        # So we need u, v, w such that no edges (u,v), (v,w), (u,w) in H.
        # This is a triangle in G.
        # But we can just check H.
        for i in range(n):
            for j in range(i+1, n):
                if (H_adj[i] >> j) & 1:
                    continue # Edge exists in H, so not independent
                # i, j are non-adjacent in H
                # Find common non-neighbors?
                # k is non-adjacent to i and j in H.
                # non_neighbors_i = ~H_adj[i] & mask
                # non_neighbors_j = ~H_adj[j] & mask
                # intersection must have a bit k != i, j
                mask = (1 << n) - 1
                non_i = (~H_adj[i]) & mask
                non_j = (~H_adj[j]) & mask
                common_non = non_i & non_j
                # Remove i and j from common_non
                common_non &= ~(1 << i)
                common_non &= ~(1 << j)
                if common_non:
                    return True
        return False

    def has_ind_set_4(H_adj):
        # Check if H has an independent set of size 4
        # This is for the constraint alpha(H) <= 3.
        # If has_ind_set_4 is True, alpha >= 4, invalid.
        # We can use a simple check.
        # To be fast, we can check if any 4-tuple is independent.
        # But 3060 tuples is fine.
        mask = (1 << n) - 1
        for i in range(n):
            non_i = (~H_adj[i]) & mask
            non_i &= ~(1 << i)
            for j in range(i+1, n):
                if not (non_i >> j) & 1:
                    continue
                # i, j non-adjacent
                non_ij = non_i & ((~H_adj[j]) & mask)
                non_ij &= ~(1 << i)
                non_ij &= ~(1 << j)
                # Check if non_ij has 2 non-adjacent vertices
                # Or just check if any k in non_ij is non-adjacent to any l in non_ij
                # Simpler: iterate k in non_ij
                k_mask = non_ij
                while k_mask:
                    lsb = k_mask & -k_mask
                    k = lsb.bit_length() - 1
                    k_mask ^= lsb
                    # Check if k has a non-neighbor in non_ij \ {k}
                    non_k = ((~H_adj[k]) & mask) & non_ij
                    non_k &= ~(1 << k)
                    if non_k:
                        return True
        return False

    def max_no_ind_3_size(H_adj):
        # Returns largest S such that H[S] has no independent set of size 3.
        # i.e. G[S] is triangle-free.
        mask = (1 << n) - 1
        for s in range(n, 0, -1):
            for comb in itertools.combinations(range(n), s):
                # Check if comb is independent set of size 3 free in H
                # i.e. no 3 vertices in comb are mutually non-adjacent in H.
                # This is equivalent to: G[comb] has no triangle.
                is_tf = True
                # Check triangles in G[comb]
                # G edge (u,v) exists if H edge (u,v) does NOT exist.
                for u in comb:
                    for v in comb:
                        if u >= v: continue
                        if (H_adj[u] >> v) & 1:
                            continue # No edge in G
                        # Edge (u,v) in G
                        # Check common neighbors in G within comb
                        # w in comb, w != u,v
                        # (u,w) in G and (v,w) in G
                        # (u,w) in G iff (H_adj[u] >> w) == 0
                        # (v,w) in G iff (H_adj[v] >> w) == 0
                        # So w must be non-neighbor of u and v in H.
                        # And w in comb.
                        # We can just check all w in comb
                        for w in comb:
                            if w == u or w == v: continue
                            if not ((H_adj[u] >> w) & 1) and not ((H_adj[v] >> w) & 1):
                                is_tf = False
                                break
                        if not is_tf:
                            break
                    if not is_tf:
                        break
                if is_tf:
                    return s
        return 0

    current_score = max_no_ind_3_size(adj_H)
    
    start_time = time.time()
    timeout = 110
    
    # Greedy removal of edges from H (adding to G)
    # We want to reduce current_score.
    # Removing edge (u,v) from H adds edge (u,v) to G.
    # This might create a triangle in G, reducing triangle-free subsets.
    # But we must ensure alpha(H) <= 3, i.e., no independent set of size 4 in H.
    
    while time.time() - start_time < timeout:
        # Pick random edge to remove from H
        u, v = random.sample(range(n), 2)
        
        # Check if edge exists in H
        if not ((adj_H[u] >> v) & 1):
            continue
            
        # Remove edge from H
        adj_H[u] &= ~(1 << v)
        adj_H[v] &= ~(1 << u)
        
        # Check constraint: alpha(H) <= 3
        if has_ind_set_4(adj_H):
            # Revert
            adj_H[u] |= (1 << v)
            adj_H[v] |= (1 << u)
            continue
            
        # Evaluate new score
        new_score = max_no_ind_3_size(adj_H)
        
        if new_score < current_score:
            current_score = new_score
        else:
            # Revert if no improvement (greedy)
            adj_H[u] |= (1 << v)
            adj_H[v] |= (1 << u)
            
    # Convert H to G edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if not ((adj_H[u] >> v) & 1):
                edges.append([u, v])
                
    return edges
