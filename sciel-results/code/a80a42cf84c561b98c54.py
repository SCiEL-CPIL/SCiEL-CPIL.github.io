import itertools
import random

def construct():
    n = 16
    vertices = list(range(n))
    
    # 1. Map edges to indices
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            idx_to_edge.append((u, v))
            idx += 1
    num_edges = len(idx_to_edge) # 120
    
    # 2. Pre-compute Q3 subgraphs
    # For each edge index, store a list of Q3 bitmasks containing that edge
    q3s_for_edge = [[] for _ in range(num_edges)]
    
    # To speed up, we can pre-calculate the Q3 structures on a canonical set of 8 vertices
    # and then map them. But mapping is tricky.
    # Let's just generate them directly.
    
    # Optimization: Pre-calculate the 840 bitmasks for vertices 0-7?
    # No, the edge indices are global.
    
    # We will iterate over all 8-vertex subsets
    # This is the slow part.
    # 12870 subsets.
    # For each, 840 Q3s.
    
    # Let's try to be as fast as possible.
    # We can use a helper to get edge index fast.
    
    all_q3_masks = set() # To avoid duplicates? No, each Q3 is unique per vertex set?
    # Actually, different 8-sets produce different Q3s.
    # But we don't need to store all, just distribute to edges.
    
    for verts in itertools.combinations(vertices, 8):
        # Generate all 35 bipartitions
        # Use a standard way to generate 4-subsets
        v_list = list(verts)
        # Generate all subsets of size 4
        subsets_4 = list(itertools.combinations(v_list, 4))
        seen_part = set()
        for U in subsets_4:
            # Construct V
            U_set = set(U)
            V = tuple(v for v in v_list if v not in U_set)
            # Canonical form for partition
            if U < V:
                part_key = (U, V)
            else:
                part_key = (V, U)
            if part_key in seen_part:
                continue
            seen_part.add(part_key)
            
            U_list = list(U)
            V_list = list(V)
            
            # Generate 24 perfect matchings
            for perm in itertools.permutations(V_list):
                # Matching is set of pairs (u[i], perm[i])
                matching_set = set(zip(U_list, perm))
                
                # Construct Q3 edge mask
                mask = 0
                edge_indices_in_q3 = []
                for u in U_list:
                    for v in V_list:
                        # Edge (u,v) is in Q3 if NOT in matching
                        if (u, v) not in matching_set and (v, u) not in matching_set:
                            # Note: matching_set contains (u,v) where u in U, v in V.
                            # So we just check (u,v)
                            pass 
                        # Actually, matching is defined on U x V.
                        # If (u,v) is in matching, edge is absent.
                        # If not, edge is present.
                        if (u, v) not in matching_set:
                            e_idx = edge_to_idx[(u, v)] if u < v else edge_to_idx[(v, u)]
                            mask |= (1 << e_idx)
                            edge_indices_in_q3.append(e_idx)
                
                # Add this Q3 mask to the lists of its 12 edges
                for e_idx in edge_indices_in_q3:
                    q3s_for_edge[e_idx].append(mask)

    # 3. Random Greedy Search
    best_edge_count = 0
    best_edges = []
    
    # Run multiple trials
    num_trials = 20
    
    for _ in range(num_trials):
        current_edges = set()
        edge_mask = 0
        shuffled = list(range(num_edges))
        random.shuffle(shuffled)
        
        for e_idx in shuffled:
            # Check if adding e_idx creates a Q3
            new_mask = edge_mask | (1 << e_idx)
            is_valid = True
            for q3_mask in q3s_for_edge[e_idx]:
                if (new_mask & q3_mask) == q3_mask:
                    is_valid = False
                    break
            if is_valid:
                current_edges.add(e_idx)
                edge_mask = new_mask
        
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            best_edges = [list(idx_to_edge[i]) for i in current_edges]
            
    return best_edges
