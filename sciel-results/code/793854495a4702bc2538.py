import numpy as np
from itertools import product, combinations
import random

def kissing_number11() -> np.ndarray:
    d = 11
    
    # Generate candidate vectors with norm squared 4
    # Type A: one +/-2
    candidates = []
    indices = []
    
    # Type A
    for i in range(d):
        for s in [1, -1]:
            v = [0]*d
            v[i] = 2*s
            candidates.append(tuple(v))
            indices.append(len(candidates)-1)
            
    num_A = len(candidates)
    
    # Type C: four +/-1
    type_c_indices = []
    for support in combinations(range(d), 4):
        for signs in product([1, -1], repeat=4):
            v = [0]*d
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(tuple(v))
            type_c_indices.append(len(candidates)-1)
            
    num_total = len(candidates)
    num_C = len(type_c_indices)
    
    # Build conflict graph for Type C vectors
    # Conflict if dot > 2
    # We know conflicts only occur if supports intersect in 3 or 4.
    # If intersect in 4 (same support), dot <= 2 for distinct vectors.
    # So only intersect in 3.
    
    # Map from support to list of Type C indices having that support
    support_to_indices = {}
    for idx in type_c_indices:
        v = candidates[idx]
        support = tuple(i for i in range(d) if v[i] != 0)
        if support not in support_to_indices:
            support_to_indices[support] = []
        support_to_indices[support].append(idx)
        
    # Precompute neighbors for each Type C vector
    # neighbors[i] is a set of indices in type_c_indices that conflict with i
    neighbors = [set() for _ in range(num_C)]
    
    # Iterate over pairs of supports that intersect in 3
    supports = list(support_to_indices.keys())
    for i in range(len(supports)):
        for j in range(i+1, len(supports)):
            s1 = supports[i]
            s2 = supports[j]
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                # Find conflicting pairs
                idxs1 = support_to_indices[s1]
                idxs2 = support_to_indices[s2]
                
                # For each pair, check dot product
                # Optimization: 
                # The intersection indices are fixed.
                # u in s1, v in s2.
                # u and v conflict if u[k]==v[k] for all k in inter.
                
                # We can group by the signs on the intersection
                # But 16x16 = 256 checks per pair of supports is fine.
                # Number of such support pairs:
                # Total supports 330.
                # Pairs intersecting in 3:
                # Each support has 28 such partners.
                # 330 * 28 / 2 = 4620 pairs.
                # 4620 * 256 ~ 1.2M checks. Fast.
                
                for u_idx in idxs1:
                    u = candidates[u_idx]
                    for v_idx in idxs2:
                        v = candidates[v_idx]
                        # Calculate dot product on intersection
                        dot = 0
                        for k in inter:
                            dot += u[k] * v[k]
                        if dot > 2:
                            # Conflict
                            # Map global index to local Type C index
                            # type_c_indices is a list of global indices.
                            # We need local index for the neighbors array.
                            # Let's create a map global_idx -> local_idx for Type C
                            pass
                
    # Let's use a global index -> local index map for Type C
    global_to_local = {g: l for l, g in enumerate(type_c_indices)}
    
    # Re-do neighbor construction with mapping
    neighbors = [set() for _ in range(num_C)]
    for i in range(len(supports)):
        for j in range(i+1, len(supports)):
            s1 = supports[i]
            s2 = supports[j]
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                idxs1 = support_to_indices[s1]
                idxs2 = support_to_indices[s2]
                
                for u_g_idx in idxs1:
                    u = candidates[u_g_idx]
                    u_l_idx = global_to_local[u_g_idx]
                    for v_g_idx in idxs2:
                        v = candidates[v_g_idx]
                        v_l_idx = global_to_local[v_g_idx]
                        # Calculate dot product
                        dot = 0
                        for k in inter:
                            dot += u[k] * v[k]
                        if dot > 2:
                            neighbors[u_l_idx].add(v_l_idx)
                            neighbors[v_l_idx].add(u_l_idx)
                            
    # Greedy MIS with random restarts
    best_size = 0
    best_selected_C = []
    
    for seed in range(50):
        random.seed(seed)
        available = set(range(num_C))
        selected = []
        
        while available:
            # Pick random element
            v = random.choice(list(available))
            selected.append(v)
            available.remove(v)
            # Remove neighbors
            for n in neighbors[v]:
                if n in available:
                    available.remove(n)
                    
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected_C = selected
            
    # Construct final array
    # All Type A are included
    final_indices = list(range(num_A))
    # Add best Type C
    for l_idx in best_selected_C:
        final_indices.append(type_c_indices[l_idx])
        
    points = [candidates[i] for i in final_indices]
    points_arr = np.array(points, dtype=np.int64)
    return points_arr
