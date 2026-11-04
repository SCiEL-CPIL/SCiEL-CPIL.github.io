import numpy as np
import itertools
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate all candidate vectors:
    # Support: 4-subset of range(11)
    # Signs: Even parity (product of signs is 1, i.e., even number of -1s)
    
    supports = list(itertools.combinations(range(D), 4))
    even_signs = []
    for s in itertools.product([1, -1], repeat=4):
        if s.count(-1) % 2 == 0:
            even_signs.append(s)
            
    candidates = []
    # Map (support, sign_tuple) to vector index?
    # We just store the vectors and their indices.
    # To check incompatibility efficiently, we can precompute neighbors.
    
    # Total candidates: C(11,4) * 8 = 330 * 8 = 2640
    # Incompatibility: v_i dot v_j == 3
    # This happens if they share 3 indices and have same signs on those indices.
    
    # Let's build the incompatibility graph.
    # Node ID: index in candidates list.
    # We need a fast way to find which candidates are incompatible with a given candidate.
    
    # For a candidate v with support S and signs sigma:
    # Neighbors are vectors w with support T, signs tau such that:
    # T shares 3 indices with S, and tau matches sigma on those 3 indices.
    # As derived, for each of the 4 indices in S to drop, and each of the 7 indices outside S to add,
    # there is exactly 1 such vector w in our candidate set (determined by parity).
    
    # Let's map (support, signs) to index.
    # support is a tuple of 4 ints. signs is a tuple of 4 ints.
    # But signs are relative to the sorted support?
    # In our generation, signs are assigned in the order of the support tuple.
    # So (support, signs) uniquely identifies a vector.
    
    index_map = {}
    for i, sup in enumerate(supports):
        for j, sig in enumerate(even_signs):
            index_map[(sup, sig)] = i * 8 + j
            # Wait, this mapping assumes fixed order of even_signs.
            # Let's just create a list of (sup, sig) pairs.
            
    vec_list = []
    sup_sig_pairs = []
    for sup in supports:
        for sig in even_signs:
            vec_list.append((sup, sig))
            
    # Now build neighbors
    # For each (sup, sig), find the 28 neighbors.
    # sup = (a,b,c,d)
    # sig = (s_a, s_b, s_c, s_d)
    
    neighbors = [[] for _ in range(len(vec_list))]
    
    for i, (sup, sig) in enumerate(vec_list):
        # Try dropping each of the 4 indices
        for drop_idx in range(4):
            # Remaining 3 indices and their signs
            rem_indices = [sup[k] for k in range(4) if k != drop_idx]
            rem_signs = [sig[k] for k in range(4) if k != drop_idx]
            
            # Add each of the 7 indices not in sup
            for add_idx in range(D):
                if add_idx in sup:
                    continue
                
                # New support
                new_sup = tuple(sorted(rem_indices + [add_idx]))
                
                # We need to find the sign assignment for new_sup such that
                # the signs on rem_indices match rem_signs.
                # The position of each rem_index in new_sup determines which sign in new_sig it corresponds to.
                # Let new_sig be the signs in the order of new_sup.
                # We know new_sig[pos] = rem_signs[k] where new_sup[pos] = rem_indices[k].
                
                # Determine the sign at the position of add_idx in new_sup.
                # Let pos_add be the index of add_idx in new_sup.
                pos_add = new_sup.index(add_idx)
                
                # Construct new_sig
                new_sig = [0] * 4
                for k, idx in enumerate(rem_indices):
                    pos = new_sup.index(idx)
                    new_sig[pos] = rem_signs[k]
                
                # Calculate sign at pos_add to satisfy even parity (product 1)
                # Product of all 4 signs must be 1.
                # prod(new_sig) = 1
                # new_sig[pos_add] = 1 / (prod of other 3)
                # Since signs are +/-1, new_sig[pos_add] = prod of other 3 (because x * x = 1)
                prod_others = 1
                for k in range(4):
                    if k != pos_add:
                        prod_others *= new_sig[k]
                        
                new_sig[pos_add] = prod_others
                
                # Check if this new_sig is in our even_signs list?
                # Yes, by construction it has even parity.
                # But we need to match it to the specific tuple in even_signs to find the index.
                # We can just look up (new_sup, tuple(new_sig)) in index_map.
                
                key = (new_sup, tuple(new_sig))
                if key in index_map:
                    neighbors[i].append(index_map[key])
    
    # Now we have the graph.
    # We want to find a large Independent Set.
    # Greedy:
    # 1. Shuffle node indices.
    # 2. Iterate. If node not blocked, select it and block its neighbors.
    
    best_size = 0
    best_set = []
    n_nodes = len(vec_list)
    
    # Run multiple trials
    for _ in range(500):
        order = list(range(n_nodes))
        random.shuffle(order)
        
        selected = set()
        blocked = set()
        
        for node in order:
            if node in blocked:
                continue
            if node in selected:
                continue
            
            # Select node
            selected.add(node)
            # Block neighbors
            for nb in neighbors[node]:
                blocked.add(nb)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
            
            # If we reach a high number, we can stop or continue.
            # Max possible?
            # 2640 nodes, degree 28.
            # Greedy usually gets around N/(d+1) ~ 2640/29 ~ 91?
            # Or better if graph structure is favorable.
            # Let's see if we can get close to 100-150.
            
    # Construct the final vector array
    vectors = []
    for idx in best_set:
        sup, sig = vec_list[idx]
        v = [0] * D
        for s, val in zip(sup, sig):
            v[s] = val
        vectors.append(v)
        
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
