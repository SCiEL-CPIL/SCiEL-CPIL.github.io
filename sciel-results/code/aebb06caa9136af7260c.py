import time
import random

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # 1. Build Edge List
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
                
    # 2. Build Face List
    face_edges = [] # list of lists of 4 edge IDs
    edge_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                face_id = len(face_edges)
                face_edges.append([e1, e2, e3, e4])
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
                    
    n_faces = len(face_edges)
    
    # Precompute face masks for bitset operations
    face_mask = [0] * n_faces
    for f, es in enumerate(face_edges):
        m = 0
        for e in es:
            m |= (1 << e)
        face_mask[f] = m

    # 3. Initialization: Greedy Set Cover to minimize Removed Edges (R)
    # This maximizes Selected Edges (S)
    # R is the set of removed edges.
    # We want to hit all faces with R.
    
    removed_mask = 0
    face_hit_count = [0] * n_faces
    unhit_count = n_faces
    
    # Greedy: Pick edge e that maximizes number of faces f with face_hit_count[f]==0
    while unhit_count > 0:
        best_e = -1
        best_gain = 0
        for e in range(n_edges):
            if (removed_mask >> e) & 1:
                continue
            gain = 0
            for f in edge_faces[e]:
                if face_hit_count[f] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_e = e
        if best_e == -1:
            break
        # Add best_e to removed
        removed_mask |= (1 << best_e)
        for f in edge_faces[best_e]:
            if face_hit_count[f] == 0:
                unhit_count -= 1
            face_hit_count[f] += 1
            
    # Initial Selected Mask
    all_edges_mask = (1 << n_edges) - 1
    selected_mask = all_edges_mask ^ removed_mask
    current_size = selected_mask.bit_count()
    
    best_mask = selected_mask
    best_size = current_size
    
    # 4. Simulated Annealing on Selected Mask
    start_time = time.time()
    timeout = 115.0 # Leave 5s margin
    
    T = 1.0
    cooling_rate = 0.9999
    T_min = 0.01
    
    # Precompute list of all edge indices for random selection
    all_edge_ids = list(range(n_edges))
    
    # Helper: Check if adding e is valid given current selected_mask
    def can_add(e, smask):
        for f in edge_faces[e]:
            # Count selected edges in face f
            # (smask & face_mask[f]) gives the bits of selected edges in this face
            if (smask & face_mask[f]).bit_count() == 3:
                return False
        return True

    # Helper: Check if swap e_out -> e_in is valid
    def can_swap(e_out, e_in, smask):
        # We remove e_out and add e_in.
        # Faces containing e_out: count decreases by 1. Always valid (was <=3, now <=2).
        # Faces containing e_in:
        #   If e_out is also in this face: count stays same. Was <=3, so valid.
        #   If e_out is NOT in this face: count increases by 1. Was <=3, must be <=2 to become <=3.
        for f in edge_faces[e_in]:
            # Check if e_out is in this face
            # e_out is in face f if (face_mask[f] >> e_out) & 1
            if (face_mask[f] >> e_out) & 1:
                continue # Count doesn't change, so valid
            else:
                # Count increases by 1. Current count must be < 3.
                if (smask & face_mask[f]).bit_count() == 3:
                    return False
        return True

    while time.time() - start_time < timeout:
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to Add
            # Pick random edge
            e = random.choice(all_edge_ids)
            if not (selected_mask >> e) & 1:
                if can_add(e, selected_mask):
                    selected_mask |= (1 << e)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = selected_mask
                        T = max(T, 0.5) # Reheat slightly
        elif move_type < 0.9:
            # Try to Swap
            e_out = random.choice(all_edge_ids)
            if (selected_mask >> e_out) & 1:
                e_in = random.choice(all_edge_ids)
                if not (selected_mask >> e_in) & 1:
                    if can_swap(e_out, e_in, selected_mask):
                        # Perform swap
                        selected_mask &= ~(1 << e_out)
                        selected_mask |= (1 << e_in)
                        # Size unchanged
                        # Accept with probability 1 to explore
        else:
            # Try to Remove
            e = random.choice(all_edge_ids)
            if (selected_mask >> e) & 1:
                # Accept with low probability to escape local maxima
                # Delta E = -1 (size decreases)
                # Prob = exp(-(-1)/T) = exp(1/T) > 1? 
                # In SA for maximization, we accept worse moves (smaller size) with prob exp((E_new - E_old)/T).
                # E_new = E_old - 1.
                # Prob = exp(-1/T).
                if random.random() < (2.718281828 ** (-1.0 / T)):
                    selected_mask &= ~(1 << e)
                    current_size -= 1
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # 5. Final Polish: Greedy Add on best_mask
    # Try to add any edge that can be added to best_mask
    final_mask = best_mask
    # Iterate through all edges, try to add if possible
    # This is a simple local search pass
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        for e in all_edge_ids:
            if not (final_mask >> e) & 1:
                if can_add(e, final_mask):
                    final_mask |= (1 << e)
                    improved = True

    # 6. Additional Polish: Swap-based local search on final_mask
    # Try to swap edges to improve structure, though size is fixed.
    # Actually, since we want to MAXIMIZE size, and we just did greedy add,
    # the size is locally maximal. Swaps don't change size.
    # However, a swap might allow a subsequent add?
    # Let's try a few random swaps and see if we can add more.
    
    # Let's stick to the result from step 5 as it is a valid local maximum.
    # If we have time, we can try to restart SA from best_mask with higher T?
    # No, the SA already ran for a long time.
    
    # Construct result
    result = []
    for i, (u, v) in enumerate(edges):
        if (final_mask >> i) & 1:
            result.append([u, v])
            
    return result
