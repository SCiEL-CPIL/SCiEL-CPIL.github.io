import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    d = 7
    n = 2 ** d
    
    # 1. Precompute edges and faces
    edges = []
    edge_id_map = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edge_id = len(edges)
                edges.append((x, y))
                edge_id_map[(x, y)] = edge_id
                edge_id_map[(y, x)] = edge_id

    n_edges = len(edges)
    
    face_edges = []
    for v in range(n):
        for i in range(d):
            for j in range(i+1, d):
                v_i = v ^ (1 << i)
                v_j = v ^ (1 << j)
                v_ij = v ^ (1 << i) ^ (1 << j)
                
                e1 = edge_id_map[(min(v, v_i), max(v, v_i))]
                e2 = edge_id_map[(min(v_i, v_ij), max(v_i, v_ij))]
                e3 = edge_id_map[(min(v, v_j), max(v, v_j))]
                e4 = edge_id_map[(min(v_j, v_ij), max(v_j, v_ij))]
                
                face_edges.append([e1, e2, e3, e4])

    n_faces = len(face_edges)
    
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, f_edges in enumerate(face_edges):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)

    # 2. Greedy Initialization of Kept Set (via Hitting Set)
    # We find a small hitting set R, then K = All \ R
    def greedy_hitting_set(rng):
        covered = [False] * n_faces
        R = set()
        benefit = [len(edge_faces[e]) for e in range(n_edges)]
        
        for _ in range(n_faces):
            best_e = -1
            max_ben = -1
            max_edges = []
            for e in range(n_edges):
                if e not in R:
                    if benefit[e] > max_ben:
                        max_ben = benefit[e]
                        max_edges = [e]
                    elif benefit[e] == max_ben:
                        max_edges.append(e)
            
            if not max_edges or max_ben == 0:
                break
                
            chosen_e = rng.choice(max_edges)
            R.add(chosen_e)
            
            for f in edge_faces[chosen_e]:
                if not covered[f]:
                    covered[f] = True
                    for e in face_edges[f]:
                        if e != chosen_e and e not in R:
                            benefit[e] -= 1
            
            if all(covered):
                break
            if time.time() - start_time > time_limit - 10:
                break
                
        return R

    def get_kept_set(R):
        return set(e for e in range(n_edges) if e not in R)

    # 3. Local Search on Kept Set with Validity Preserving Moves
    def local_search_kept(K_init, rng, time_budget):
        K = set(K_init)
        # Initialize face counts
        face_counts = [0] * n_faces
        for e in K:
            for f in edge_faces[e]:
                face_counts[f] += 1
        
        # Verify validity (should be valid)
        for f in range(n_faces):
            if face_counts[f] == 4:
                # This shouldn't happen if K_init is valid
                # But if it does, we need to fix it
                # Remove one edge from the face
                e_to_remove = face_edges[f][0]
                K.remove(e_to_remove)
                for ff in edge_faces[e_to_remove]:
                    face_counts[ff] -= 1
                break

        best_K = set(K)
        best_size = len(K)
        
        T_start = 1.0
        T_end = 0.05
        cooling = 0.99
        T = T_start
        
        end_time = time.time() + time_budget
        current_size = len(K)
        
        while time.time() < end_time:
            if T < T_end:
                T = T_start
            
            # Generate a move
            # 1. Try to add an edge
            # 2. Try to remove an edge
            # 3. Swap (optional, might be slower)
            
            move_type = rng.random()
            if move_type < 0.7:
                # Try Add
                # Pick random edge not in K
                e = rng.randint(0, n_edges - 1)
                if e in K:
                    continue
                
                # Check if adding e is valid
                valid = True
                for f in edge_faces[e]:
                    if face_counts[f] == 3:
                        valid = False
                        break
                
                if valid:
                    # Accept Add (improvement)
                    K.add(e)
                    for f in edge_faces[e]:
                        face_counts[f] += 1
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_K = set(K)
                else:
                    # Reject Add (would be invalid)
                    # In SA, we might accept invalid states? No, we must stay valid.
                    # So we just don't do it.
                    pass
                    
            elif move_type < 0.9:
                # Try Remove
                if not K:
                    continue
                e = rng.choice(list(K))
                
                # Calculate delta: -1
                delta = -1
                # Accept with probability 1 (always valid) or SA prob
                # Since it decreases size, we use SA to accept
                if rng.random() < pow(2.718281828, -delta / T):
                    K.remove(e)
                    for f in edge_faces[e]:
                        face_counts[f] -= 1
                    current_size -= 1
            else:
                # Try Swap: Remove e_out, Add e_in
                # Pick random e_out in K
                if not K:
                    continue
                e_out = rng.choice(list(K))
                # Pick random e_in not in K
                e_in = rng.randint(0, n_edges - 1)
                while e_in in K:
                    e_in = rng.randint(0, n_edges - 1)
                
                # Check if K - {e_out} + {e_in} is valid
                # Temporarily remove e_out
                # We need to check if e_in can be added to K \ {e_out}
                
                # Count for e_in in (K \ {e_out})
                # For each face f containing e_in:
                #   count_in_K_minus_eout = face_counts[f] - (1 if e_out in face_edges[f] else 0)
                #   if count_in_K_minus_eout == 3, then adding e_in makes it 4 -> Invalid
                #   Also, we must ensure that removing e_out doesn't leave any face with count 4?
                #   No, removing an edge never creates a C4.
                
                valid = True
                for f in edge_faces[e_in]:
                    # Check if e_out is in this face
                    # We can precompute face membership or just check
                    # face_edges[f] is a list of 4 edges
                    # if e_out in face_edges[f]:
                    #    current_count_in_face = face_counts[f] - 1
                    # else:
                    #    current_count_in_face = face_counts[f]
                    
                    # Optimization: Precompute for each edge e, the set of faces it's in.
                    # And for each face, the edges.
                    # We can just iterate face_edges[f]
                    e_out_in_face = False
                    for e_check in face_edges[f]:
                        if e_check == e_out:
                            e_out_in_face = True
                            break
                    
                    new_count = face_counts[f] - (1 if e_out_in_face else 0)
                    if new_count == 3:
                        valid = False
                        break
                
                if valid:
                    # Perform Swap
                    K.remove(e_out)
                    for f in edge_faces[e_out]:
                        face_counts[f] -= 1
                    
                    K.add(e_in)
                    for f in edge_faces[e_in]:
                        face_counts[f] += 1
                    # Size remains same
                    # We might accept this to explore
                    # Since size is same, we accept if it leads to future improvements
                    # In SA, we accept neutral moves with prob 1 or high
                    if rng.random() < 0.5: # Heuristic: accept half the neutral swaps
                         pass # Already applied
                    else:
                         # Undo Swap
                         K.remove(e_in)
                         for f in edge_faces[e_in]:
                             face_counts[f] -= 1
                         
                         K.add(e_out)
                         for f in edge_faces[e_out]:
                             face_counts[f] += 1

            T *= cooling
            
        return best_K, best_size

    # 4. Multi-Start
    best_K = set()
    best_size = 0
    
    # Phase 1: Get a good initial solution
    rng_init = random.Random(42)
    R_init = greedy_hitting_set(rng_init)
    K_init = get_kept_set(R_init)
    
    # Run Local Search
    remaining_time = time_limit - 5 - (time.time() - start_time)
    if remaining_time > 5:
        K_best, size_best = local_search_kept(K_init, random.Random(123), remaining_time)
        best_K = K_best
        best_size = size_best

    # Convert K (edge IDs) to list of [u, v]
    result = []
    for e_id in best_K:
        u, v = edges[e_id]
        result.append([u, v])
        
    return result
