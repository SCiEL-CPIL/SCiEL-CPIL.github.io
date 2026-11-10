import random
import time
import numpy as np

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill
    d = 7
    n = 1 << d  # 128 vertices
    best_count = 0
    best_chosen = None
    
    # All edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
    
    num_edges = len(all_edges)
    
    # Map edge tuple to index
    edge_tuple_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_tuple_to_idx[(u, v)] = idx
    
    # Precompute face edge indices for fast lookup
    faces = []
    
    # Generate all 448*6/2? No, number of faces is 2^(7-2) * C(7,2) = 32 * 21 = 672 faces.
    for i in range(d):
        for j in range(i + 1, d):
            for mask in range(1 << d):
                if (mask & (1 << i)) or (mask & (1 << j)):
                    continue
                
                v1 = mask
                v2 = mask ^ (1 << i)
                v3 = mask ^ (1 << j)
                v4 = mask ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                
                idx1 = edge_tuple_to_idx[e1]
                idx2 = edge_tuple_to_idx[e2]
                idx3 = edge_tuple_to_idx[e3]
                idx4 = edge_tuple_to_idx[e4]
                
                faces.append((idx1, idx2, idx3, idx4))

    num_faces = len(faces)
    
    # Map edge index to list of face indices it belongs to
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_to_faces[e_idx].append(f_idx)
    
    # Precompute for each edge, the set of other edges that share a face with it
    # This is useful for local search
    edge_neighbors = [set() for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e1 in face:
            for e2 in face:
                if e1 != e2:
                    edge_neighbors[e1].add(e2)

    def try_greedy(order_seed=None, weight_seed=None):
        rng = random.Random(order_seed)
        chosen = [False] * num_edges
        count = 0
        
        # If weight_seed is provided, use weighted random selection
        if weight_seed is not None:
            wrng = random.Random(weight_seed)
            # Simple weighted random: assign random weights
            weights = [wrng.random() for _ in range(num_edges)]
            indices = list(range(num_edges))
            # Sort by weight descending
            indices.sort(key=lambda x: weights[x], reverse=True)
        else:
            indices = list(range(num_edges))
            rng.shuffle(indices)
        
        # face_counts[f] = number of chosen edges in face f
        face_counts = [0] * num_faces
        
        for idx in indices:
            can_add = True
            for f_idx in edge_to_faces[idx]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] += 1
        
        return count, chosen, face_counts

    def local_search(chosen, face_counts, count, deadline):
        # Returns improved count, chosen, face_counts
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each chosen edge
            for r_idx in range(num_edges):
                if not chosen[r_idx]:
                    continue
                
                # Tentatively remove r_idx
                chosen[r_idx] = False
                count -= 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] -= 1
                
                # Collect candidate edges: unchosen edges sharing a face with r_idx
                candidate_set = set()
                for f_idx in edge_to_faces[r_idx]:
                    for e_idx in faces[f_idx]:
                        if e_idx != r_idx and not chosen[e_idx]:
                            candidate_set.add(e_idx)
                
                candidates = []
                for e_idx in candidate_set:
                    valid = True
                    for f_idx in edge_to_faces[e_idx]:
                        if face_counts[f_idx] == 3:
                            valid = False
                            break
                    if valid:
                        candidates.append(e_idx)
                
                # Try "remove 1, add 2" move
                added = []
                for i in range(len(candidates)):
                    e1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        e2 = candidates[j]
                        
                        # Check if e1 and e2 are compatible
                        compatible = True
                        for f_idx in edge_to_faces[e1]:
                            if e2 in faces[f_idx]:
                                if face_counts[f_idx] >= 2:
                                    compatible = False
                                    break
                        if compatible:
                            added.append(e1)
                            added.append(e2)
                            break
                    if len(added) >= 2:
                        break
                
                if len(added) >= 2:
                    # Commit the move
                    for e_idx in added:
                        chosen[e_idx] = True
                        count += 1
                        for f_idx in edge_to_faces[e_idx]:
                            face_counts[f_idx] += 1
                    improved = True
                    break
                
                # If no improvement, restore r_idx
                chosen[r_idx] = True
                count += 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] += 1
        
        return count, chosen, face_counts

    attempt = 0
    while time.time() < deadline:
        # Use a mix of random and weighted orders
        if attempt % 3 == 0:
            count, chosen, face_counts = try_greedy(attempt)
        elif attempt % 3 == 1:
            count, chosen, face_counts = try_greedy(attempt, weight_seed=attempt + 1000)
        else:
            count, chosen, face_counts = try_greedy(attempt, weight_seed=attempt + 2000)
        
        # Local search
        count, chosen, face_counts = local_search(chosen, face_counts, count, deadline)
        
        if count > best_count:
            best_count = count
            best_chosen = chosen
        attempt += 1
    
    out = []
    for idx in range(num_edges):
        if best_chosen[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out
