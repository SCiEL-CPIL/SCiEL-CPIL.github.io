import time
import random
import sys
import math

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
                
    # 2. Build Face List and Adjacency
    # face_edges[f] = list of 4 edge IDs
    # edge_faces[e] = list of face IDs containing edge e
    face_edges = []
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
    
    # Precompute for each edge, the set of faces it belongs to as a bitmask
    edge_face_mask = [0] * n_edges
    for e in range(n_edges):
        m = 0
        for f in edge_faces[e]:
            m |= (1 << f)
        edge_face_mask[e] = m

    # 3. Initialization: Greedy Set Cover to minimize Removed Edges (R)
    removed_mask = 0
    face_hit_count = [0] * n_faces
    unhit_count = n_faces
    
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
        removed_mask |= (1 << best_e)
        for f in edge_faces[best_e]:
            if face_hit_count[f] == 0:
                unhit_count -= 1
            face_hit_count[f] += 1
            
    all_edges_mask = (1 << n_edges) - 1
    selected_mask = all_edges_mask ^ removed_mask
    current_size = selected_mask.bit_count()
    
    best_mask = selected_mask
    best_size = current_size
    
    start_time = time.time()
    timeout = 112.0  # Leave 8s margin for polish and reconstruction
    
    T = 2.0
    cooling_rate = 0.99995
    T_min = 0.005
    
    all_edge_ids = list(range(n_edges))
    
    def can_add(e, smask):
        for f in edge_faces[e]:
            if (smask & face_mask[f]).bit_count() == 3:
                return False
        return True

    def can_swap(e_out, e_in, smask):
        for f in edge_faces[e_in]:
            if (face_mask[f] >> e_out) & 1:
                continue
            if (smask & face_mask[f]).bit_count() == 3:
                return False
        return True
    
    def can_multi_swap(e_outs, e_ins, smask):
        """Check if removing all e_outs and adding all e_ins is valid."""
        # Compute new mask
        new_smask = smask
        for e in e_outs:
            new_smask &= ~(1 << e)
        for e in e_ins:
            new_smask |= (1 << e)
        
        # Check all faces
        for f in range(n_faces):
            count = (new_smask & face_mask[f]).bit_count()
            if count > 3:
                return False
        return True
    
    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time
        progress = elapsed / timeout
        # Adaptive cooling: start hot, cool gradually
        T = max(T_min, T * cooling_rate)
        
        move_type = random.random()
        
        if move_type < 0.35:
            # Try to Add (1-for-0)
            e = random.choice(all_edge_ids)
            if not (selected_mask >> e) & 1:
                if can_add(e, selected_mask):
                    selected_mask |= (1 << e)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = selected_mask
                        T = max(T, 0.3)
        elif move_type < 0.65:
            # Try 1-for-1 Swap
            e_out = random.choice(all_edge_ids)
            if (selected_mask >> e_out) & 1:
                e_in = random.choice(all_edge_ids)
                if not (selected_mask >> e_in) & 1:
                    if can_swap(e_out, e_in, selected_mask):
                        selected_mask &= ~(1 << e_out)
                        selected_mask |= (1 << e_in)
        elif move_type < 0.80:
            # Try 2-for-1 Swap: remove 2 selected, add 1 unselected
            # This can help escape local optima
            e_out1 = random.choice(all_edge_ids)
            if (selected_mask >> e_out1) & 1:
                e_out2 = random.choice(all_edge_ids)
                if (selected_mask >> e_out2) & 1 and e_out2 != e_out1:
                    e_in = random.choice(all_edge_ids)
                    if not (selected_mask >> e_in) & 1 and e_in != e_out1 and e_in != e_out2:
                        if can_multi_swap([e_out1, e_out2], [e_in], selected_mask):
                            selected_mask &= ~(1 << e_out1)
                            selected_mask &= ~(1 << e_out2)
                            selected_mask |= (1 << e_in)
                            current_size -= 1
        elif move_type < 0.90:
            # Try 1-for-2 Swap: remove 1 selected, add 2 unselected
            e_out = random.choice(all_edge_ids)
            if (selected_mask >> e_out) & 1:
                e_in1 = random.choice(all_edge_ids)
                if not (selected_mask >> e_in1) & 1:
                    e_in2 = random.choice(all_edge_ids)
                    if not (selected_mask >> e_in2) & 1 and e_in2 != e_in1:
                        if can_multi_swap([e_out], [e_in1, e_in2], selected_mask):
                            selected_mask &= ~(1 << e_out)
                            selected_mask |= (1 << e_in1)
                            selected_mask |= (1 << e_in2)
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_mask = selected_mask
                                T = max(T, 0.3)
        else:
            # Try to Remove (escape local maxima)
            e = random.choice(all_edge_ids)
            if (selected_mask >> e) & 1:
                if random.random() < (math.exp(-1.0 / T)):
                    selected_mask &= ~(1 << e)
                    current_size -= 1

    # 5. Final Polish: Multiple passes of greedy add
    final_mask = best_mask
    for _ in range(10):  # Multiple passes to ensure convergence
        improved = True
        while improved:
            improved = False
            for e in all_edge_ids:
                if not (final_mask >> e) & 1:
                    if can_add(e, final_mask):
                        final_mask |= (1 << e)
                        improved = True
    
    # Additional polish: try 1-for-1 swaps that don't change size but might enable more adds
    for _ in range(5):
        improved = True
        while improved:
            improved = False
            for e_out in all_edge_ids:
                if (final_mask >> e_out) & 1:
                    for e_in in all_edge_ids:
                        if not (final_mask >> e_in) & 1:
                            if can_swap(e_out, e_in, final_mask):
                                new_mask = (final_mask & ~(1 << e_out)) | (1 << e_in)
                                # Check if this enables any additions
                                can_improve = False
                                for e_add in all_edge_ids:
                                    if not (new_mask >> e_add) & 1:
                                        if can_add(e_add, new_mask):
                                            can_improve = True
                                            break
                                if can_improve:
                                    final_mask = new_mask
                                    improved = True
                                    break
                    if improved:
                        break
    
    # Reconstruct result
    result = []
    for i in range(n_edges):
        if (final_mask >> i) & 1:
            result.append(list(edges[i]))
            
    return result
