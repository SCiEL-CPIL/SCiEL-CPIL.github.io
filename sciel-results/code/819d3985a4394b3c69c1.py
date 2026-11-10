import random
import time
import math

def construct():
    deadline = time.time() + 110
    d = 7
    n = 1 << d
    
    # Build all edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
    
    num_edges = len(all_edges)
    
    edge_tuple_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_tuple_to_idx[(u, v)] = idx
    
    # Build faces
    faces = []
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
    
    # Build incidence structures
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_to_faces[e_idx].append(f_idx)
    
    face_to_edges = [list(face) for face in faces]
    
    best_count = 0
    best_state = None
    
    # Helper to calculate score
    # Score = num_edges - LAMBDA * num_bad_faces
    LAMBDA = 1000
    
    def simulate_annealing(start_state, max_iters, start_temp, end_temp):
        selected = start_state[:]
        face_counts = [0] * num_faces
        num_edges_cur = 0
        num_bad_faces_cur = 0
        
        for e_idx in range(num_edges):
            if selected[e_idx]:
                num_edges_cur += 1
                for f_idx in edge_to_faces[e_idx]:
                    face_counts[f_idx] += 1
        
        for f_idx in range(num_faces):
            if face_counts[f_idx] == 4:
                num_bad_faces_cur += 1
                
        current_score = num_edges_cur - LAMBDA * num_bad_faces_cur
        
        best_local_count = 0
        best_local_state = None
        
        if num_bad_faces_cur == 0:
            best_local_count = num_edges_cur
            best_local_state = selected[:]
            
        rng = random.Random()
        
        for i in range(max_iters):
            # Cooling
            t = start_temp * (end_temp / start_temp) ** (i / max_iters)
            
            # Pick random edge
            e_idx = rng.randint(0, num_edges - 1)
            
            # Calculate delta
            delta_edges = 0
            delta_bad = 0
            
            if selected[e_idx]:
                # Removing
                delta_edges = -1
                for f_idx in edge_to_faces[e_idx]:
                    if face_counts[f_idx] == 4:
                        # Becomes 3, so bad face count decreases by 1
                        delta_bad -= 1
            else:
                # Adding
                delta_edges = 1
                for f_idx in edge_to_faces[e_idx]:
                    if face_counts[f_idx] == 3:
                        # Becomes 4, so bad face count increases by 1
                        delta_bad += 1
            
            delta_score = delta_edges - LAMBDA * delta_bad
            
            accept = False
            if delta_score > 0:
                accept = True
            else:
                if t > 0:
                    accept_prob = math.exp(delta_score / t)
                    if rng.random() < accept_prob:
                        accept = True
            
            if accept:
                # Apply move
                if selected[e_idx]:
                    selected[e_idx] = False
                    for f_idx in edge_to_faces[e_idx]:
                        face_counts[f_idx] -= 1
                else:
                    selected[e_idx] = True
                    for f_idx in edge_to_faces[e_idx]:
                        face_counts[f_idx] += 1
                
                num_edges_cur += delta_edges
                num_bad_faces_cur += delta_bad
                current_score += delta_score
                
                if num_bad_faces_cur == 0:
                    if num_edges_cur > best_local_count:
                        best_local_count = num_edges_cur
                        best_local_state = selected[:]
            else:
                # Revert (do nothing, state unchanged)
                pass
        
        return best_local_count, best_local_state

    # Initial state: All edges selected
    init_state = [True] * num_edges
    
    # Run multiple SA restarts with different random seeds/temps
    attempt = 0
    while time.time() < deadline - 10:
        # Vary temperature and iterations
        start_temp = 50 + attempt * 5
        max_iters = 5000 + attempt * 1000
        
        count, state = simulate_annealing(init_state, max_iters, start_temp, 0.01)
        
        if count > best_count:
            best_count = count
            best_state = state
            
        # Local Search: Try to add any valid edges
        if best_state:
            # Greedy add
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Re-calculate face counts for current best_state
                face_counts = [0] * num_faces
                for e_idx in range(num_edges):
                    if best_state[e_idx]:
                        for f_idx in edge_to_faces[e_idx]:
                            face_counts[f_idx] += 1
                
                for e_idx in range(num_edges):
                    if not best_state[e_idx]:
                        can_add = True
                        for f_idx in edge_to_faces[e_idx]:
                            if face_counts[f_idx] == 3:
                                can_add = False
                                break
                        if can_add:
                            best_state[e_idx] = True
                            best_count += 1
                            for f_idx in edge_to_faces[e_idx]:
                                face_counts[f_idx] += 1
                            improved = True
                            
        attempt += 1

    if best_state is None:
        return []
        
    out = []
    for idx in range(num_edges):
        if best_state[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out
