import random
import time
import math

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = n_vertices * d // 2  # 448

    # Precompute all edges
    all_edges = []
    edge_to_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_to_index[e] = len(all_edges)
                all_edges.append(e)

    # Precompute faces
    faces = []
    edge_faces = [[] for _ in range(total_edges)]
    
    for u in range(n_vertices):
        for i in range(d):
            if u & (1 << i):
                continue
            for j in range(i + 1, d):
                if u & (1 << j):
                    continue
                
                v = u ^ (1 << i)
                w = u ^ (1 << j)
                z = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(u, v), max(u, v))
                e2 = (min(v, z), max(v, z))
                e3 = (min(z, w), max(z, w))
                e4 = (min(w, u), max(w, u))
                
                e1_idx = edge_to_index[e1]
                e2_idx = edge_to_index[e2]
                e3_idx = edge_to_index[e3]
                e4_idx = edge_to_index[e4]
                
                face_idx = len(faces)
                faces.append((e1_idx, e2_idx, e3_idx, e4_idx))
                
                for e_idx in (e1_idx, e2_idx, e3_idx, e4_idx):
                    edge_faces[e_idx].append(face_idx)
                    
    num_faces = len(faces) # Should be 672
    
    def greedy_init():
        mask = 0
        face_counts = [0] * num_faces
        order = list(range(total_edges))
        random.shuffle(order)
        for e_idx in order:
            can_add = True
            for f_idx in edge_faces[e_idx]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                mask |= (1 << e_idx)
                for f_idx in edge_faces[e_idx]:
                    face_counts[f_idx] += 1
        return mask, face_counts

    start_mask, start_counts = greedy_init()
    best_mask = start_mask
    best_size = bin(start_mask).count('1')
    
    start_time = time.time()
    deadline = start_time + 145
    
    T = 500.0
    cooling = 0.99999
    min_T = 0.01
    
    current_mask = start_mask
    current_counts = start_counts[:]
    current_edges = bin(current_mask).count('1')
    current_violations = sum(1 for c in current_counts if c == 4)
    
    current_energy = 10000 * current_violations - current_edges
    
    restarts = 0
    max_restarts = 100
    
    while time.time() < deadline:
        if T < min_T:
            if restarts >= max_restarts:
                break
            restarts += 1
            # Restart
            new_mask, new_counts = greedy_init()
            new_edges = bin(new_mask).count('1')
            if new_edges > best_size:
                best_size = new_edges
                best_mask = new_mask
            
            current_mask = new_mask
            current_counts = new_counts
            current_edges = new_edges
            current_violations = sum(1 for c in current_counts if c == 4)
            current_energy = 10000 * current_violations - current_edges
            T = 500.0
            continue
            
        # Pick random edge
        e_idx = random.randint(0, total_edges - 1)
        is_in_set = (current_mask >> e_idx) & 1
        
        delta_edges = 0
        delta_violations = 0
        
        if is_in_set:
            delta_edges = -1
            for f_idx in edge_faces[e_idx]:
                if current_counts[f_idx] == 4:
                    delta_violations -= 1
                current_counts[f_idx] -= 1
        else:
            delta_edges = 1
            for f_idx in edge_faces[e_idx]:
                if current_counts[f_idx] == 3:
                    delta_violations += 1
                current_counts[f_idx] += 1
        
        new_energy = current_energy + 10000 * delta_violations - delta_edges
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or (T > 0 and random.random() < math.exp(-delta_E / T)):
            if is_in_set:
                current_mask &= ~(1 << e_idx)
            else:
                current_mask |= (1 << e_idx)
            current_edges += delta_edges
            current_violations += delta_violations
            current_energy = new_energy
            
            if current_violations == 0 and current_edges > best_size:
                best_size = current_edges
                best_mask = current_mask
        else:
            if is_in_set:
                for f_idx in edge_faces[e_idx]:
                    current_counts[f_idx] += 1
            else:
                for f_idx in edge_faces[e_idx]:
                    current_counts[f_idx] -= 1
        
        T *= cooling

    result = []
    for e_idx in range(total_edges):
        if (best_mask >> e_idx) & 1:
            u, v = all_edges[e_idx]
            result.append([u, v])
            
    return result
