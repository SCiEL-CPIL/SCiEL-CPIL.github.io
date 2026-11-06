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
    # Face defined by (u, i, j) with i < j
    # Edges: (u, u^i), (u^i, u^i^j), (u^i^j, u^j), (u^j, u)
    faces = []
    edge_faces = [[] for _ in range(total_edges)]
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
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
                    
    num_faces = len(faces)
    
    # Greedy initialization
    def greedy_init():
        mask = 0
        face_counts = [0] * num_faces
        # Random order
        order = list(range(total_edges))
        random.shuffle(order)
        for e_idx in order:
            # Check if adding e_idx causes violation
            # e_idx is in faces edge_faces[e_idx]
            # If any face has count 3, adding makes it 4 -> violation
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

    start_time = time.time()
    # Safety margin: 150s kill, aim to finish by 100s to be safe
    global_deadline = start_time + 100
    
    best_mask = 0
    best_size = 0
    
    # Run multiple SA phases
    # Each phase starts from a random greedy solution
    # SA tries to improve it
    
    while time.time() < global_deadline - 5: # Leave 5s for final processing
        
        # Start with a random greedy solution
        current_mask, current_counts = greedy_init()
        current_edges = bin(current_mask).count('1')
        
        if current_edges > best_size:
            best_size = current_edges
            best_mask = current_mask
            
        # Simulated Annealing for this phase
        # Time limit for this SA phase: 20 seconds
        sa_start = time.time()
        sa_deadline = sa_start + 20
        
        T = 50.0
        cooling = 0.9999
        min_T = 0.1
        
        # Precompute list of edge indices for faster random selection? 
        # random.randint is fast enough.
        
        while time.time() < sa_deadline and T > min_T:
            e_idx = random.randint(0, total_edges - 1)
            
            is_in_set = (current_mask >> e_idx) & 1
            
            # Calculate delta violations
            # We maintain current_counts to track how many edges are in each face
            # Violation exists if count == 4.
            # We want to minimize violations, then maximize edges.
            
            if is_in_set:
                # Remove edge
                delta_violations = 0
                for f_idx in edge_faces[e_idx]:
                    if current_counts[f_idx] == 4:
                        delta_violations -= 1
                    current_counts[f_idx] -= 1
                delta_edges = -1
            else:
                # Add edge
                delta_violations = 0
                for f_idx in edge_faces[e_idx]:
                    if current_counts[f_idx] == 3:
                        delta_violations += 1
                    current_counts[f_idx] += 1
                delta_edges = 1
            
            # Energy: Penalty for violations is high.
            # E = 100 * violations - edges
            # We accept if new_E < old_E or with prob exp(-delta_E/T)
            # delta_E = 100 * delta_violations - delta_edges
            
            delta_E = 100 * delta_violations - delta_edges
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                if is_in_set:
                    current_mask &= ~(1 << e_idx)
                else:
                    current_mask |= (1 << e_idx)
                current_edges += delta_edges
                
                if delta_violations == 0 and current_edges > best_size:
                    best_size = current_edges
                    best_mask = current_mask
            else:
                # Reject, undo count changes
                if is_in_set:
                    for f_idx in edge_faces[e_idx]:
                        current_counts[f_idx] += 1
                else:
                    for f_idx in edge_faces[e_idx]:
                        current_counts[f_idx] -= 1
            
            T *= cooling
            
            # If T drops too low, restart SA with a new greedy or break?
            # Breaking and starting a new outer loop iteration is cleaner.
            if T <= min_T:
                break

    # Convert best_mask to list of edges
    result = []
    for e_idx in range(total_edges):
        if (best_mask >> e_idx) & 1:
            u, v = all_edges[e_idx]
            result.append([u, v])
            
    return result
