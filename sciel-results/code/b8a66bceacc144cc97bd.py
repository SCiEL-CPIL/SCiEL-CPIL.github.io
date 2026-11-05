import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))  # 7 * 64 = 448
    
    # 1. Precompute edges and faces
    # Map (u, v) with u < v to index
    edge_map = {}
    edges = []
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_map[(u, v)] = idx
    
    # Generate faces
    # A face is identified by (dim_i, dim_j, fixed_bits_mask, fixed_bits_val)
    # But we just need the list of 4 edge indices for each face.
    face_edges_list = []
    # We also need to know for each edge, which faces it belongs to
    edge_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all fixed bits for the other d-2 dimensions
            # There are 2^(d-2) such combinations
            for mask in range(1 << (d - 2)):
                # Construct the base vertex
                # We need to assign the fixed bits to the dimensions other than i and j
                # Let's iterate through dimensions k != i, j
                # It's easier to just iterate all vertices and group them?
                # No, explicit construction is better.
                
                # Let's create a function to get the base vertex
                # We need to map the 'mask' bits to the specific dimensions.
                # Let dims_fixed = [k for k in range(d) if k != i and k != j]
                dims_fixed = [k for k in range(d) if k != i and k != j]
                
                base = 0
                for idx, k in enumerate(dims_fixed):
                    if mask & (1 << idx):
                        base |= (1 << k)
                
                # The 4 vertices of the face
                v0 = base
                v1 = base ^ (1 << i)
                v2 = base ^ (1 << j)
                v3 = base ^ (1 << i) ^ (1 << j)
                
                # The 4 edges
                e01 = (min(v0, v1), max(v0, v1))
                e13 = (min(v1, v3), max(v1, v3))
                e32 = (min(v3, v2), max(v3, v2))
                e20 = (min(v2, v0), max(v2, v0))
                
                e01_idx = edge_map[e01]
                e13_idx = edge_map[e13]
                e32_idx = edge_map[e32]
                e20_idx = edge_map[e20]
                
                face_idx = len(face_edges_list)
                face_edges_list.append([e01_idx, e13_idx, e32_idx, e20_idx])
                
                edge_faces[e01_idx].append(face_idx)
                edge_faces[e13_idx].append(face_idx)
                edge_faces[e32_idx].append(face_idx)
                edge_faces[e20_idx].append(face_idx)
                
    n_faces = len(face_edges_list)
    
    # 2. Simulated Annealing
    best_score = -1
    best_edges = None
    
    # Run multiple restarts
    while time.time() < deadline:
        # Initialize with Greedy
        random.seed() # Or use a fixed seed for reproducibility in tests? No, random is fine.
        order = list(range(n_edges))
        random.shuffle(order)
        
        selected = [False] * n_edges
        face_counts = [0] * n_faces
        num_edges = 0
        violations = 0
        
        # Greedy Add
        for e_idx in order:
            # Check if adding e_idx causes violation
            ok = True
            for f_idx in edge_faces[e_idx]:
                if face_counts[f_idx] == 3:
                    ok = False
                    break
            if ok:
                selected[e_idx] = True
                num_edges += 1
                for f_idx in edge_faces[e_idx]:
                    face_counts[f_idx] += 1
                    if face_counts[f_idx] == 4:
                        violations += 1
        # Note: Greedy ensures no violations, so violations should be 0.
        
        current_score = num_edges - 100 * violations
        if violations == 0 and num_edges > best_score:
            best_score = num_edges
            best_edges = selected[:]
            
        # SA Loop
        T = 5.0
        cooling = 0.995
        steps_per_restart = 50000
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            e_idx = random.randint(0, n_edges - 1)
            
            # Calculate delta if we flip e_idx
            old_val = selected[e_idx]
            
            delta_edges = 1 if not old_val else -1
            delta_violations = 0
            
            if not old_val:
                # Adding
                for f_idx in edge_faces[e_idx]:
                    if face_counts[f_idx] == 3:
                        delta_violations += 1
                    # If face_counts[f_idx] == 4, it was already a violation, adding one more doesn't change the fact it's a violation?
                    # Wait, violation is "count == 4".
                    # If count is 4, it's a violation.
                    # If we add an edge to a face with count 3, count becomes 4 -> new violation.
                    # If we add an edge to a face with count 4, count becomes 5.
                    # But max count is 4. So this case doesn't happen if we maintain counts correctly.
                    # Actually, if count is 4, the edge is already selected? No, face has 4 edges.
                    # If we are adding an edge, it was 0.
                    # The other 3 edges in the face are 1. So count is 3.
                    # So count becomes 4.
                    # So delta_violations is number of faces where count is 3.
            else:
                # Removing
                for f_idx in edge_faces[e_idx]:
                    if face_counts[f_idx] == 4:
                        delta_violations -= 1
                    # If count is 4, removing makes it 3, so violation resolved.
                    # If count is < 4, no change in violation status.
            
            delta_score = delta_edges - 100 * delta_violations
            
            accept = False
            if delta_score >= 0:
                accept = True
            else:
                if random.random() < pow(2.718281828, -delta_score / T):
                    accept = True
            
            if accept:
                # Apply move
                selected[e_idx] = not old_val
                num_edges += delta_edges
                violations += delta_violations
                for f_idx in edge_faces[e_idx]:
                    if old_val:
                        face_counts[f_idx] -= 1
                    else:
                        face_counts[f_idx] += 1
                
                current_score = num_edges - 100 * violations
                if violations == 0 and num_edges > best_score:
                    best_score = num_edges
                    best_edges = selected[:]
            
            T *= cooling
            if T < 0.1:
                T = 0.1
    
    # Convert best_edges to list of [u, v]
    if best_edges is None:
        # Fallback: return empty or simple greedy
        return []
        
    result = []
    for i, is_sel in enumerate(best_edges):
        if is_sel:
            u, v = edges[i]
            result.append([u, v])
            
    return result

