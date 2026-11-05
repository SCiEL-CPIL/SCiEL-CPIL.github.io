"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    d = 7
    n_vertices = 2 ** d
    n_edges = d * 2 ** (d - 1)  # 448
    
    # Build edge list
    edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    # Build edge index mapping
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
    
    # Build list of all 4-cycles (faces)
    faces = []
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    v0 = x
                    v1 = x ^ (1 << i)
                    v2 = x ^ (1 << j)
                    v3 = x ^ (1 << i) ^ (1 << j)
                    
                    e01 = (min(v0, v1), max(v0, v1))
                    e02 = (min(v0, v2), max(v0, v2))
                    e13 = (min(v1, v3), max(v1, v3))
                    e23 = (min(v2, v3), max(v2, v3))
                    
                    idx01 = edge_to_idx[e01]
                    idx02 = edge_to_idx[e02]
                    idx13 = edge_to_idx[e13]
                    idx23 = edge_to_idx[e23]
                    
                    faces.append([idx01, idx02, idx13, idx23])
    
    # For each edge, find which faces it belongs to
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # Greedy initialization
    random.seed(42)
    selected = [False] * n_edges
    selected_count = 0
    face_complete = [0] * len(faces)
    
    edge_order = list(range(n_edges))
    random.shuffle(edge_order)
    
    for e_idx in edge_order:
        can_add = True
        for f_idx in edge_faces[e_idx]:
            if face_complete[f_idx] == 3:
                can_add = False
                break
        if can_add:
            selected[e_idx] = True
            selected_count += 1
            for f_idx in edge_faces[e_idx]:
                face_complete[f_idx] += 1
    
    best_count = selected_count
    best_selected = selected[:]
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.99995
    
    while time.time() < deadline and T > T_min:
        # Pick a random edge to flip
        e_idx = random.randint(0, n_edges - 1)
        
        if selected[e_idx]:
            # Consider removing this edge
            old_count = selected_count
            selected[e_idx] = False
            selected_count -= 1
            for f_idx in edge_faces[e_idx]:
                face_complete[f_idx] -= 1
            
            delta = selected_count - old_count  # -1
            # Accept with probability exp(delta/T) = exp(-1/T)
            if delta >= 0 or random.random() < math.exp(delta / T):
                if selected_count > best_count:
                    best_count = selected_count
                    best_selected = selected[:]
            else:
                # Reject
                selected[e_idx] = True
                selected_count = old_count
                for f_idx in edge_faces[e_idx]:
                    face_complete[f_idx] += 1
        else:
            # Consider adding this edge
            can_add = True
            for f_idx in edge_faces[e_idx]:
                if face_complete[f_idx] == 3:
                    can_add = False
                    break
            
            if can_add:
                old_count = selected_count
                selected[e_idx] = True
                selected_count += 1
                for f_idx in edge_faces[e_idx]:
                    face_complete[f_idx] += 1
                
                delta = selected_count - old_count  # +1
                # Always accept improvements
                if delta >= 0:
                    if selected_count > best_count:
                        best_count = selected_count
                        best_selected = selected[:]
                else:
                    if random.random() >= math.exp(delta / T):
                        selected[e_idx] = False
                        selected_count = old_count
                        for f_idx in edge_faces[e_idx]:
                            face_complete[f_idx] -= 1
        
        T *= cooling
    
    # Return best solution
    result = []
    for e_idx in range(n_edges):
        if best_selected[e_idx]:
            result.append(list(edges[e_idx]))
    
    return result
# EVOLVE-BLOCK-END
