import random
import time
import math

def construct():
    d = 7
    n_v = 1 << d
    deadline = time.time() + 130  # Safety margin before 150s kill

    # Build edges
    edges = []
    edge_idx = {}
    for x in range(n_v):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y, i))
                edge_idx[(x, y)] = idx

    n_edges = len(edges)  # 448

    # Build faces
    faces = []
    edge_to_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            other_bits = [k for k in range(d) if k != i and k != j]
            for mask in range(1 << (d - 2)):
                base = 0
                for k_idx, k in enumerate(other_bits):
                    if mask & (1 << k_idx):
                        base |= (1 << k)
                
                v0 = base
                v1 = base ^ (1 << i)
                v2 = base ^ (1 << j)
                v3 = base ^ (1 << i) ^ (1 << j)
                
                e01 = edge_idx[(min(v0, v1), max(v0, v1))]
                e02 = edge_idx[(min(v0, v2), max(v0, v2))]
                e13 = edge_idx[(min(v1, v3), max(v1, v3))]
                e23 = edge_idx[(min(v2, v3), max(v2, v3))]
                
                face_edges = [e01, e02, e13, e23]
                face_id = len(faces)
                faces.append(face_edges)
                for e in face_edges:
                    edge_to_faces[e].append(face_id)

    n_faces = len(faces)  # 672

    def greedy_init(rng):
        selected = [False] * n_edges
        face_count = [0] * n_faces
        count = 0
        order = list(range(n_edges))
        rng.shuffle(order)
        
        for e in order:
            valid = True
            for f in edge_to_faces[e]:
                if face_count[f] >= 3:
                    valid = False
                    break
            if valid:
                selected[e] = True
                count += 1
                for f in edge_to_faces[e]:
                    face_count[f] += 1
        return selected, face_count, count

    def simulate_annealing(selected, face_count, count, time_limit, rng):
        best_count = count
        best_selected = selected[:]
        
        T = 1.0
        cooling_rate = 0.99995
        
        selected_edges = [e for e in range(n_edges) if selected[e]]
        unselected_edges = [e for e in range(n_edges) if not selected[e]]
        
        # Precompute sets of faces for each edge for faster swap checks?
        # edge_to_faces is a list. Converting to set for each swap is O(6).
        # It's fine.
        
        start_time = time.time()
        while time.time() < time_limit:
            move_type = rng.random()
            if move_type < 0.3: # Add
                if not unselected_edges:
                    continue
                e = rng.choice(unselected_edges)
                valid = True
                for f in edge_to_faces[e]:
                    if face_count[f] >= 3:
                        valid = False
                        break
                if valid:
                    selected[e] = True
                    count += 1
                    unselected_edges.remove(e)
                    selected_edges.append(e)
                    for f in edge_to_faces[e]:
                        face_count[f] += 1
                    
                    if count > best_count:
                        best_count = count
                        best_selected = selected[:]
                        
            elif move_type < 0.6: # Remove
                if not selected_edges:
                    continue
                e = rng.choice(selected_edges)
                # Delta C = -1
                # P = exp(-1/T)
                if rng.random() < math.exp(-1.0 / T):
                    selected[e] = False
                    count -= 1
                    selected_edges.remove(e)
                    unselected_edges.append(e)
                    for f in edge_to_faces[e]:
                        face_count[f] -= 1
                        
            else: # Swap
                if not selected_edges or not unselected_edges:
                    continue
                u = rng.choice(selected_edges)
                v = rng.choice(unselected_edges)
                
                faces_u = set(edge_to_faces[u])
                valid = True
                for f in edge_to_faces[v]:
                    if f not in faces_u:
                        if face_count[f] > 2:
                            valid = False
                            break
                
                if valid:
                    # Remove u
                    selected[u] = False
                    selected_edges.remove(u)
                    unselected_edges.append(u)
                    for f in edge_to_faces[u]:
                        face_count[f] -= 1
                        
                    # Add v
                    selected[v] = True
                    unselected_edges.remove(v)
                    selected_edges.append(v)
                    for f in edge_to_faces[v]:
                        face_count[f] += 1
                    
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
                
        return best_count, best_selected

    best_overall_count = 0
    best_overall_selected = None
    
    rng = random.Random(42)
    
    sel, fc, cnt = greedy_init(rng)
    best_overall_count = cnt
    best_overall_selected = sel
    
    sa_deadline = time.time() + 120
    cnt, sel = simulate_annealing(sel, fc, cnt, sa_deadline, rng)
    
    if cnt > best_overall_count:
        best_overall_count = cnt
        best_overall_selected = sel
        
    # If time remains, do another restart?
    if time.time() < deadline - 5:
         sel2, fc2, cnt2 = greedy_init(rng)
         sa_deadline2 = time.time() + 5
         if time.time() < deadline - 5:
             cnt2, sel2 = simulate_annealing(sel2, fc2, cnt2, sa_deadline2, rng)
         if cnt2 > best_overall_count:
             best_overall_count = cnt2
             best_overall_selected = sel2

    out = []
    for e in range(n_edges):
        if best_overall_selected[e]:
            x, y, i = edges[e]
            out.append([x, y])
    
    return out
