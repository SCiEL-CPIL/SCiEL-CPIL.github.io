import time
import random
import math

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = n_vertices * d // 2  # 448
    n_faces = (d * (d - 1) // 2) * (1 << (d - 2))  # 672
    
    # Time limit
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Precomputation
    edges = []
    edge_index = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_index[(u, v)] = idx
                edges.append((u, v))
                idx += 1
    assert idx == n_edges
    
    faces = []
    face_index = {}
    f_idx = 0
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                verts = sorted([u, v1, v2, v3])
                key = tuple(verts)
                if key not in face_index:
                    face_index[key] = f_idx
                    face_edges = []
                    for a in range(4):
                        for b in range(a + 1, 4):
                            va, vb = verts[a], verts[b]
                            if (va ^ vb) & (va ^ vb - 1) == 0:
                                if va < vb:
                                    e_idx = edge_index[(va, vb)]
                                else:
                                    e_idx = edge_index[(vb, va)]
                                face_edges.append(e_idx)
                    faces.append(face_edges)
                    f_idx += 1
    assert f_idx == n_faces
    
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx in range(n_faces):
        for e_idx in faces[f_idx]:
            edge_faces[e_idx].append(f_idx)
            
    selected = [False] * n_edges
    face_count = [0] * n_faces
    
    # Greedy Initialization
    for e_idx in range(n_edges):
        can_add = True
        for f_idx in edge_faces[e_idx]:
            if face_count[f_idx] == 3:
                can_add = False
                break
        if can_add:
            selected[e_idx] = True
            for f_idx in edge_faces[e_idx]:
                face_count[f_idx] += 1
                
    current_size = sum(selected)
    best_size = current_size
    best_selected = selected[:]
    
    def find_min_hitting_set(bad_faces, excluded_edge):
        uncovered = set(bad_faces)
        R = []
        while uncovered:
            candidates = {}
            for f_idx in uncovered:
                for e_idx in faces[f_idx]:
                    if e_idx == excluded_edge:
                        continue
                    if e_idx in candidates:
                        candidates[e_idx] += 1
                    else:
                        candidates[e_idx] = 1
            if not candidates:
                return None
            best_e = max(candidates, key=lambda x: candidates[x])
            R.append(best_e)
            for f_idx in faces[best_e]:
                if f_idx in uncovered:
                    uncovered.remove(f_idx)
        
        if len(R) <= 1:
            return R
            
        # Redundancy removal
        changed = True
        while changed:
            changed = False
            for i in range(len(R)):
                R_test = R[:i] + R[i+1:]
                covered = set()
                for e_idx in R_test:
                    for f_idx in faces[e_idx]:
                        covered.add(f_idx)
                if all(f in covered for f in bad_faces):
                    R = R_test
                    changed = True
                    break
        return R

    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Pick random unselected edge
        e_add = random.randint(0, n_edges - 1)
        if selected[e_add]:
            continue
            
        bad_faces = []
        for f_idx in edge_faces[e_add]:
            if face_count[f_idx] == 3:
                bad_faces.append(f_idx)
        
        if not bad_faces:
            selected[e_add] = True
            for f_idx in edge_faces[e_add]:
                face_count[f_idx] += 1
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_selected = selected[:]
        else:
            R = find_min_hitting_set(bad_faces, e_add)
            if R is None:
                continue
            delta_size = 1 - len(R)
            
            if delta_size > 0:
                accept = True
            elif delta_size == 0:
                accept = True
            else:
                # Accept with probability exp(delta_size / T)
                # delta_size is negative
                p = math.exp(delta_size / T)
                accept = (random.random() < p)
            
            if accept:
                selected[e_add] = True
                for f_idx in edge_faces[e_add]:
                    face_count[f_idx] += 1
                for e_rem in R:
                    selected[e_rem] = False
                    for f_idx in edge_faces[e_rem]:
                        face_count[f_idx] -= 1
                current_size += delta_size
                if current_size > best_size:
                    best_size = current_size
                    best_selected = selected[:]
        
        T *= cooling_rate

    out = []
    for i in range(n_edges):
        if best_selected[i]:
            out.append(list(edges[i]))
    return out
