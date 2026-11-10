import random
import time

def construct():
    d = 7
    n = 1 << d
    edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
    
    edge_index = {e: i for i, e in enumerate(edges)}
    E = len(edges)
    
    faces = []
    face_edges = []
    edge_faces = [set() for _ in range(E)]
    
    for i in range(d):
        for j in range(i + 1, d):
            other_bits = [k for k in range(d) if k != i and k != j]
            for mask in range(1 << len(other_bits)):
                u = 0
                for k, bit in enumerate(other_bits):
                    if mask & (1 << k):
                        u |= (1 << bit)
                
                v1 = u | (1 << i)
                v2 = u | (1 << j)
                v3 = u | (1 << i) | (1 << j)
                
                e1 = (u, v1)
                e2 = (u, v2)
                e3 = (v1, v3)
                e4 = (v2, v3)
                
                e1 = (min(e1), max(e1))
                e2 = (min(e2), max(e2))
                e3 = (min(e3), max(e3))
                e4 = (min(e4), max(e4))
                
                face_id = len(faces)
                faces.append((e1, e2, e3, e4))
                face_edges.append([edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]])
                
                for ei in face_edges[-1]:
                    edge_faces[ei].add(face_id)
                    
    F = len(faces)
    
    def greedy_removal_full(seed):
        random.seed(seed)
        active_faces = [True] * F
        face_count = [0] * E
        for f in range(F):
            for e in face_edges[f]:
                face_count[e] += 1
                
        removed = [False] * E
        active_face_count = F
        
        while active_face_count > 0:
            max_count = -1
            best_edges = []
            for e in range(E):
                if not removed[e]:
                    if face_count[e] > max_count:
                        max_count = face_count[e]
                        best_edges = [e]
                    elif face_count[e] == max_count:
                        best_edges.append(e)
            
            if not best_edges:
                break
                
            e = random.choice(best_edges)
            removed[e] = True
            
            for f in edge_faces[e]:
                if active_faces[f]:
                    active_faces[f] = False
                    active_face_count -= 1
                    for oe in face_edges[f]:
                        if oe != e:
                            face_count[oe] -= 1
                            
        kept = []
        for e in range(E):
            if not removed[e]:
                kept.append(list(edges[e]))
        return kept, len(kept)

    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = []
    best_val = 0
    seed = 0
    
    # Run at least one
    edges_list, val = greedy_removal_full(seed)
    if val > best_val:
        best_val = val
        best_edges = edges_list
    seed += 1
    
    while time.time() < deadline:
        edges_list, val = greedy_removal_full(seed)
        if val > best_val:
            best_val = val
            best_edges = edges_list
        seed += 1
        
    return best_edges
