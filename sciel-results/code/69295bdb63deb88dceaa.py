import itertools

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # 1. Map edges to indices
    edge_to_idx = {}
    idx_to_edge = []
    
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # 2. Generate all C4 faces
    faces = set()
    for x in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v0 = x
                v1 = x ^ (1 << i)
                v2 = x ^ (1 << j)
                v3 = x ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v0, v1), max(v0, v1))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v3, v2), max(v3, v2))
                e4 = (min(v2, v0), max(v2, v0))
                
                try:
                    i1 = edge_to_idx[e1]
                    i2 = edge_to_idx[e2]
                    i3 = edge_to_idx[e3]
                    i4 = edge_to_idx[e4]
                    face = frozenset([i1, i2, i3, i4])
                    faces.add(face)
                except KeyError:
                    pass
                    
    face_list = list(faces)
    n_faces = len(face_list)
    
    # Precompute edge_to_faces
    edge_to_faces = [[] for _ in range(n_edges)]
    for f_idx, f in enumerate(face_list):
        for e in f:
            edge_to_faces[e].append(f_idx)
            
    # 3. Greedy Hitting Set
    active_faces = set(range(n_faces))
    
    face_count = [0] * n_edges
    for f in face_list:
        for e in f:
            face_count[e] += 1
            
    removed = [False] * n_edges
    
    while active_faces:
        max_count = -1
        best_edge = -1
        
        for e in range(n_edges):
            if not removed[e] and face_count[e] > max_count:
                max_count = face_count[e]
                best_edge = e
                
        if best_edge == -1:
            break
            
        removed[best_edge] = True
        
        for f_idx in edge_to_faces[best_edge]:
            if f_idx in active_faces:
                active_faces.remove(f_idx)
                for e in face_list[f_idx]:
                    if e != best_edge:
                        face_count[e] -= 1
                        
    # 4. Local Search / Pruning Phase
    # Goal: Remove edges from 'removed' if possible (i.e., make them selected)
    # An edge e in 'removed' can be removed from 'removed' (added back to solution)
    # if no face containing e has exactly 1 removed edge (which is e).
    # If a face has 1 removed edge, that edge is critical.
    
    # Initialize face_removed_count
    face_removed_count = [0] * n_faces
    for f_idx, f in enumerate(face_list):
        count = 0
        for e in f:
            if removed[e]:
                count += 1
        face_removed_count[f_idx] = count
        
    # Pruning loop
    # We iterate until no more edges can be added back
    # To avoid infinite loops or excessive time, we limit iterations or rely on convergence
    # Since we are strictly decreasing the size of 'removed', it must terminate.
    
    improved = True
    while improved:
        improved = False
        for e in range(n_edges):
            if removed[e]:
                # Check if e is critical for any face
                is_critical = False
                for f_idx in edge_to_faces[e]:
                    if face_removed_count[f_idx] == 1:
                        is_critical = True
                        break
                
                if not is_critical:
                    # Safe to remove e from 'removed'
                    removed[e] = False
                    improved = True
                    # Update face_removed_count
                    for f_idx in edge_to_faces[e]:
                        face_removed_count[f_idx] -= 1
                        
    # Construct result
    result = []
    for e_idx in range(n_edges):
        if not removed[e_idx]:
            u, v = idx_to_edge[e_idx]
            result.append([u, v])
            
    return result
