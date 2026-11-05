import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    d = 7
    n_vertices = 1 << d
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}

    # Generate all 2D faces (C4s)
    faces = []
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                if (x & (1 << i)) != 0 or (x & (1 << j)) != 0:
                    continue
                v00 = x
                v10 = x ^ (1 << i)
                v01 = x ^ (1 << j)
                v11 = x ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v00, v10), max(v00, v10))
                e2 = (min(v10, v11), max(v10, v11))
                e3 = (min(v01, v11), max(v01, v11))
                e4 = (min(v00, v01), max(v00, v01))
                
                face_edges = [edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]]
                faces.append(face_edges)

    # MILP Formulation
    # Maximize sum(x_e)
    # Subject to sum(x_e for e in face) <= 3 for all faces
    # 0 <= x_e <= 1, x_e integer
    
    A = np.zeros((len(faces), num_edges))
    for k, face in enumerate(faces):
        for idx in face:
            A[k, idx] = 1.0
    
    b = np.ones(len(faces)) * 3.0
    c = -np.ones(num_edges)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(num_edges)
    
    res = milp(c=c, constraints=LinearConstraint(A, ub=b), bounds=bounds, integrality=integrality)
    
    if res.status == 0 and res.x is not None:
        chosen_indices = [i for i in range(num_edges) if res.x[i] > 0.5]
        # Safety check to ensure validity
        chosen_set = set(chosen_indices)
        for face in faces:
            if all(idx in chosen_set for idx in face):
                chosen_set.remove(face[-1])
        return [[all_edges[i][0], all_edges[i][1]] for i in chosen_set]
    else:
        # Fallback: Simple greedy (deterministic)
        # This is unlikely to be reached.
        chosen = set()
        # Precompute face membership for speed
        edge_faces = [[] for _ in range(num_edges)]
        for k, face in enumerate(faces):
            for idx in face:
                edge_faces[idx].append(k)
                
        for i in range(num_edges):
            ok = True
            for fk in edge_faces[i]:
                face = faces[fk]
                # Check if other 3 edges are in chosen
                others = [x for x in face if x != i]
                if all(x in chosen for x in others):
                    ok = False
                    break
            if ok:
                chosen.add(i)
        return [[all_edges[i][0], all_edges[i][1]] for i in chosen]
