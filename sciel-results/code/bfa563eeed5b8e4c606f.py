import time
import numpy as np
import random

def construct():
    deadline = time.time() + 115  # safety margin before 120s kill
    
    d = 7
    n = 1 << d  # 128 vertices
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
    num_edges = len(all_edges)  # 448
    
    edge_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_idx[(u, v)] = idx
    
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
                faces.append([edge_idx[e1], edge_idx[e2], edge_idx[e3], edge_idx[e4]])
    num_faces = len(faces)  # 672
    
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_to_faces[e_idx].append(f_idx)
    
    best_count = 0
    best_chosen = None
    
    # Try ILP first - this should find the optimal or near-optimal solution
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        c = -np.ones(num_edges)
        A = np.zeros((num_faces, num_edges))
        for f_idx, face in enumerate(faces):
            for e_idx in face:
                A[f_idx, e_idx] = 1.0
        constraints = LinearConstraint(A, ub=3.0)
        integrality = np.ones(num_edges)
        bounds = Bounds(0.0, 1.0)
        
        remaining = deadline - time.time() - 10
        time_limit = max(5, int(remaining))
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": time_limit})
        
        if res.x is not None:
            chosen = res.x.astype(int)
            count = int(chosen.sum())
            # Verify validity
            valid = True
            for face in faces:
                if sum(chosen[e] for e in face) > 3:
                    valid = False
                    break
            if valid and count > best_count:
                best_count = count
                best_chosen = chosen
    except Exception:
        pass
    
    # Local search fallback / improvement
    if time.time() < deadline - 3:
        def try_greedy(seed):
            rng = random.Random(seed)
            chosen = [False] * num_edges
            face_counts = [0] * num_faces
            count = 0
            indices = list(range(num_edges))
            rng.shuffle(indices)
            for idx in indices:
                can_add = True
                for f_idx in edge_to_faces[idx]:
                    if face_counts[f_idx] == 3:
                        can_add = False
                        break
                if can_add:
                    chosen[idx] = True
                    count += 1
                    for f_idx in edge_to_faces[idx]:
                        face_counts[f_idx] += 1
            return count, chosen, face_counts
        
        def local_search(chosen, face_counts, count, deadline):
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for r_idx in range(num_edges):
                    if time.time() >= deadline:
                        return count, chosen, face_counts
                    if not chosen[r_idx]:
                        continue
                    chosen[r_idx] = False
                    count -= 1
                    for f_idx in edge_to_faces[r_idx]:
                        face_counts[f_idx] -= 1
                    
                    # Find all unchosen edges that share a face with r_idx
                    candidate_set = set()
                    for f_idx in edge_to_faces[r_idx]:
                        for e_idx in faces[f_idx]:
                            if e_idx != r_idx and not chosen[e_idx]:
                                candidate_set.add(e_idx)
                    
                    # Find single-edge additions (1-for-1 swaps)
                    best_single = -1
                    for e_idx in candidate_set:
                        valid = True
                        for f_idx in edge_to_faces[e_idx]:
                            if face_counts[f_idx] == 3:
                                valid = False
                                break
                        if valid:
                            best_single = e_idx
                            break
                    
                    # Find 2-edge additions (1-for-2 swaps)
                    best_pair = None
                    for e1 in candidate_set:
                        if time.time() >= deadline:
                            break
                        # Check if e1 is individually valid
                        e1_valid = True
                        for f_idx in edge_to_faces[e1]:
                            if face_counts[f_idx] == 3:
                                e1_valid = False
                                break
                        if not e1_valid:
                            continue
                        # Check if e1 can be added (would make some faces count 3)
                        # For each face containing e1, see if count would become 3
                        for e2 in candidate_set:
                            if e2 <= e1:
                                continue
                            # Check e2 individually valid
                            e2_valid = True
                            for f_idx in edge_to_faces[e2]:
                                if face_counts[f_idx] == 3:
                                    e2_valid = False
                                    break
                            if not e2_valid:
                                continue
                            # Check if e1 and e2 together are valid
                            compatible = True
                            for f_idx in edge_to_faces[e1]:
                                if e2 in faces[f_idx]:
                                    if face_counts[f_idx] >= 2:
                                        compatible = False
                                        break
                            if compatible:
                                best_pair = (e1, e2)
                                break
                        if best_pair is not None:
                            break
                    
                    if best_pair is not None:
                        for e_idx in best_pair:
                            chosen[e_idx] = True
                            count += 1
                            for f_idx in edge_to_faces[e_idx]:
                                face_counts[f_idx] += 1
                        improved = True
                        break
                    elif best_single >= 0:
                        chosen[best_single] = True
                        count += 1
                        for f_idx in edge_to_faces[best_single]:
                            face_counts[f_idx] += 1
                        improved = True
                        break
                    else:
                        # No improvement, restore
                        chosen[r_idx] = True
                        count += 1
                        for f_idx in edge_to_faces[r_idx]:
                            face_counts[f_idx] += 1
            return count, chosen, face_counts
        
        # Use ILP result as starting point if available
        if best_chosen is not None and time.time() < deadline - 3:
            chosen = [bool(x) for x in best_chosen]
            face_counts = [0] * num_faces
            count = 0
            for idx in range(num_edges):
                if chosen[idx]:
                    count += 1
                    for f_idx in edge_to_faces[idx]:
                        face_counts[f_idx] += 1
            count, chosen, face_counts = local_search(chosen, face_counts, count, deadline)
            if count > best_count:
                best_count = count
                best_chosen = chosen
        
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() >= deadline - 3:
                break
            count, chosen, face_counts = try_greedy(attempt)
            count, chosen, face_counts = local_search(chosen, face_counts, count, deadline)
            if count > best_count:
                best_count = count
                best_chosen = chosen
    
    # Final fallback
    if best_chosen is None or best_count == 0:
        chosen = [False] * num_edges
        face_counts = [0] * num_faces
        count = 0
        for idx in range(num_edges):
            can_add = True
            for f_idx in edge_to_faces[idx]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] += 1
        best_count = count
        best_chosen = chosen
    
    out = []
    for idx in range(num_edges):
        if best_chosen[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out
