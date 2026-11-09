import numpy as np
import random
import time

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    d = 7
    n = 1 << d
    total_edges = d * (n // 2)
    
    # Precompute all edges with their dimensions
    # Edge representation: (u, v, dim) where u < v, v = u ^ (1 << dim)
    edges = []
    edge_index = {}  # (u, v) -> index
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v, i))
                edge_index[(u, v)] = idx
    
    E = len(edges)  # 448
    
    # Precompute for each edge, the other 3 edges of each 2-face it belongs to
    # An edge (u, v, dim) is in a face with dimension j (j != dim)
    # The face has vertices u, v, u^(1<<j), v^(1<<j)
    # The other 3 edges are:
    #   (u, u^(1<<j)), (v, v^(1<<j)), (u^(1<<j), v^(1<<j))
    
    # face_edges[e] = list of (other_edge_idx1, other_edge_idx2, other_edge_idx3) for each face
    face_edges = [[] for _ in range(E)]
    for e_idx, (u, v, dim) in enumerate(edges):
        for j in range(d):
            if j == dim:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            # Three other edges in this face
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            other_indices = []
            for (a, b) in [e1, e2, e3]:
                if (a, b) in edge_index:
                    other_indices.append(edge_index[(a, b)])
            face_edges[e_idx].append(tuple(other_indices))
    
    # Convert to numpy for speed
    # For each edge, store its faces as arrays
    # Max faces per edge = d-1 = 6
    max_faces = d - 1
    faces_array = np.full((E, max_faces, 3), -1, dtype=np.int32)
    for e_idx in range(E):
        for fi, face in enumerate(face_edges[e_idx]):
            if len(face) == 3:
                faces_array[e_idx, fi] = face
    
    # Face count: number of selected edges in each face
    # A face is identified by (u, dim1, dim2) where u < u^(1<<dim1) < u^(1<<dim1)^(1<<dim2)
    # Total faces: C(7,2) * 2^(7-2) = 21 * 32 = 672
    
    # Let's enumerate all faces
    faces = []  # list of (v1, v2, v3, v4, e1, e2, e3, e4)
    for i in range(d):
        for j in range(i+1, d):
            for base in range(n):
                # Check that base is the minimum vertex in this face
                # Face vertices: base, base^(1<<i), base^(1<<j), base^(1<<(i|j))
                v1 = base
                v2 = base ^ (1 << i)
                v3 = base ^ (1 << j)
                v4 = base ^ (1 << i) ^ (1 << j)
                # Only count if v1 is the smallest
                if v1 == min(v1, v2, v3, v4):
                    # Get the 4 edges
                    e_a = (min(v1, v2), max(v1, v2))
                    e_b = (min(v1, v3), max(v1, v3))
                    e_c = (min(v2, v4), max(v2, v4))
                    e_d = (min(v3, v4), max(v3, v4))
                    idxs = []
                    valid = True
                    for (a, b) in [e_a, e_b, e_c, e_d]:
                        if (a, b) in edge_index:
                            idxs.append(edge_index[(a, b)])
                        else:
                            valid = False
                            break
                    if valid:
                        faces.append(tuple(idxs))
    
    F = len(faces)  # should be 672
    faces_np = np.array(faces, dtype=np.int32)  # shape (F, 4)
    
    # For each edge, which faces does it belong to?
    edge_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    edge_faces_np = np.array([ef if ef else [-1] for ef in edge_faces], dtype=np.int32)
    # Actually let's keep it as a list for flexibility
    
    def greedy_random(seed=42):
        rng = random.Random(seed)
        order = list(range(E))
        rng.shuffle(order)
        
        selected = np.zeros(E, dtype=bool)
        face_count = np.zeros(F, dtype=np.int32)
        
        count = 0
        for e_idx in order:
            # Check if adding this edge would make any face have 4 edges
            can_add = True
            for f_idx in edge_faces[e_idx]:
                if face_count[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                selected[e_idx] = True
                count += 1
                for f_idx in edge_faces[e_idx]:
                    face_count[f_idx] += 1
        
        return selected, face_count, count
    
    def local_search(selected, face_count, count, time_limit=10.0):
        """Try to improve by removing one edge and adding two others (or other moves)."""
        rng = random.Random(12345)
        start = time.time()
        improved = True
        best_count = count
        
        while improved and time.time() < start + time_limit:
            improved = False
            # Try removing one edge and adding back two
            # Actually, let's try a simpler move: try to add an edge that was rejected
            # First, find edges that can be added
            rejected = [i for i in range(E) if not selected[i]]
            rng.shuffle(rejected)
            
            for e_add in rejected:
                if time.time() > start + time_limit:
                    break
                # Check which faces would be violated
                violating_faces = []
                for f_idx in edge_faces[e_add]:
                    if face_count[f_idx] == 3:
                        violating_faces.append(f_idx)
                
                if not violating_faces:
                    # Can add directly
                    selected[e_add] = True
                    count += 1
                    best_count = count
                    for f_idx in edge_faces[e_add]:
                        face_count[f_idx] += 1
                    improved = True
                    continue
                
                # Try to remove one edge from each violating face to make room
                # For each violating face, we need to remove one of the other 3 edges
                # Try all combinations (small number of violating faces)
                if len(violating_faces) <= 2:
                    # Try removing one edge to unblock
                    candidates_to_remove = set()
                    for f_idx in violating_faces:
                        for e_other in faces_np[f_idx]:
                            if e_other != e_add and selected[e_other]:
                                candidates_to_remove.add(e_other)
                    
                    for e_remove in candidates_to_remove:
                        # Check if removing e_remove and adding e_add is valid
                        # After removing e_remove, check if e_add can be added
                        new_face_count = face_count.copy()
                        for f_idx in edge_faces[e_remove]:
                            new_face_count[f_idx] -= 1
                        for f_idx in edge_faces[e_add]:
                            if new_face_count[f_idx] >= 3:
                                break
                        else:
                            # Valid move: remove e_remove, add e_add
                            selected[e_remove] = False
                            selected[e_add] = True
                            for f_idx in edge_faces[e_remove]:
                                face_count[f_idx] -= 1
                            for f_idx in edge_faces[e_add]:
                                face_count[f_idx] += 1
                            # Count stays same, but we might be able to add more later
                            improved = True
                            break
    
        return selected, face_count, count
    
    best_result = None
    best_count = 0
    seed = 42
    
    # Run multiple greedy with different seeds
    while time.time() < deadline - 10.0:
        selected, face_count, count = greedy_random(seed)
        if count > best_count:
            best_count = count
            best_result = selected.copy()
        
        # Try local search
        if time.time() < deadline - 5.0:
            selected_ls, face_count_ls, count_ls = local_search(selected, face_count, count, time_limit=5.0)
            if count_ls > best_count:
                best_count = count_ls
                best_result = selected_ls.copy()
        
        seed += 1
        if seed % 50 == 0:
            # Periodically try a different approach
            pass
    
    # Final attempt: try to add more edges to best_result
    if best_result is not None and time.time() < deadline - 3.0:
        selected = best_result.copy()
        face_count = np.zeros(F, dtype=np.int32)
        for e_idx in range(E):
            if selected[e_idx]:
                for f_idx in edge_faces[e_idx]:
                    face_count[f_idx] += 1
        
        # Try adding any remaining edges
        for e_idx in range(E):
            if not selected[e_idx]:
                can_add = True
                for f_idx in edge_faces[e_idx]:
                    if face_count[f_idx] == 3:
                        can_add = False
                        break
                if can_add:
                    selected[e_idx] = True
                    for f_idx in edge_faces[e_idx]:
                        face_count[f_idx] += 1
        
        final_count = np.sum(selected)
        if final_count > best_count:
            best_count = final_count
            best_result = selected
    
    # Convert to output format
    out = []
    for e_idx in range(E):
        if best_result is not None and best_result[e_idx]:
            u, v, dim = edges[e_idx]
            out.append([u, v])
    
    return out
