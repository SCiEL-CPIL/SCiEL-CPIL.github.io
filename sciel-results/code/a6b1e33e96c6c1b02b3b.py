import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Maximize C4-free edge set in Q7.
    Uses Complementary Deletion (start full, remove edges) and fast Local Search.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    d = 7
    n_vertices = 1 << d
    n_edges_total = d * (1 << (d - 1))  # 448
    
    # Precompute edges
    edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                
    n_edges = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Precompute faces (4-cycles)
    # A face is defined by (base, dim1, dim2) where base has 0 in dim1 and dim2
    faces = []
    for base in range(n_vertices):
        for d1 in range(d):
            if (base >> d1) & 1:
                continue
            for d2 in range(d1 + 1, d):
                if (base >> d2) & 1:
                    continue
                v1 = base
                v2 = base ^ (1 << d1)
                v3 = base ^ (1 << d2)
                v4 = base ^ (1 << d1) ^ (1 << d2)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                
                face_edges = (edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4])
                faces.append(face_edges)
                
    n_faces = len(faces)
    
    # Incidence: edge -> list of faces
    edge_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for ei in face:
            edge_faces[ei].append(fi)
            
    # Convert to numpy for vectorized ops if needed, but list is fast for 448 edges
    faces_np = np.array(faces, dtype=np.int32)
    
    def time_left():
        return deadline - time.time()
        
    def is_valid(sol):
        sol_arr = np.array(sol, dtype=np.bool_)
        # Check if any face has all 4 edges
        # faces_np shape (n_faces, 4)
        counts = sol_arr[faces_np].sum(axis=1)
        return not np.any(counts == 4)
        
    def count_edges(sol):
        return sum(sol)
        
    def complementary_deletion(seed):
        """Start with all edges, remove to break C4s."""
        rng = random.Random(seed)
        sol = [True] * n_edges
        face_counts = [4] * n_faces
        bad_faces = set(range(n_faces))
        
        # We want to remove edges to minimize total edges removed.
        # Greedy: pick a bad face, remove an edge from it that is in the most other bad faces.
        
        while bad_faces and time_left() > 5:
            # Pick a random bad face
            fi = rng.choice(list(bad_faces))
            
            # Find edge in this face that hits the most other bad faces
            best_ei = -1
            max_hits = -1
            for ei in faces[fi]:
                if not sol[ei]:
                    continue
                hits = 0
                for f2 in edge_faces[ei]:
                    if f2 in bad_faces:
                        hits += 1
                if hits > max_hits:
                    max_hits = hits
                    best_ei = ei
                    
            if best_ei != -1:
                sol[best_ei] = False
                for f2 in edge_faces[best_ei]:
                    face_counts[f2] -= 1
                    if face_counts[f2] == 3:
                        bad_faces.discard(f2)
                        
        return sol

    def local_search(sol, seed):
        """
        Fast local search.
        State: sol (list of bools), face_counts (list of ints), bad_faces (set), addable (set)
        """
        rng = random.Random(seed)
        sol = list(sol)
        face_counts = [0] * n_faces
        bad_faces = set()
        
        # Initialize face_counts and bad_faces
        for ei in range(n_edges):
            if sol[ei]:
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
                    if face_counts[fi] == 4:
                        bad_faces.add(fi)
                        
        # Initialize addable set: edges not in sol that can be added
        # An edge can be added if for all faces f containing it, face_counts[f] < 3
        addable = set()
        for ei in range(n_edges):
            if not sol[ei]:
                can_add = True
                for fi in edge_faces[ei]:
                    if face_counts[fi] >= 3:
                        can_add = False
                        break
                if can_add:
                    addable.add(ei)
                    
        improved = True
        iter_count = 0
        max_iter = 100000
        
        while improved and time_left() > 10 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            if bad_faces:
                # Repair: Remove edge from a bad face
                # Choose bad face with most "flexible" edges? Or just random.
                # Let's choose edge that is in the most bad faces.
                best_ei = -1
                max_hits = -1
                # Sample a few bad faces to speed up
                sample_size = min(5, len(bad_faces))
                sample_faces = rng.sample(list(bad_faces), sample_size)
                for fi in sample_faces:
                    for ei in faces[fi]:
                        if sol[ei]:
                            hits = 0
                            for f2 in edge_faces[ei]:
                                if f2 in bad_faces:
                                    hits += 1
                            if hits > max_hits:
                                max_hits = hits
                                best_ei = ei
                                
                if best_ei != -1:
                    sol[best_ei] = False
                    for f2 in edge_faces[best_ei]:
                        face_counts[f2] -= 1
                        if face_counts[f2] == 3:
                            bad_faces.discard(f2)
                        # Update addable: this edge might become addable? No, we removed it.
                        # But other edges in these faces might become addable.
                        # We will recompute addable if it becomes empty, or just check specific edges.
                    improved = True
                    
            else:
                # No bad faces. Try to add edges.
                if addable:
                    ei = rng.choice(list(addable))
                    sol[ei] = True
                    addable.discard(ei)
                    for f2 in edge_faces[ei]:
                        face_counts[f2] += 1
                        # This might make other edges non-addable
                        # Specifically, edges sharing a face with ei.
                        # We don't update addable set incrementally for "removal" of addability here for simplicity,
                        # but we can. If face_counts[f2] becomes 3, then any other edge in f2 is no longer addable.
                        if face_counts[f2] == 3:
                            for e_other in faces[f2]:
                                if e_other in addable:
                                    addable.discard(e_other)
                    improved = True
                else:
                    # Stuck. Perturb.
                    # Remove a random edge to break symmetry or allow different path
                    if time_left() > 20:
                        selected = [e for e in range(n_edges) if sol[e]]
                        if selected:
                            ei = rng.choice(selected)
                            sol[ei] = False
                            for f2 in edge_faces[ei]:
                                face_counts[f2] -= 1
                                if face_counts[f2] == 3:
                                    # e_other might become addable
                                    for e_other in faces[f2]:
                                        if not sol[e_other]:
                                            # Check if it can be added
                                            can_add = True
                                            for f3 in edge_faces[e_other]:
                                                if face_counts[f3] >= 3:
                                                    can_add = False
                                                    break
                                            if can_add:
                                                addable.add(e_other)
                            improved = True
                    else:
                        break
                        
        return sol

    best_count = 0
    best_solution = None
    
    # Run multiple restarts
    for i in range(200):
        if time_left() < 15:
            break
            
        seed = i * 12345 + 67890
        # Alternate between complementary deletion and random greedy
        if i % 2 == 0:
            sol = complementary_deletion(seed)
        else:
            # Random Greedy
            sol = [False] * n_edges
            face_counts = [0] * n_faces
            order = list(range(n_edges))
            random.Random(seed).shuffle(order)
            for ei in order:
                can_add = True
                for fi in edge_faces[ei]:
                    if face_counts[fi] == 3:
                        can_add = False
                        break
                if can_add:
                    sol[ei] = True
                    for fi in edge_faces[ei]:
                        face_counts[fi] += 1
            # Repair if necessary (greedy should be valid)
            
        # Local Search
        sol = local_search(sol, seed)
        
        cnt = count_edges(sol)
        if is_valid(sol):
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
                
    # Fallback
    if best_solution is None:
        # Simple deterministic greedy
        sol = [False] * n_edges
        face_counts = [0] * n_faces
        for ei in range(n_edges):
            can_add = True
            for fi in edge_faces[ei]:
                if face_counts[fi] == 3:
                    can_add = False
                    break
            if can_add:
                sol[ei] = True
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        best_solution = sol
        best_count = count_edges(sol)
        
    result = []
    for ei in range(n_edges):
        if best_solution[ei]:
            result.append(list(edges[ei]))
            
    return result
