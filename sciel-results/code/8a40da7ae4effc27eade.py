import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 148  # safety margin before 150s kill
    
    d = 7
    n_vertices = 1 << d  # 128
    n_edges_total = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    n_edges = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Build face list: each face is 4 edge indices
    faces = []
    for base in range(n_vertices):
        for d1 in range(d):
            if (base >> d1) & 1:
                continue
            for d2 in range(d1 + 1, d):
                if (base >> d2) & 1:
                    continue
                e1 = (base, base ^ (1 << d1))
                e2 = (base, base ^ (1 << d2))
                e3 = (base ^ (1 << d1), base ^ (1 << d1) ^ (1 << d2))
                e4 = (base ^ (1 << d2), base ^ (1 << d1) ^ (1 << d2))
                
                e1 = (min(e1), max(e1))
                e2 = (min(e2), max(e2))
                e3 = (min(e3), max(e3))
                e4 = (min(e4), max(e4))
                
                face_edges = (edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4])
                faces.append(face_edges)
    
    n_faces = len(faces)
    
    # For each edge, list of faces it belongs to
    edge_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for ei in face:
            edge_faces[ei].append(fi)
    
    # Convert to numpy arrays for vectorized operations
    faces_np = np.array(faces, dtype=np.int32)  # shape (n_faces, 4)
    
    # Build adjacency: for each edge, the set of edges that share a face
    # Two edges conflict if they are in the same face
    # For each edge, precompute the mask of edges that would create a C4 if both added
    # Actually, we need: if we add edge e, it creates a C4 if all other 3 edges in any face are present
    
    # Precompute for each edge, the faces it's in (as numpy array)
    edge_faces_arr = []
    for ei in range(n_edges):
        edge_faces_arr.append(np.array(edge_faces[ei], dtype=np.int32))
    
    def time_left():
        return deadline - time.time()
    
    def validate(sol_np):
        """Check no face has all 4 edges. Returns True if valid."""
        # sol_np is boolean array of shape (n_edges,)
        # faces_np is (n_faces, 4)
        # A face is bad if all 4 edges are selected
        face_sums = np.sum(sol_np[faces_np], axis=1)
        return not np.any(face_sums == 4)
    
    def count_edges(sol_np):
        return int(np.sum(sol_np))
    
    def greedy_initial(rng, mode=0):
        """Generate greedy initial solution with different strategies."""
        sol = np.zeros(n_edges, dtype=np.bool_)
        face_counts = np.zeros(n_faces, dtype=np.int32)
        
        if mode == 0:
            # Random order
            order = np.arange(n_edges)
            rng.shuffle(order.tolist())
            order = np.array(order)
        elif mode == 1:
            # Bit-dimension priority: process edges by dimension
            order = []
            for dim in range(d):
                for x in range(n_vertices):
                    y = x ^ (1 << dim)
                    if x < y:
                        order.append(edge_to_idx[(x, y)])
            order = np.array(order)
            rng.shuffle(order.tolist())
            order = np.array(order)
        elif mode == 2:
            # Gray code order
            order = []
            for x in range(n_vertices):
                for dim in range(d):
                    y = x ^ (1 << dim)
                    if x < y:
                        order.append(edge_to_idx[(x, y)])
            order = np.array(order)
        else:
            # Reverse bit-dimension
            order = []
            for dim in range(d - 1, -1, -1):
                for x in range(n_vertices):
                    y = x ^ (1 << dim)
                    if x < y:
                        order.append(edge_to_idx[(x, y)])
            order = np.array(order)
            rng.shuffle(order.tolist())
            order = np.array(order)
        
        for ei in order:
            # Check if adding this edge creates a C4
            # It creates a C4 if in any face containing ei, the other 3 edges are all present
            fis = edge_faces_arr[ei]
            if len(fis) == 0:
                sol[ei] = True
                continue
            # For each face, check if other 3 edges are selected
            # face_counts[fi] counts how many edges in face fi are selected
            # If face_counts[fi] == 3, adding ei makes it 4
            if np.any(face_counts[fis] == 3):
                continue
            sol[ei] = True
            face_counts[fis] += 1
        
        return sol, face_counts
    
    def local_search(sol, face_counts, rng, max_time=140):
        """Improve solution via local search."""
        sol = sol.copy()
        face_counts = face_counts.copy()
        
        # Track bad faces
        bad_faces = set(np.where(face_counts == 4)[0].tolist())
        
        while time_left() > 15:
            if bad_faces:
                # Repair: remove edges from bad faces
                # Find edge that appears in most bad faces
                bad_face_list = list(bad_faces)
                rng.shuffle(bad_face_list)
                
                # Count how many bad faces each edge in bad faces appears in
                edge_bad_count = {}
                for fi in bad_face_list:
                    for ei in faces[fi]:
                        if sol[ei]:
                            edge_bad_count[ei] = edge_bad_count.get(ei, 0) + 1
                
                if not edge_bad_count:
                    break
                    
                # Pick edge with max bad face count
                best_ei = max(edge_bad_count, key=lambda e: edge_bad_count[e])
                
                sol[best_ei] = False
                for fi in edge_faces_arr[best_ei]:
                    face_counts[fi] -= 1
                    if face_counts[fi] == 3 and fi in bad_faces:
                        bad_faces.discard(fi)
            else:
                # No bad faces: try to add edges
                # Find addable edges: edges where no face has count >= 3
                # Vectorized check
                addable = []
                for ei in range(n_edges):
                    if sol[ei]:
                        continue
                    fis = edge_faces_arr[ei]
                    if len(fis) == 0:
                        addable.append(ei)
                    elif not np.any(face_counts[fis] >= 3):
                        addable.append(ei)
                
                if addable:
                    rng.shuffle(addable)
                    added_any = False
                    for ei in addable:
                        if time_left() < 15:
                            break
                        fis = edge_faces_arr[ei]
                        if len(fis) == 0 or not np.any(face_counts[fis] >= 3):
                            sol[ei] = True
                            face_counts[fis] += 1
                            added_any = True
                    if not added_any:
                        break
                else:
                    # Stuck: perturbation
                    if time_left() > 30:
                        selected = np.where(sol)[0].tolist()
                        if len(selected) > 20:
                            k = min(5, len(selected) // 15)
                            to_remove = rng.sample(selected, k)
                            for ei in to_remove:
                                sol[ei] = False
                                for fi in edge_faces_arr[ei]:
                                    face_counts[fi] -= 1
                                    if fi in bad_faces:
                                        bad_faces.discard(fi)
                            continue
                    else:
                        break
                        
            # Rebuild bad_faces periodically
            if len(bad_faces) > 100:
                bad_faces = set(np.where(face_counts == 4)[0].tolist())
        
        return sol, face_counts
    
    def swap_local_search(sol, face_counts, rng):
        """Try 1-removal + 1-addition swaps to improve."""
        sol = sol.copy()
        face_counts = face_counts.copy()
        
        bad_faces = set(np.where(face_counts == 4)[0].tolist())
        
        while time_left() > 15:
            if bad_faces:
                # Remove edge from bad face
                bad_face_list = list(bad_faces)
                rng.shuffle(bad_face_list)
                
                edge_bad_count = {}
                for fi in bad_face_list:
                    for ei in faces[fi]:
                        if sol[ei]:
                            edge_bad_count[ei] = edge_bad_count.get(ei, 0) + 1
                
                if not edge_bad_count:
                    break
                best_ei = max(edge_bad_count, key=lambda e: edge_bad_count[e])
                
                sol[best_ei] = False
                for fi in edge_faces_arr[best_ei]:
                    face_counts[fi] -= 1
                    if fi in bad_faces:
                        bad_faces.discard(fi)
            else:
                # Try to add an edge
                added = False
                # Random sample of unselected edges
                unselected = np.where(~sol)[0].tolist()
                if not unselected:
                    break
                rng.shuffle(unselected)
                for ei in unselected[:50]:  # try up to 50
                    fis = edge_faces_arr[ei]
                    if len(fis) == 0 or not np.any(face_counts[fis] >= 3):
                        sol[ei] = True
                        face_counts[fis] += 1
                        added = True
                        break
                if not added:
                    # Perturbation
                    if time_left() > 30:
                        selected = np.where(sol)[0].tolist()
                        if len(selected) > 20:
                            k = min(3, len(selected) // 20)
                            to_remove = rng.sample(selected, k)
                            for ei in to_remove:
                                sol[ei] = False
                                for fi in edge_faces_arr[ei]:
                                    face_counts[fi] -= 1
                            continue
                    else:
                        break
        
        return sol, face_counts
    
    best_count = 0
    best_solution = None
    
    # Multiple restarts with different strategies
    attempts = 0
    while time_left() > 20:
        attempts += 1
        seed = attempts * 7919 + 42
        rng = random.Random(seed)
        
        mode = attempts % 4
        sol, face_counts = greedy_initial(rng, mode)
        
        # Local search
        sol, face_counts = local_search(sol, face_counts, rng)
        
        # Verify
        if validate(sol):
            cnt = count_edges(sol)
            if cnt > best_count:
                best_count = cnt
                best_solution = sol.copy()
        
        # If we have time, try swap local search on best solution
        if best_solution is not None and time_left() > 30:
            sol2 = best_solution.copy()
            fc2 = np.zeros(n_faces, dtype=np.int32)
            for ei in range(n_edges):
                if sol2[ei]:
                    fc2[edge_faces_arr[ei]] += 1
            sol2, fc2 = swap_local_search(sol2, fc2, rng)
            if validate(sol2):
                cnt2 = count_edges(sol2)
                if cnt2 > best_count:
                    best_count = cnt2
                    best_solution = sol2.copy()
        
        if time_left() < 25:
            break
    
    # Final fallback
    if best_solution is None or best_count < 250:
        # Simple deterministic greedy
        chosen = set()
        out = []
        for x in range(n_vertices):
            for i in range(d):
                y = x ^ (1 << i)
                if x < y:
                    ok = True
                    for j in range(d):
                        if j != i:
                            c = x ^ (1 << j)
                            e = y ^ (1 << j)
                            if all(p in chosen for p in ((min(x, c), max(x, c)), (min(y, e), max(y, e)), (min(c, e), max(c, e)))):
                                ok = False
                                break
                    if ok:
                        chosen.add((x, y))
                        out.append([x, y])
        return out
    
    # Convert to list of edges
    result = []
    for ei in range(n_edges):
        if best_solution[ei]:
            result.append(list(edges[ei]))
    
    return result
