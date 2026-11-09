import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Maximize C4-free edge set in Q7 using a different structural approach:
    1. Parity-based deterministic construction
    2. Larger-neighborhood local search (2-swap, 3-move)
    3. Iterated local search with restarts
    """
    start_time = time.time()
    deadline = start_time + 145
    
    d = 7
    n_vertices = 1 << d
    n_edges_total = d * (1 << (d - 1))
    
    edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    n_edges = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
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
    
    edge_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for ei in face:
            edge_faces[ei].append(fi)
    
    faces_np = np.array(faces, dtype=np.int32)
    
    def time_left():
        return deadline - time.time()
    
    def is_valid(sol):
        sol_arr = np.array(sol, dtype=np.bool_)
        for fi in range(n_faces):
            if np.all(sol_arr[faces_np[fi]]):
                return False
        return True
    
    def count_edges(sol):
        return sum(1 for e in range(n_edges) if sol[e])
    
    def parity_construction(seed=None):
        """
        Different construction: use bit-parity patterns.
        For each edge (u,v) in dimension i, include it based on 
        a hash-like function of u and v.
        """
        rng = random.Random(seed)
        sol = [False] * n_edges
        face_counts = [0] * n_faces
        
        # Try multiple parity patterns and keep the best
        best_local = None
        best_local_count = 0
        
        for pattern_seed in range(20):
            prng = random.Random(seed + pattern_seed * 1000)
            sol_local = [False] * n_edges
            fc = [0] * n_faces
            
            # Randomly assign "thresholds" to faces
            # An edge can be added if it doesn't complete any face
            # Use a weighted random order
            order = list(range(n_edges))
            weights = []
            for ei, (u, v) in enumerate(edges):
                # Weight based on vertex parity and random
                p = bin(u).count('1') + bin(v).count('1')
                w = prng.random() + 0.01 * p
                weights.append(w)
            
            order.sort(key=lambda e: weights[e], reverse=True)
            
            for ei in order:
                can_add = True
                for fi in edge_faces[ei]:
                    if fc[fi] == 3:
                        can_add = False
                        break
                if can_add:
                    sol_local[ei] = True
                    for fi in edge_faces[ei]:
                        fc[fi] += 1
            
            cnt = count_edges(sol_local)
            if cnt > best_local_count:
                best_local_count = cnt
                best_local = list(sol_local)
        
        return best_local
    
    def large_neighborhood_search(sol, seed=None, max_iter=20000):
        """
        Local search with larger moves:
        - 1-edge add/remove
        - 2-edge swap (remove one, add another)
        - 2-edge remove + 2-edge add (2-for-2 swap)
        """
        rng = random.Random(seed)
        sol = list(sol)
        face_counts = [0] * n_faces
        for ei in range(n_edges):
            if sol[ei]:
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        
        bad_faces = set()
        for fi in range(n_faces):
            if face_counts[fi] == 4:
                bad_faces.add(fi)
        
        improved = True
        iter_count = 0
        no_improve = 0
        
        while improved and time_left() > 15 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            if bad_faces:
                # Must fix bad faces: remove an edge from a bad face
                bad_list = list(bad_faces)
                rng.shuffle(bad_list)
                
                best_removal = None
                best_score = -1
                for fi in bad_list[:20]:
                    for ei in faces[fi]:
                        if sol[ei]:
                            cnt_bad = sum(1 for f2 in edge_faces[ei] if f2 in bad_faces)
                            score = cnt_bad + rng.random() * 0.05
                            if score > best_score:
                                best_score = score
                                best_removal = ei
                
                if best_removal is not None:
                    ei = best_removal
                    sol[ei] = False
                    for f2 in edge_faces[ei]:
                        face_counts[f2] -= 1
                        if face_counts[f2] == 3 and f2 in bad_faces:
                            bad_faces.discard(f2)
                    improved = True
                    no_improve = 0
            else:
                # No bad faces: try to add or swap
                # First try simple adds
                added = False
                addable = []
                for ei in range(n_edges):
                    if not sol[ei]:
                        can_add = True
                        for fi in edge_faces[ei]:
                            if face_counts[fi] >= 3:
                                can_add = False
                                break
                        if can_add:
                            addable.append(ei)
                
                if addable:
                    rng.shuffle(addable)
                    for ei in addable:
                        if time_left() < 15:
                            break
                        can_add = True
                        for fi in edge_faces[ei]:
                            if face_counts[fi] >= 3:
                                can_add = False
                                break
                        if can_add:
                            sol[ei] = True
                            for fi in edge_faces[ei]:
                                face_counts[fi] += 1
                            improved = True
                            added = True
                            no_improve = 0
                            break
                
                if not added:
                    # Try 2-for-2 swap: remove 2 edges, add 2 edges
                    # This can escape local optima where single swaps don't help
                    selected = [e for e in range(n_edges) if sol[e]]
                    unselected = [e for e in range(n_edges) if not sol[e]]
                    
                    if len(selected) >= 2 and len(unselected) >= 2:
                        # Try a few random 2-for-2 swaps
                        for _ in range(10):
                            if time_left() < 15:
                                break
                            rem1, rem2 = rng.sample(selected, 2)
                            add1, add2 = rng.sample(unselected, 2)
                            
                            # Check if removing rem1, rem2 and adding add1, add2 is valid
                            temp_fc = face_counts.copy()
                            for ei in [rem1, rem2]:
                                for f2 in edge_faces[ei]:
                                    temp_fc[f2] -= 1
                            
                            valid = True
                            for ei in [add1, add2]:
                                for f2 in edge_faces[ei]:
                                    if temp_fc[f2] >= 3:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            
                            if valid:
                                # Check that after adding, no face has count > 4
                                for ei in [add1, add2]:
                                    for f2 in edge_faces[ei]:
                                        temp_fc[f2] += 1
                                if all(temp_fc[f] <= 4 for f in range(n_faces)):
                                    # Apply
                                    for ei in [rem1, rem2]:
                                        sol[ei] = False
                                        for f2 in edge_faces[ei]:
                                            face_counts[f2] -= 1
                                    for ei in [add1, add2]:
                                        sol[ei] = True
                                        for f2 in edge_faces[ei]:
                                            face_counts[f2] += 1
                                    improved = True
                                    no_improve = 0
                                    break
                    
                    if not improved:
                        # Perturbation: remove k random edges
                        no_improve += 1
                        if no_improve > 3 and time_left() > 25:
                            k = rng.choice([3, 5, 7])
                            to_remove = rng.sample(selected, min(k, len(selected) // 10))
                            for ei in to_remove:
                                sol[ei] = False
                                for f2 in edge_faces[ei]:
                                    face_counts[f2] -= 1
                                    if f2 in bad_faces:
                                        bad_faces.discard(f2)
                            # Rebuild bad_faces
                            bad_faces = set()
                            for fi in range(n_faces):
                                if face_counts[fi] == 4:
                                    bad_faces.add(fi)
                            improved = True
                            no_improve = 0
                        elif no_improve > 10:
                            break
        
        return sol
    
    def iterated_local_search(sol, seed=None):
        """
        Iterated local search: apply local search, perturb, repeat.
        """
        rng = random.Random(seed)
        best_sol = list(sol)
        best_count = count_edges(sol)
        
        for iteration in range(50):
            if time_left() < 20:
                break
            
            # Perturb
            perturbed = list(best_sol)
            selected = [e for e in range(n_edges) if perturbed[e]]
            if selected:
                k = rng.choice([2, 3, 5, 7])
                to_remove = rng.sample(selected, min(k, len(selected) // 8))
                for ei in to_remove:
                    perturbed[ei] = False
            
            # Local search
            result = large_neighborhood_search(perturbed, seed + iteration * 100)
            
            cnt = count_edges(result)
            if is_valid(result) and cnt > best_count:
                best_count = cnt
                best_sol = list(result)
        
        return best_sol
    
    # Start with parity construction
    best_count = 0
    best_solution = None
    
    # Multiple parity constructions with different seeds
    for s in range(5):
        sol = parity_construction(s * 12345)
        cnt = count_edges(sol)
        if is_valid(sol) and cnt > best_count:
            best_count = cnt
            best_solution = list(sol)
    
    # If parity didn't give good results, try simple greedy
    if best_count < 280:
        sol = [False] * n_edges
        fc = [0] * n_faces
        rng = random.Random(42)
        order = list(range(n_edges))
        rng.shuffle(order)
        for ei in order:
            can_add = True
            for fi in edge_faces[ei]:
                if fc[fi] == 3:
                    can_add = False
                    break
            if can_add:
                sol[ei] = True
                for fi in edge_faces[ei]:
                    fc[fi] += 1
        cnt = count_edges(sol)
        if is_valid(sol) and cnt > best_count:
            best_count = cnt
            best_solution = list(sol)
    
    # Iterated local search
    if best_solution and time_left() > 30:
        best_solution = iterated_local_search(best_solution, 42)
        cnt = count_edges(best_solution)
        if is_valid(best_solution):
            best_count = cnt
    
    # Final polish
    if best_solution and time_left() > 20:
        sol = large_neighborhood_search(best_solution, 99999, max_iter=30000)
        cnt = count_edges(sol)
        if is_valid(sol) and cnt > best_count:
            best_count = cnt
            best_solution = list(sol)
    
    # Fallback
    if best_solution is None or best_count < 250:
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
    
    result = []
    for ei in range(n_edges):
        if best_solution[ei]:
            result.append(list(edges[ei]))
    
    return result
