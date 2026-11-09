import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Combined approach for maximizing C4-free edge set in Q7.
    Uses greedy construction, local search with swap moves, perturbation,
    and a final polish pass. Multiple restarts with different seeds.
    """
    start_time = time.time()
    deadline = start_time + 142  # Safety margin before 150s kill
    
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
    
    # Build face list
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
    
    # Convert to numpy for speed
    edge_faces_np = [np.array(edge_faces[ei], dtype=np.int32) for ei in range(n_edges)]
    faces_np = np.array(faces, dtype=np.int32)
    
    best_count = 0
    best_solution = None
    
    def time_left():
        return deadline - time.time()
    
    def is_valid(sol):
        """Check if solution has no face with all 4 edges."""
        sol_arr = np.array(sol, dtype=np.bool_)
        face_all = np.all(sol_arr[faces_np], axis=1)
        return not np.any(face_all)
    
    def count_edges(sol):
        return int(np.sum(np.array(sol, dtype=np.int8)))
    
    def deterministic_greedy():
        """The original seed's deterministic greedy approach."""
        chosen = set()
        sol = [False] * n_edges
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
                        ei = edge_to_idx[(x, y)]
                        sol[ei] = True
        return sol
    
    def greedy_initial(seed=None):
        """Generate a greedy initial solution with random edge order."""
        rng = random.Random(seed)
        sol = [False] * n_edges
        face_counts = np.zeros(n_faces, dtype=np.int8)
        
        order = list(range(n_edges))
        rng.shuffle(order)
        
        for ei in order:
            ef = edge_faces_np[ei]
            if np.any(face_counts[ef] >= 3):
                continue
            sol[ei] = True
            face_counts[ef] += 1
        
        return sol
    
    def compute_face_counts(sol):
        """Compute face counts from a solution using numpy."""
        sol_arr = np.array(sol, dtype=np.int8)
        fc = np.sum(sol_arr[faces_np], axis=1)
        return fc
    
    def local_search(sol, seed=None, max_iter=15000):
        """Improve solution via local search with swap moves and perturbation."""
        rng = random.Random(seed)
        sol = list(sol)
        face_counts = compute_face_counts(sol)
        
        # Find bad faces (count == 4)
        bad_faces_set = set(np.where(face_counts == 4)[0].tolist())
        
        improved = True
        iter_count = 0
        consecutive_no_improve = 0
        
        while improved and time_left() > 15 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            if bad_faces_set:
                # Remove an edge from a bad face - choose edge in most bad faces
                bad_face_list = list(bad_faces_set)
                rng.shuffle(bad_face_list)
                
                best_removal = None
                best_score = -1
                for fi in bad_face_list:
                    for ei in faces[fi]:
                        if sol[ei]:
                            cnt = 0
                            for f2 in edge_faces[ei]:
                                if f2 in bad_faces_set:
                                    cnt += 1
                            if cnt > best_score:
                                best_score = cnt
                                best_removal = ei
                
                if best_removal is not None:
                    ei = best_removal
                    sol[ei] = False
                    ef = edge_faces_np[ei]
                    face_counts[ef] -= 1
                    for f2 in ef:
                        if face_counts[f2] < 4:
                            bad_faces_set.discard(int(f2))
                    improved = True
                    consecutive_no_improve = 0
            else:
                # No bad faces, try to add edges
                addable = []
                for ei in range(n_edges):
                    if sol[ei]:
                        continue
                    ef = edge_faces_np[ei]
                    if np.any(face_counts[ef] >= 3):
                        continue
                    addable.append(ei)
                
                if addable:
                    rng.shuffle(addable)
                    for ei in addable:
                        if time_left() < 15:
                            break
                        ef = edge_faces_np[ei]
                        if np.any(face_counts[ef] >= 3):
                            continue
                        sol[ei] = True
                        face_counts[ef] += 1
                        improved = True
                        consecutive_no_improve = 0
                else:
                    # Stuck - try swap moves first, then perturbation
                    if time_left() > 30:
                        selected = [e for e in range(n_edges) if sol[e]]
                        unselected = [e for e in range(n_edges) if not sol[e]]
                        
                        if selected and unselected:
                            # Try systematic swaps: for each unselected edge, find a removable edge
                            # that allows adding it
                            swap_found = False
                            # Build list of edges that could be added if we remove one specific edge
                            for ei_add in unselected:
                                if swap_found:
                                    break
                                if time_left() < 15:
                                    break
                                ef_add = edge_faces_np[ei_add]
                                # Check which faces would go to 3 if we add this edge
                                blocking_faces = set()
                                for f2 in ef_add:
                                    if face_counts[f2] == 2:
                                        blocking_faces.add(int(f2))
                                    elif face_counts[f2] >= 3:
                                        # Can't add this edge at all
                                        break
                                else:
                                    if not blocking_faces:
                                        continue
                                    # For each blocking face, try removing one of its selected edges
                                    for fi in list(blocking_faces):
                                        if swap_found:
                                            break
                                        for ei_remove in faces[fi]:
                                            if not sol[ei_remove]:
                                                continue
                                            # Check if removing ei_remove allows adding ei_add
                                            ef_remove = edge_faces_np[ei_remove]
                                            # We need to check if (face_counts - ef_remove + ef_add) <= 3 for all faces
                                            # And (face_counts - ef_remove) >= 0 for all faces (trivially true)
                                            can_swap = True
                                            # Check faces in ef_add
                                            for f3 in ef_add:
                                                new_c = face_counts[f3] - (1 if f3 in ef_remove else 0) + 1
                                                if new_c > 3:
                                                    can_swap = False
                                                    break
                                            if not can_swap:
                                                continue
                                            # Check faces in ef_remove that are not in ef_add
                                            # new_c = face_counts[f3] - 1
                                            # This is always <= 2 if old was <= 3.
                                            # So no new bad faces (count 4) can be created by removal.
                                            # And count 4 is the only "bad" state.
                                            # So removal is always safe.
                                            # Perform the swap
                                            sol[ei_remove] = False
                                            sol[ei_add] = True
                                            face_counts[ef_remove] -= 1
                                            face_counts[ef_add] += 1
                                            for f2 in ef_remove:
                                                if face_counts[f2] < 4:
                                                    bad_faces_set.discard(int(f2))
                                            for f2 in ef_add:
                                                if face_counts[f2] == 4:
                                                    bad_faces_set.add(int(f2))
                                            if bad_faces_set:
                                                # Revert
                                                sol[ei_add] = False
                                                sol[ei_remove] = True
                                                face_counts[ef_add] -= 1
                                                face_counts[ef_remove] += 1
                                                for f2 in ef_add:
                                                    if face_counts[f2] < 4:
                                                        bad_faces_set.discard(int(f2))
                                                for f2 in ef_remove:
                                                    if face_counts[f2] == 4:
                                                        bad_faces_set.add(int(f2))
                                                continue
                                            improved = True
                                            consecutive_no_improve = 0
                                            swap_found = True
                                            break
                            
                            if not improved:
                                # Perturbation: remove 3-8 random edges
                                if len(selected) > 15:
                                    n_remove = rng.randint(3, 8)
                                    to_remove = rng.sample(selected, min(n_remove, len(selected) // 6))
                                    for ei in to_remove:
                                        sol[ei] = False
                                        ef = edge_faces_np[ei]
                                        face_counts[ef] -= 1
                                        for f2 in ef:
                                            if face_counts[f2] < 4:
                                                bad_faces_set.discard(int(f2))
                                    improved = True
                                    consecutive_no_improve = 0
                                else:
                                    consecutive_no_improve += 1
                                    if consecutive_no_improve > 8:
                                        break
                        else:
                            consecutive_no_improve += 1
                            if consecutive_no_improve > 8:
                                break
    
        return sol
    
    def polish(sol, seed=None):
        """Final high-precision polish: iterative swap optimization."""
        rng = random.Random(seed)
        sol = np.array(sol, dtype=np.bool_)
        face_counts = compute_face_counts(sol)
        
        # First, add any possible edges
        changed = True
        while changed and time_left() > 20:
            changed = False
            for ei in range(n_edges):
                if sol[ei]:
                    continue
                ef = edge_faces_np[ei]
                if np.any(face_counts[ef] >= 3):
                    continue
                sol[ei] = True
                face_counts[ef] += 1
                changed = True
        
        # Multi-pass 1-for-1 swaps
        if time_left() > 25:
            # Precompute sets of faces for each edge for faster intersection checks
            # We use a list of sets for faces containing each edge
            edge_faces_sets = [set(ef.tolist()) for ef in edge_faces_np]
            
            swap_improved = True
            while swap_improved and time_left() > 25:
                swap_improved = False
                # Iterate over all unselected edges
                for ei_add in range(n_edges):
                    if sol[ei_add]:
                        continue
                    ef_add = edge_faces_np[ei_add]
                    ef_add_set = edge_faces_sets[ei_add]
                    
                    # Quick check: if any face in ef_add has count 3, we can't add without removal
                    # But we are swapping, so we need to check if removal helps
                    # If any face has count 3, the removal MUST be in that face.
                    # If all faces have count < 3, we can add directly (handled above), 
                    # but maybe we want to swap to free up space for future additions?
                    # No, if we can add directly, we did it in the first phase.
                    # So here, at least one face in ef_add has count 3.
                    
                    candidate_removals = []
                    for f in ef_add:
                        if face_counts[f] >= 3:
                            # Must remove one of the edges in this face
                            for ei_remove in faces[f]:
                                if sol[ei_remove]:
                                    candidate_removals.append(ei_remove)
                    
                    if not candidate_removals:
                        continue
                    
                    # Try each candidate removal
                    for ei_remove in candidate_removals:
                        if ei_remove == ei_add:
                            continue
                        ef_remove = edge_faces_np[ei_remove]
                        ef_remove_set = edge_faces_sets[ei_remove]
                        
                        # Check validity of swap
                        valid = True
                        # For each face in ef_add:
                        # new_count = old_count - (1 if face in ef_remove) + 1
                        # Must be <= 3
                        for f in ef_add:
                            old_c = face_counts[f]
                            new_c = old_c - (1 if f in ef_remove_set else 0) + 1
                            if new_c > 3:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # For each face in ef_remove (not in ef_add):
                        # new_count = old_count - 1
                        # Must be >= 0 (always true) and <= 3 (always true if old<=3)
                        # But we must ensure we don't create a situation where 
                        # a face drops to 0? No, that's fine.
                        # Wait, the constraint is ONLY max 3. Min is 0.
                        # So removals are always safe regarding the upper bound.
                        # However, we must ensure that we don't "unblock" something incorrectly?
                        # No, the logic holds.
                        
                        # Execute swap
                        sol[ei_remove] = False
                        sol[ei_add] = True
                        face_counts[ef_remove] -= 1
                        face_counts[ef_add] += 1
                        swap_improved = True
                        break # Restart scan? Or continue? 
                        # To be safe and simple, break and restart the outer loop
                        # to re-evaluate face counts.
                    if swap_improved:
                        break
            
        return sol.tolist()
    
    # First, get the deterministic greedy baseline
    sol = deterministic_greedy()
    cnt = count_edges(sol)
    if is_valid(sol):
        best_count = cnt
        best_solution = list(sol)
    
    # Multiple restarts with different seeds
    attempts = 0
    while time_left() > 25:
        attempts += 1
        seed = attempts * 7919 + 42
        
        if attempts % 6 == 1:
            sol = greedy_initial(seed)
        elif attempts % 6 == 2:
            sol = greedy_initial(seed + 100000)
        elif attempts % 6 == 3:
            if best_solution:
                sol = list(best_solution)
                rng = random.Random(seed)
                selected = [e for e in range(n_edges) if sol[e]]
                if selected:
                    to_remove = rng.sample(selected, min(5, len(selected) // 7))
                    for ei in to_remove:
                        sol[ei] = False
            else:
                sol = greedy_initial(seed)
        elif attempts % 6 == 4:
            if best_solution:
                sol = list(best_solution)
                rng = random.Random(seed + 777)
                selected = [e for e in range(n_edges) if sol[e]]
                if selected:
                    to_remove = rng.sample(selected, min(8, len(selected) // 5))
                    for ei in to_remove:
                        sol[ei] = False
            else:
                sol = greedy_initial(seed)
        elif attempts % 6 == 5:
            sol = greedy_initial(seed + 200000)
            rng = random.Random(seed + 300000)
            selected = [e for e in range(n_edges) if sol[e]]
            if selected:
                to_remove = rng.sample(selected, min(4, len(selected) // 9))
                for ei in to_remove:
                    sol[ei] = False
        else:
            # Hybrid: start from greedy, then perturb more aggressively
            sol = greedy_initial(seed + 400000)
            rng = random.Random(seed + 500000)
            selected = [e for e in range(n_edges) if sol[e]]
            if selected:
                to_remove = rng.sample(selected, min(6, len(selected) // 7))
                for ei in to_remove:
                    sol[ei] = False
        
        sol = local_search(sol, seed)
        sol = polish(sol, seed + 55555)
        
        cnt = count_edges(sol)
        if is_valid(sol):
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
        
        if time_left() < 40:
            break
    
    # Final fallback
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
