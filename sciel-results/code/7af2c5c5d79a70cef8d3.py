import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Maximize C4-free edge set in Q7 using complementary deletion initialization
    and iterated local search with 1-flip and 2-swap moves.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
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
    edge_faces = [[] for _ in range(n_edges)]
    for fi, face in enumerate(faces):
        for ei in face:
            edge_faces[ei].append(fi)
    
    # Convert to numpy for vectorized face count updates if needed, 
    # but for local search, lists might be faster for individual updates.
    # We will use lists for face_counts and sol for speed in the inner loop.
    
    def time_left():
        return deadline - time.time()
    
    def is_valid(sol):
        """Check if solution has no face with all 4 edges."""
        # Using numpy for final validation check
        sol_arr = np.array(sol, dtype=np.bool_)
        faces_np = np.array(faces, dtype=np.int32)
        face_all = np.all(sol_arr[faces_np], axis=1)
        return not np.any(face_all)
    
    def count_edges(sol):
        return sum(sol)
        
    def init_complementary_deletion(seed=None):
        """
        Start with all edges, remove edges to break all C4s.
        Greedy strategy: remove edge in most bad faces.
        """
        rng = random.Random(seed)
        sol = [True] * n_edges
        face_counts = [4] * n_faces
        bad_faces = set(range(n_faces))
        
        # Precompute face incidence for fast updates
        # edge_faces is already built
        
        while bad_faces and time_left() > 30:
            # Pick a random bad face to break ties or just iterate
            # To be efficient, pick the bad face that has the most "flexibility" or just arbitrary
            # Better: Pick a bad face, find the edge in it that is in the most other bad faces
            # To speed up, we can just pick a random bad face and then pick the best edge in it
            
            if not bad_faces:
                break
                
            # Pick a random bad face
            f = rng.choice(list(bad_faces))
            
            best_edge = -1
            max_bad_count = -1
            
            for ei in faces[f]:
                if not sol[ei]:
                    continue
                # Count how many bad faces this edge is in
                cnt = 0
                for f2 in edge_faces[ei]:
                    if f2 in bad_faces:
                        cnt += 1
                if cnt > max_bad_count:
                    max_bad_count = cnt
                    best_edge = ei
                    
            if best_edge == -1:
                # Should not happen if f is bad
                break
                
            # Remove best_edge
            sol[best_edge] = False
            for f2 in edge_faces[best_edge]:
                face_counts[f2] -= 1
                if face_counts[f2] < 4:
                    bad_faces.discard(f2)
                    
        return sol, face_counts, bad_faces

    def local_search(sol, face_counts, bad_faces, seed=None, max_iter=20000):
        """
        Improve solution via 1-flip (add) and 2-swap (remove+add) moves.
        """
        rng = random.Random(seed)
        sol = list(sol)
        face_counts = list(face_counts)
        bad_faces = set(bad_faces)
        
        improved = True
        iter_count = 0
        
        while improved and time_left() > 15:
            improved = False
            iter_count += 1
            
            # Phase 1: Try to add edges (1-flip)
            # We can do this in a single pass if we just check all unselected edges
            # But we need to update face_counts.
            
            # To make it faster, we can just iterate through all edges not in sol
            # and try to add them.
            
            # Optimization: Only try to add if it doesn't create a bad face.
            # An edge e can be added if for all f in edge_faces[e], face_counts[f] < 3.
            
            added_any = False
            # Shuffle order to avoid bias
            unselected = [i for i in range(n_edges) if not sol[i]]
            rng.shuffle(unselected)
            
            for ei in unselected:
                if time_left() < 15:
                    break
                # Check if ei can be added
                can_add = True
                for f in edge_faces[ei]:
                    if face_counts[f] >= 3:
                        can_add = False
                        break
                
                if can_add:
                    sol[ei] = True
                    for f in edge_faces[ei]:
                        face_counts[f] += 1
                        if face_counts[f] == 4:
                            bad_faces.add(f)
                    added_any = True
                    # Note: Adding an edge might create a bad face. 
                    # We allow this temporarily if we plan to fix it, 
                    # but standard local search usually maintains validity.
                    # The problem requires the FINAL solution to be valid.
                    # If we add an edge that creates a C4, the solution is invalid.
                    # So we should ONLY add if it doesn't create a C4.
                    # My check `face_counts[f] >= 3` ensures that if we add, 
                    # the count becomes 4, which is BAD.
                    # So the condition should be `face_counts[f] < 3`?
                    # No, if face_counts[f] is 3, adding makes it 4 (Bad).
                    # If face_counts[f] is 2, adding makes it 3 (Good).
                    # So condition is: max(face_counts[f] for f in edge_faces[ei]) < 3.
                    
                    # Wait, my logic above:
                    # `if face_counts[f] >= 3: can_add = False`
                    # This means if any face has 3 edges, we cannot add.
                    # This is correct for maintaining validity.
                    
                    # However, I added it and then added to bad_faces?
                    # No, if can_add is True, no face had count >= 3.
                    # So max count was 2. Adding makes it 3.
                    # So no bad face is created.
                    # So `bad_faces` should not change.
                    
                    # Let's re-verify.
                    # If I add an edge, and it doesn't create a C4, then no face count becomes 4.
                    # So `bad_faces` remains unchanged (or empty).
                    
                    # So the logic holds.
                    
                    # But wait, if I add an edge, I should check if it *removes* a bad face?
                    # No, adding an edge only increases counts. It cannot fix a bad face.
                    # So `bad_faces` is monotonically non-decreasing in size during addition phase?
                    # No, if I start with a valid solution, `bad_faces` is empty.
                    # Adding edges keeps it empty if I'm careful.
                    
                    # So in a valid solution, `bad_faces` is always empty.
                    # The local search should maintain validity.
                    
                    # So, if `bad_faces` is not empty, we are in an invalid state.
                    # The `init_complementary_deletion` ends with a valid state (bad_faces empty).
                    # So `local_search` starts with `bad_faces` empty.
                    # And maintains it empty.
                    
                    # So the `bad_faces` set is actually redundant if we maintain validity.
                    # We can just rely on `face_counts`.
                    
                    improved = True
                    break # Restart the unselected loop? Or just continue?
                    # If we add an edge, it might block other additions.
                    # It's better to restart the scan or just continue.
                    # Continuing is fine, as we shuffle.
                    
            if added_any:
                continue # Restart the main while loop to try adding more
                
            # Phase 2: If no additions possible, try swaps (2-flip)
            # Remove one edge, add another.
            # This is more complex.
            # Strategy: 
            # 1. Pick a random selected edge e_out.
            # 2. Remove it. (Update counts)
            # 3. Try to add random unselected edges until stuck.
            # 4. If we added more than 1 edge (net gain > 0), keep the change.
            # 5. Else, revert.
            
            # Simpler Swap:
            # Try to find a pair (e_out, e_in) such that removing e_out and adding e_in is valid.
            # This doesn't increase edge count.
            # We need a "kick" move: Remove e_out, then add MULTIPLE edges.
            
            if time_left() > 30:
                # Perturbation / Kick
                selected = [i for i in range(n_edges) if sol[i]]
                if selected:
                    # Remove 1-3 random edges
                    n_remove = rng.randint(1, 3)
                    to_remove = rng.sample(selected, min(n_remove, len(selected)))
                    
                    # Save state
                    old_sol = list(sol)
                    old_fc = list(face_counts)
                    
                    for ei in to_remove:
                        sol[ei] = False
                        for f in edge_faces[ei]:
                            face_counts[f] -= 1
                    
                    # Now try to add as many edges as possible
                    added_count = 0
                    unselected = [i for i in range(n_edges) if not sol[i]]
                    rng.shuffle(unselected)
                    
                    for ei in unselected:
                        can_add = True
                        for f in edge_faces[ei]:
                            if face_counts[f] >= 3:
                                can_add = False
                                break
                        if can_add:
                            sol[ei] = True
                            for f in edge_faces[ei]:
                                face_counts[f] += 1
                            added_count += 1
                            
                    # If we added more edges than we removed, it's an improvement
                    if added_count > len(to_remove):
                        improved = True
                        # bad_faces is still empty (we maintained validity)
                    else:
                        # Revert
                        sol = old_sol
                        face_counts = old_fc
                        # Try again or give up
                        # To avoid infinite loop, we can break after some failures
                        # But `improved` is False, so the while loop will exit if we don't set it True.
                        # We should probably try a few kicks.
                        # For simplicity, we just let the while loop exit if no improvement.
                        # But we want to explore more.
                        # Let's set a counter for consecutive failures.
                        pass
                else:
                    break
                    
            if not improved:
                # Stuck. Try a more aggressive kick?
                # Or just break to restart the whole process.
                break
                
        return sol, face_counts, bad_faces

    best_count = 0
    best_solution = None
    
    # Initial run
    sol, fc, bf = init_complementary_deletion(42)
    cnt = count_edges(sol)
    if is_valid(sol):
        best_count = cnt
        best_solution = list(sol)
    
    attempts = 0
    while time_left() > 20:
        attempts += 1
        seed = attempts * 1000 + 7
        
        # Perturb the best solution or start fresh
        if attempts % 3 == 0 and best_solution:
            sol = list(best_solution)
            # Perturb: remove 5-10 random edges
            rng = random.Random(seed)
            selected = [i for i in range(n_edges) if sol[i]]
            if selected:
                to_remove = rng.sample(selected, min(rng.randint(5, 10), len(selected)))
                for ei in to_remove:
                    sol[ei] = False
            # Recompute face counts
            fc = [0] * n_faces
            for ei in range(n_edges):
                if sol[ei]:
                    for f in edge_faces[ei]:
                        fc[f] += 1
            bf = set(i for i in range(n_faces) if fc[i] == 4)
        else:
            sol, fc, bf = init_complementary_deletion(seed)
            
        sol, fc, bf = local_search(sol, fc, bf, seed)
        
        cnt = count_edges(sol)
        if is_valid(sol):
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
                
        if time_left() < 30:
            break

    # Final Fallback
    if best_solution is None:
        # Deterministic greedy
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
