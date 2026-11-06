import time
import random

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # 1. Build Edge List and Face Data
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
                
    # Identify faces
    edge_faces = [[] for _ in range(n_edges)]
    faces = []
    
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                face_id = len(faces)
                faces.append([e1, e2, e3, e4])
                
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
                    
    n_faces = len(faces)
    
    # 2. Initialize Removed Set using Greedy Hitting Set
    removed = set()
    face_counts = [0] * n_faces
    
    def add_to_removed(e):
        removed.add(e)
        for f in edge_faces[e]:
            face_counts[f] += 1
            
    def remove_from_removed(e):
        removed.remove(e)
        for f in edge_faces[e]:
            face_counts[f] -= 1
            
    uncovered_faces = set(range(n_faces))
    
    while uncovered_faces:
        # Pick a random uncovered face
        f = random.choice(list(uncovered_faces))
        
        best_e = -1
        max_cover = -1
        for e in faces[f]:
            if e in removed:
                continue
            cover = 0
            for ff in edge_faces[e]:
                if ff in uncovered_faces:
                    cover += 1
            if cover > max_cover:
                max_cover = cover
                best_e = e
        
        if best_e == -1:
            best_e = faces[f][0]
            
        add_to_removed(best_e)
        
        for ff in edge_faces[best_e]:
            if face_counts[ff] == 1:
                uncovered_faces.remove(ff)
                
    best_removed = set(removed)
    best_size = len(removed)
    
    # 3. Local Search
    start_time = time.time()
    timeout = 115
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Phase 1: Deletions
        # We need to iterate until no more deletions are possible in a single pass
        # or we can just do one pass and check if any were made.
        # Doing one pass per outer loop iteration is fine if we loop back.
        for e in list(removed):
            critical = False
            for f in edge_faces[e]:
                if face_counts[f] == 1:
                    critical = True
                    break
            if not critical:
                remove_from_removed(e)
                improved = True
                if len(removed) < best_size:
                    best_size = len(removed)
                    best_removed = set(removed)
        
        # Phase 2: Swaps
        if not improved:
            # Try swaps
            # To be efficient, we can limit the number of swap attempts
            # or iterate through all edges if n_edges is small (448 is small)
            # Let's iterate through all removed edges and try to swap them.
            # For each e_out, find valid e_in.
            # This is O(N * F * E) which is fine for N=448.
            
            swap_found = False
            # Shuffle to avoid bias?
            # Let's just iterate.
            for e_out in list(removed):
                critical_faces = []
                for f in edge_faces[e_out]:
                    if face_counts[f] == 1:
                        critical_faces.append(f)
                
                if not critical_faces:
                    # Should have been deleted in Phase 1
                    continue
                
                # Find e_in in intersection of critical_faces, not in removed
                # Intersection of faces
                # Since each face has 4 edges, the intersection of critical faces
                # is likely small.
                
                # Start with edges of first critical face
                candidates = set(faces[critical_faces[0]])
                for f in critical_faces[1:]:
                    candidates &= set(faces[f])
                
                candidates.discard(e_out)
                
                # Filter candidates not in removed
                valid_candidates = [e for e in candidates if e not in removed]
                
                if valid_candidates:
                    # Pick one randomly or deterministically
                    e_in = random.choice(valid_candidates)
                    
                    # Perform swap
                    remove_from_removed(e_out)
                    add_to_removed(e_in)
                    improved = True
                    swap_found = True
                    # We could break here to restart the outer loop
                    # or continue to find more swaps in this pass
                    # Continuing might allow multiple swaps in one pass
                    # which is good.
                    # But we must be careful: e_in might become critical later?
                    # No, we just added it. It covers some faces.
                    # The faces it covers might now have count 1?
                    # If e_in covers a face f, and f was covered only by e_out,
                    # then count becomes 1.
                    # But e_in is in candidates, so it covers all critical faces of e_out.
                    # So those faces are still covered.
                    # Other faces covered by e_in were covered by other edges (since e_in was not in removed).
                    # So they had count >= 1. Now count >= 2.
                    # So no new critical faces are created by adding e_in?
                    # Wait.
                    # If e_in covers face f, and f was covered by k edges in R.
                    # k >= 1 because R is a hitting set.
                    # After swap, f is covered by k+1 edges?
                    # No, e_out might have covered f too.
                    # If e_out covered f, then f is in critical_faces?
                    # If f is in critical_faces, then count_R(f) == 1.
                    # Since e_out is in R and covers f, e_out is the only edge in R covering f.
                    # So e_in must cover f to keep it covered.
                    # So for all f in critical_faces, count_{R'}(f) >= 1.
                    # For other faces g covered by e_in:
                    # count_R(g) >= 1.
                    # If e_out also covered g, then count_{R'}(g) = count_R(g) (e_out removed, e_in added).
                    # If e_out did not cover g, then count_{R'}(g) = count_R(g) + 1.
                    # So no face becomes uncovered.
                    # And no face becomes critical (count 1) unless it was already critical?
                    # If count_R(g) == 1 and e_out did not cover g, then count_{R'}(g) = 2.
                    # If count_R(g) == 1 and e_out covered g, then g is in critical_faces, so e_in covers g, so count_{R'}(g) = 1.
                    # So the set of critical faces might change, but validity is maintained.
                    
                    # We continue the loop to find more swaps in this pass.
                    # But we must update the list of removed edges?
                    # We are iterating over a snapshot `list(removed)`.
                    # e_in is now in removed, but it wasn't in the snapshot.
                    # e_out is no longer in removed, but it is in the snapshot.
                    # We should not try to swap e_out again (it's not in removed).
                    # We should not try to swap e_in in this pass (it wasn't in the snapshot).
                    # This is safe.
                    
            if not swap_found and not improved:
                # No deletions, no swaps.
                # Perturbation: Add a random edge to removed
                # This increases size, but might allow better structure.
                # Then run local search again.
                # To avoid infinite loops, we should only do this a few times.
                # Or just break.
                # Let's try a few perturbations.
                # But we need to track best.
                
                # If we are stuck, we might try to restart the whole process?
                # No, that's expensive.
                # Let's try a "kick": add a random edge, then try to delete.
                # But if we add an edge, size increases.
                # We only care about best_size.
                # So we can afford to explore larger sets.
                
                # Let's do 10 perturbations max per stuck state?
                # Or just break if we've been stuck for a while.
                # Given the time, breaking is safer.
                break

    # Final Polish
    # We have best_removed.
    # Let's do one more pass of deletions on best_removed to ensure it's minimal.
    # Reconstruct face counts for best_removed.
    face_counts_final = [0] * n_faces
    for e in best_removed:
        for f in edge_faces[e]:
            face_counts_final[f] += 1
            
    changed = True
    while changed:
        changed = False
        for e in list(best_removed):
            critical = False
            for f in edge_faces[e]:
                if face_counts_final[f] == 1:
                    critical = True
                    break
            if not critical:
                best_removed.remove(e)
                for f in edge_faces[e]:
                    face_counts_final[f] -= 1
                changed = True
                
    result = []
    for i in range(n_edges):
        if i not in best_removed:
            result.append(list(edges[i]))
            
    return result
