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
    # edge_id -> list of face_ids
    # face_id -> list of edge_ids
    edge_faces = [[] for _ in range(n_edges)]
    faces = []
    
    # Enumerate all faces
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
    
    # Helper to add edge to removed set
    def add_to_removed(e):
        removed.add(e)
        for f in edge_faces[e]:
            face_counts[f] += 1
            
    # Helper to remove edge from removed set
    def remove_from_removed(e):
        removed.remove(e)
        for f in edge_faces[e]:
            face_counts[f] -= 1
            
    # Greedy Initialization
    # We want to cover all faces.
    # While there is an uncovered face, pick an edge to cover.
    uncovered_faces = set(range(n_faces))
    
    while uncovered_faces:
        # Pick a random uncovered face to avoid bias
        f = random.choice(list(uncovered_faces))
        
        # Find edge in this face that covers most uncovered faces
        best_e = -1
        max_cover = -1
        for e in faces[f]:
            if e in removed:
                continue
            # Count how many uncovered faces this edge covers
            cover = 0
            for ff in edge_faces[e]:
                if ff in uncovered_faces:
                    cover += 1
            if cover > max_cover:
                max_cover = cover
                best_e = e
        
        if best_e == -1:
            # Should not happen if face is uncovered
            # Pick any edge in face
            best_e = faces[f][0]
            
        add_to_removed(best_e)
        
        # Update uncovered faces
        for ff in edge_faces[best_e]:
            if face_counts[ff] == 1: # Just became covered
                uncovered_faces.remove(ff)
                
    current_removed_size = len(removed)
    best_removed_size = current_removed_size
    best_removed = set(removed)
    
    # 3. Local Search to Minimize Removed Set
    start_time = time.time()
    timeout = 115
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Phase 1: Try to delete edges (reduce size)
        # We iterate through a copy of removed to allow modification
        for e in list(removed):
            # Check if e is critical (only removed edge in any face)
            critical = False
            for f in edge_faces[e]:
                if face_counts[f] == 1:
                    critical = True
                    break
            
            if not critical:
                remove_from_removed(e)
                current_removed_size -= 1
                improved = True
                if current_removed_size < best_removed_size:
                    best_removed_size = current_removed_size
                    best_removed = set(removed)
        
        # Phase 2: Try swaps (keep size, change composition)
        # Only if no deletions were possible in the last pass? 
        # Or always try swaps to escape local minima.
        if not improved:
            # Try a few random swaps
            # We need to find e_out in removed, e_in not in removed
            # such that swapping maintains validity.
            
            # To make it efficient, we can pick random e_out
            # and check if it has critical faces.
            # If it does, e_in must be in the intersection of those critical faces.
            
            # Let's try up to 1000 swaps per iteration
            swap_attempts = 0
            while swap_attempts < 1000:
                swap_attempts += 1
                if not removed:
                    break
                
                e_out = random.choice(list(removed))
                
                # Find critical faces for e_out
                critical_faces = []
                for f in edge_faces[e_out]:
                    if face_counts[f] == 1:
                        critical_faces.append(f)
                        
                if not critical_faces:
                    # This is a deletion candidate, but Phase 1 should have caught it.
                    # If we are here, maybe it became non-critical after previous swaps in this loop?
                    # But we are in a fresh loop iteration.
                    # If Phase 1 found nothing, all edges are critical.
                    # So this case shouldn't happen unless Phase 1 logic was flawed.
                    # Actually, if Phase 1 runs and finds nothing, all edges in 'removed' are critical.
                    # So critical_faces should not be empty.
                    continue
                
                # We need e_in that is in ALL critical_faces
                # And e_in is NOT in removed
                # And e_in != e_out (e_out is in removed, so e_in != e_out is satisfied if e_in not in removed)
                
                # Intersection of edges in critical_faces
                # Start with edges of the first critical face
                candidate_edges = set(faces[critical_faces[0]])
                for f in critical_faces[1:]:
                    candidate_edges &= set(faces[f])
                
                # Remove e_out from candidates (it's in removed, but we want e_in not in removed)
                candidate_edges.discard(e_out)
                
                # Filter out edges already in removed
                valid_candidates = [e for e in candidate_edges if e not in removed]
                
                if valid_candidates:
                    e_in = random.choice(valid_candidates)
                    
                    # Perform swap
                    remove_from_removed(e_out)
                    add_to_removed(e_in)
                    improved = True
                    # Note: Size doesn't change, so best_removed_size doesn't update here directly
                    # But it might enable deletions in next Phase 1
                    
        if not improved:
            # No deletions, no swaps found in this iteration
            # To avoid getting stuck in a specific local minimum, we can do a random restart 
            # or perturb the current state.
            # Given the time budget, we can try a few random restarts.
            # But first, let's see if we can do a "2-swap" or similar.
            # For now, if stuck, we break or restart.
            
            # Let's try a simple restart with a different random seed for initialization?
            # Or just break and return best.
            # The problem is small enough that greedy+local search usually finds a very good solution.
            # Let's add a small random perturbation: remove a random edge from removed (invalidating) 
            # and repair? No, that's complex.
            
            # Let's try to break out by doing a random swap even if it doesn't look promising?
            # No, swaps must be valid.
            
            # If we are stuck, we can try to restart the whole process with a different random seed.
            # But we have limited time.
            
            # Let's check if we've been stuck for a while.
            # If so, break.
            break

    # 4. Final Polish: Greedy Add (on Selected set, which is equivalent to Greedy Delete on Removed)
    # We already did greedy delete in Phase 1.
    # Let's do one more pass of Phase 1 on the best solution found.
    
    selected = set(range(n_edges)) - best_removed
    
    # Verify validity of best_removed
    # face_counts should be >= 1 for all faces if best_removed is valid.
    # We reconstructed best_removed from a valid state, so it should be valid.
    
    # Just to be safe, let's run a final cleanup on best_removed
    # Re-initialize face counts for best_removed
    face_counts_final = [0] * n_faces
    for e in best_removed:
        for f in edge_faces[e]:
            face_counts_final[f] += 1
            
    # Check for any deletable edges
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
                
    # Reconstruct result
    result = []
    for i in range(n_edges):
        if i not in best_removed:
            result.append(list(edges[i]))
            
    return result
