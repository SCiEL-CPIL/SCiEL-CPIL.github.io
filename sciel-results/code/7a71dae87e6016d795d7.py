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
    
    # 3. Local Search with improved moves
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, 40s safety margin
    
    def try_deletions():
        """Try to remove any non-critical edges. Returns True if any were removed."""
        nonlocal removed, face_counts, best_size, best_removed
        any_removed = False
        for e in list(removed):
            critical = False
            for f in edge_faces[e]:
                if face_counts[f] == 1:
                    critical = True
                    break
            if not critical:
                remove_from_removed(e)
                any_removed = True
                if len(removed) < best_size:
                    best_size = len(removed)
                    best_removed = set(removed)
        return any_removed
    
    def try_swaps():
        """Try 1-opt swaps: replace one removed edge with another. Returns True if any swaps made."""
        nonlocal removed, face_counts, best_size, best_removed
        any_swap = False
        for e_out in list(removed):
            critical_faces = []
            for f in edge_faces[e_out]:
                if face_counts[f] == 1:
                    critical_faces.append(f)
            
            if not critical_faces:
                continue
            
            # Find edges that cover all critical faces
            candidates = set(faces[critical_faces[0]])
            for f in critical_faces[1:]:
                candidates &= set(faces[f])
            
            candidates.discard(e_out)
            
            # Filter to edges not in removed
            valid_candidates = [e for e in candidates if e not in removed]
            
            if valid_candidates:
                # Try each candidate and pick the best one
                best_e_in = -1
                best_gain = -1
                
                for e_in in valid_candidates:
                    # Calculate how many faces e_in covers that are currently at count 1
                    # (these would become count 2, which is fine)
                    # and how many faces it covers that are at count >= 2
                    # The key insight: we want e_in to cover as many critical faces as possible
                    # but all critical faces are already covered by e_in by construction
                    # So we just need to pick any valid one
                    # To break ties, prefer edges that cover more faces overall
                    gain = len(edge_faces[e_in])
                    if gain > best_gain:
                        best_gain = gain
                        best_e_in = e_in
                
                if best_e_in != -1:
                    remove_from_removed(e_out)
                    add_to_removed(best_e_in)
                    any_swap = True
                    if len(removed) < best_size:
                        best_size = len(removed)
                        best_removed = set(removed)
        
        return any_swap
    
    def try_2opt():
        """Try 2-opt moves: remove two edges, add one or two different edges.
        This is more complex but can escape local optima."""
        nonlocal removed, face_counts, best_size, best_removed
        any_move = False
        
        # Try removing one edge and adding two others
        for e_out in list(removed):
            critical_faces = []
            for f in edge_faces[e_out]:
                if face_counts[f] == 1:
                    critical_faces.append(f)
            
            if not critical_faces:
                continue
            
            # Find pairs of edges that together cover all critical faces
            # Start with edges from the first critical face
            first_face_edges = set(faces[critical_faces[0]]) - {e_out}
            
            for e_in1 in first_face_edges:
                if e_in1 in removed:
                    continue
                
                # Check which critical faces e_in1 covers
                remaining_critical = []
                for f in critical_faces:
                    if f not in edge_faces[e_in1]:
                        remaining_critical.append(f)
                
                if not remaining_critical:
                    # e_in1 alone covers all critical faces - this is a 1-opt swap
                    continue
                
                # Try to find e_in2 that covers the remaining critical faces
                candidates2 = set(faces[remaining_critical[0]])
                for f in remaining_critical[1:]:
                    candidates2 &= set(faces[f])
                
                candidates2.discard(e_out)
                candidates2.discard(e_in1)
                
                valid_candidates2 = [e for e in candidates2 if e not in removed]
                
                if valid_candidates2:
                    e_in2 = valid_candidates2[0]
                    
                    # Perform the 2-opt move
                    remove_from_removed(e_out)
                    add_to_removed(e_in1)
                    add_to_removed(e_in2)
                    any_move = True
                    if len(removed) < best_size:
                        best_size = len(removed)
                        best_removed = set(removed)
                    
                    # If size didn't improve, revert
                    if len(removed) >= best_size:
                        remove_from_removed(e_in1)
                        remove_from_removed(e_in2)
                        add_to_removed(e_out)
        
        return any_move
    
    # Main local search loop
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Deletions
        if try_deletions():
            improved = True
        
        # Phase 2: Swaps
        if not improved:
            if try_swaps():
                improved = True
        
        # Phase 3: 2-opt (less frequently, as it's more expensive)
        if not improved and random.random() < 0.1:
            if try_2opt():
                improved = True
        
        if not improved:
            # Stuck - try perturbation
            # Add a random edge to removed, then try to clean up
            non_removed = [e for e in range(n_edges) if e not in removed]
            if non_removed:
                e_perturb = random.choice(non_removed)
                add_to_removed(e_perturb)
                
                # Try to clean up
                for _ in range(10):
                    if not try_deletions():
                        break
                    if not try_swaps():
                        break
                
                # If we're still worse, revert
                if len(removed) > best_size:
                    # Try to get back to best state
                    for e in list(removed):
                        if e == e_perturb:
                            remove_from_removed(e)
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
