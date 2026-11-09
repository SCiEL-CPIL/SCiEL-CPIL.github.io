import time
import random
import numpy as np

def construct():
    """
    Finds a large C4-free edge set in Q7.
    Returns a list of edges [u, v].
    """
    d = 7
    n_vertices = 2 ** d
    total_edges = d * (2 ** (d - 1)) # 448
    
    # 1. Precompute edges
    edges = []
    edge_index = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = idx
    
    # 2. Precompute faces
    faces = []
    face_edges = []
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            for base in range(n_vertices):
                if (base & mask_i) == 0 and (base & mask_j) == 0:
                    v0 = base
                    v1 = base ^ mask_i
                    v2 = base ^ mask_j
                    v3 = base ^ mask_i ^ mask_j
                    
                    e1 = (min(v0, v1), max(v0, v1))
                    e2 = (min(v2, v3), max(v2, v3))
                    e3 = (min(v0, v2), max(v0, v2))
                    e4 = (min(v1, v3), max(v1, v3))
                    
                    idxs = [
                        edge_index[e1],
                        edge_index[e2],
                        edge_index[e3],
                        edge_index[e4]
                    ]
                    faces.append(idxs)
                    face_edges.append(idxs)
    
    num_faces = len(faces) # 672
    
    # 3. Precompute edge-to-face mapping
    edge_faces = [[] for _ in range(total_edges)]
    for f_idx, f_edges in enumerate(face_edges):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)
    
    # Precompute for each face, the list of edges (as indices)
    # Also precompute: for each edge, which faces it belongs to (already done)
    
    # 4. Search Function with 2-swap moves
    def local_search(initial_selection, time_limit, rng):
        selection = initial_selection.copy()
        face_count = np.zeros(num_faces, dtype=np.int8)
        
        for e_idx in range(total_edges):
            if selection[e_idx]:
                for f_idx in edge_faces[e_idx]:
                    face_count[f_idx] += 1
        
        current_size = int(np.sum(selection))
        current_violations = int(np.sum(face_count == 4))
        lambda_penalty = 50.0
        current_score = current_size - lambda_penalty * current_violations
        
        best_selection = selection.copy()
        best_size = current_size
        best_violations = current_violations
        
        T = 2.0
        cooling = 0.999
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < time_limit:
            step += 1
            # Choose move type: 70% single flip, 30% 2-swap
            if rng.random() < 0.7:
                # Single flip
                e_idx = rng.randint(0, total_edges - 1)
                
                if selection[e_idx]:
                    delta_size = -1
                    delta_viol = 0
                    for f_idx in edge_faces[e_idx]:
                        if face_count[f_idx] == 4:
                            delta_viol -= 1
                    
                    new_score = (current_size - 1) - lambda_penalty * (current_violations + delta_viol)
                    delta_score = new_score - current_score
                else:
                    delta_size = 1
                    delta_viol = 0
                    for f_idx in edge_faces[e_idx]:
                        if face_count[f_idx] == 3:
                            delta_viol += 1
                    
                    new_score = (current_size + 1) - lambda_penalty * (current_violations + delta_viol)
                    delta_score = new_score - current_score
                
                # Acceptance
                accept = False
                if delta_score >= 0:
                    accept = True
                else:
                    if T > 0.01:
                        prob = np.exp(delta_score / T)
                        if rng.random() < prob:
                            accept = True
                
                if accept:
                    selection[e_idx] = not selection[e_idx]
                    current_size += delta_size
                    current_violations += delta_viol
                    current_score = new_score
                    
                    for f_idx in edge_faces[e_idx]:
                        if selection[e_idx]:
                            face_count[f_idx] += 1
                        else:
                            face_count[f_idx] -= 1
                    
                    if current_violations == 0 and current_size > best_size:
                        best_size = current_size
                        best_violations = 0
                        best_selection = selection.copy()
                    elif current_violations < best_violations and current_size >= best_size:
                        best_size = current_size
                        best_violations = current_violations
                        best_selection = selection.copy()
                        
            else:
                # 2-swap: remove one edge, add another
                # Find a violating face, pick an edge to remove from it
                if current_violations > 0:
                    # Pick a random violating face
                    viol_faces = np.where(face_count == 4)[0]
                    if len(viol_faces) > 0:
                        f_idx = viol_faces[rng.randint(0, len(viol_faces) - 1)]
                        # Pick an edge to remove from this face
                        e_remove = face_edges[f_idx][rng.randint(0, 3)]
                        
                        if selection[e_remove]:
                            # Calculate effect of removing e_remove
                            delta_remove_size = -1
                            delta_remove_viol = 0
                            for ff in edge_faces[e_remove]:
                                if face_count[ff] == 4:
                                    delta_remove_viol -= 1
                            
                            # Now try to add an edge that is compatible
                            # Find edges not in conflict with the current state after removal
                            # Simulate: after removing e_remove, check which additions are safe
                            best_add = -1
                            best_add_score = -1e9
                            
                            # Try a few candidate additions
                            for _ in range(5):
                                e_add = rng.randint(0, total_edges - 1)
                                if e_add == e_remove or selection[e_add]:
                                    continue
                                
                                # Check if adding e_add is safe after removing e_remove
                                # First, check conflicts with current selection (excluding e_remove)
                                safe = True
                                for ff in edge_faces[e_add]:
                                    # Count selected edges in this face, excluding e_remove if it's in this face
                                    cnt = 0
                                    for ee in face_edges[ff]:
                                        if ee == e_remove:
                                            continue  # would be removed
                                        if selection[ee]:
                                            cnt += 1
                                    if cnt >= 3:
                                        safe = False
                                        break
                                
                                if safe:
                                    # Calculate score improvement
                                    delta_add_size = 1
                                    delta_add_viol = 0
                                    for ff in edge_faces[e_add]:
                                        # Count selected edges in this face after removal of e_remove
                                        cnt = 0
                                        for ee in face_edges[ff]:
                                            if ee == e_remove:
                                                continue
                                            if selection[ee]:
                                                cnt += 1
                                        if cnt == 3:
                                            delta_add_viol += 1
                                    
                                    # Total delta
                                    total_delta_size = delta_remove_size + delta_add_size
                                    total_delta_viol = delta_remove_viol + delta_add_viol
                                    add_score = total_delta_size - lambda_penalty * total_delta_viol
                                    
                                    if add_score > best_add_score:
                                        best_add_score = add_score
                                        best_add = e_add
                            
                            if best_add >= 0:
                                # Apply the 2-swap
                                # Remove e_remove
                                selection[e_remove] = False
                                current_size += delta_remove_size
                                current_violations += delta_remove_viol
                                for ff in edge_faces[e_remove]:
                                    face_count[ff] -= 1
                                
                                # Add best_add
                                selection[best_add] = True
                                for ff in edge_faces[best_add]:
                                    face_count[ff] += 1
                                current_size += 1
                                # Recalculate violations for affected faces
                                # Actually, let's just recalculate delta_viol properly
                                # This is approximate; the face_count updates are correct
                                
                                # Track best
                                if current_violations == 0 and current_size > best_size:
                                    best_size = current_size
                                    best_violations = 0
                                    best_selection = selection.copy()
                                elif current_violations < best_violations and current_size >= best_size:
                                    best_size = current_size
                                    best_violations = current_violations
                                    best_selection = selection.copy()
                
                # If no violations, just do a single flip as fallback
                else:
                    e_idx = rng.randint(0, total_edges - 1)
                    if not selection[e_idx]:
                        delta_viol = 0
                        for f_idx in edge_faces[e_idx]:
                            if face_count[f_idx] == 3:
                                delta_viol += 1
                        if delta_viol == 0:
                            selection[e_idx] = True
                            current_size += 1
                            for f_idx in edge_faces[e_idx]:
                                face_count[f_idx] += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_violations = 0
                                best_selection = selection.copy()
            
            # Cool down
            T *= cooling
            if T < 0.01:
                T = 0.01
            
            # Check time
            if time.time() - start_time > time_limit:
                break
                
        return best_selection, best_size

    def repair(selection, fc=None):
        if fc is None:
            fc = np.zeros(num_faces, dtype=np.int8)
            for e_idx in range(total_edges):
                if selection[e_idx]:
                    for f_idx in edge_faces[e_idx]:
                        fc[f_idx] += 1
        
        changed = True
        while changed:
            changed = False
            for f_idx in range(num_faces):
                if fc[f_idx] == 4:
                    # Pick the edge in this face that appears in the most other violating faces
                    best_e = -1
                    best_count = -1
                    for e_idx in face_edges[f_idx]:
                        if selection[e_idx]:
                            # Count how many other violating faces contain this edge
                            cnt = 0
                            for ff in edge_faces[e_idx]:
                                if ff != f_idx and fc[ff] == 4:
                                    cnt += 1
                            if cnt > best_count:
                                best_count = cnt
                                best_e = e_idx
                    if best_e >= 0:
                        selection[best_e] = False
                        for ff in edge_faces[best_e]:
                            fc[ff] -= 1
                        changed = True
        return selection, fc

    def greedy_fill(selection, fc):
        # Try to add any edge that doesn't create a violation
        improved = True
        while improved:
            improved = False
            for e_idx in range(total_edges):
                if not selection[e_idx]:
                    ok = True
                    for f_idx in edge_faces[e_idx]:
                        if fc[f_idx] == 3:
                            ok = False
                            break
                    if ok:
                        selection[e_idx] = True
                        for f_idx in edge_faces[e_idx]:
                            fc[f_idx] += 1
                        improved = True
        return selection, fc

    # Main Loop
    start_time = time.time()
    time_limit = 110.0
    
    best_final_selection = None
    best_final_size = 0
    
    rng = random.Random(42)
    
    num_restarts = 15
    restart_time = time_limit / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with varying densities
        init_density = 0.4 + 0.2 * rng.random()
        init_sel = np.array([rng.random() < init_density for _ in range(total_edges)], dtype=bool)
        
        # Run local search
        sel, size = local_search(init_sel, restart_time, rng)
        
        # Repair
        sel, fc = repair(sel, None)
        
        # Greedy Fill
        sel, fc = greedy_fill(sel, fc)
        
        final_size = int(np.sum(sel))
        
        if final_size > best_final_size:
            best_final_size = final_size
            best_final_selection = sel.copy()
        
        # If we have time, do a polish pass on the best solution
        elapsed = time.time() - start_time
        if elapsed < time_limit - 5 and best_final_selection is not None:
            polish_time = min(3.0, time_limit - elapsed - 2)
            if polish_time > 1:
                polished, _ = local_search(best_final_selection, polish_time, rng)
                polished, fc2 = repair(polished, None)
                polished, fc2 = greedy_fill(polished, fc2)
                polish_size = int(np.sum(polished))
                if polish_size > best_final_size:
                    best_final_size = polish_size
                    best_final_selection = polished.copy()
            
    # Convert to list of edges
    if best_final_selection is None:
        chosen = set()
        out = []
        for u in range(n_vertices):
            for i in range(d):
                v = u ^ (1 << i)
                if u < v:
                    ok = True
                    for j in range(d):
                        if j != i:
                            c, e = u ^ (1 << j), v ^ (1 << j)
                            if all(p in chosen for p in ((min(u, c), max(u, c)), (min(v, e), max(v, e)), (min(c, e), max(c, e)))):
                                ok = False
                                break
                    if ok:
                        chosen.add((u, v))
                        out.append([u, v])
        return out

    out = []
    for e_idx, is_sel in enumerate(best_final_selection):
        if is_sel:
            out.append(list(edges[e_idx]))
            
    return out
