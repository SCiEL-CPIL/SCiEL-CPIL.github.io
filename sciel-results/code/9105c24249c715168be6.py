import random
import time
import math

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = d * (2 ** (d - 1))

    # Generate all edges
    edges = []
    edge_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                edge_index[(x, y)] = len(edges) - 1
                edge_index[(y, x)] = len(edges) - 1

    # Precompute faces (C4s)
    # A face is defined by a vertex and two distinct bits i, j
    # Edges in face: (v, v^i), (v^i, v^i^j), (v^j, v^i^j), (v, v^j)
    # We need to map each edge to the faces it belongs to.
    
    edge_to_faces = [[] for _ in range(total_edges)]
    face_edges = [] # List of 4 edge indices for each face
    face_id = 0
    
    # Iterate over all possible faces
    # To avoid duplicates, we can iterate over all vertices and pairs of bits,
    # but each face is counted 4 times (once per vertex).
    # Or simply iterate over all combinations of 2 bits and all vertices, 
    # and use a set to store unique faces? 
    # A face is uniquely identified by the set of 4 vertices.
    # Or by (min_vertex, bit_i, bit_j) where bit_i < bit_j and min_vertex has bit_i=0, bit_j=0?
    # Let's just generate all 672 faces.
    
    seen_faces = set()
    
    for v in range(n_vertices):
        for i in range(d):
            for j in range(i+1, d):
                # Face vertices: v, v^i, v^j, v^i^j
                # Canonical representation of the face: sorted tuple of 4 vertices
                verts = [v, v ^ (1<<i), v ^ (1<<j), v ^ (1<<i ^ (1<<j))]
                verts.sort()
                key = tuple(verts)
                
                if key in seen_faces:
                    continue
                seen_faces.add(key)
                
                # Get edge indices for the 4 edges of this face
                e1 = edge_index[(min(v, v^(1<<i)), max(v, v^(1<<i)))]
                e2 = edge_index[(min(v^(1<<i), v^(1<<i^(1<<j))), max(v^(1<<i), v^(1<<i^(1<<j))))]
                e3 = edge_index[(min(v^(1<<j), v^(1<<i^(1<<j))), max(v^(1<<j), v^(1<<i^(1<<j))))]
                e4 = edge_index[(min(v, v^(1<<j)), max(v, v^(1<<j)))]
                
                face_idx = len(face_edges)
                face_edges.append([e1, e2, e3, e4])
                
                # Update edge_to_faces
                for e in [e1, e2, e3, e4]:
                    edge_to_faces[e].append(face_idx)
    
    # Simulated Annealing
    # State: selected (boolean array or set)
    # Score: num_selected - PENALTY * num_violations
    
    PENALTY = 1000
    start_time = time.time()
    time_limit = 115
    
    rng = random.Random(42)
    
    # Initialize: All edges selected
    selected = [True] * total_edges
    num_selected = total_edges
    
    # Count violations
    # A face is violated if all 4 edges are selected
    violations = 0
    face_violated = [False] * len(face_edges)
    
    for f_idx, e_list in enumerate(face_edges):
        if all(selected[e] for e in e_list):
            violations += 1
            face_violated[f_idx] = True
            
    current_score = num_selected - PENALTY * violations
    best_score = current_score
    best_selected = selected[:]
    
    # SA Parameters
    T = 1.0
    cooling = 0.9995
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor
        move_type = rng.random()
        if move_type < 0.4:
            # Add an edge
            # Find a random unselected edge
            # To make this efficient, we can just pick a random index and check
            # If selected, skip or try another. 
            # Better: maintain a list of unselected? 
            # For now, simple rejection sampling
            attempts = 0
            while True:
                idx = rng.randint(0, total_edges - 1)
                if not selected[idx]:
                    break
                attempts += 1
                if attempts > 10:
                    break
            
            if attempts > 10:
                continue
                
            # Calculate delta
            delta_violations = 0
            for f_idx in edge_to_faces[idx]:
                # Check if the other 3 edges are selected
                # If yes, adding idx makes this face violated (if it wasn't already)
                # If it was already violated, it stays violated (delta 0)
                # If it wasn't violated, it becomes violated (delta +1)
                e_list = face_edges[f_idx]
                # Count selected edges in this face excluding idx
                # Since idx is not selected, we check the other 3
                other_selected = 0
                for e in e_list:
                    if e != idx and selected[e]:
                        other_selected += 1
                
                if other_selected == 3:
                    # This face will become violated
                    if not face_violated[f_idx]:
                        delta_violations += 1
                        
            delta_score = 1 - PENALTY * delta_violations
            
            if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                # Accept
                selected[idx] = True
                num_selected += 1
                violations += delta_violations
                # Update face_violated status for affected faces
                for f_idx in edge_to_faces[idx]:
                    if not face_violated[f_idx]:
                        # Check if it's now violated
                        if all(selected[e] for e in face_edges[f_idx]):
                            face_violated[f_idx] = True
                            
                current_score += delta_score
                if current_score > best_score:
                    best_score = current_score
                    best_selected = selected[:]

        elif move_type < 0.8:
            # Remove an edge
            attempts = 0
            while True:
                idx = rng.randint(0, total_edges - 1)
                if selected[idx]:
                    break
                attempts += 1
                if attempts > 10:
                    break
                    
            if attempts > 10:
                continue
                
            # Calculate delta
            delta_violations = 0
            for f_idx in edge_to_faces[idx]:
                # If the face is currently violated, removing idx might make it non-violated
                # Check if the other 3 edges are selected
                e_list = face_edges[f_idx]
                other_selected = 0
                for e in e_list:
                    if e != idx and selected[e]:
                        other_selected += 1
                        
                if face_violated[f_idx]:
                    if other_selected < 3:
                        # It was violated, now it's not (because we removed one edge, 
                        # so at most 3 are left, but actually if other_selected < 3, 
                        # it means before removal, total selected was <= 3? 
                        # Wait. If face_violated is True, all 4 were selected.
                        # We remove idx. Now 3 are selected. It is no longer violated.
                        # So delta_violations -= 1
                        delta_violations -= 1
                        
            delta_score = -1 - PENALTY * delta_violations
            
            if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                # Accept
                selected[idx] = False
                num_selected -= 1
                violations += delta_violations
                # Update face_violated
                for f_idx in edge_to_faces[idx]:
                    if face_violated[f_idx]:
                        # Check if it's still violated
                        if not all(selected[e] for e in face_edges[f_idx]):
                            face_violated[f_idx] = False
                            
                current_score += delta_score
                if current_score > best_score:
                    best_score = current_score
                    best_selected = selected[:]

        else:
            # Swap: remove one, add another
            # Pick random selected and unselected
            attempts = 0
            while True:
                idx_rem = rng.randint(0, total_edges - 1)
                idx_add = rng.randint(0, total_edges - 1)
                if selected[idx_rem] and not selected[idx_add] and idx_rem != idx_add:
                    break
                attempts += 1
                if attempts > 20:
                    break
                    
            if attempts > 20:
                continue
                
            # Calculate delta
            # Remove idx_rem
            delta_v_rem = 0
            for f_idx in edge_to_faces[idx_rem]:
                if face_violated[f_idx]:
                    e_list = face_edges[f_idx]
                    other_selected = 0
                    for e in e_list:
                        if e != idx_rem and selected[e]:
                            other_selected += 1
                    # Before removal, all 4 selected. After, 3 selected. Not violated.
                    delta_v_rem -= 1
                    
            # Add idx_add
            delta_v_add = 0
            # Note: The state for "other selected" should reflect the state AFTER removal?
            # Yes, swap is simultaneous.
            # But idx_rem and idx_add might be in the same face.
            # Let's just calculate the change in violations for the whole set.
            # Simpler: 
            # 1. Temporarily remove idx_rem
            # 2. Check delta for adding idx_add
            # This is complex to do analytically without state change.
            # Let's just skip Swap in this initial version or implement carefully.
            # Given time, let's stick to Add/Remove which is sufficient for SA.
            pass

        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor

    # Final cleanup: Ensure validity
    # If best_selected has violations, remove edges from violated faces
    final_selected = best_selected[:]
    
    # Re-evaluate violations for final_selected
    face_violated_final = [False] * len(face_edges)
    for f_idx, e_list in enumerate(face_edges):
        if all(final_selected[e] for e in e_list):
            face_violated_final[f_idx] = True
            
    if any(face_violated_final):
        # Greedy removal
        # While there are violations, pick a violated face and remove a random edge from it
        # To be safe, remove the edge that is in the most violated faces?
        # Or just random.
        while True:
            violated_faces = [i for i, v in enumerate(face_violated_final) if v]
            if not violated_faces:
                break
            f_idx = rng.choice(violated_faces)
            e_list = face_edges[f_idx]
            # Pick a random edge from this face to remove
            # Prefer edges that are in many violated faces
            candidates = []
            for e in e_list:
                if final_selected[e]:
                    # Count how many other violated faces this edge is in
                    count = 0
                    for ef in edge_to_faces[e]:
                        if face_violated_final[ef] and ef != f_idx:
                            count += 1
                    candidates.append((count, e))
            candidates.sort(reverse=True)
            if candidates:
                rem_edge = candidates[0][1]
                final_selected[rem_edge] = False
                # Update violated faces
                for ef in edge_to_faces[rem_edge]:
                    if face_violated_final[ef]:
                        if not all(final_selected[e] for e in face_edges[ef]):
                            face_violated_final[ef] = False
            else:
                break # Should not happen

    # Convert to list of edges
    result = []
    for i, is_sel in enumerate(final_selected):
        if is_sel:
            result.append(list(edges[i]))
            
    return result
