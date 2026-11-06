import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448
    
    # Precompute edges
    edges = []
    edge_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_index[(x, y)] = idx
                edge_index[(y, x)] = idx

    # Precompute faces and edge-face relationships
    # A face is a set of 4 edges.
    # We represent a face by a frozenset of edge indices.
    faces = []
    edge_to_faces = [[] for _ in range(n_edges)]
    
    seen_faces = set()
    for i in range(d):
        for j in range(i + 1, d):
            mask_ij = (1 << i) | (1 << j)
            for u in range(n_vertices):
                if (u & mask_ij) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ mask_ij
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    idx1 = edge_index[e1]
                    idx2 = edge_index[e2]
                    idx3 = edge_index[e3]
                    idx4 = edge_index[e4]
                    
                    face_key = frozenset([idx1, idx2, idx3, idx4])
                    if face_key not in seen_faces:
                        seen_faces.add(face_key)
                        face_id = len(faces)
                        faces.append(face_key)
                        for e_idx in face_key:
                            edge_to_faces[e_idx].append(face_id)
    
    n_faces = len(faces)
    
    # Helper to count uncovered faces given a set of removed edges R
    # We maintain a count of how many removed edges are in each face.
    # If count == 0, the face is uncovered.
    
    def greedy_init():
        # Greedy Hitting Set
        R = set()
        # face_hit_count[f] = number of edges in R that are in face f
        face_hit_count = [0] * n_faces
        uncovered = n_faces
        
        # Precompute for each edge, which faces it hits
        # We already have edge_to_faces
        
        while uncovered > 0 and time.time() < deadline - 5:
            best_edge = -1
            best_gain = 0
            # Iterate all edges to find the one covering most uncovered faces
            # Optimization: Only consider edges that cover at least 1 uncovered face
            # For d=7, n_edges=448, n_faces=672. This loop is fast enough.
            for e in range(n_edges):
                if e in R:
                    continue
                gain = 0
                for f_idx in edge_to_faces[e]:
                    if face_hit_count[f_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_edge = e
            
            if best_edge == -1:
                break # Should not happen unless uncovered==0
            
            R.add(best_edge)
            for f_idx in edge_to_faces[best_edge]:
                if face_hit_count[f_idx] == 0:
                    uncovered -= 1
                face_hit_count[f_idx] += 1
                
        return R, face_hit_count, uncovered

    def polish(R, face_hit_count):
        # Remove redundant edges from R
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of R
            for e in list(R):
                # Check if e is redundant
                # e is redundant if for all faces f containing e, face_hit_count[f] > 1
                redundant = True
                for f_idx in edge_to_faces[e]:
                    if face_hit_count[f_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    R.remove(e)
                    for f_idx in edge_to_faces[e]:
                        face_hit_count[f_idx] -= 1
                    changed = True
        return R, face_hit_count

    # Initial solution
    R, face_hit_count, uncovered = greedy_init()
    R, face_hit_count = polish(R, face_hit_count)
    
    best_R = set(R)
    best_size = len(R)
    best_score = n_edges - best_size
    
    # Simulated Annealing on R
    # Cost function: |R| + P * uncovered
    # We want to minimize this.
    P = 10.0 # Penalty for uncovered faces
    
    T = 5.0
    T_min = 0.01
    alpha = 0.99995
    
    current_R = set(R)
    current_face_hit_count = list(face_hit_count)
    current_uncovered = uncovered
    current_cost = len(current_R) + P * current_uncovered
    
    steps = 0
    max_steps = 5000000
    
    # Lists for faster random selection if needed, but set is fine for now
    # We need to pick random e in R and e not in R.
    
    while time.time() < deadline:
        steps += 1
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to remove an edge from R
            if len(current_R) == 0:
                continue
            # Pick random edge from R
            # Converting set to list every time is O(N). 
            # For 448 edges, it's okay, but we can optimize if needed.
            # Let's just pick random index and check membership? 
            # Or maintain a list of removed edges?
            # Let's use random.sample or just random.choice(list(current_R))
            # Given the small size, list conversion is acceptable.
            e = random.choice(list(current_R))
            
            # Calculate delta
            # Removing e: |R| decreases by 1.
            # Uncovered faces increase by the number of faces f where face_hit_count[f] == 1 and e in f.
            delta_uncovered = 0
            for f_idx in edge_to_faces[e]:
                if current_face_hit_count[f_idx] == 1:
                    delta_uncovered += 1
            
            delta_cost = -1 + P * delta_uncovered
            
            if delta_cost <= 0 or random.random() < (2.71828 ** (-delta_cost / T)):
                # Accept
                current_R.remove(e)
                for f_idx in edge_to_faces[e]:
                    if current_face_hit_count[f_idx] == 1:
                        current_uncovered += 1
                    current_face_hit_count[f_idx] -= 1
                current_cost += delta_cost
                
                if current_uncovered == 0:
                    # Valid solution
                    if len(current_R) < best_size:
                        best_size = len(current_R)
                        best_R = set(current_R)
                        best_score = n_edges - best_size
                        
        elif move_type < 0.8:
            # Try to add an edge to R
            # Pick random edge not in R
            # We can pick random index 0..447 and check if in R
            e = random.randint(0, n_edges - 1)
            if e in current_R:
                continue
            
            # Calculate delta
            # Adding e: |R| increases by 1.
            # Uncovered faces decrease by the number of faces f where face_hit_count[f] == 0 and e in f.
            delta_uncovered = 0
            for f_idx in edge_to_faces[e]:
                if current_face_hit_count[f_idx] == 0:
                    delta_uncovered -= 1 # Uncovered decreases
            
            # Note: delta_uncovered is negative here.
            # Let's stick to positive counts for clarity.
            # newly_covered = count of f where hit_count[f]==0
            newly_covered = 0
            for f_idx in edge_to_faces[e]:
                if current_face_hit_count[f_idx] == 0:
                    newly_covered += 1
            
            delta_cost = 1 - P * newly_covered
            
            if delta_cost <= 0 or random.random() < (2.71828 ** (-delta_cost / T)):
                # Accept
                current_R.add(e)
                for f_idx in edge_to_faces[e]:
                    if current_face_hit_count[f_idx] == 0:
                        current_uncovered -= 1
                    current_face_hit_count[f_idx] += 1
                current_cost += delta_cost
                
                if current_uncovered == 0:
                    if len(current_R) < best_size:
                        best_size = len(current_R)
                        best_R = set(current_R)
                        best_score = n_edges - best_size
                        
        else:
            # Swap: Remove e1 from R, Add e2 to R
            if len(current_R) == 0:
                continue
            e1 = random.choice(list(current_R))
            e2 = random.randint(0, n_edges - 1)
            if e2 in current_R:
                continue
            
            # Calculate delta for removing e1
            delta_uncovered_remove = 0
            for f_idx in edge_to_faces[e1]:
                if current_face_hit_count[f_idx] == 1:
                    delta_uncovered_remove += 1
            
            # Calculate delta for adding e2
            # Note: If e1 and e2 are in the same face, the interaction is complex.
            # Simpler: Apply remove, then apply add.
            # But we need the net delta for SA acceptance.
            
            # Let's just simulate the change in a temporary structure? 
            # Or calculate net change carefully.
            
            # Net change in uncovered:
            # Faces affected are those containing e1 or e2.
            # Let's just do a quick check.
            
            # Actually, swaps are less critical if Add/Remove are good.
            # Let's implement a simple swap logic.
            
            # Temporarily remove e1
            temp_hit_count = list(current_face_hit_count)
            for f_idx in edge_to_faces[e1]:
                if temp_hit_count[f_idx] == 1:
                    temp_hit_count[f_idx] = 0 # Becomes uncovered
                else:
                    temp_hit_count[f_idx] -= 1
            
            # Temporarily add e2
            newly_covered = 0
            for f_idx in edge_to_faces[e2]:
                if temp_hit_count[f_idx] == 0:
                    temp_hit_count[f_idx] = 1
                    newly_covered += 1
                else:
                    temp_hit_count[f_idx] += 1
            
            # Calculate new uncovered count
            new_uncovered = sum(1 for c in temp_hit_count if c == 0)
            delta_cost = 0 + P * (new_uncovered - current_uncovered) # |R| size unchanged
            
            if delta_cost <= 0 or random.random() < (2.71828 ** (-delta_cost / T)):
                # Accept
                current_R.remove(e1)
                current_R.add(e2)
                
                # Update state
                for f_idx in edge_to_faces[e1]:
                    if current_face_hit_count[f_idx] == 1:
                        current_uncovered += 1
                    current_face_hit_count[f_idx] -= 1
                for f_idx in edge_to_faces[e2]:
                    if current_face_hit_count[f_idx] == 0:
                        current_uncovered -= 1
                    current_face_hit_count[f_idx] += 1
                current_cost += delta_cost
                
                if current_uncovered == 0:
                    if len(current_R) < best_size:
                        best_size = len(current_R)
                        best_R = set(current_R)
                        best_score = n_edges - best_size
                        
        T *= alpha
        if T < T_min:
            T = T_min

    # Final Polish on best_R
    # Reconstruct face_hit_count for best_R
    best_face_hit_count = [0] * n_faces
    for e in best_R:
        for f_idx in edge_to_faces[e]:
            best_face_hit_count[f_idx] += 1
            
    best_R, best_face_hit_count = polish(best_R, best_face_hit_count)
    
    # Convert back to list of [u, v]
    result = []
    for e_idx in best_R:
        u, v = edges[e_idx]
        result.append([u, v])
        
    return result

