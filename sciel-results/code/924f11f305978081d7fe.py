import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    d = 7
    n = 1 << d
    m = d * n // 2 # 448
    
    # Build edges
    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
    
    # Build faces and face-edge incidence
    faces = []
    edge_faces = [[] for _ in range(m)]
    for x in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                xi = x ^ (1 << i)
                xj = x ^ (1 << j)
                xij = x ^ (1 << i) ^ (1 << j)
                e1 = edge_idx[(min(x, xi), max(x, xi))]
                e2 = edge_idx[(min(x, xj), max(x, xj))]
                e3 = edge_idx[(min(xi, xij), max(xi, xij))]
                e4 = edge_idx[(min(xj, xij), max(xj, xij))]
                face = (e1, e2, e3, e4)
                fid = len(faces)
                faces.append(face)
                for e in face:
                    edge_faces[e].append(fid)
    
    F = len(faces)
    
    # State
    selected = np.ones(m, dtype=bool)
    face_counts = np.zeros(F, dtype=int)
    
    # Initialize face_counts
    for e in range(m):
        if selected[e]:
            for f in edge_faces[e]:
                face_counts[f] += 1
                
    # Count initial violations
    violation_count = np.sum(face_counts == 4)
    
    # Repair: Remove edges to make valid
    while violation_count > 0:
        if time.time() > deadline - 5:
            break
        # Find a violated face
        # For speed, we can keep a list, but F is small.
        violated_faces = [i for i in range(F) if face_counts[i] == 4]
        if not violated_faces:
            break
        f = random.choice(violated_faces)
        # Pick a random edge from this face
        e = random.choice(faces[f])
        # Remove e
        selected[e] = False
        for f2 in edge_faces[e]:
            old_c = face_counts[f2]
            face_counts[f2] = old_c - 1
            if old_c == 4:
                violation_count -= 1
            # If old_c was 3, it becomes 2, no change in violation count (was 0, now 0)
            # If old_c was 4, it becomes 3, violation removed.
            
    # Record initial best
    best_valid_edges = np.sum(selected)
    best_valid_selected = selected.copy()
    best_violation_count = violation_count
    
    # SA Parameters
    alpha = 1.0
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    # Lists for random selection
    # To pick random selected/unselected edge, we can just pick random index and check state.
    # Or maintain lists. Picking random index is O(1) average.
    
    steps = 0
    while time.time() < deadline - 5:
        steps += 1
        
        # Pick a move type
        # 1: Add, 2: Remove, 3: Swap
        move_type = random.randint(1, 3)
        
        accepted = False
        delta_score = 0
        delta_violations = 0
        delta_edges = 0
        
        if move_type == 1:
            # Add
            e = random.randint(0, m - 1)
            if not selected[e]:
                delta_edges = 1
                # Calculate delta violations
                for f in edge_faces[e]:
                    if face_counts[f] == 3:
                        delta_violations += 1
                
                delta_score = delta_edges - alpha * delta_violations
                
                if delta_score > 0 or (T > 0 and random.random() < math.exp(delta_score / T)):
                    # Accept
                    selected[e] = True
                    for f in edge_faces[e]:
                        old_c = face_counts[f]
                        face_counts[f] = old_c + 1
                        if old_c == 3:
                            violation_count += 1
                    accepted = True
                    
        elif move_type == 2:
            # Remove
            e = random.randint(0, m - 1)
            if selected[e]:
                delta_edges = -1
                for f in edge_faces[e]:
                    if face_counts[f] == 4:
                        delta_violations -= 1
                
                delta_score = delta_edges - alpha * delta_violations
                
                if delta_score > 0 or (T > 0 and random.random() < math.exp(delta_score / T)):
                    # Accept
                    selected[e] = False
                    for f in edge_faces[e]:
                        old_c = face_counts[f]
                        face_counts[f] = old_c - 1
                        if old_c == 4:
                            violation_count -= 1
                    accepted = True
                    
        else:
            # Swap
            # Pick random e_out (selected) and e_in (unselected)
            # To ensure they are different and valid states, we pick indices.
            e_out = random.randint(0, m - 1)
            e_in = random.randint(0, m - 1)
            
            if selected[e_out] and not selected[e_in] and e_out != e_in:
                delta_edges = 0
                
                # Calculate delta violations
                # We need to consider the union of faces
                # To avoid double counting or missing interactions, 
                # we can just iterate over unique faces in edge_faces[e_out] U edge_faces[e_in]
                
                # Simple approach: 
                # 1. Tentatively remove e_out, add e_in? No, just calculate.
                
                # Create a set of faces to check
                faces_to_check = set(edge_faces[e_out])
                faces_to_check.update(edge_faces[e_in])
                
                for f in faces_to_check:
                    old_c = face_counts[f]
                    new_c = old_c
                    if e_out in faces[f]: # This check is O(1) if faces are tuples? 
                        # faces[f] is a tuple of 4 ints. `in` is O(4).
                        new_c -= 1
                    if e_in in faces[f]:
                        new_c += 1
                    
                    if old_c == 4 and new_c == 3:
                        delta_violations -= 1
                    elif old_c == 3 and new_c == 4:
                        delta_violations += 1
                
                delta_score = delta_edges - alpha * delta_violations
                
                if delta_score > 0 or (T > 0 and random.random() < math.exp(delta_score / T)):
                    # Accept
                    selected[e_out] = False
                    selected[e_in] = True
                    
                    # Update counts
                    for f in edge_faces[e_out]:
                        old_c = face_counts[f]
                        face_counts[f] = old_c - 1
                        if old_c == 4:
                            violation_count -= 1
                    for f in edge_faces[e_in]:
                        old_c = face_counts[f]
                        face_counts[f] = old_c + 1
                        if old_c == 3:
                            violation_count += 1
                    accepted = True

        if accepted:
            # Check if valid and better
            if violation_count == 0:
                curr_edges = np.sum(selected)
                if curr_edges > best_valid_edges:
                    best_valid_edges = curr_edges
                    best_valid_selected = selected.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Optional: Restart T if stuck?
        # If no improvement for 10000 steps, reset T?
        # For now, just run.
        
    # Restore best valid solution
    selected = best_valid_selected
    
    out = [[edges[i][0], edges[i][1]] for i in range(m) if selected[i]]
    return out
