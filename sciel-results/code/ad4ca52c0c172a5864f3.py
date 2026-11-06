import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                idx_to_edge[idx] = (x, y)
    
    # Precompute all faces (2D subcubes)
    # A face is determined by (u, i, j) where i<j and u has bits i,j = 0
    faces = []  # list of (edge_idx1, edge_idx2, edge_idx3, edge_idx4)
    face_of_edge = [[] for _ in range(n_edges)]  # for each edge, list of faces containing it
    
    for i in range(d):
        for j in range(i+1, d):
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
                    if e1 in edge_to_idx and e2 in edge_to_idx and e3 in edge_to_idx and e4 in edge_to_idx:
                        f = (edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4])
                        faces.append(f)
                        face_of_edge[f[0]].append(len(faces)-1)
                        face_of_edge[f[1]].append(len(faces)-1)
                        face_of_edge[f[2]].append(len(faces)-1)
                        face_of_edge[f[3]].append(len(faces)-1)
    
    n_faces = len(faces)
    
    # Build conflict graph: two edges conflict if they share a face
    # Actually, for C4-free, we need no face with all 4 edges.
    # This is a 4-uniform hypergraph constraint, not a pairwise constraint.
    # So we can't directly use MIS on a graph.
    
    # Alternative approach: Use a different formulation.
    # The constraint is: for each face f with edges (e1,e2,e3,e4), not all 4 are selected.
    # This is equivalent to: for each face, at least one edge is not selected.
    
    # Let's use a local search with a different strategy:
    # Start with a random solution, then iteratively improve.
    
    # Better approach: Use a structured construction based on linear codes or algebraic methods.
    # For Q7, there are known constructions that achieve 304 edges.
    
    # Let's try a different approach: model as a SAT-like problem and use local search.
    # Variables: x_e for each edge e (0 or 1)
    # Constraints: for each face f, sum of x_e over e in f <= 3
    
    # We'll use a randomized local search with simulated annealing.
    
    def count_violations(selected):
        """Count the number of faces with all 4 edges selected."""
        face_counts = np.zeros(n_faces, dtype=np.int8)
        for e in range(n_edges):
            if selected[e]:
                for f in face_of_edge[e]:
                    face_counts[f] += 1
        return np.sum(face_counts == 4)
    
    def get_violating_faces(selected):
        """Return list of faces with all 4 edges selected."""
        face_counts = np.zeros(n_faces, dtype=np.int8)
        for e in range(n_edges):
            if selected[e]:
                for f in face_of_edge[e]:
                    face_counts[f] += 1
        return np.where(face_counts == 4)[0]
    
    # Simulated annealing
    def simulated_annealing(initial_selected, max_time=130.0):
        selected = initial_selected.copy()
        # Compute initial face counts
        face_counts = np.zeros(n_faces, dtype=np.int8)
        for e in range(n_edges):
            if selected[e]:
                for f in face_of_edge[e]:
                    face_counts[f] += 1
        
        count = np.sum(selected)
        violations = np.sum(face_counts == 4)
        
        best_selected = selected.copy()
        best_count = count
        best_violations = violations
        
        temperature = 10.0
        cooling_rate = 0.9995
        
        while time.time() < max_time and temperature > 0.01:
            # Pick a random edge to flip
            e = random.randint(0, n_edges - 1)
            
            # Compute delta in violations if we flip e
            # If selected[e] is True, flipping to False:
            #   For each face f containing e, face_counts[f] decreases by 1
            #   Violations change: faces that were 4 become 3 (no longer violating)
            #   Faces that were 3 become 2 (no change in violation status)
            # If selected[e] is False, flipping to True:
            #   For each face f containing e, face_counts[f] increases by 1
            #   Violations change: faces that were 3 become 4 (now violating)
            
            if selected[e]:
                # Flipping to False
                delta_violations = 0
                for f in face_of_edge[e]:
                    if face_counts[f] == 4:
                        delta_violations -= 1  # One fewer violation
                    elif face_counts[f] == 3:
                        delta_violations += 0  # No change
                delta_count = -1
            else:
                # Flipping to True
                delta_violations = 0
                for f in face_of_edge[e]:
                    if face_counts[f] == 3:
                        delta_violations += 1  # One more violation
                    elif face_counts[f] == 2:
                        delta_violations += 0
                delta_count = 1
            
            new_violations = violations + delta_violations
            new_count = count + delta_count
            
            # Accept if better or with some probability
            if new_violations < violations or (new_violations == violations and new_count > count):
                # Always accept
                selected[e] = not selected[e]
                for f in face_of_edge[e]:
                    face_counts[f] += delta_count
                violations = new_violations
                count = new_count
                if violations == 0 and count > best_count:
                    best_count = count
                    best_selected = selected.copy()
            else:
                # Accept with probability exp(-delta / temperature)
                # Delta is the "energy" change: we want to minimize violations and maximize count
                # Let's define energy = violations * 1000 - count
                old_energy = violations * 1000 - count
                new_energy = new_violations * 1000 - new_count
                delta_energy = new_energy - old_energy
                if delta_energy < 0 or random.random() < np.exp(-delta_energy / temperature):
                    selected[e] = not selected[e]
                    for f in face_of_edge[e]:
                        face_counts[f] += delta_count
                    violations = new_violations
                    count = new_count
                    if violations == 0 and count > best_count:
                        best_count = count
                        best_selected = selected.copy()
            
            temperature *= cooling_rate
        
        return best_selected, best_count
    
    # Generate multiple random initial solutions and optimize
    best_selected = None
    best_count = 0
    
    n_attempts = 50
    for attempt in range(n_attempts):
        if time.time() > deadline - 10:
            break
        
        # Random initial solution: select each edge with probability p
        p = random.uniform(0.5, 0.8)
        selected = np.random.random(n_edges) < p
        
        # Quick fix: remove edges from violating faces
        face_counts = np.zeros(n_faces, dtype=np.int8)
        for e in range(n_edges):
            if selected[e]:
                for f in face_of_edge[e]:
                    face_counts[f] += 1
        
        # Remove edges from faces with count 4
        changed = True
        while changed:
            changed = False
            for f in range(n_faces):
                if face_counts[f] == 4:
                    e = faces[f][random.randint(0, 3)]
                    if selected[e]:
                        selected[e] = False
                        face_counts[f] -= 1
                        for f2 in face_of_edge[e]:
                            face_counts[f2] -= 1
                        changed = True
                        break
        
        # Now optimize with simulated annealing
        remaining_time = deadline - 2 - time.time()
        if remaining_time < 5:
            break
        
        sel, cnt = simulated_annealing(selected, max_time=min(remaining_time, 3.0))
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()
    
    # Final optimization on the best solution
    if best_selected is not None:
        remaining_time = deadline - 2 - time.time()
        if remaining_time > 5:
            sel, cnt = simulated_annealing(best_selected, max_time=remaining_time)
            if cnt > best_count:
                best_count = cnt
                best_selected = sel
    
    # Convert to output format
    result = []
    for e in range(n_edges):
        if best_selected[e]:
            u, v = idx_to_edge[e]
            result.append([u, v])
    
    return result
