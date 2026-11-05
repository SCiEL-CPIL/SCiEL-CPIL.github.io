import time
import random
import numpy as np

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * 2 ** (d - 1)
    
    # Precompute edges
    # Edge index mapping: (u, v) where u < v and u^v is a power of 2
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                
    # Precompute faces
    # A face is defined by 2 bits (i, j) and the values of the other d-2 bits.
    # There are C(d, 2) * 2^(d-2) faces.
    face_edges = []
    edge_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all combinations of fixed bits for the other d-2 dimensions
            # The free bits are i and j.
            # We can iterate over all vertices, but that's inefficient.
            # Better: iterate over the 2^(d-2) patterns for the fixed bits.
            
            # Let's just iterate over all vertices and group them?
            # Or construct directly.
            # For a fixed i, j, a face consists of 4 vertices:
            # x, x^(1<<i), x^(1<<j), x^(1<<i)^(1<<j)
            # where x has 0 at i and j, and arbitrary bits elsewhere.
            
            mask_i = 1 << i
            mask_j = 1 << j
            mask_rest = 0
            for k in range(d):
                if k != i and k != j:
                    mask_rest |= (1 << k)
            
            # Iterate over all combinations of mask_rest
            # There are 2^(d-2) such combinations.
            # We can just iterate 0 to 2^d-1 and filter? 
            # Or use itertools.product for the d-2 bits.
            
            # Simpler: iterate all x in 0..2^d-1
            # If x has bit i = 0 and bit j = 0, it represents a unique face base.
            # But this is O(2^d * d^2) which is 128 * 21 = 2688 checks. Very fast.
            
            pass # We will do this in a loop below
            
    # Let's re-implement face generation more cleanly
    face_edges = []
    edge_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            # Iterate over all vertices x where bit i is 0 and bit j is 0
            # This ensures we don't duplicate faces.
            for x in range(n_vertices):
                if (x & mask_i) == 0 and (x & mask_j) == 0:
                    # The 4 vertices of this face are:
                    v1 = x
                    v2 = x | mask_i
                    v3 = x | mask_j
                    v4 = x | mask_i | mask_j
                    
                    # The 4 edges are:
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    idxs = []
                    for e in [e1, e2, e3, e4]:
                        idxs.append(edge_to_idx[e])
                    
                    face_idx = len(face_edges)
                    face_edges.append(idxs)
                    for idx in idxs:
                        edge_faces[idx].append(face_idx)
                        
    n_faces = len(face_edges)
    
    # State
    # edge_present: boolean array of length n_edges
    # face_counts: integer array of length n_faces
    
    # Initial state: Greedy or Random?
    # Let's start with a random subset of edges, maybe 50% density?
    # Or start with the greedy seed to have a good baseline.
    # The prompt says "Keep the function names...". It doesn't forbid using the seed logic internally.
    # But "Rewrite the program from scratch".
    # I will implement a fast greedy initialization to get a good start, then SA.
    
    edge_present = np.zeros(n_edges, dtype=bool)
    face_counts = np.zeros(n_faces, dtype=np.int32)
    
    # Fast Greedy Initialization
    # Order edges randomly to break symmetry
    edge_order = list(range(n_edges))
    random.shuffle(edge_order)
    
    for idx in edge_order:
        # Check if adding this edge creates a violation
        # An edge is in 6 faces.
        # If any of those faces already has 3 edges, adding this makes it 4 (violation).
        # So we can only add if all incident faces have < 3 edges.
        can_add = True
        for f in edge_faces[idx]:
            if face_counts[f] >= 3:
                can_add = False
                break
        if can_add:
            edge_present[idx] = True
            for f in edge_faces[idx]:
                face_counts[f] += 1
                
    best_edges = np.sum(edge_present)
    best_edge_present = edge_present.copy()
    
    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    T = 1.0
    cooling = 0.9995
    # We want to allow violations temporarily to escape local optima
    # Penalty for violation: M
    M = 10.0
    
    current_edges = np.sum(edge_present)
    current_violations = 0 # Should be 0 from greedy
    
    # Precompute incident faces for quick access
    # edge_faces is a list of lists.
    
    # Move generation:
    # 1. Add a random edge
    # 2. Remove a random edge
    # 3. Swap: Remove one, Add another
    
    while time.time() < deadline:
        # Choose a move
        move_type = random.random()
        if move_type < 0.3:
            # Add
            idx = random.randint(0, n_edges - 1)
            if edge_present[idx]:
                continue
            # Calculate delta
            delta_viol = 0
            for f in edge_faces[idx]:
                if face_counts[f] == 3:
                    delta_viol += 1
                elif face_counts[f] == 4:
                    delta_viol -= 1 # Should not happen if we track correctly, but good for robustness
            # New violations = current_violations + delta_viol
            # Energy change = -1 (edge added) + M * delta_viol
            delta_E = -1.0 + M * delta_viol
        elif move_type < 0.6:
            # Remove
            # Find a present edge
            # Using random index and checking is O(1) avg if density is not tiny
            idx = random.randint(0, n_edges - 1)
            if not edge_present[idx]:
                continue
            delta_viol = 0
            for f in edge_faces[idx]:
                if face_counts[f] == 4:
                    delta_viol -= 1
                elif face_counts[f] == 3:
                    delta_viol += 1
            delta_E = 1.0 + M * delta_viol # Edge removed (+1 to energy if minimizing -edges), Violations change
        else:
            # Swap
            # Remove idx1, Add idx2
            idx1 = random.randint(0, n_edges - 1)
            if not edge_present[idx1]:
                continue
            idx2 = random.randint(0, n_edges - 1)
            if edge_present[idx2]:
                continue
            
            # Calculate delta for removing idx1 and adding idx2
            # This is slightly more complex as they might share faces.
            # Delta E = (Cost of removing idx1) + (Cost of adding idx2)
            # But if they share a face, the net change in that face is 0.
            
            # Let's just compute total delta violations by simulating the change on a temporary count?
            # No, that's slow.
            # Let's compute delta_viol_remove and delta_viol_add separately, but correct for shared faces.
            
            # Faces affected by idx1: F1
            # Faces affected by idx2: F2
            # Shared faces: F1 ∩ F2
            
            # For f in F1 \ F2:
            #   if count[f] == 4: removing idx1 reduces viol by 1
            #   if count[f] == 3: removing idx1 increases viol by 1
            # For f in F2 \ F1:
            #   if count[f] == 3: adding idx2 increases viol by 1
            #   if count[f] == 4: adding idx2 reduces viol by 1
            # For f in F1 ∩ F2:
            #   count changes by -1 + 1 = 0. No change in violation status?
            #   Wait. If count was 3, becomes 2 (remove), then 3 (add). No change.
            #   If count was 4, becomes 3 (remove), then 4 (add). No change.
            #   So shared faces don't affect the delta in violations!
            
            delta_viol = 0
            for f in edge_faces[idx1]:
                if f in edge_faces[idx2]: # This check is slow (list membership)
                    continue
                if face_counts[f] == 4:
                    delta_viol -= 1
                elif face_counts[f] == 3:
                    delta_viol += 1
            for f in edge_faces[idx2]:
                if f in edge_faces[idx1]:
                    continue
                if face_counts[f] == 3:
                    delta_viol += 1
                elif face_counts[f] == 4:
                    delta_viol -= 1
            
            delta_E = M * delta_viol # Edges count stays same
            
        # Acceptance
        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
            # Apply move
            if move_type < 0.3:
                # Add
                edge_present[idx] = True
                for f in edge_faces[idx]:
                    face_counts[f] += 1
                current_edges += 1
                current_violations += delta_viol
            elif move_type < 0.6:
                # Remove
                edge_present[idx] = False
                for f in edge_faces[idx]:
                    face_counts[f] -= 1
                current_edges -= 1
                current_violations += delta_viol
            else:
                # Swap
                edge_present[idx1] = False
                for f in edge_faces[idx1]:
                    face_counts[f] -= 1
                edge_present[idx2] = True
                for f in edge_faces[idx2]:
                    face_counts[f] += 1
                current_violations += delta_viol
                # current_edges unchanged
                
            # Update best if feasible
            if current_violations == 0:
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_edge_present = edge_present.copy()
                    
            T *= cooling
            if T < 1e-4:
                T = 1e-4
        else:
            # Rejection, just cool down slightly or keep T?
            T *= cooling
            if T < 1e-4:
                T = 1e-4

    # Final check: ensure best_edge_present is valid
    # It should be, as we only updated it when violations == 0.
    
    # Construct output
    out = []
    for idx, is_present in enumerate(best_edge_present):
        if is_present:
            u, v = edges[idx]
            out.append([u, v])
            
    return out
