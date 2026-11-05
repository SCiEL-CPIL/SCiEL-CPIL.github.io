import numpy as np
import random

def construct():
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Generate all edges of Q7
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                edge_to_idx[(y, x)] = idx
    
    m = len(edges)  # 448
    
    # Precompute faces
    # A face is defined by a pair of bits (i, j) and a pattern of the other 5 bits.
    # To avoid duplicates, we fix the bits i and j to 0 in the base vertex.
    faces = [] # List of lists of edge indices
    face_idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all combinations of the other bits
            # There are 2^(d-2) = 32 such combinations
            for mask in range(2 ** (d - 2)):
                # Construct the base vertex u where bits i and j are 0
                # We need to map mask (bits 0..4) to the actual bit positions excluding i, j
                # This is a bit tricky. 
                # Alternative: Iterate all vertices u, and check if u has 0 in i and j.
                # But that's 128 * 21 = 2688 checks. Fine.
                pass
            
    # Simpler face generation:
    # Iterate all vertices u.
    # For each pair (i, j), if u has 0 in i and j, then u is the "base" for a face.
    # The face edges are:
    # e1 = (u, u ^ (1<<i))
    # e2 = (u ^ (1<<j), u ^ (1<<i) ^ (1<<j))
    # e3 = (u, u ^ (1<<j))
    # e4 = (u ^ (1<<i), u ^ (1<<i) ^ (1<<j))
    
    faces = []
    for u in range(n):
        for i in range(d):
            if u & (1 << i): continue
            for j in range(i + 1, d):
                if u & (1 << j): continue
                # u is base
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (u, v1)
                e2 = (v2, v3)
                e3 = (u, v2)
                e4 = (v1, v3)
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                faces.append([idx1, idx2, idx3, idx4])
                
    num_faces = len(faces)
    
    # Precompute edge_faces
    edge_faces = [[] for _ in range(m)]
    for f_idx, f_edges in enumerate(faces):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)
            
    # Greedy function
    def greedy_run(seed_val):
        random.seed(seed_val)
        candidates = set(range(m))
        blocked = set()
        face_counts = [0] * num_faces
        selected = []
        
        while candidates:
            # Find valid candidates
            valid = [e for e in candidates if e not in blocked]
            if not valid:
                break
            
            # Calculate risk
            min_risk = float('inf')
            best_candidates = []
            for e in valid:
                risk = 0
                for f_idx in edge_faces[e]:
                    if face_counts[f_idx] == 2:
                        risk += 1
                if risk < min_risk:
                    min_risk = risk
                    best_candidates = [e]
                elif risk == min_risk:
                    best_candidates.append(e)
            
            # Pick randomly from best_candidates
            e = random.choice(best_candidates)
            selected.append(e)
            candidates.remove(e)
            
            # Update face counts and block
            for f_idx in edge_faces[e]:
                face_counts[f_idx] += 1
                if face_counts[f_idx] == 3:
                    # Find the 4th edge
                    for e4 in faces[f_idx]:
                        if e4 not in selected:
                            # e4 is the one not selected
                            # But we need to be sure it's the one completing the face
                            # Since count is 3, exactly 3 are selected.
                            # So e4 is the unselected one.
                            if e4 in candidates:
                                blocked.add(e4)
                                # We don't remove from candidates yet, just block
                            break
                            
        return len(selected)

    # Run multiple times
    best_count = 0
    best_edges = []
    
    # We need to return the edges, not just count.
    # So we should store the selected edges from the best run.
    
    def greedy_run_with_edges(seed_val):
        random.seed(seed_val)
        candidates = set(range(m))
        blocked = set()
        face_counts = [0] * num_faces
        selected = []
        
        while candidates:
            valid = [e for e in candidates if e not in blocked]
            if not valid:
                break
            
            min_risk = float('inf')
            best_candidates = []
            for e in valid:
                risk = 0
                for f_idx in edge_faces[e]:
                    if face_counts[f_idx] == 2:
                        risk += 1
                if risk < min_risk:
                    min_risk = risk
                    best_candidates = [e]
                elif risk == min_risk:
                    best_candidates.append(e)
            
            e = random.choice(best_candidates)
            selected.append(e)
            candidates.remove(e)
            
            for f_idx in edge_faces[e]:
                face_counts[f_idx] += 1
                if face_counts[f_idx] == 3:
                    for e4 in faces[f_idx]:
                        if e4 not in selected:
                            if e4 in candidates:
                                blocked.add(e4)
                            break
                            
        return selected

    best_len = 0
    best_sel = []
    for i in range(100):
        sel = greedy_run_with_edges(i)
        if len(sel) > best_len:
            best_len = len(sel)
            best_sel = sel
            
    result = []
    for idx in best_sel:
        x, y = edges[idx]
        result.append([x, y])
        
    return result
