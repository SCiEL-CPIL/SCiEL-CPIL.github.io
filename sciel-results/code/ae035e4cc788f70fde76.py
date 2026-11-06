import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
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
    
    # Precompute for each edge, the 6 other edges in the faces it belongs to
    # An edge (u,v) with diff bit i belongs to faces with bits i and j for j != i
    # The face has vertices u, v, u^(1<<j), v^(1<<j)
    # Edges in face: (u,v), (u^(1<<j), v^(1<<j)), (u, u^(1<<j)), (v, v^(1<<j))
    # We need to track for each edge, which other edges form a C4 with it
    # Actually, for C4-free, we need that no 2D face has all 4 edges.
    # A 2D face is defined by two bits i,j and a vertex u with those bits 0.
    # The 4 edges of the face are:
    # (u, u^(1<<i)), (u, u^(1<<j)), (u^(1<<i), u^(1<<i|1<<j)), (u^(1<<j), u^(1<<i|1<<j))
    
    # Let's precompute all faces (2D subcubes)
    # A face is determined by (u, i, j) where i<j and u has bits i,j = 0
    faces = []  # list of (edge_idx1, edge_idx2, edge_idx3, edge_idx4)
    face_of_edge = [[] for _ in range(n_edges)]  # for each edge, list of faces containing it
    
    for i in range(d):
        for j in range(i+1, d):
            # u must have bits i and j set to 0
            # Iterate over all vertices with bits i,j = 0
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
    
    # Initial construction: use a structured approach
    # Try multiple random linear code constructions and pick the best
    best_edges_set = set()
    best_count = 0
    
    # Strategy: Use a random affine subspace-based construction
    # For Q7, we can try to construct edges based on a random linear map
    
    # Method: Random greedy with randomization, then local search
    # But let's try a structured approach first: 
    # Use a random permutation of bits and apply a known construction
    
    # Known good construction for C4-free subgraphs of hypercubes:
    # Take edges where the "weight" of the vertex in some ordering satisfies a condition
    
    # Let's use a randomized approach: 
    # 1. Generate a random ordering of edges
    # 2. Greedily add edges that don't complete a C4
    # 3. Then do local search: try to add removed edges, or swap edges
    
    # For speed, let's use a vectorized face count approach
    # face_counts[f] = number of edges in face f that are selected
    # A face is "bad" if face_counts[f] == 4
    
    # Local search moves:
    # - Add an edge e if for all faces f containing e, face_counts[f] < 3
    #   (adding e would make face_counts[f] become 4 if it was 3)
    #   Actually, we can add e if for all faces f containing e, face_counts[f] <= 2
    #   Wait: if face_counts[f] == 3, adding e makes it 4 -> bad
    #   So we can add e only if for all f in face_of_edge[e], face_counts[f] <= 2
    # - Remove an edge e to unblock other edges
    # - Swap: remove e1, add e2
    
    # Let's implement a fast local search
    # State: selected (boolean array of length n_edges), face_counts (array of length n_faces)
    
    def evaluate_and_optimize(initial_selected, max_time=130.0):
        selected = initial_selected.copy()
        # Compute face counts
        face_counts = np.zeros(n_faces, dtype=np.int8)
        for e in range(n_edges):
            if selected[e]:
                for f in face_of_edge[e]:
                    face_counts[f] += 1
        
        # Count valid edges
        count = np.sum(selected)
        
        # Local search: repeatedly try to add edges, and if stuck, try swaps
        # For speed, maintain a list of "addable" edges
        # An edge e is addable if for all f in face_of_edge[e], face_counts[f] <= 2
        
        # We'll do a simple iterative improvement
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try to add all possible edges
            for e in range(n_edges):
                if not selected[e]:
                    can_add = True
                    for f in face_of_edge[e]:
                        if face_counts[f] >= 3:
                            can_add = False
                            break
                    if can_add:
                        selected[e] = True
                        count += 1
                        for f in face_of_edge[e]:
                            face_counts[f] += 1
                        improved = True
            if not improved:
                # Try random swaps: remove one edge, then try to add others
                # Or just break if no progress
                # Try a few random swaps
                for _ in range(100):
                    if time.time() > max_time:
                        break
                    # Pick a random selected edge to remove
                    sel_indices = np.where(selected)[0]
                    if len(sel_indices) == 0:
                        break
                    e1 = sel_indices[random.randint(0, len(sel_indices)-1)]
                    # Remove e1
                    selected[e1] = False
                    count -= 1
                    for f in face_of_edge[e1]:
                        face_counts[f] -= 1
                    # Try to add other edges
                    added_any = False
                    for e2 in range(n_edges):
                        if not selected[e2]:
                            can_add = True
                            for f in face_of_edge[e2]:
                                if face_counts[f] >= 3:
                                    can_add = False
                                    break
                            if can_add:
                                selected[e2] = True
                                count += 1
                                for f in face_of_edge[e2]:
                                    face_counts[f] += 1
                                added_any = True
                    if not added_any:
                        # Restore e1
                        selected[e1] = True
                        count += 1
                        for f in face_of_edge[e1]:
                            face_counts[f] += 1
        
        return selected, count
    
    # Generate multiple random initial solutions and optimize
    best_selected = None
    best_count = 0
    
    n_attempts = 20
    for attempt in range(n_attempts):
        if time.time() > deadline - 10:
            break
        
        # Random initial solution: select each edge with probability p
        # Then fix violations by removing edges
        p = random.uniform(0.5, 0.8)
        selected = np.random.random(n_edges) < p
        
        # Fix violations: iteratively remove edges from full faces
        # This is slow, so let's just use greedy from this random start
        # Actually, let's do a quick greedy fix
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
                    # Remove one random edge from this face
                    e = faces[f][random.randint(0, 3)]
                    if selected[e]:
                        selected[e] = False
                        face_counts[f] -= 1
                        for f2 in face_of_edge[e]:
                            face_counts[f2] -= 1
                        changed = True
                        break
        
        # Now optimize
        sel, cnt = evaluate_and_optimize(selected, max_time=min(deadline - 2, start_time + 5 + attempt * 0.5))
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()
    
    # Final optimization on the best solution
    if best_selected is not None:
        sel, cnt = evaluate_and_optimize(best_selected, max_time=deadline - 2)
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
