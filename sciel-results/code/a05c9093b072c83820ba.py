import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 7 * 64 = 448
    
    # Generate all edges and map them to indices
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = len(edges)
                edges.append((u, v))
    
    # Generate all faces (squares)
    # A face is defined by a pair of varying bits (i, j) and fixed bits
    # There are C(7,2) = 21 pairs of varying bits
    # For each pair, there are 2^(7-2) = 32 combinations of fixed bits
    faces = []
    face_edges = []  # List of lists of edge indices for each face
    
    varying_pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            varying_pairs.append((i, j))
    
    for i, j in varying_pairs:
        # Iterate over all combinations of the other 5 bits
        # We can iterate 0 to 2^5 - 1 and map to the actual bits
        # But it's easier to just iterate all vertices and check if they form the base?
        # No, let's construct the base mask.
        # The varying bits are i and j.
        # The fixed bits are the others.
        # Let's iterate over all vertices u.
        # If u has 0 in bit i and 0 in bit j, it is the "base" for a unique face?
        # Yes, for a specific face defined by varying i,j and fixed others,
        # the base vertex is the one with 0 in i and 0 in j.
        # So we only consider u where (u & (1<<i)) == 0 and (u & (1<<j)) == 0.
        
        for u in range(n_vertices):
            if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                # u is the base vertex (0 in i, 0 in j)
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                # The 4 edges of this face
                e1 = (min(u, v1), max(u, v1))
                e2 = (min(v2, v3), max(v2, v3))
                e3 = (min(u, v2), max(u, v2))
                e4 = (min(v1, v3), max(v1, v3))
                
                # Get indices
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_edges.append([idx1, idx2, idx3, idx4])
    
    n_faces = len(face_edges) # Should be 672
    
    # Try to solve using scipy.optimize.milp
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Objective: Maximize sum(x) -> Minimize -sum(x)
        c = -np.ones(n_edges)
        
        # Constraints: For each face, sum(x_e for e in face) <= 3
        # A is n_faces x n_edges
        A = np.zeros((n_faces, n_edges), dtype=np.float64)
        for f_idx, edge_indices in enumerate(face_edges):
            for e_idx in edge_indices:
                A[f_idx, e_idx] = 1.0
        
        # LinearConstraint(A, lb, ub)
        # lb = -inf, ub = 3
        constraints = LinearConstraint(A, lb=-np.inf, ub=3)
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: 1 for binary
        integrality = np.ones(n_edges)
        
        # Solve
        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if result.success:
            x = result.x
            # Extract edges where x > 0.5
            selected_edges = []
            for i in range(n_edges):
                if x[i] > 0.5:
                    selected_edges.append(list(edges[i]))
            return selected_edges
        else:
            # Fallback to heuristic if MILP fails
            pass
            
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback: Randomized Greedy
    # This is very fast, so we can run multiple iterations
    best_count = 0
    best_edges = []
    
    # Precompute face membership for each edge for faster checking
    # edge_faces[e_idx] = list of face indices containing e_idx
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, edge_indices in enumerate(face_edges):
        for e_idx in edge_indices:
            edge_faces[e_idx].append(f_idx)
    
    def greedy_randomized():
        random.shuffle(edges)
        # We need to shuffle the indices corresponding to the shuffled edges
        # It's easier to shuffle the indices 0..n_edges-1
        idxs = list(range(n_edges))
        random.shuffle(idxs)
        
        selected = [False] * n_edges
        face_counts = [0] * n_faces
        count = 0
        
        for e_idx in idxs:
            # Check if adding this edge creates a conflict
            # Conflict if any face containing this edge already has 3 selected edges
            ok = True
            for f_idx in edge_faces[e_idx]:
                if face_counts[f_idx] >= 3:
                    ok = False
                    break
            
            if ok:
                selected[e_idx] = True
                count += 1
                for f_idx in edge_faces[e_idx]:
                    face_counts[f_idx] += 1
        
        return count, selected

    # Run multiple randomized greeds
    max_iters = 2000
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        count, selected = greedy_randomized()
        if count > best_count:
            best_count = count
            best_edges = [list(edges[i]) for i in range(n_edges) if selected[i]]
            
    return best_edges
