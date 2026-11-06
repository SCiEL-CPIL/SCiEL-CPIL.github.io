import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448
    
    # 1. Precompute Edge Indices and Face Constraints
    # Edge index e corresponds to (u, i) where u < u^(1<<i)
    # We map edge tuple (u, v) to index
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # Faces: 2D subcubes.
    # A face is defined by two dimensions i, j and a base vertex u (with bits i,j = 0 for uniqueness? 
    # Actually, let's just generate all unique sets of 4 edges).
    # There are C(7,2) * 2^(7-2) = 21 * 32 = 672 faces.
    
    # For each face, we need the 4 edge indices.
    # To make vectorization easy, we can represent a face by the indices of its 4 edges.
    # However, checking if a face is "full" requires checking if all 4 edges are present.
    
    # Alternative representation for fast checking:
    # For each edge e, we store the list of faces it belongs to.
    # But for vectorized parallel SA, we need to quickly check if adding edge e creates a C4.
    # Adding e creates a C4 if there exists a face F containing e such that the other 3 edges of F are already present.
    
    # Let's precompute for each edge e, a bitmask of the other edges that form a C4 with e?
    # No, that's not quite right. It's about specific faces.
    
    # Let's use a different approach for vectorized SA:
    # State: a boolean array of length 448 (0 or 1).
    # Move: flip a random edge.
    # Check: If we flip 0->1, is it valid?
    #   Valid iff for all faces containing e, the count of other 3 edges in the face is < 3.
    #   i.e., sum(state[other_edges_in_face]) < 3.
    
    # Precompute for each edge e:
    #   faces_with_e: list of faces
    #   For each face, the indices of the other 3 edges.
    
    edge_faces = [[] for _ in range(num_edges)]
    # We need to iterate over all faces.
    # Face defined by i, j, and base u where bit i=0, bit j=0?
    # Let's just generate all 4 vertices and map to edge indices.
    
    # To be efficient, let's build the face list first.
    # A face is a set of 4 edges.
    # We can represent a face by a tuple of 4 edge indices.
    # We need to avoid duplicates.
    
    seen_faces = set()
    face_data = [] # List of lists of 4 edge indices
    
    for i in range(d):
        for j in range(i+1, d):
            # Iterate over all possible base vertices?
            # A face is determined by the two bits i and j.
            # The vertices are u, u^i, u^j, u^i^j.
            # To get unique faces, we can fix bits i and j to 0 in u?
            # No, u can have other bits set.
            # If we iterate u from 0 to 127, we will generate each face 4 times.
            # We can just use a set of frozensets of edge indices to deduplicate.
            
            for u in range(n):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                # Get edges
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idxs = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                key = frozenset(idxs)
                
                if key not in seen_faces:
                    seen_faces.add(key)
                    face_data.append(idxs)
                    
    # Now, for each edge, find the faces it belongs to and the other edges in those faces.
    # edge_faces[e] will be a list of arrays, where each array contains the 3 other edge indices.
    
    for f_idx, face_edges in enumerate(face_data):
        for e in face_edges:
            others = [x for x in face_edges if x != e]
            edge_faces[e].append(np.array(others, dtype=np.int32))
            
    # Convert edge_faces to a list of numpy arrays for faster access?
    # Or just keep as list of lists of arrays.
    
    # 2. Parallel Simulated Annealing
    
    num_threads = 64 # Number of parallel trajectories
    # We will simulate these in a loop, but vectorize the state updates where possible?
    # Actually, Python loop over 64 trajectories is fine if each step is fast.
    # But checking validity is the bottleneck.
    
    # Let's try to vectorize the validity check for a batch of proposed moves?
    # Or just optimize the single-threaded check and run many restarts.
    
    # Let's stick to a multi-start local search with fast checks.
    # We will run multiple independent SA runs.
    
    best_edges = None
    best_count = 0
    
    # Precompute "conflict masks" or similar?
    # For each edge e, we have a list of "other 3" arrays.
    # To check if adding e is valid:
    #   for each arr in edge_faces[e]:
    #       if np.sum(state[arr]) == 3: return False
    #   return True
    
    # This is still a Python loop over faces for each edge.
    # Average faces per edge: 672 faces * 4 edges / 448 edges = 6 faces per edge.
    # So 6 checks per move. Each check is a numpy sum over 3 elements.
    # This should be reasonably fast.
    
    # Let's run 100 restarts, each with limited time.
    # Or one long run with parallel threads simulated via list of states?
    # Given Python GIL, true parallelism is hard. But we can interleave steps.
    
    # Let's do 20 independent SA runs, each getting (deadline - now) / 20 time.
    
    num_runs = 20
    time_per_run = (deadline - start_time) / num_runs
    
    rng = np.random.default_rng(42)
    
    for run_idx in range(num_runs):
        run_deadline = start_time + (run_idx + 1) * time_per_run
        if time.time() > deadline - 2: break
        
        # Initialize state
        # Greedy initialization:
        state = np.zeros(num_edges, dtype=np.uint8)
        # Shuffle edges
        order = rng.permutation(num_edges)
        count = 0
        for e in order:
            valid = True
            for others in edge_faces[e]:
                if np.sum(state[others]) == 3:
                    valid = False
                    break
            if valid:
                state[e] = 1
                count += 1
        
        current_state = state
        current_count = count
        best_state_run = current_state.copy()
        best_count_run = current_count
        
        T = 5.0
        cooling = 0.9995
        
        while time.time() < run_deadline:
            # Pick random edge
            e = rng.integers(0, num_edges)
            
            if current_state[e] == 1:
                # Remove
                # Always valid
                current_state[e] = 0
                current_count -= 1
                delta = -1
            else:
                # Add
                valid = True
                for others in edge_faces[e]:
                    if np.sum(current_state[others]) == 3:
                        valid = False
                        break
                if not valid:
                    continue
                current_state[e] = 1
                current_count += 1
                delta = 1
                
            # Acceptance
            if delta > 0:
                if current_count > best_count_run:
                    best_count_run = current_count
                    best_state_run = current_state.copy()
            else:
                # delta = -1
                # Accept with prob exp(1/T) ? No, exp(-delta/T) = exp(1/T) > 1?
                # Standard SA: accept if delta < 0 with prob exp(delta/T).
                # Here delta = -1. prob = exp(-1/T).
                if rng.random() < np.exp(-1.0 / T):
                    pass
                else:
                    # Reject
                    current_state[e] = 1
                    current_count += 1
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        if best_count_run > best_count:
            best_count = best_count_run
            best_edges = best_state_run.copy()
            
    # Convert best_edges to list of lists
    if best_edges is None:
        # Fallback: return empty or some valid set
        return []
        
    result = []
    for i in range(num_edges):
        if best_edges[i] == 1:
            result.append(list(idx_to_edge[i]))
            
    return result
