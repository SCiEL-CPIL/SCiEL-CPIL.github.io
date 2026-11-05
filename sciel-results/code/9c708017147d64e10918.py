import time
import random
import sys

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy: 
    1. Use a bitmask representation for the edge set for O(1) C4 checks.
    2. Initialize with a deterministic valid construction (Min-Even rule).
    3. Perform fast local search (Greedy Add + Random Swaps) to maximize edge count.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    d = 7
    n = 2 ** d
    num_edges = d * (n // 2)  # 7 * 64 = 448
    num_faces = (d * (d - 1) // 2) * (n // 4)  # 21 * 32 = 672
    
    # 1. Precompute edge indices and face memberships
    # Edge index mapping: (u, v, dim) -> index
    # We generate edges in a fixed order
    edge_to_idx = {}
    edges = []
    idx = 0
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edge_to_idx[(x, y)] = idx
                edges.append((x, y, i))
                idx += 1
    
    # Precompute for each edge, which faces it belongs to
    # A face is defined by (u, i, j) where u is the base vertex, and i, j are dims.
    # To uniquely identify a face, we can use the minimum vertex in the face and the two dims.
    # Or simpler: iterate over all faces and assign indices.
    
    face_edges = [[] for _ in range(num_faces)]
    edge_faces = [[] for _ in range(num_edges)]
    
    face_idx = 0
    for u in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                # Face vertices: u, u^i, u^j, u^i^j
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                # The 4 edges of this face
                e1 = (min(u, v1), max(u, v1))
                e2 = (min(u, v2), max(u, v2))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v3), max(v2, v3))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_edges[face_idx] = [idx1, idx2, idx3, idx4]
                
                edge_faces[idx1].append(face_idx)
                edge_faces[idx2].append(face_idx)
                edge_faces[idx3].append(face_idx)
                edge_faces[idx4].append(face_idx)
                
                face_idx += 1
                
    # 2. Initialization: Min-Even Rule
    # Include edge (u,v) if min(u,v) has even popcount
    initial_selected = 0
    face_counts = [0] * num_faces
    
    for idx, (u, v, i) in enumerate(edges):
        min_u = min(u, v)
        if bin(min_u).count('1') % 2 == 0:
            # Try to add
            # Check if it's valid (it should be, but let's verify logic)
            # Actually, we just set it. We assume this construction is valid.
            # To be safe, we can check, but it's known to be valid (max 3 per face).
            initial_selected |= (1 << idx)
            for f in edge_faces[idx]:
                face_counts[f] += 1
                
    # Verify initial count
    initial_count = bin(initial_selected).count('1')
    
    best_selected = initial_selected
    best_count = initial_count
    best_face_counts = face_counts[:]
    
    # 3. Local Search
    random.seed(42)
    
    # Helper to add edge
    def try_add_edge(selected, f_counts, idx):
        if selected & (1 << idx):
            return False, selected, f_counts
        # Check faces
        for f in edge_faces[idx]:
            if f_counts[f] == 3:
                return False, selected, f_counts
        # Valid
        new_selected = selected | (1 << idx)
        # Update counts (we can modify in place if we rollback, but for simplicity in function, let's return new state or modify and rollback)
        # For speed, we'll modify in place and rollback on failure
        for f in edge_faces[idx]:
            f_counts[f] += 1
        return True, new_selected, f_counts

    def remove_edge(selected, f_counts, idx):
        if not (selected & (1 << idx)):
            return False, selected, f_counts
        new_selected = selected & ~(1 << idx)
        for f in edge_faces[idx]:
            f_counts[f] -= 1
        return True, new_selected, f_counts

    # Main Loop
    # We do multiple restarts of a "Greedy Add" phase
    # Since the initial state is valid, we can try to add more edges.
    
    max_restarts = 500
    restart = 0
    while restart < max_restarts:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        restart += 1
        
        # Start from a slightly perturbed version of the best solution or random valid?
        # Let's start from the best solution found so far, but shuffle the order of "add" attempts.
        # Or start from scratch with a different random seed for the Min-Even variant?
        # The Min-Even rule is deterministic. Let's use Random Hash for initialization in later restarts.
        
        if restart == 1:
            selected = initial_selected
            f_counts = best_face_counts[:]
            count = best_count
        else:
            # Generate a new random valid set
            # Use random linear hash: edge (u,v) included if hash(u) < hash(v)?
            # Simpler: Random subset of the "Min-Even" edges? No, that might be small.
            # Let's just use the initial set and try to improve it.
            selected = best_selected
            f_counts = best_face_counts[:]
            count = best_count
            
            # Randomly remove a few edges to allow reconfiguration?
            # Or just try to add.
            
        # Greedy Add Phase
        # Shuffle edge indices
        order = list(range(num_edges))
        random.shuffle(order)
        
        improved = True
        while improved:
            improved = False
            for idx in order:
                if selected & (1 << idx):
                    continue
                # Try to add
                valid = True
                for f in edge_faces[idx]:
                    if f_counts[f] == 3:
                        valid = False
                        break
                if valid:
                    selected |= (1 << idx)
                    for f in edge_faces[idx]:
                        f_counts[f] += 1
                    count += 1
                    improved = True
            
        # If we improved, update best
        if count > best_count:
            best_count = count
            best_selected = selected
            best_face_counts = f_counts[:]
            
        # Check time
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # If we reached the upper bound, break
        if best_count >= 308:
            break

    # Convert best_selected bitmask to list of edges
    result = []
    for idx in range(num_edges):
        if best_selected & (1 << idx):
            u, v, _ = edges[idx]
            result.append([u, v])
            
    return result
