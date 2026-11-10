import time
import random

def construct():
    """
    Construct a C4-free edge set in Q7.
    Uses randomized local search (hill climbing with swaps) to maximize the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448
    
    # 1. Precomputation
    # Generate all edges
    edges = []
    edge_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_idx[(u, v)] = idx
    
    # Generate all faces (4-cycles)
    # A face is defined by two distinct dimensions i, j and a base vertex u with bits i,j = 0 (or just any base, but we need unique representation)
    # Vertices of face: u, u^(1<<i), u^(1<<j), u^(1<<i)^(1<<j)
    # To ensure unique face representation, we can require u has 0 in bits i and j.
    faces = []
    face_idx = {}
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            mask_ij = mask_i | mask_j
            # Iterate over all vertices u such that (u & mask_ij) == 0
            # There are 2^(d-2) such vertices
            for u in range(n_vertices):
                if (u & mask_ij) == 0:
                    v1 = u ^ mask_i
                    v2 = u ^ mask_j
                    v3 = u ^ mask_ij
                    
                    e1 = edge_idx.get((min(u, v1), max(u, v1)))
                    e2 = edge_idx.get((min(u, v2), max(u, v2)))
                    e3 = edge_idx.get((min(v1, v3), max(v1, v3)))
                    e4 = edge_idx.get((min(v2, v3), max(v2, v3)))
                    
                    if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                        face = (e1, e2, e3, e4)
                        faces.append(face)
                        face_idx[face] = len(faces) - 1
    
    n_faces = len(faces)
    
    # For each edge, list the indices of faces containing it
    edge_to_faces = [[] for _ in range(n_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_to_faces[e_idx].append(f_idx)
    
    # 2. State Representation
    selected = [False] * n_edges
    face_counts = [0] * n_faces
    
    def is_valid_add(e_idx):
        # Check if adding e_idx keeps all face counts < 4
        for f_idx in edge_to_faces[e_idx]:
            if face_counts[f_idx] >= 4:
                return False
        return True
    
    def add_edge(e_idx):
        selected[e_idx] = True
        for f_idx in edge_to_faces[e_idx]:
            face_counts[f_idx] += 1
    
    def remove_edge(e_idx):
        selected[e_idx] = False
        for f_idx in edge_to_faces[e_idx]:
            face_counts[f_idx] -= 1
    
    def try_add_random():
        # Try to add a random unselected edge
        # To be fast, we can just pick a random index and check if it's unselected
        # Since n_edges is small (448), this is fine.
        for _ in range(10): # Try a few times
            e_idx = random.randint(0, n_edges - 1)
            if not selected[e_idx]:
                if is_valid_add(e_idx):
                    add_edge(e_idx)
                    return True
        return False

    def try_swap_random():
        # Pick random e_out in selected, e_in not in selected
        # If swap is valid, perform it
        # Validity of swap:
        # 1. Remove e_out
        # 2. Check if e_in can be added
        # 3. If yes, add e_in
        
        # Optimization: 
        # Removing e_out decreases counts for faces containing e_out.
        # Adding e_in increases counts for faces containing e_in.
        # We need to ensure that for all faces F containing e_in, 
        # the count after removal of e_out and addition of e_in is < 4.
        
        # Let F be a face containing e_in.
        # New count = face_counts[F] - (1 if e_out in F else 0) + 1
        # We need New count < 4 => face_counts[F] - (1 if e_out in F else 0) < 3
        
        # We pick e_out and e_in randomly.
        # To make it efficient, we can just pick them and check.
        
        # Pick e_out
        # We need a list of selected edges to pick from efficiently?
        # Or just pick random index and check if selected.
        e_out = -1
        e_in = -1
        
        # Find a valid e_out
        while e_out == -1:
            candidate = random.randint(0, n_edges - 1)
            if selected[candidate]:
                e_out = candidate
        
        # Find a valid e_in
        while e_in == -1:
            candidate = random.randint(0, n_edges - 1)
            if not selected[candidate]:
                e_in = candidate
        
        # Check validity
        # Temporarily simulate
        valid = True
        for f_idx in edge_to_faces[e_in]:
            # Count of e_out in this face?
            # e_out is in f_idx if e_out is one of the 4 edges of face f_idx
            # We can check this by seeing if e_out is in the face tuple
            # But face tuple is (e1, e2, e3, e4). 
            # Is e_out in faces[f_idx]?
            # This is O(1) if we precompute a bitmask or set for each face?
            # Or just check if e_out is in the list of edges of the face.
            # Since we have edge_to_faces, we can precompute face_to_edges as a set for O(1) lookup?
            # n_faces is 336. Storing sets is fine.
            
            # Let's use a simple check: 
            # We know e_in is in f_idx.
            # We need to know if e_out is in f_idx.
            # We can precompute a boolean matrix or just check membership in the face tuple.
            # Tuple membership is O(4) which is constant.
            
            face_edges = faces[f_idx]
            e_out_in_face = (e_out == face_edges[0] or e_out == face_edges[1] or e_out == face_edges[2] or e_out == face_edges[3])
            
            current_count = face_counts[f_idx]
            new_count = current_count - (1 if e_out_in_face else 0) + 1
            if new_count >= 4:
                valid = False
                break
        
        if valid:
            remove_edge(e_out)
            add_edge(e_in)
            return True
        return False

    # 3. Main Search Loop
    best_size = 0
    
    # Run multiple restarts or a single long run?
    # Given the time limit, a single long run with swaps is usually good.
    # Or a few restarts.
    
    # Let's do a loop until time runs out.
    # Start with a random greedy initialization
    random.seed(42) # Fixed seed for reproducibility? No, random is better for exploration.
    # Actually, let's use a fixed seed to ensure consistent behavior if needed, but random is fine.
    
    # Initialization: Random Greedy
    order = list(range(n_edges))
    random.shuffle(order)
    for e_idx in order:
        if is_valid_add(e_idx):
            add_edge(e_idx)
    
    current_size = sum(selected)
    best_size = current_size
    best_selected = selected[:]
    best_face_counts = face_counts[:]
    
    stuck_counter = 0
    
    while time.time() < deadline:
        # Try to improve size
        if try_add_random():
            current_size += 1
            stuck_counter = 0
            if current_size > best_size:
                best_size = current_size
                best_selected = selected[:]
                best_face_counts = face_counts[:]
        else:
            # Try to swap to potentially enable future adds
            if try_swap_random():
                stuck_counter = 0
                # Size unchanged, but configuration changed
            else:
                stuck_counter += 1
                # If stuck for a long time, restart?
                # Or just continue.
                if stuck_counter > 1000:
                    # Restart with a new random greedy
                    # Reset state
                    selected = [False] * n_edges
                    face_counts = [0] * n_faces
                    order = list(range(n_edges))
                    random.shuffle(order)
                    for e_idx in order:
                        if is_valid_add(e_idx):
                            add_edge(e_idx)
                    current_size = sum(selected)
                    if current_size > best_size:
                        best_size = current_size
                        best_selected = selected[:]
                        best_face_counts = face_counts[:]
                    stuck_counter = 0
    
    # Return the best found set
    result = []
    for i in range(n_edges):
        if best_selected[i]:
            result.append(list(edges[i]))
            
    return result

