import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges_total = d * (1 << (d - 1))  # 448
    
    # Generate all edges and assign indices
    edges = []
    edge_index = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_index[(u, v)] = len(edges)
                edges.append((u, v))
    
    n_edges = len(edges)
    
    # Precompute for each edge, the 6 companion edges that form C4s with it
    # For edge (u,v) differing in bit i, for each j != i:
    #   w1 = u ^ (1<<j), w2 = v ^ (1<<j)
    #   companions: (u,w1), (v,w2), (w1,w2)
    edge_companions = [[] for _ in range(n_edges)]
    for e_idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            mask = 1 << j
            w1 = u ^ mask
            w2 = v ^ mask
            e1 = (min(u, w1), max(u, w1))
            e2 = min(v, w2), max(v, w2)
            e2 = (e2[0], e2[1])
            e3 = (min(w1, w2), max(w1, w2))
            edge_companions[e_idx].append((edge_index[e1], edge_index[e2], edge_index[e3]))
    
    # Conflict count: for each edge, how many of its companion edges are selected
    # When we add edge e, it conflicts with edge f if f is a companion of e
    # Actually, we need: when adding edge e, check if any C4 is completed
    # A C4 is completed if all 4 edges of some face are selected.
    # Edge e participates in 6 faces. For each face, the other 3 edges are companions.
    # So adding e creates a C4 iff there exists a face where all 3 companions are already selected.
    
    # Use a "conflict count" per edge: for each edge, count how many of the 6 faces
    # have 2 of the other 3 edges selected (so adding the 3rd would complete the face).
    # Actually simpler: maintain for each face how many of its 4 edges are selected.
    # If a face has 3 edges selected, the 4th edge cannot be added.
    
    # Let's precompute all faces (2D subcubes)
    # A face is determined by choosing 2 bits (i,j) and fixing the other 5 bits.
    # Each face has 4 vertices and 4 edges.
    faces = []  # list of sets of edge indices
    face_of_edge = [[] for _ in range(n_edges)]  # for each edge, list of face indices
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_ij = (1 << i) | (1 << j)
            # For each assignment of the other d-2 bits
            for mask in range(1 << d):
                if mask & mask_ij != 0:
                    continue
                # The 4 vertices of this face
                v0 = mask
                v1 = mask | (1 << i)
                v2 = mask | (1 << j)
                v3 = mask | (1 << i) | (1 << j)
                
                # The 4 edges
                e1 = (min(v0, v1), max(v0, v1))
                e2 = (min(v0, v2), max(v0, v2))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v3), max(v2, v3))
                
                face_edges = (edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4])
                face_idx = len(faces)
                faces.append(face_edges)
                for e_idx in face_edges:
                    face_of_edge[e_idx].append(face_idx)
    
    n_faces = len(faces)
    
    # Face count: number of selected edges in each face
    face_count = [0] * n_faces
    
    def can_add(e_idx):
        """Check if edge e_idx can be added without creating a C4."""
        for f_idx in face_of_edge[e_idx]:
            if face_count[f_idx] == 3:
                return False
        return True
    
    def add_edge(e_idx):
        """Add edge e_idx to the set, updating face counts."""
        selected[e_idx] = True
        for f_idx in face_of_edge[e_idx]:
            face_count[f_idx] += 1
    
    def remove_edge(e_idx):
        """Remove edge e_idx from the set, updating face counts."""
        selected[e_idx] = False
        for f_idx in face_of_edge[e_idx]:
            face_count[f_idx] -= 1
    
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_set = [False] * n_edges
    best_count = 0
    
    # Structured initialization: use parity-based construction
    # For a 7-cube, we can use a linear map over F2 to define which edges to select.
    # Edge in direction i between u and u^(1<<i) is selected if some function of u and i is true.
    # Try: select edge (u, u^e_i) if popcount(u) mod 2 == i mod 2 (or some variant)
    
    def try_structured_init():
        """Try various structured initializations."""
        nonlocal best_set, best_count
        
        # Variant 1: select edge i of u if (u & (1<<i)) == 0 and popcount(u) % 2 == 0
        # Actually let's try: for each direction i, select edge if the bit i of u is 0
        # and some condition on other bits
        
        # Simple structured: for each edge (u,v) in direction i, select if u has bit i = 0
        # This gives exactly half the edges in each direction = 224 edges
        # But this might create C4s.
        
        # Better: use a random linear map. For each direction i, define a linear function
        # L_i: F2^7 -> F2, and select edge in direction i at vertex u if L_i(u) = 0.
        # This gives ~half edges per direction.
        
        for trial in range(50):
            selected = [False] * n_edges
            face_count = [0] * n_faces
            count = 0
            
            # Random linear maps for each direction
            for i in range(d):
                # Random linear function: random 7-bit vector, dot product with u
                vec = random.getrandbits(d)
                # Ensure vec is not zero and not just the i-th bit
                while vec == 0 or vec == (1 << i):
                    vec = random.getrandbits(d)
                
                for u in range(n_vertices):
                    v = u ^ (1 << i)
                    if u >= v:
                        continue
                    # Compute dot product mod 2
                    dot = 0
                    for k in range(d):
                        if (vec >> k) & 1:
                            dot ^= (u >> k) & 1
                    if dot == 0:
                        e_idx = edge_index[(u, v)]
                        # Check if we can add
                        ok = True
                        for f_idx in face_of_edge[e_idx]:
                            if face_count[f_idx] == 3:
                                ok = False
                                break
                        if ok:
                            selected[e_idx] = True
                            for f_idx in face_of_edge[e_idx]:
                                face_count[f_idx] += 1
                            count += 1
            
            if count > best_count:
                best_set = selected[:]
                best_count = count
                return True
        return False
    
    try_structured_init()
    
    # Phase 1: Randomized greedy with conflict counts
    edge_order = list(range(n_edges))
    
    while time.time() < deadline:
        random.shuffle(edge_order)
        selected = [False] * n_edges
        face_count = [0] * n_faces
        count = 0
        
        for e_idx in edge_order:
            if can_add_with_local(face_count, face_of_edge, e_idx):
                selected[e_idx] = True
                for f_idx in face_of_edge[e_idx]:
                    face_count[f_idx] += 1
                count += 1
        
        if count > best_count:
            best_set = selected[:]
            best_count = count
            # If we got a good result, do local search
            if time.time() < deadline - 5:
                best_set, best_count = local_search(best_set, best_count, face_of_edge, face_count, deadline)
    
    # Convert to list of lists
    result = []
    for e_idx in range(n_edges):
        if best_set[e_idx]:
            u, v = edges[e_idx]
            result.append([u, v])
    
    return result

def can_add_with_local(face_count, face_of_edge, e_idx):
    """Check if edge can be added using local face counts."""
    for f_idx in face_of_edge[e_idx]:
        if face_count[f_idx] == 3:
            return False
    return True

def local_search(selected, count, face_of_edge, face_count_init, deadline):
    """Local search: try to improve by swapping edges."""
    face_count = face_count_init[:]
    improved = True
    iterations = 0
    
    while improved and time.time() < deadline - 2:
        improved = False
        iterations += 1
        
        # Try removing one edge and adding another
        for e_remove in range(len(selected)):
            if not selected[e_remove]:
                continue
            if time.time() > deadline - 2:
                break
            
            # Remove e_remove
            selected[e_remove] = False
            for f_idx in face_of_edge[e_remove]:
                face_count[f_idx] -= 1
            count -= 1
            
            # Try to add some unselected edge
            added = False
            for e_add in range(len(selected)):
                if selected[e_add]:
                    continue
                ok = True
                for f_idx in face_of_edge[e_add]:
                    if face_count[f_idx] == 3:
                        ok = False
                        break
                if ok:
                    selected[e_add] = True
                    for f_idx in face_of_edge[e_add]:
                        face_count[f_idx] += 1
                    count += 1
                    added = True
                    improved = True
                    break
            
            if not added:
                # Put e_remove back
                selected[e_remove] = True
                for f_idx in face_of_edge[e_remove]:
                    face_count[f_idx] += 1
                count += 1
            else:
                # Keep e_remove removed and e_add added
                pass
        
        # Also try just adding an edge if possible
        for e_add in range(len(selected)):
            if selected[e_add]:
                continue
            ok = True
            for f_idx in face_of_edge[e_add]:
                if face_count[f_idx] == 3:
                    ok = False
                    break
            if ok:
                selected[e_add] = True
                for f_idx in face_of_edge[e_add]:
                    face_count[f_idx] += 1
                count += 1
                improved = True
    
    return selected, count
