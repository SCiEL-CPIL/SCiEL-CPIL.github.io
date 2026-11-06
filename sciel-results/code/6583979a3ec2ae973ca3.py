import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # Map edges to indices 0..447
    # Edge (u, v) where u < v and u^v is a power of 2
    # We can just generate all edges and assign IDs
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
                
    # Precompute faces
    # A face is determined by two dimensions i, j and the other 5 bits fixed.
    # There are C(7,2) = 21 pairs of dimensions.
    # For each pair, there are 2^(7-2) = 32 faces.
    # Total 672 faces.
    
    face_edges = [] # list of lists of 4 edge IDs
    edge_faces = [[] for _ in range(n_edges)]
    
    # Generate faces
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all assignments of the other bits
            # The bits not in {i, j} are fixed.
            # Let's iterate over all vertices, and group them by their projection on bits not in {i,j}?
            # Easier: iterate over all 2^(d-2) combinations of the other bits.
            # Let's just iterate over all vertices u.
            # For a given u, the face containing u with dimensions i, j has vertices:
            # u, u^1<<i, u^1<<j, u^1<<i^1<<j
            # But we need to identify the face uniquely.
            # A face is uniquely identified by (i, j, fixed_bits).
            # fixed_bits is the value of u with bits i and j cleared.
            
            # Let's collect faces in a set to avoid duplicates?
            # No, iterating u from 0 to 127 will visit each face 4 times.
            # We can just use a set of tuples of sorted edge IDs to represent the face?
            # Or just generate them systematically.
            
            # Systematic generation:
            # For each pair (i, j):
            #   For each mask of the other d-2 bits:
            #     Construct the 4 vertices.
            
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                # Construct base vertex
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                # The 4 vertices of the face
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                # The 4 edges
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                face_id = len(face_edges)
                face_edges.append([e1, e2, e3, e4])
                
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
                    
    n_faces = len(face_edges)
    
    best_k = n_edges # Start with all edges removed (worst case)
    best_removed = set(range(n_edges))
    
    start_time = time.time()
    timeout = 110 # Leave 10s margin
    
    while time.time() - start_time < timeout:
        # Phase 1: Greedy Set Cover
        removed = set()
        face_hit_count = [0] * n_faces
        unhit_count = n_faces
        
        # Helper to add edge
        def add_edge(e):
            nonlocal unhit_count
            if e in removed:
                return
            removed.add(e)
            for f in edge_faces[e]:
                if face_hit_count[f] == 0:
                    unhit_count -= 1
                face_hit_count[f] += 1

        # Helper to remove edge
        def remove_edge(e):
            nonlocal unhit_count
            if e not in removed:
                return
            removed.remove(e)
            for f in edge_faces[e]:
                face_hit_count[f] -= 1
                if face_hit_count[f] == 0:
                    unhit_count += 1

        # Greedy
        # To speed up, we can pick edges that cover max unhit faces.
        # But calculating gain for all 448 edges every step is 448*6 ops.
        # 672 steps max. 672 * 448 * 6 ~ 1.8M ops. Very fast.
        
        while unhit_count > 0:
            # Find edge with max gain
            best_gain = -1
            best_e = -1
            # Optimization: Only consider edges that are incident to unhit faces?
            # Or just scan all.
            # Scanning all is simpler.
            for e in range(n_edges):
                if e in removed:
                    continue
                gain = 0
                for f in edge_faces[e]:
                    if face_hit_count[f] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_e = e
            
            if best_e == -1:
                break # Should not happen if unhit_count > 0
            
            add_edge(best_e)
            
        # Phase 2: Local Search (Minimize)
        # Try to remove edges
        # Shuffle order to break symmetry
        candidates = list(removed)
        random.shuffle(candidates)
        
        improved = True
        while improved:
            improved = False
            random.shuffle(candidates)
            for e in candidates:
                if e not in removed:
                    continue
                # Check if we can remove e
                # e is in removed.
                # If any face f in edge_faces[e] has face_hit_count[f] == 1, we cannot remove e.
                can_remove = True
                for f in edge_faces[e]:
                    if face_hit_count[f] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    remove_edge(e)
                    improved = True
                    # Restart shuffle? Or continue?
                    # Continuing is fine, but we might miss opportunities.
                    # Restarting is safer for local minima escape?
                    # Let's just break and restart the while loop.
                    break
        
        if len(removed) < best_k:
            best_k = len(removed)
            best_removed = set(removed)
            
        # If we hit the theoretical lower bound or something, break?
        # Lower bound for hitting set:
        # Each edge hits 6 faces.
        # Total faces 672.
        # 672 / 6 = 112.
        # But overlaps exist.
        # 140 is a good target.
        
        if best_k <= 140:
            break

    # Construct the output list of edges
    # We kept edges NOT in best_removed
    result = []
    for i, (u, v) in enumerate(edges):
        if i not in best_removed:
            result.append([u, v])
            
    return result
