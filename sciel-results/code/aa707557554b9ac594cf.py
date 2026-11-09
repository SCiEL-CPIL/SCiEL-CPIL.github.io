import time
import random
import sys

def _precompute():
    """Precompute all edges and faces."""
    d = 7
    n = 2 ** d
    edges = []
    edge_to_idx = {}
    
    # Generate all edges
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx
                
    num_edges = len(edges)
    
    # Generate all faces (2D squares)
    # A face is defined by two distinct dimensions i, j and a vertex u such that
    # u, u^i, u^j, u^i^j are the corners.
    # To avoid duplicates, we can iterate u and ensure u is the "smallest" in some sense, 
    # or just generate all and deduplicate by edge sets.
    # Total faces in Q7: C(7,2) * 2^(7-2) = 21 * 32 = 672 faces.
    
    faces = []
    face_to_edges = [] # List of list of edge indices
    edge_to_faces = [[] for _ in range(num_edges)]
    
    seen_faces = set()
    
    for u in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                # Corners
                c0 = u
                c1 = u ^ (1 << i)
                c2 = u ^ (1 << j)
                c3 = u ^ (1 << i) ^ (1 << j)
                
                # Get edge indices for the 4 edges of this face
                e1 = edge_to_idx.get((min(c0, c1), max(c0, c1)))
                e2 = edge_to_idx.get((min(c1, c3), max(c1, c3)))
                e3 = edge_to_idx.get((min(c3, c2), max(c3, c2)))
                e4 = edge_to_idx.get((min(c2, c0), max(c2, c0)))
                
                if e1 is None or e2 is None or e3 is None or e4 is None:
                    continue
                
                # Create a canonical representation of the face to deduplicate
                face_edges = tuple(sorted([e1, e2, e3, e4]))
                
                if face_edges in seen_faces:
                    continue
                    
                seen_faces.add(face_edges)
                face_idx = len(faces)
                faces.append(face_edges)
                face_to_edges.append(face_edges)
                
                for e in face_edges:
                    edge_to_faces[e].append(face_idx)
                    
    return edges, face_to_edges, edge_to_faces

# Precompute once at module load
_EDGES, _FACES, _EDGE_TO_FACES = _precompute()
_NUM_EDGES = len(_EDGES)
_NUM_FACES = len(_FACES)

def _run_one_trial(deadline, seed):
    random.seed(seed)
    
    # State: 1 if edge is present, 0 otherwise
    present = [1] * _NUM_EDGES
    # Count of present edges in each face
    face_count = [0] * _NUM_FACES
    
    # Initialize face_count
    for f_idx in range(_NUM_FACES):
        # All edges present initially, so count is 4
        face_count[f_idx] = 4
        
    # List of edges to remove? No, we remove one by one.
    # We need to efficiently find faces with count 4.
    # And for each edge, how many faces with count 4 it is in.
    
    # face_count is 4 for all initially.
    # We will remove edges until no face has count 4.
    
    # To speed up, we can maintain a list of "active" faces (count==4).
    # But updating this list dynamically is tricky.
    # Given small size (672 faces), we can just scan or maintain a counter.
    
    # Let's use a simpler approach:
    # While there exists a face with count 4:
    #   Pick a random face with count 4.
    #   Pick a random edge from that face.
    #   Remove it.
    #   Update counts.
    
    # Optimization: Instead of random face, pick the edge that is in the most "full" faces.
    
    full_faces = set(range(_NUM_FACES))
    
    while full_faces:
        if time.time() > deadline:
            break
            
        # Pick a face from full_faces
        # To be greedy, we could pick the edge with highest "full-degree".
        # But scanning all edges every time is O(E * F).
        # Let's just pick a random full face, then a random edge in it.
        # This is O(1) per removal.
        
        f_idx = random.choice(list(full_faces))
        face_edges = _FACES[f_idx]
        
        # Pick an edge to remove from this face
        # Heuristic: Remove the edge that is in the most other full faces.
        best_edge = -1
        best_score = -1
        
        for e_idx in face_edges:
            score = 0
            for f2 in _EDGE_TO_FACES[e_idx]:
                if f2 in full_faces:
                    score += 1
            if score > best_score:
                best_score = score
                best_edge = e_idx
                
        if best_edge == -1:
            # Should not happen if face is full
            break
            
        # Remove best_edge
        present[best_edge] = 0
        
        # Update face counts
        for f2 in _EDGE_TO_FACES[best_edge]:
            face_count[f2] -= 1
            if face_count[f2] == 3 and f2 in full_faces:
                full_faces.remove(f2)
                
    # Now we have a valid C4-free set.
    # Phase 2: Local Search to add back edges
    # We can try to add edges that were removed.
    # An edge e can be added if no face containing e has count 3 (which would become 4).
    # i.e., for all f in edge_to_faces[e], face_count[f] < 3.
    
    removed_edges = [i for i in range(_NUM_EDGES) if present[i] == 0]
    
    # Shuffle removed edges for randomization
    random.shuffle(removed_edges)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e_idx in removed_edges:
            if present[e_idx]:
                continue
                
            # Check if we can add e_idx
            can_add = True
            for f_idx in _EDGE_TO_FACES[e_idx]:
                if face_count[f_idx] == 3:
                    can_add = False
                    break
                    
            if can_add:
                present[e_idx] = 1
                for f_idx in _EDGE_TO_FACES[e_idx]:
                    face_count[f_idx] += 1
                    # It becomes 4? No, it was 2 or less, so becomes 3 or less.
                    # Wait, if it was 3, we wouldn't add. So it becomes <= 3.
                    # So it never becomes a full face (4) by itself.
                improved = True
                # Remove from removed_edges list? Or just mark present.
                # We iterate through the list, so if we add it, it's present.
                
    # Count result
    count = sum(present)
    result_edges = []
    for i in range(_NUM_EDGES):
        if present[i]:
            result_edges.append(list(_EDGES[i]))
            
    return count, result_edges

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    best_count = 0
    best_result = []
    
    # Run multiple trials with different seeds
    # The greedy deletion is stochastic due to random face/edge selection if we implement it that way.
    # In my current _run_one_trial, I used a heuristic for edge selection (highest score) but random face.
    # Let's make it fully random to explore different local optima.
    
    # Modify _run_one_trial to accept a 'randomness' factor or just rely on seed.
    # The current _run_one_trial uses random.choice for face, and a deterministic heuristic for edge.
    # Let's add randomness to edge selection too.
    
    # Revised _run_one_trial logic for randomness:
    # Instead of best_edge, pick random edge from face_edges.
    # This allows different structures.
    
    num_trials = 500 # Adjust based on speed
    
    for i in range(num_trials):
        if time.time() > deadline:
            break
            
        # Slight modification: Use random edge selection in the greedy phase
        # I will implement a local version of the trial to allow random edge picking.
        
        present = [1] * _NUM_EDGES
        face_count = [4] * _NUM_FACES
        full_faces = set(range(_NUM_FACES))
        
        while full_faces:
            if time.time() > deadline:
                break
            f_idx = random.choice(list(full_faces))
            face_edges = _FACES[f_idx]
            
            # Random edge selection
            e_idx = random.choice(face_edges)
            
            present[e_idx] = 0
            for f2 in _EDGE_TO_FACES[e_idx]:
                face_count[f2] -= 1
                if face_count[f2] == 3 and f2 in full_faces:
                    full_faces.remove(f2)
                    
        # Local Search: Add back edges
        removed_edges = [i for i in range(_NUM_EDGES) if present[i] == 0]
        random.shuffle(removed_edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for e_idx in removed_edges:
                if present[e_idx]:
                    continue
                can_add = True
                for f_idx in _EDGE_TO_FACES[e_idx]:
                    if face_count[f_idx] == 3:
                        can_add = False
                        break
                if can_add:
                    present[e_idx] = 1
                    for f_idx in _EDGE_TO_FACES[e_idx]:
                        face_count[f_idx] += 1
                    improved = True
                    
        count = sum(present)
        if count > best_count:
            best_count = count
            best_result = [[u, v] for i, (u, v) in enumerate(_EDGES) if present[i]]
            
    return best_result
