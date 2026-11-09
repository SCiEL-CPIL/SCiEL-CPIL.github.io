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
