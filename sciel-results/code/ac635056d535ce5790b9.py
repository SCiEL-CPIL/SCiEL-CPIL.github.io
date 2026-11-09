import time
import random
import sys

def _precompute():
    """
    Precompute edges and faces of Q7.
    Returns:
    edges: list of (u, v) tuples
    faces: list of frozensets of edge indices
    edge_to_faces: list of sets of face indices for each edge
    """
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
    
    # Generate all faces
    # A face is defined by two distinct dimensions i, j and a base vertex u
    # such that bits i and j are 0 in u. This generates each face exactly once.
    faces = []
    edge_to_faces = [set() for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n):
                # Ensure u is the "base" corner with 0s in bits i and j
                if (u & (1 << i)) or (u & (1 << j)):
                    continue
                
                v0 = u
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                # Edges of the face: (v0,v1), (v1,v3), (v3,v2), (v2,v0)
                e1 = edge_to_idx.get((min(v0, v1), max(v0, v1)))
                e2 = edge_to_idx.get((min(v1, v3), max(v1, v3)))
                e3 = edge_to_idx.get((min(v3, v2), max(v3, v2)))
                e4 = edge_to_idx.get((min(v2, v0), max(v2, v0)))
                
                if e1 is None or e2 is None or e3 is None or e4 is None:
                    continue
                
                face_edges = frozenset([e1, e2, e3, e4])
                f_idx = len(faces)
                faces.append(face_edges)
                
                for e in face_edges:
                    edge_to_faces[e].add(f_idx)
                    
    return edges, faces, edge_to_faces

_EDGES, _FACES, _EDGE_TO_FACES = _precompute()
_NUM_EDGES = len(_EDGES)
_NUM_FACES = len(_FACES)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_removed_size = _NUM_EDGES
    best_removed_set = None
    
    # We will run multiple trials with different random seeds for the local search swaps
    # The greedy initialization is deterministic, so we need randomness in the search phase
    # to escape local minima.
    
    num_trials = 100 # Adjust based on speed
    
    for trial_seed in range(num_trials):
        if time.time() > deadline:
            break
            
        rng = random.Random(trial_seed)
        
        # Phase 1: Greedy Hitting Set Initialization
        # We want to find a set R of edges such that every face has at least one edge in R.
        # Greedy: repeatedly pick the edge that covers the most uncovered faces.
        
        uncovered_faces = set(range(_NUM_FACES))
        removed = set()
        
        # Precompute face counts for each edge to speed up greedy?
        # Or just calculate on the fly. 448 edges, 672 faces.
        # Calculating coverage for all 448 edges each step is 448 * 4 (avg faces per edge is small)
        # Actually, each edge is in C(6,1) * 2^(7-3) = 6 * 8 = 48 faces? No.
        # An edge is defined by dimension i and vertex u (bit i is 0, say).
        # Faces containing this edge:
        # The edge is in dimension i. A face is defined by dims (i, j).
        # For each j != i (6 choices), the face is fixed if the other bits are fixed.
        # The vertex u has 6 other bits. 2^6 = 64 faces?
        # Let's check: edge (u, u^i). u has bit i=0.
        # Face (i, j) containing this edge must have base vertex w such that w_i=0, w_j=0.
        # The edge connects w^i and w. So w must be u or u^j?
        # If the edge is (u, u^i), it is in faces defined by (i, j) for all j!=i.
        # For a specific j, the face corners are w, w^i, w^j, w^i^j.
        # The edge (u, u^i) is one of these if {w, w^i} = {u, u^i}.
        # So w = u or w = u^i.
        # Since w_j=0, if u_j=0, w=u is valid. If u_j=1, w=u^i is valid (since i!=j, u^i has bit j same as u).
        # So for each j != i, there is exactly 1 face.
        # Total faces per edge = 6.
        
        # So checking coverage is fast.
        
        while uncovered_faces:
            if time.time() > deadline - 5.0:
                break
                
            best_edge = -1
            max_cov = -1
            
            # Optimization: Only check edges that are in at least one uncovered face.
            # We can maintain a list of candidate edges.
            # Or just iterate all edges. 448 * 6 operations is trivial.
            
            for e in range(_NUM_EDGES):
                cov = 0
                for f in _EDGE_TO_FACES[e]:
                    if f in uncovered_faces:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_edge = e
                    
            if best_edge == -1:
                # Should not happen if uncovered_faces is not empty
                break
                
            removed.add(best_edge)
            
            # Update uncovered faces
            for f in _EDGE_TO_FACES[best_edge]:
                if f in uncovered_faces:
                    uncovered_faces.remove(f)
                    
        # Initial solution size
        current_size = len(removed)
        
        # Phase 2: Local Search
        # We have a valid hitting set R (removed).
        # Goal: Minimize |R|.
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # 1. Deletion Step
            # Try to remove edges from R if they are redundant.
            # An edge e is redundant if for every face f containing e, there is another edge e' in R \cap face[f].
            
            # We can iterate through edges in R.
            # To be efficient, we can check faces.
            # Let's iterate through faces. If a face has exactly 1 removed edge, that edge is "critical".
            # If a face has >1 removed edges, any of them could potentially be removed (if not critical in other faces).
            
            # Simpler: Iterate through e in R.
            # For each e, check if it is critical in any face.
            # Critical means: exists face f such that e is the ONLY removed edge in f.
            
            # We can precompute for each edge e, the faces it is in.
            # For each such face f, count removed edges in f.
            
            # To speed up, we can maintain a count of removed edges per face.
            # But updating it on swap is complex.
            # Given small size, we can just recompute counts for faces involved in the candidate edge.
            
            # Let's try a random order of deletion candidates to break symmetry.
            # Or just iterate.
            
            # Optimization: If we delete an edge, we might create new redundancies.
            # So we loop until no deletion is possible.
            
            deleted_any = False
            # Use a list to allow modification during iteration? No, just check all.
            for e in list(removed):
                if time.time() > deadline - 2.0:
                    break
                    
                is_critical = False
                for f in _EDGE_TO_FACES[e]:
                    # Count removed edges in f
                    count = 0
                    for ee in _FACES[f]:
                        if ee in removed:
                            count += 1
                            if count > 1:
                                break
                    if count == 1:
                        is_critical = True
                        break
                        
                if not is_critical:
                    removed.discard(e)
                    improved = True
                    deleted_any = True
                    # Restart the loop? Or continue?
                    # Continuing is fine, but new redundancies might appear.
                    # The outer 'while improved' loop will catch them.
                    
            if deleted_any:
                continue # Restart the while loop to check for more deletions
            
            # 2. Swap Step
            # If no deletion is possible, try to swap an edge to enable future deletions.
            # Pick a random edge e from R.
            # Find its critical faces (faces where e is the only removed edge).
            # Let C be the set of critical faces for e.
            # If C is empty, e is not critical anywhere, so it should have been deleted in step 1.
            # So C is non-empty.
            # To swap e with e', e' must hit all faces in C.
            # i.e., e' must be in the intersection of the edge sets of all faces in C.
            
            if removed:
                e = rng.choice(list(removed))
                
                critical_faces = []
                for f in _EDGE_TO_FACES[e]:
                    count = 0
                    for ee in _FACES[f]:
                        if ee in removed:
                            count += 1
                            if count > 1:
                                break
                    if count == 1:
                        critical_faces.append(f)
                        
                if critical_faces:
                    # Find intersection of edges in critical faces
                    # Start with edges of the first critical face
                    candidates = set(_FACES[critical_faces[0]])
                    for f in critical_faces[1:]:
                        candidates.intersection_update(_FACES[f])
                    
                    # Remove e itself and any other removed edges (they are already in R, swapping with them is useless or invalid)
                    # Actually, if e' is in R, swapping e with e' just changes R by removing e and adding e' (which is already there).
                    # So e' must NOT be in R.
                    candidates -= removed
                    
                    if candidates:
                        e_new = rng.choice(list(candidates))
                        # Perform swap
                        removed.discard(e)
                        removed.add(e_new)
                        improved = True
                        # The loop will restart, checking for deletions again.
                        
        # End of local search for this trial
        
        if len(removed) < best_removed_size:
            best_removed_size = len(removed)
            best_removed_set = set(removed)
            
    # Construct the final list of edges
    if best_removed_set is None:
        # Fallback: return empty set if something went wrong? 
        # But greedy should always produce a valid set.
        best_removed_set = set()
        
    result_edges = []
    for i in range(_NUM_EDGES):
        if i not in best_removed_set:
            result_edges.append(list(_EDGES[i]))
            
    return result_edges

# Allow running as a script for testing
if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Edges: {len(res)}")
    # Verify
    # Re-run precompute to verify
    e, f, e2f = _precompute()
    removed = set()
    kept = set()
    for i, (u, v) in enumerate(e):
        if [u, v] in res or [v, u] in res:
            kept.add(i)
        else:
            removed.add(i)
            
    valid = True
    for face in f:
        if len(face & removed) == 0:
            valid = False
            break
    print(f"Valid: {valid}")
    print(f"Removed: {len(removed)}, Kept: {len(kept)}")
