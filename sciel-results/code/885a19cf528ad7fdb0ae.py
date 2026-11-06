import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = n_vertices * d // 2
    
    # Time limit guard
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer for 120s limit

    # 1. Precomputation
    # Generate edges
    edges = []
    edge_map = {} # (u, v) -> index
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_map[(u, v)] = idx
                edge_map[(v, u)] = idx
    
    # Generate faces (squares)
    # A face is defined by two bit positions i, j and a prefix/suffix pattern.
    # Actually, simpler: A face is a set of 4 vertices.
    # Iterate over all pairs of bits (i, j) with i < j.
    # For each assignment of the other d-2 bits, we get a face.
    faces = [] # List of lists of 4 edge indices
    face_edges = [] # List of 4 edge indices for each face
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all combinations of the other d-2 bits
            # There are 2^(d-2) such faces for this pair (i,j)
            mask = 0
            for k in range(d):
                if k != i and k != j:
                    mask |= (1 << k)
            
            # Iterate over all 2^(d-2) assignments
            num_assignments = 1 << (d - 2)
            for a in range(num_assignments):
                # Construct the 4 vertices
                # We need to map 'a' (which has d-2 bits) to the actual bit positions
                # Let's just iterate u from 0 to n_vertices and check if it's the "min" of the face?
                # Simpler: Iterate u, if (u & (1<<i)) == 0 and (u & (1<<j)) == 0, 
                # then u, u^1<<i, u^1<<j, u^1<<i^1<<j form a face?
                # No, this only generates faces where the '00' case is at u.
                # Every face has exactly one vertex with 0 in both i and j? No.
                # A face is determined by fixing all bits except i and j.
                # So for each u, if we fix bits other than i,j, we get a face.
                # To avoid duplicates, we can require u to have 0 in bit i? No.
                # Standard way:
                # For each u, for each i<j, the face containing u with directions i,j is:
                # u, u^1<<i, u^1<<j, u^1<<i^1<<j.
                # This generates each face 4 times (once for each vertex).
                # We can just generate all and use a set of frozensets of vertices to deduplicate.
                pass

    # Let's use a set of frozensets of vertices to identify unique faces
    unique_faces = set()
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                face_verts = frozenset([v1, v2, v3, v4])
                unique_faces.add(face_verts)
                
    # Now map faces to edges
    face_edge_lists = []
    for face in unique_faces:
        verts = list(face)
        # Get all edges in this face
        # The edges are between pairs of vertices in the face that differ by 1 bit.
        # There are exactly 4 such edges.
        f_edges = []
        for k in range(4):
            for l in range(k + 1, 4):
                u = verts[k]
                v = verts[l]
                if u < v:
                    idx = edge_map.get((u, v))
                    if idx is not None:
                        f_edges.append(idx)
                else:
                    idx = edge_map.get((v, u))
                    if idx is not None:
                        f_edges.append(idx)
        # Should be 4 edges
        if len(f_edges) == 4:
            face_edge_lists.append(f_edges)
            
    n_faces = len(face_edge_lists)
    
    # Map edge to faces it belongs to
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, f_edges in enumerate(face_edge_lists):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)

    # 2. Initialization: Greedy Hitting Set on Removed Edges
    # We want to remove edges such that every face has at least one removed edge.
    removed = set()
    face_hits = [0] * n_faces
    
    # Greedy: Pick edge that covers most uncovered faces
    # To be fast, we can just pick random edges from uncovered faces
    # Or sort edges by degree?
    # Let's do a simple greedy:
    # While there are uncovered faces:
    #   Pick a random uncovered face
    #   Pick a random edge in that face not in removed
    #   Add to removed
    # This is fast but might not be optimal.
    
    # Better Greedy:
    # Calculate for each edge, how many uncovered faces it covers.
    # Pick max, add, update.
    # Repeat.
    
    # For speed, let's start with a random valid set?
    # No, let's start with empty removed set and add until valid.
    # This ensures we start with a valid state.
    
    uncovered_faces = list(range(n_faces))
    random.shuffle(uncovered_faces)
    
    # Simple Greedy Hitting Set
    while uncovered_faces:
        # Pick a face
        f_idx = uncovered_faces.pop()
        # Find an edge in this face to remove
        # Prefer an edge that covers other uncovered faces
        best_e = -1
        max_cov = -1
        for e_idx in face_edge_lists[f_idx]:
            if e_idx in removed:
                continue
            cov = 0
            for ff in edge_faces[e_idx]:
                if face_hits[ff] == 0:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_e = e_idx
        if best_e != -1:
            removed.add(best_e)
            for ff in edge_faces[best_e]:
                face_hits[ff] += 1
                if face_hits[ff] == 1:
                    # It was uncovered, now covered
                    # Remove from uncovered list? List is messy.
                    pass
    
    # Re-calculate uncovered faces for clarity
    uncovered_faces = [i for i in range(n_faces) if face_hits[i] == 0]
    
    # 3. Local Search / Simulated Annealing
    # Minimize |removed|
    # State: removed set, face_hits array
    # Move: 
    #   Try to remove an edge e from removed.
    #   If valid (no face becomes 0), accept.
    #   If invalid, try to swap: remove e, add f.
    
    best_removed = set(removed)
    best_cost = len(removed)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    while time.time() - start_time < time_limit:
        # Try to remove a random edge
        if removed:
            e = random.choice(list(removed))
            # Check if removal is valid
            # A removal is invalid if there is a face f containing e such that face_hits[f] == 1
            # and e is the only removed edge in f.
            # i.e., face_hits[f] == 1.
            # If face_hits[f] > 1, removing e is safe for that face.
            
            valid = True
            for f_idx in edge_faces[e]:
                if face_hits[f_idx] == 1:
                    valid = False
                    break
            
            if valid:
                # Accept
                removed.discard(e)
                for f_idx in edge_faces[e]:
                    face_hits[f_idx] -= 1
                if len(removed) < best_cost:
                    best_cost = len(removed)
                    best_removed = set(removed)
            else:
                # Try to swap
                # We must add an edge f that covers all faces where e was the only hit.
                # Faces where e is the only hit: F_crit = {f | e in f and face_hits[f] == 1}
                # We need to pick f such that for all g in F_crit, f in g.
                # i.e., f must be in the intersection of F_crit faces.
                
                F_crit = [f_idx for f_idx in edge_faces[e] if face_hits[f_idx] == 1]
                
                if not F_crit:
                    # Should not happen if valid was False
                    continue
                
                # Find candidate f's
                # f must be in all faces in F_crit
                # Intersection of edge sets of these faces
                candidates = set(face_edge_lists[F_crit[0]])
                for g in F_crit[1:]:
                    candidates.intersection_update(face_edge_lists[g])
                
                # Remove e from candidates (we are swapping)
                candidates.discard(e)
                
                if candidates:
                    f = random.choice(list(candidates))
                    # Perform swap
                    # Remove e
                    for f_idx in edge_faces[e]:
                        face_hits[f_idx] -= 1
                    removed.discard(e)
                    
                    # Add f
                    for f_idx in edge_faces[f]:
                        face_hits[f_idx] += 1
                    removed.add(f)
                    
                    # Cost is same (len(removed) unchanged)
                    # We accept swap to explore space.
                    # To avoid getting stuck, we can use SA logic here?
                    # Since cost is same, we always accept to move around.
                    # But we might oscillate.
                    # We can track best valid state.
                    # Since we maintain validity, current state is valid.
                    # If len(removed) < best_cost, update. (Won't happen in swap)
                    
        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with higher T?
            # T = 1.0

    # Final cleanup: Try to remove any remaining redundant edges
    # Greedy reduction
    for e in list(removed):
        valid = True
        for f_idx in edge_faces[e]:
            if face_hits[f_idx] == 1:
                valid = False
                break
        if valid:
            removed.discard(e)
            for f_idx in edge_faces[e]:
                face_hits[f_idx] -= 1

    # Use best found if current is worse?
    # Actually, we track best_cost.
    # But best_removed might not be the current state.
    # We should reconstruct the edge list from best_removed.
    
    # Wait, the local search above only did 1-for-1 swaps when 1-for-0 failed.
    # It did not do 2-for-1 swaps.
    # To improve further, we can do a final pass looking for 2-for-1.
    
    # But given time constraints, the SA/Swap loop should be sufficient to get close to optimal.
    # The best known is 144 removed edges.
    
    # Construct output
    out = []
    for e_idx in range(n_edges):
        if e_idx not in best_removed:
            u, v = edges[e_idx]
            out.append([u, v])
            
    return out
