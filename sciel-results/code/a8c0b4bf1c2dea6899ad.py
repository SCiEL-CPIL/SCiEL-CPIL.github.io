import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges_total = n_vertices * d // 2
    
    # Time limit guard
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for 120s limit (or 150s if that's the real limit)
    
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
    
    n_edges = len(edges)
    
    # Generate faces (squares)
    # A face is defined by fixing d-2 bits and varying the other 2.
    # We represent a face by the set of 4 edge indices it contains.
    # To avoid duplicates, we iterate over all pairs of bits (i, j) with i < j.
    # For each assignment of the other d-2 bits, we get a unique face.
    
    face_edge_lists = []
    seen_faces = set()
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all assignments of the other bits
            # We can just iterate u from 0 to n_vertices-1 and check if bits i and j are 0?
            # No, that only generates half the faces for a fixed (i,j) pair if we're not careful.
            # Actually, for a fixed pair (i,j), there are 2^(d-2) faces.
            # A face is uniquely identified by the values of bits NOT in {i,j}.
            # Let's iterate over all u, and for each u, consider the face spanned by directions i and j.
            # The vertices are u, u^1<<i, u^1<<j, u^1<<i^1<<j.
            # This generates each face 4 times (once for each vertex u in the face).
            # To avoid duplicates, we can require that u has bit i = 0 and bit j = 0?
            # No, that would miss faces where the "base" vertex doesn't have 0s.
            # Wait, every face has exactly one vertex with bit i=0 and bit j=0? No.
            # In a 2D square, the vertices are (0,0), (1,0), (0,1), (1,1) relative to the face.
            # So yes, for any fixed face, there is exactly one vertex where both i and j are 0 (relative to the face's local coordinates, but in global coordinates, it's just one of the 4 vertices).
            # Actually, if we fix the other d-2 bits, there are 4 combinations for bits i and j.
            # Only one of these 4 has bit i=0 and bit j=0? No, that's not right.
            # Let's just use a set of frozensets of edge indices to deduplicate.
            
            for u in range(n_vertices):
                # Check if u is the "canonical" representative of this face for this (i,j) pair?
                # Let's just generate the face and check if we've seen it.
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                # Get edge indices
                e1 = edge_map.get((min(v1, v2), max(v1, v2)))
                e2 = edge_map.get((min(v1, v3), max(v1, v3)))
                e3 = edge_map.get((min(v2, v4), max(v2, v4)))
                e4 = edge_map.get((min(v3, v4), max(v3, v4)))
                
                if e1 is None or e2 is None or e3 is None or e4 is None:
                    continue
                
                face_key = frozenset([e1, e2, e3, e4])
                if face_key not in seen_faces:
                    seen_faces.add(face_key)
                    face_edge_lists.append([e1, e2, e3, e4])
    
    n_faces = len(face_edge_lists)
    
    # Map edge to faces it belongs to
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, f_edges in enumerate(face_edge_lists):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)
    
    # 2. Initialization: Greedy Hitting Set
    # We want to remove edges such that every face has at least one removed edge.
    # State: removed set, face_hits array (count of removed edges in each face)
    removed = set()
    face_hits = np.zeros(n_faces, dtype=np.int8)
    
    # Greedy: While there are uncovered faces, pick an edge that covers the most uncovered faces.
    # For speed, we can approximate or use a simpler heuristic.
    # Let's use a randomized greedy:
    # 1. Start with all faces uncovered.
    # 2. Pick a random uncovered face.
    # 3. Pick an edge from that face that covers the most other uncovered faces.
    # 4. Add to removed, update face_hits.
    
    # To speed this up, we can maintain a list of uncovered faces.
    # But updating "most uncovered faces covered" is expensive.
    # Let's try a simpler approach: Random Greedy.
    # Pick random edge, if it covers any uncovered face, add it.
    # This is O(n_edges * avg_face_degree).
    
    # Better: 
    # Initialize face_hits to 0.
    # While any face_hits == 0:
    #   Pick a random face f with face_hits[f] == 0.
    #   Pick a random edge e in f.
    #   Add e to removed.
    #   Update face_hits for all faces containing e.
    
    # This is fast but might not be optimal.
    # Let's try to be slightly smarter:
    # For the chosen face, pick the edge that appears in the most OTHER uncovered faces.
    
    uncovered_faces = list(range(n_faces))
    random.shuffle(uncovered_faces)
    
    # We need a way to quickly find uncovered faces.
    # A list is fine if we filter it.
    
    while uncovered_faces:
        # Pick a random uncovered face
        f_idx = uncovered_faces.pop()
        if face_hits[f_idx] > 0:
            continue # Already covered by a previous choice
        
        # Find best edge in this face
        best_e = -1
        max_cov = -1
        for e_idx in face_edge_lists[f_idx]:
            if e_idx in removed:
                continue
            # Count how many uncovered faces this edge covers
            # We can just count total faces it covers, since we only add to uncovered ones effectively?
            # No, we need to know if it covers OTHER uncovered faces.
            # Let's just count total faces it touches.
            cov = len(edge_faces[e_idx])
            if cov > max_cov:
                max_cov = cov
                best_e = e_idx
        
        if best_e != -1:
            removed.add(best_e)
            for ff in edge_faces[best_e]:
                face_hits[ff] += 1
            
            # Update uncovered_faces list?
            # Instead of maintaining a list, we can just re-scan or use a different structure.
            # Given n_faces is 448, re-scanning is cheap.
            # But we are popping from a list. Let's just restart the loop with a new list?
            # No, that's inefficient.
            # Let's just keep the list and ignore covered faces when popping.
            # But the list shrinks. If we cover many faces, the list might become stale.
            # Let's just re-generate the list of uncovered faces every time?
            # n_faces is small (448). 448 * 448 operations is ~200k, which is fast.
            # But we are doing this in a loop that runs ~144 times. 144 * 200k = 28M ops. Acceptable.
            
            # Actually, the current approach pops from the list. If we cover a face that is still in the list,
            # we will eventually pop it and skip it. This is fine.
            # But if we cover a face that was already popped, we don't care.
            # The loop condition is `while uncovered_faces`.
            # If we cover all faces, the list might still have items, but they will be skipped.
            # To be safe, let's just break if all faces are covered.
            
            if np.all(face_hits > 0):
                uncovered_faces = []
                break

    # 3. Local Search / Simulated Annealing
    # Minimize |removed|
    # State: removed set, face_hits array
    # Moves:
    # 1. Try to remove an edge e from removed.
    #    Valid if for all f in edge_faces[e], face_hits[f] > 1.
    # 2. Try to swap edge e (in removed) with edge f (not in removed).
    #    Valid if for all f_face in edge_faces[e] U edge_faces[f], the new count is >= 1.
    #    Specifically, faces only in edge_faces[e] must have count > 1 (so they remain >= 1 after removing e).
    #    Faces only in edge_faces[f] will have count + 1, so they are fine.
    #    Faces in intersection will have count unchanged.
    
    best_removed = set(removed)
    best_cost = len(removed)
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Check time more frequently
        if time.time() - start_time > time_limit:
            break
            
        # Try to remove a random edge
        if removed:
            e = random.choice(list(removed))
            # Check if removal is valid
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
                # Find a candidate f such that swapping e with f is valid.
                # Conditions:
                # 1. f not in removed.
                # 2. For all f_face in edge_faces[e] with face_hits[f_face] == 1, f must be in f_face.
                #    (Because removing e would make count 0, so adding f must restore it to 1).
                # 3. For all f_face in edge_faces[f] NOT in edge_faces[e], face_hits[f_face] >= 0 (always true).
                #    Adding f increases count, so no new violations.
                
                # Find critical faces: faces where e is the ONLY removed edge.
                critical_faces = [f_idx for f_idx in edge_faces[e] if face_hits[f_idx] == 1]
                
                if not critical_faces:
                    # Should not happen if valid was False, but just in case
                    continue
                
                # Candidate f must be in ALL critical faces.
                # Intersection of edge sets of critical faces.
                candidates = set(face_edge_lists[critical_faces[0]])
                for g in critical_faces[1:]:
                    candidates.intersection_update(face_edge_lists[g])
                
                # Remove e from candidates (we are swapping, so f != e)
                candidates.discard(e)
                
                # Also, f must not be in removed.
                candidates -= removed
                
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
                    
                    # Cost is same. We accept to explore.
                    # We don't update best_cost here since it's the same.
        
        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min
    
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
    # Wait, if we did swaps, the current state is valid.
    # If we did removals, the current state is valid.
    # So the current state is always valid.
    # best_removed is a snapshot of a valid state with minimal cost found so far.
    # It's possible that the current state has the same cost but different edges.
    # Or current state has higher cost? No, we only accept valid moves.
    # Removals decrease cost. Swaps keep cost same.
    # So current cost <= best_cost? No, best_cost is the min seen.
    # So len(removed) >= best_cost.
    # We should use best_removed to be safe, or just use the current one if it's better?
    # Actually, if we found a better one, we updated best_removed.
    # So best_removed is the best we found.
    # Let's use best_removed.
    
    final_removed = best_removed
    
    # Construct output
    out = []
    for e_idx in range(n_edges):
        if e_idx not in final_removed:
            u, v = edges[e_idx]
            out.append([u, v])
            
    return out
