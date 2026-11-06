import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                idx_to_edge[idx] = (x, y)
    
    # Precompute conflict graph
    # Two edges conflict if they are part of the same 2D face.
    # In Q7, an edge (u,v) with diff bit i conflicts with (u',v') if they form a C4.
    # This is equivalent to: edges e1 and e2 conflict if they share a face.
    # We can compute this by iterating over all faces.
    
    # A face is defined by 4 vertices forming a square.
    # Let's build the conflict adjacency list using bitsets for speed.
    # n_edges = 448. We can use Python integers as bitsets or numpy arrays.
    # Python integers are arbitrary precision, so they can handle 448 bits easily.
    
    conflict_mask = [0] * n_edges
    
    # Iterate over all 2D faces
    # A face is determined by two distinct dimensions i, j and a vertex u 
    # such that bits i and j are 0 in u.
    # The vertices of the face are:
    # u, u^(1<<i), u^(1<<j), u^(1<<i|1<<j)
    # The 4 edges of this face are:
    # (u, u^(1<<i)), (u, u^(1<<j)), (u^(1<<i), u^(1<<i|1<<j)), (u^(1<<j), u^(1<<i|1<<j))
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_ij = (1 << i) | (1 << j)
            # Iterate over all u where bits i and j are 0
            # We can iterate over all vertices and filter, or generate them.
            # For d=7, 128 vertices is small.
            for u in range(n_vertices):
                if (u & mask_ij) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ mask_ij
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    if e1 in edge_to_idx and e2 in edge_to_idx and e3 in edge_to_idx and e4 in edge_to_idx:
                        idx1 = edge_to_idx[e1]
                        idx2 = edge_to_idx[e2]
                        idx3 = edge_to_idx[e3]
                        idx4 = edge_to_idx[e4]
                        
                        # In this face, all 4 edges are mutually "conflicting" in the sense 
                        # that we cannot select all 4. 
                        # Wait, the constraint is "No 2-dimensional face may have all four edges selected".
                        # This means we cannot select the set {e1, e2, e3, e4}.
                        # This is NOT a pairwise conflict. It is a hyperedge constraint of size 4.
                        # My previous thought about MIS in line graph was slightly off for the exact constraint.
                        # If it were "no two edges in a face", it would be independent set.
                        # But it's "no face has ALL 4 edges".
                        
                        # So the "conflict" is not pairwise. 
                        # However, we can reformulate.
                        # Let's stick to the local search approach but make it much faster 
                        # using bitsets and a smarter move generation, OR 
                        # use a different structural insight.
                        
                        # Actually, let's look at the constraint again.
                        # "No 2-dimensional face may have all four edges selected."
                        # This is equivalent to: For every face F, |E_selected \cap E_F| <= 3.
                        
                        # The current program uses local search.
                        # The directive suggests: "Model the problem as a maximum independent set in the line graph... branch-and-bound".
                        # But as noted, the constraint is not pairwise. 
                        # UNLESS we transform the problem.
                        
                        # Is there a way to map this to an Independent Set?
                        # If we select a subset of edges S such that no face has 4 edges.
                        # This is a 3-uniform hypergraph independent set problem? No, it's a 4-uniform constraint.
                        
                        # Let's reconsider the "different strategy".
                        # Maybe we can use a different representation: 
                        # Instead of selecting edges, let's select "forbidden" edges to remove?
                        # Total edges 448. Maximize selected = Minimize removed.
                        # Constraint: For each face, at least 1 edge must be removed.
                        # This is the Minimum Hitting Set (or Set Cover) problem!
                        # We have a collection of sets (faces), each of size 4.
                        # We want to find the smallest subset of edges R such that every face contains at least one edge from R.
                        # Then the selected edges are E \ R.
                        # Maximize |E \ R| is equivalent to Minimize |R|.
                        
                        # This is a Set Cover problem with universe size 21*64 = 1344 faces?
                        # Number of faces in Q7: C(7,2) * 2^(7-2) = 21 * 32 = 672 faces.
                        # Each face has 4 edges.
                        # We want to hit all 672 faces with the minimum number of edges.
                        # This is a hitting set problem.
                        # The current program does local search on the selected edges.
                        # Switching to a Hitting Set formulation on the removed edges might be faster 
                        # because the "cost" of adding a removal is uniform, and we can use greedy set cover 
                        # or local search on the hitting set.
                        
                        # Let's try a Greedy Set Cover approach first to get a good initial bound, 
                        # then Local Search on the Hitting Set (remove edges).
                        
                        # Greedy Set Cover:
                        # 1. Start with all edges selected. All 672 faces are "unhit" (violation).
                        # 2. Find the edge that is in the most unhit faces.
                        # 3. Remove that edge (add to R). Mark all faces containing it as "hit".
                        # 4. Repeat until all faces are hit.
                        # This gives a valid solution. 
                        # Then, we can try to "un-remove" edges (remove from R) if it doesn't create a violation.
                        # This is a local search on R.
                        
                        # Let's implement this.
                        pass # Placeholder for logic below

    # Re-implementing with Hitting Set formulation
    
    # Build face_to_edges and edge_to_faces
    face_to_edges = []
    edge_to_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_ij = (1 << i) | (1 << j)
            for u in range(n_vertices):
                if (u & mask_ij) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ mask_ij
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    if e1 in edge_to_idx and e2 in edge_to_idx and e3 in edge_to_idx and e4 in edge_to_idx:
                        idx1 = edge_to_idx[e1]
                        idx2 = edge_to_idx[e2]
                        idx3 = edge_to_idx[e3]
                        idx4 = edge_to_idx[e4]
                        face_idx = len(face_to_edges)
                        face_to_edges.append([idx1, idx2, idx3, idx4])
                        edge_to_faces[idx1].append(face_idx)
                        edge_to_faces[idx2].append(face_idx)
                        edge_to_faces[idx3].append(face_idx)
                        edge_to_faces[idx4].append(face_idx)
    
    n_faces = len(face_to_edges)
    
    # Initial Hitting Set via Greedy
    # State: 
    # removed: boolean array of size n_edges (True if in R)
    # face_hit_count: array of size n_faces, count of removed edges in that face.
    # A face is "satisfied" if face_hit_count[f] >= 1.
    
    def greedy_hitting_set():
        removed = np.zeros(n_edges, dtype=bool)
        face_hit_count = np.zeros(n_faces, dtype=np.int8)
        
        # Count how many unhit faces each edge covers
        # Initially all faces are unhit.
        # edge_cover_count[e] = number of faces containing e that are currently unhit.
        
        # To speed up, we can just pick edges randomly or by degree.
        # Degree of an edge is 6 (belongs to 6 faces).
        # Let's just iterate and pick the best.
        
        # For efficiency, we'll just do a simple greedy:
        # While there are unhit faces:
        #   Pick an edge in an unhit face.
        #   To be smart, pick the edge that hits the most unhit faces.
        
        # Optimization: Precompute edge degrees (all 6).
        # We can use a priority queue or just scan.
        # Since n_edges=448, scanning is fast.
        
        unhit_faces = set(range(n_faces))
        
        while unhit_faces:
            if time.time() > deadline - 5:
                break
                
            # Find edge that maximizes hits in unhit_faces
            best_edge = -1
            best_hits = 0
            for e in range(n_edges):
                if removed[e]:
                    continue
                hits = 0
                for f in edge_to_faces[e]:
                    if f in unhit_faces:
                        hits += 1
                if hits > best_hits:
                    best_hits = hits
                    best_edge = e
                    if best_hits == 6: # Max possible
                        break
            
            if best_edge == -1:
                # Should not happen if unhit_faces is not empty
                break
            
            # Add best_edge to removed
            removed[best_edge] = True
            # Update face_hit_count and unhit_faces
            for f in edge_to_faces[best_edge]:
                face_hit_count[f] += 1
                if face_hit_count[f] == 1:
                    unhit_faces.discard(f)
            
        return removed, face_hit_count
    
    # Local Search on Hitting Set
    # Objective: Minimize |removed|.
    # Move: Remove an edge from R (un-remove it) if it doesn't cause any face to become unhit.
    # i.e., if for all faces f containing e, face_hit_count[f] > 1.
    # If such an edge exists, remove it from R.
    # If stuck, try swaps: Remove e1 from R, add e2 to R? 
    # Actually, if we are stuck in local search (no single un-removal is possible), 
    # we can try to swap: 
    # Pick an edge e1 in R. Temporarily remove it. 
    # This makes some faces "critical" (count becomes 1).
    # We need to add some edges to R to fix those faces.
    # If we can add fewer edges than we removed (i.e., 0, which is the un-removal case, or 1 if we swap 1 for 1 but gain something else?), 
    # wait, the objective is just to minimize |R|.
    # So a swap that keeps |R| constant doesn't help unless it enables future un-removals.
    # So we should use Simulated Annealing or Random Restarts.
    
    def local_search_hitting_set(removed, face_hit_count, max_time=130.0):
        # Make a copy
        removed = removed.copy()
        face_hit_count = face_hit_count.copy()
        
        # Precompute edge_to_faces for fast access
        # We already have it.
        
        # Iterate
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try to un-remove any edge
            for e in range(n_edges):
                if removed[e]:
                    can_unremove = True
                    for f in edge_to_faces[e]:
                        if face_hit_count[f] <= 1:
                            can_unremove = False
                            break
                    if can_unremove:
                        removed[e] = False
                        for f in edge_to_faces[e]:
                            face_hit_count[f] -= 1
                        improved = True
            if not improved:
                # Try random swaps to escape local minima
                # Pick a random e1 in R.
                # Remove it.
                # Find faces that are now critical (count 1).
                # Find the cheapest way to re-cover them?
                # Or just add random edges from the critical faces.
                # Since we want to minimize size, we only accept swaps if we can reduce size later.
                # So we do a "kick": remove e1, add a set of edges to cover the new violations.
                # If the set of added edges is smaller than 1, we improved (unlikely).
                # If it is 1, size is same, but structure changed.
                # If it is >1, size increases, which is bad.
                # So we should only do kicks that result in a valid solution with size <= current.
                # Actually, if we remove 1 and add 1, size is same.
                # If we remove 1 and add 0, size decreases (handled above).
                # So kicks should be: Remove e1. Add minimal edges to restore validity.
                # If minimal edges added < 1, we are in the "improved" case.
                # If minimal edges added == 1, we have a valid solution of same size.
                # We can accept this to escape local minima.
                
                # Let's try a few random kicks
                for _ in range(20):
                    if time.time() > max_time:
                        break
                    sel_indices = np.where(removed)[0]
                    if len(sel_indices) == 0:
                        break
                    e1 = sel_indices[random.randint(0, len(sel_indices)-1)]
                    
                    # Temporarily remove e1
                    removed[e1] = False
                    for f in edge_to_faces[e1]:
                        face_hit_count[f] -= 1
                    
                    # Find critical faces
                    critical_faces = []
                    for f in edge_to_faces[e1]:
                        if face_hit_count[f] == 0:
                            critical_faces.append(f)
                    
                    if not critical_faces:
                        # We un-removed successfully, this is an improvement
                        improved = True
                        break
                    
                    # We need to add edges to cover critical_faces
                    # Greedily pick edges from critical faces
                    added_edges = []
                    covered = set(critical_faces)
                    while covered:
                        # Pick a face from covered
                        f = next(iter(covered))
                        # Pick an edge from face_to_edges[f]
                        # Prefer edges that cover more uncovered faces
                        best_e = -1
                        best_cov = 0
                        for e in face_to_edges[f]:
                            if removed[e]:
                                continue # already removed? No, removed[e] is True if in R.
                                # Wait, removed[e] is True if edge is in R.
                                # We are adding to R. So we want edges NOT in R.
                                # But we just removed e1 from R.
                                # So e1 is a candidate.
                                # Also other edges not in R.
                                pass
                        # This logic is getting complex.
                        # Let's simplify: Just pick a random edge from the face that is NOT in R.
                        candidates = [e for e in face_to_edges[f] if not removed[e]]
                        if not candidates:
                            # Should not happen
                            break
                        e_add = random.choice(candidates)
                        removed[e_add] = True
                        added_edges.append(e_add)
                        for f2 in edge_to_faces[e_add]:
                            face_hit_count[f2] += 1
                            if f2 in covered:
                                covered.discard(f2)
                    
                    # Check if we improved
                    # We removed 1 edge (e1).
                    # We added len(added_edges) edges.
                    # If len(added_edges) < 1, improved.
                    # If len(added_edges) == 1, same size.
                    # If len(added_edges) > 1, worse.
                    
                    if len(added_edges) < 1:
                        improved = True
                        break
                    elif len(added_edges) == 1:
                        # Same size, accept to change state
                        # Do not set improved=True to allow more kicks
                        pass
                    else:
                        # Worse, revert
                        for e in added_edges:
                            removed[e] = False
                            for f in edge_to_faces[e]:
                                face_hit_count[f] -= 1
                        removed[e1] = True
                        for f in edge_to_faces[e1]:
                            face_hit_count[f] += 1
        
        return removed, face_hit_count

    # Run multiple attempts
    best_removed = None
    best_size = n_edges
    
    n_attempts = 50
    for attempt in range(n_attempts):
        if time.time() > deadline - 10:
            break
        
        # Greedy initialization
        removed, face_hit_count = greedy_hitting_set()
        size = np.sum(removed)
        
        # Local search
        removed, face_hit_count = local_search_hitting_set(removed, face_hit_count, max_time=deadline - 2 - attempt * 0.1)
        size = np.sum(removed)
        
        if size < best_size:
            best_size = size
            best_removed = removed.copy()
    
    # Final polish
    if best_removed is not None:
        # Reconstruct face_hit_count
        face_hit_count = np.zeros(n_faces, dtype=np.int8)
        for e in range(n_edges):
            if best_removed[e]:
                for f in edge_to_faces[e]:
                    face_hit_count[f] += 1
        best_removed, face_hit_count = local_search_hitting_set(best_removed, face_hit_count, max_time=deadline - 2)
        best_size = np.sum(best_removed)

    # Convert to selected edges
    selected = ~best_removed
    result = []
    for e in range(n_edges):
        if selected[e]:
            u, v = idx_to_edge[e]
            result.append([u, v])
    
    return result
