import time
import random
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d  # 128
    num_edges_total = n_vertices * d // 2  # 448
    
    # Precompute all edges
    all_edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                idx_to_edge[idx] = (u, v)
                
    num_edges = len(all_edges) # 448

    # Precompute all faces (C4s)
    # A face is defined by two distinct dimensions i, j and a vertex u 
    # such that the bits i and j are 0 in u (to avoid duplicates).
    # The 4 edges of the face are:
    # (u, u^1<<i), (u, u^1<<j), 
    # (u^1<<i, u^1<<i^1<<j), (u^1<<j, u^1<<i^1<<j)
    
    faces = []
    face_edge_indices = [] # List of lists of 4 edge indices
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i+1, d):
                # To ensure unique faces, we can just iterate all u and pairs i,j
                # but a face is generated 4 times (once for each corner).
                # To avoid duplicates, we can require that the lowest set bit 
                # among the two dimensions in u is 0? No, simpler:
                # A face is uniquely identified by the set of 4 vertices.
                # Let's just generate all and use a set of frozensets of vertices?
                # Or better: generate all 4 edges, sort them, and use a set of tuples of edge indices.
                
                # Actually, there are exactly 672 faces.
                # 128 vertices * C(7,2) pairs / 4 corners per face = 128 * 21 / 4 = 672.
                # So if we iterate u, i, j, we generate each face 4 times.
                # We can just check if the face is already seen.
                
                e1 = (min(u, u ^ (1<<i)), max(u, u ^ (1<<i)))
                e2 = (min(u, u ^ (1<<j)), max(u, u ^ (1<<j)))
                e3 = (min(u ^ (1<<i), u ^ (1<<i) ^ (1<<j)), max(u ^ (1<<i), u ^ (1<<i) ^ (1<<j)))
                e4 = (min(u ^ (1<<j), u ^ (1<<i) ^ (1<<j)), max(u ^ (1<<j), u ^ (1<<i) ^ (1<<j)))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_key = tuple(sorted([idx1, idx2, idx3, idx4]))
                
                # We can just collect unique faces
                # Since we know there are 672, we can stop if we have that many?
                # Or just use a set.
                
                # For efficiency in the main loop, we want a list of faces.
                # Let's build a set first to deduplicate.
                pass
                
    # Re-do face generation properly with deduplication
    unique_faces_set = set()
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i+1, d):
                e1 = (min(u, u ^ (1<<i)), max(u, u ^ (1<<i)))
                e2 = (min(u, u ^ (1<<j)), max(u, u ^ (1<<j)))
                e3 = (min(u ^ (1<<i), u ^ (1<<i) ^ (1<<j)), max(u ^ (1<<i), u ^ (1<<i) ^ (1<<j)))
                e4 = (min(u ^ (1<<j), u ^ (1<<i) ^ (1<<j)), max(u ^ (1<<j), u ^ (1<<i) ^ (1<<j)))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_key = tuple(sorted([idx1, idx2, idx3, idx4]))
                unique_faces_set.add(face_key)
                
    faces = list(unique_faces_set)
    num_faces = len(faces) # Should be 672
    
    # Precompute for each edge, which faces it covers
    # edge_faces[e] = list of face indices
    edge_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
            
    # Precompute for each face, the 4 edges
    # face_edges[f] = [e1, e2, e3, e4]
    face_edges = [list(f) for f in faces]

    def greedy_hit_set(uncovered_faces_mask, rng):
        """
        Greedily add edges to cover all uncovered faces.
        uncovered_faces_mask is a boolean array or set of face indices to cover.
        Returns a set of edge indices.
        """
        R = set()
        uncovered = set(uncovered_faces_mask)
        
        while uncovered:
            # Find edge that covers max uncovered faces
            max_count = 0
            best_edge = -1
            # Optimization: Only consider edges that cover at least one uncovered face
            # We can iterate all edges, but checking count is O(1) if we maintain counts?
            # For simplicity and given small size, we scan all edges.
            # To speed up, we can maintain a list of candidate edges.
            
            # A faster way:
            # For each edge, count how many faces in 'uncovered' it covers.
            # Since |uncovered| decreases, this gets faster.
            
            # Let's use a simple scan. 448 edges * 672 faces is small.
            # But we do this many times.
            
            # Better: maintain a count array for edges.
            # However, initializing counts for every greedy step is expensive.
            # Let's just scan.
            
            for e in range(num_edges):
                count = 0
                for f in edge_faces[e]:
                    if f in uncovered:
                        count += 1
                        if count > max_count:
                            max_count = count
                            best_edge = e
                            if max_count == len(uncovered): # Can't do better
                                break
                if max_count == len(uncovered):
                    break
                
            if best_edge == -1:
                # Should not happen if graph is connected
                break
                
            R.add(best_edge)
            # Remove covered faces from uncovered
            for f in edge_faces[best_edge]:
                if f in uncovered:
                    uncovered.remove(f)
                    
        return R

    def is_valid_hit_set(R):
        """Check if R hits all faces."""
        covered = [False] * num_faces
        for e in R:
            for f in edge_faces[e]:
                covered[f] = True
        for f in range(num_faces):
            if not covered[f]:
                return False
        return True

    def local_search_hit_set(R, deadline, rng):
        """
        Local search to minimize |R|.
        Moves:
        1. Deletion: If removing e from R leaves all faces covered, do it.
        2. Swap: Replace e in R with f not in R if it allows a deletion later? 
           Simpler: Try removing e, then greedily refill. If new size < old size, accept.
           If new size == old size, accept with prob p (to move around).
        """
        R = set(R)
        improved = True
        no_improve_count = 0
        
        # Precompute coverage counts for current R to speed up deletion check
        # cover_count[f] = number of edges in R covering face f
        cover_count = np.zeros(num_faces, dtype=np.int32)
        for e in R:
            for f in edge_faces[e]:
                cover_count[f] += 1
        
        while improved and time.time() < deadline:
            improved = False
            R_list = list(R)
            rng.shuffle(R_list)
            
            for e_remove in R_list:
                if time.time() > deadline:
                    break
                
                # Try deleting e_remove
                # Check if any face becomes uncovered
                valid = True
                for f in edge_faces[e_remove]:
                    if cover_count[f] == 1:
                        valid = False
                        break
                
                if valid:
                    # Safe to delete
                    R.discard(e_remove)
                    for f in edge_faces[e_remove]:
                        cover_count[f] -= 1
                    improved = True
                    no_improve_count = 0
                    break # Restart loop with new R
                else:
                    # Try swap/kick: Remove e_remove, then greedily add back to cover holes
                    # This is a "kick" move
                    if rng.random() < 0.05: # Low probability kick
                        R_temp = set(R)
                        R_temp.discard(e_remove)
                        
                        # Find uncovered faces
                        uncovered = []
                        for f in edge_faces[e_remove]:
                            if cover_count[f] == 1:
                                uncovered.append(f)
                        
                        if uncovered:
                            # Greedily fill
                            R_add = greedy_hit_set(uncovered, rng)
                            R_new = R_temp.union(R_add)
                            
                            if len(R_new) < len(R):
                                R = R_new
                                # Recalculate cover_count
                                cover_count = np.zeros(num_faces, dtype=np.int32)
                                for e in R:
                                    for f in edge_faces[e]:
                                        cover_count[f] += 1
                                improved = True
                                no_improve_count = 0
                                break
                            elif len(R_new) == len(R) and rng.random() < 0.1:
                                R = R_new
                                cover_count = np.zeros(num_faces, dtype=np.int32)
                                for e in R:
                                    for f in edge_faces[e]:
                                        cover_count[f] += 1
                                improved = True
                                no_improve_count = 0
                                break
            
            if not improved:
                no_improve_count += 1
                # If stuck, try a bigger kick
                if no_improve_count > 5:
                    # Remove 3 random edges
                    if len(R) > 3:
                        to_remove = rng.sample(list(R), 3)
                        R_temp = set(R)
                        for e in to_remove:
                            R_temp.discard(e)
                        
                        # Find all uncovered faces
                        cover_count_temp = np.zeros(num_faces, dtype=np.int32)
                        for e in R_temp:
                            for f in edge_faces[e]:
                                cover_count_temp[f] += 1
                                
                        uncovered = [f for f in range(num_faces) if cover_count_temp[f] == 0]
                        
                        if uncovered:
                            R_add = greedy_hit_set(uncovered, rng)
                            R_new = R_temp.union(R_add)
                            
                            if len(R_new) <= len(R):
                                R = R_new
                                cover_count = np.zeros(num_faces, dtype=np.int32)
                                for e in R:
                                    for f in edge_faces[e]:
                                        cover_count[f] += 1
                                improved = True
                                no_improve_count = 0
                                break
                    no_improve_count = 0
                    
        return R

    best_R = set()
    best_size = num_edges # Max possible size of R is 448, min is 0
    
    start_time = time.time()
    deadline = start_time + 115.0
    rng = random.Random(42)
    
    # Initial Greedy
    R = greedy_hit_set(set(range(num_faces)), rng)
    best_R = R
    best_size = len(R)
    
    # Local Search
    R = local_search_hit_set(R, deadline, rng)
    if len(R) < best_size:
        best_R = R
        best_size = len(R)
    
    # Multiple restarts with different random seeds/strategies
    while time.time() < deadline:
        # Randomized Greedy: When multiple edges have same max count, pick randomly
        # Or just shuffle edges before greedy
        R_new = set()
        uncovered = set(range(num_faces))
        
        # To make it different, we can add noise or different tie-breaking
        # Simple: Just run greedy again with different rng state
        R_new = greedy_hit_set(uncovered, rng)
        
        # Local search
        R_new = local_search_hit_set(R_new, deadline, rng)
        
        if len(R_new) < best_size:
            best_R = R_new
            best_size = len(R_new)
            
        # Kick from best
        if time.time() < deadline and len(best_R) > 0:
            R_kick = set(best_R)
            # Remove 2 random edges
            if len(R_kick) >= 2:
                to_remove = rng.sample(list(R_kick), 2)
                for e in to_remove:
                    R_kick.discard(e)
                
                # Greedy fill
                # Need to find uncovered faces
                cover_count_temp = np.zeros(num_faces, dtype=np.int32)
                for e in R_kick:
                    for f in edge_faces[e]:
                        cover_count_temp[f] += 1
                uncovered = [f for f in range(num_faces) if cover_count_temp[f] == 0]
                
                if uncovered:
                    R_add = greedy_hit_set(uncovered, rng)
                    R_kick = R_kick.union(R_add)
                
                R_kick = local_search_hit_set(R_kick, deadline, rng)
                
                if len(R_kick) < best_size:
                    best_R = R_kick
                    best_size = len(R_kick)

    # Convert best_R (set of edges to remove) to list of kept edges
    # Kept edges are all edges NOT in best_R
    result = []
    for idx in range(num_edges):
        if idx not in best_R:
            u, v = idx_to_edge[idx]
            result.append([u, v])
            
    return result
