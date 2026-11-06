import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448
    
    # 1. Precompute edges
    idx_to_edge = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # 2. Precompute faces and mappings
    # A face is defined by two dimensions i, j and a base vertex u.
    # To be unique, we can just iterate all u, i, j and use the set of edges as key.
    seen_faces = set()
    face_edge_list = [] # List of lists of 4 edge indices
    
    for i in range(d):
        for j in range(i+1, d):
            for u in range(n):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idxs = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                key = frozenset(idxs)
                
                if key not in seen_faces:
                    seen_faces.add(key)
                    face_edge_list.append(idxs)
                    
    num_faces = len(face_edge_list)
    
    # Precompute edge -> faces mapping
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(face_edge_list):
        for e in face:
            edge_to_faces[e].append(f_idx)
            
    # State:
    # selected: boolean array of length num_edges. True if edge is in the set.
    # face_counts: array of length num_faces. Number of selected edges in that face.
    # A face is "saturated" (blocked) if face_count == 3.
    # An edge is "addable" if for all faces f containing it, face_count[f] < 3.
    # i.e., no face containing it is saturated.
    
    rng = np.random.default_rng(42)
    
    best_count = 0
    best_selected = None
    
    # We will run multiple restarts with different initializations or just one long run?
    # Let's do a few restarts to explore different local optima.
    num_restarts = 10
    
    for run in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialization: Greedy Add
        # Randomize edge order for diversity
        order = rng.permutation(num_edges)
        
        selected = np.zeros(num_edges, dtype=np.uint8)
        face_counts = np.zeros(num_faces, dtype=np.int32)
        
        # Greedy Add Phase
        # We iterate through edges in random order.
        # If an edge is addable, we add it.
        # Note: This produces a "maximal" set with respect to the order, but not necessarily maximum.
        
        for e in order:
            # Check if addable
            # An edge e is addable if no face containing it has count 3.
            is_addable = True
            for f_idx in edge_to_faces[e]:
                if face_counts[f_idx] == 3:
                    is_addable = False
                    break
            
            if is_addable:
                selected[e] = 1
                for f_idx in edge_to_faces[e]:
                    face_counts[f_idx] += 1
        
        current_count = np.sum(selected)
        
        # Local Search Phase
        # Strategy:
        # 1. Try to add any edge that is currently addable.
        # 2. If no addable edge exists, perform a swap: Remove an edge from a saturated face, 
        #    then try to add the edge that was blocking it? 
        #    Actually, a better swap: 
        #    Pick a random missing edge e_miss.
        #    Find all saturated faces containing e_miss.
        #    If none, e_miss is addable (contradiction if we checked all, but maybe we didn't scan all).
        #    If some, pick one saturated face f.
        #    Pick a random edge e_rem from f (excluding e_miss).
        #    Remove e_rem.
        #    Check if e_miss is now addable.
        #    If yes, add e_miss. (Net change 0, but landscape changed).
        #    If no, undo (re-add e_rem).
        
        # To improve efficiency, we maintain a list of "addable" edges?
        # That's expensive to update.
        # Instead, we can just scan for addable edges periodically.
        
        max_iterations = 5000 # Adjust based on time
        iter_count = 0
        last_improvement = 0
        
        while iter_count < max_iterations:
            if time.time() > deadline - 5:
                break
                
            # 1. Check for addable edges
            # We can scan all missing edges. 448 is small.
            found_addable = False
            for e in range(num_edges):
                if selected[e] == 0:
                    # Check if addable
                    can_add = True
                    for f_idx in edge_to_faces[e]:
                        if face_counts[f_idx] == 3:
                            can_add = False
                            break
                    if can_add:
                        # Add it
                        selected[e] = 1
                        for f_idx in edge_to_faces[e]:
                            face_counts[f_idx] += 1
                        current_count += 1
                        found_addable = True
                        last_improvement = iter_count
                        break # Restart scan to pick up new addable edges? Or continue?
                        # Continuing is faster, but might miss edges that became addable.
                        # Let's just break and restart the inner loop.
            
            if found_addable:
                iter_count += 1
                continue
                
            # 2. No addable edges. Perform a Swap.
            # Pick a random missing edge e_miss
            missing_edges = np.where(selected == 0)[0]
            if len(missing_edges) == 0:
                break # All edges selected? Impossible for C4-free.
                
            e_miss = missing_edges[rng.integers(0, len(missing_edges))]
            
            # Find saturated faces containing e_miss
            sat_faces = []
            for f_idx in edge_to_faces[e_miss]:
                if face_counts[f_idx] == 3:
                    sat_faces.append(f_idx)
                    
            if len(sat_faces) == 0:
                # This should not happen if the scan above was correct.
                # But if it does, it means e_miss is addable.
                # Force add?
                selected[e_miss] = 1
                for f_idx in edge_to_faces[e_miss]:
                    face_counts[f_idx] += 1
                current_count += 1
                last_improvement = iter_count
                iter_count += 1
                continue
                
            # Pick a random saturated face
            f_idx = sat_faces[rng.integers(0, len(sat_faces))]
            face = face_edge_list[f_idx]
            
            # Pick a random edge to remove from this face (must be selected, and not e_miss)
            # e_miss is not in selected, so all selected edges in face are candidates.
            candidates = [e for e in face if selected[e] == 1]
            if not candidates:
                # Should not happen, face has count 3, so 3 selected edges.
                continue
                
            e_rem = candidates[rng.integers(0, len(candidates))]
            
            # Remove e_rem
            selected[e_rem] = 0
            for f_idx2 in edge_to_faces[e_rem]:
                face_counts[f_idx2] -= 1
                
            # Check if e_miss is now addable
            can_add_miss = True
            for f_idx3 in edge_to_faces[e_miss]:
                if face_counts[f_idx3] == 3:
                    can_add_miss = False
                    break
            
            if can_add_miss:
                # Add e_miss
                selected[e_miss] = 1
                for f_idx4 in edge_to_faces[e_miss]:
                    face_counts[f_idx4] += 1
                # Count remains same.
                # This is a "neutral" move, but it changes the structure.
            else:
                # Undo removal of e_rem
                selected[e_rem] = 1
                for f_idx5 in edge_to_faces[e_rem]:
                    face_counts[f_idx5] += 1
                
            iter_count += 1
            
            # If we haven't improved in a long time, maybe restart?
            # Or just keep going.
            
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            
    # Convert best_selected to list of edges
    result = []
    if best_selected is not None:
        for i in range(num_edges):
            if best_selected[i] == 1:
                result.append(list(idx_to_edge[i]))
    else:
        # Fallback: Return a valid set (e.g., empty or simple greedy)
        # Simple greedy: select edges of dimension 0 only?
        # No, let's just return the last run's state if best is None (shouldn't happen)
        for i in range(num_edges):
            if selected[i] == 1:
                result.append(list(idx_to_edge[i]))
                
    return result
