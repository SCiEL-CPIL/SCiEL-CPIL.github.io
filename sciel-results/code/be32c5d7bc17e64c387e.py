import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    d = 7
    num_vertices = 1 << d
    num_edges = d * (1 << (d - 1))
    
    # Precompute edge representations and face memberships
    # Edge is (u, v) with u < v.
    # We need to map edge index to (u, v) and vice versa.
    edges = []
    edge_map = {}
    for u in range(num_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_map[(u, v)] = idx
                
    # Precompute faces
    # A face is defined by two dimensions (i, j) and the fixed bits.
    # Fixed bits are the d-2 bits that are not i or j.
    # There are C(7,2) = 21 pairs of dimensions.
    # For each pair, there are 2^(7-2) = 32 faces.
    # Total 672 faces.
    
    # For each edge, which faces does it belong to?
    # Edge is defined by dimension i and vertex u (where bit i is 0? No, u and v differ in i).
    # Let's say edge e is between u and v, differing in bit i.
    # The faces containing e are determined by choosing another dimension j != i.
    # The face is in the plane where bit j is fixed.
    # The fixed value of bit j is determined by u (and v, they are same in j).
    # The other d-2 bits are fixed to their values in u.
    
    # Let's create face_ids.
    # Face ID can be (i, j, fixed_mask, fixed_val)
    # But we just need indices.
    
    edge_to_faces = [[] for _ in range(num_edges)]
    face_edges = [] # List of lists of edge indices
    
    # Generate all faces
    # For each pair of dimensions (i, j) with i < j
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all combinations of fixed bits for the other d-2 dimensions
            # There are 2^(d-2) such combinations.
            # We can iterate through vertices and check if they form a face?
            # Easier: Iterate through all 2^(d-2) patterns for the non-i, non-j bits.
            
            # Let's identify the indices of the other dimensions
            other_dims = [k for k in range(d) if k != i and k != j]
            
            for val in range(1 << (d - 2)):
                # Construct the fixed bits
                # We need to map val to the specific bits
                # This is a bit tedious. Let's use a different approach.
                
                # Iterate through all vertices u.
                # A face is a set of 4 vertices.
                # Instead, let's just iterate through all edges and find their faces.
                pass

    # Alternative: Build face memberships from edges.
    # For each edge e (u, v) differing in bit i:
    # For each j != i:
    #    The face is defined by fixing bit j to u[j] (which is same as v[j]).
    #    And fixing all other bits k != i, j to u[k].
    #    We need a unique ID for this face.
    #    ID = (min(i,j), max(i,j), tuple(sorted((k, u[k]) for k in other_dims)))
    
    face_ids = set()
    face_id_map = {}
    
    for e_idx, (u, v) in enumerate(edges):
        # Determine dimension i
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            # Determine fixed bits for k != i, j
            # We can use a tuple of (bit_index, bit_value) for k != i, j
            fixed_bits = []
            for k in range(d):
                if k != i and k != j:
                    fixed_bits.append((k, (u >> k) & 1))
            fixed_bits.sort()
            
            face_key = (min(i, j), max(i, j), tuple(fixed_bits))
            
            if face_key not in face_id_map:
                face_id_map[face_key] = len(face_ids)
                face_ids.add(face_key)
            
            f_id = face_id_map[face_key]
            edge_to_faces[e_idx].append(f_id)
            
    # Now we have edge_to_faces.
    # We also need face_edges to check validity quickly?
    # We only need face_counts.
    
    num_faces = len(face_ids)
    
    def greedy_init():
        selected = [False] * num_edges
        face_counts = [0] * num_faces
        
        edge_indices = list(range(num_edges))
        random.shuffle(edge_indices)
        
        for e in edge_indices:
            ok = True
            for f in edge_to_faces[e]:
                if face_counts[f] >= 3:
                    ok = False
                    break
            if ok:
                selected[e] = True
                for f in edge_to_faces[e]:
                    face_counts[f] += 1
        return selected, face_counts

    best_selected = None
    best_count = 0
    
    # Run multiple greedy restarts
    while time.time() - start_time < 5.0:
        selected, face_counts = greedy_init()
        count = sum(selected)
        if count > best_count:
            best_count = count
            best_selected = selected
            best_face_counts = face_counts
    
    if best_selected is None:
        best_selected, best_face_counts = greedy_init()
        best_count = sum(best_selected)
    
    selected = best_selected
    face_counts = best_face_counts
    current_count = best_count
    
    # Local Search
    # Maintain a list of unselected edges? Or just scan.
    # Scanning 448 edges is fast.
    
    while time.time() - start_time < time_limit:
        # 1. Try to add an edge
        added = False
        for e in range(num_edges):
            if selected[e]:
                continue
            # Check if e can be added
            can_add = True
            for f in edge_to_faces[e]:
                if face_counts[f] >= 3:
                    can_add = False
                    break
            if can_add:
                selected[e] = True
                current_count += 1
                for f in edge_to_faces[e]:
                    face_counts[f] += 1
                added = True
                # Continue looking for more addable edges
                # Don't break, as adding one might make another addable?
                # No, adding an edge only increases counts. It makes it harder to add others.
                # So we should break and restart the scan?
                # Or just continue. If we continue, we might add multiple in one pass.
        
        if not added:
            # Stuck. Perform a swap.
            # Find an unselected edge e_in that is "close" to being addable.
            # Or just random.
            # Let's try to find a swap that allows an addition.
            
            # Pick a random unselected edge
            unselected = [e for e in range(num_edges) if not selected[e]]
            if not unselected:
                break # Maxed out? Unlikely.
            
            e_in = random.choice(unselected)
            
            # Find blocking faces for e_in
            blocking_faces = [f for f in edge_to_faces[e_in] if face_counts[f] == 3]
            
            if not blocking_faces:
                # Should have been added in previous step.
                # Unless logic error.
                continue
            
            # Find an edge e_out in S that is in all blocking_faces
            # Intersection of blocking_faces
            # For each blocking face, get its edges in S.
            
            # Optimize:
            # If len(blocking_faces) == 1:
            #   Any edge in that face (except e_in, which is not in S) works?
            #   Wait, if we remove an edge from that face, count becomes 2.
            #   Adding e_in makes it 3. Valid.
            #   So any edge in that face works.
            
            # If len(blocking_faces) > 1:
            #   e_out must be in the intersection of all blocking faces.
            
            # Let's find candidate e_outs.
            # Start with edges of the first blocking face.
            # Get edges in face f0 that are selected.
            
            f0 = blocking_faces[0]
            # We need to know which edges are in face f0.
            # We didn't build face_edges. Let's build it now or infer.
            # It's better to precompute face_edges.
            
            # Let's rebuild face_edges efficiently.
            # We have edge_to_faces.
            # We can invert it.
            face_edges = [[] for _ in range(num_faces)]
            for e in range(num_edges):
                for f in edge_to_faces[e]:
                    face_edges[f].append(e)
            
            # Now, candidates for e_out are in face_edges[f0] and selected[e]
            candidates = [e for e in face_edges[f0] if selected[e]]
            
            for e_out in candidates:
                # Check if e_out is in all other blocking faces
                valid_swap = True
                for f in blocking_faces[1:]:
                    if e_out not in face_edges[f]:
                        valid_swap = False
                        break
                
                if valid_swap:
                    # Check if swapping e_out with e_in is valid
                    # Remove e_out, Add e_in.
                    # Faces affected:
                    # 1. Faces containing e_out: count decreases.
                    # 2. Faces containing e_in: count increases.
                    
                    # We know e_in's blocking faces are handled by e_out.
                    # Are there other faces containing e_in with count 2?
                    # They will become 3. Valid.
                    # Are there other faces containing e_in with count 3?
                    # Those were in blocking_faces.
                    
                    # So the swap is valid IF removing e_out doesn't cause issues?
                    # Removing e_out never causes a 4-face.
                    
                    # So we can perform the swap.
                    selected[e_out] = False
                    selected[e_in] = True
                    # current_count stays the same
                    
                    # Update counts
                    for f in edge_to_faces[e_out]:
                        face_counts[f] -= 1
                    for f in edge_to_faces[e_in]:
                        face_counts[f] += 1
                    
                    # After swap, we might be able to add another edge.
                    # So break out of swap search and go to "Try to Add"
                    added = True # Signal to restart add loop
                    break
            
            if not added:
                # No single swap found.
                # Try a random valid swap to explore.
                # Pick random e_in from unselected.
                # Pick random e_out from selected such that swap is valid.
                # This is harder.
                # Let's just pick random e_in and try to find ANY e_out that makes it valid.
                # If none, do nothing.
                
                # To avoid getting stuck in a loop doing nothing,
                # we can try a "2-swap" or just restart greedy?
                # Restarting greedy might be better if stuck for too long.
                
                # For now, just continue.
                pass

    # Return the list of edges
    result = []
    for e in range(num_edges):
        if selected[e]:
            u, v = edges[e]
            result.append([u, v])
            
    return result

# Precomputation of face_edges should be done once outside construct if possible,
# but construct is the entry point. We can cache it in a global variable.
_face_edges_cache = None
_edge_to_faces_cache = None
_edges_cache = None
_num_edges_cache = None
_num_faces_cache = None

def _precompute():
    global _face_edges_cache, _edge_to_faces_cache, _edges_cache, _num_edges_cache, _num_faces_cache
    if _edges_cache is not None:
        return
    
    d = 7
    num_vertices = 1 << d
    _edges_cache = []
    edge_map = {}
    for u in range(num_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(_edges_cache)
                _edges_cache.append((u, v))
                edge_map[(u, v)] = idx
                
    _num_edges_cache = len(_edges_cache)
    
    _edge_to_faces_cache = [[] for _ in range(_num_edges_cache)]
    face_id_map = {}
    face_ids_set = set()
    
    for e_idx, (u, v) in enumerate(_edges_cache):
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            fixed_bits = []
            for k in range(d):
                if k != i and k != j:
                    fixed_bits.append((k, (u >> k) & 1))
            fixed_bits.sort()
            
            face_key = (min(i, j), max(i, j), tuple(fixed_bits))
            
            if face_key not in face_id_map:
                face_id_map[face_key] = len(face_ids_set)
                face_ids_set.add(face_key)
            
            f_id = face_id_map[face_key]
            _edge_to_faces_cache[e_idx].append(f_id)
            
    _num_faces_cache = len(face_ids_set)
    
    _face_edges_cache = [[] for _ in range(_num_faces_cache)]
    for e in range(_num_edges_cache):
        for f in _edge_to_faces_cache[e]:
            _face_edges_cache[f].append(e)

def construct():
    _precompute()
    start_time = time.time()
    time_limit = 110.0
    
    num_edges = _num_edges_cache
    num_faces = _num_faces_cache
    edges = _edges_cache
    edge_to_faces = _edge_to_faces_cache
    face_edges = _face_edges_cache
    
    def greedy_init():
        selected = [False] * num_edges
        face_counts = [0] * num_faces
        
        edge_indices = list(range(num_edges))
        random.shuffle(edge_indices)
        
        for e in edge_indices:
            ok = True
            for f in edge_to_faces[e]:
                if face_counts[f] >= 3:
                    ok = False
                    break
            if ok:
                selected[e] = True
                for f in edge_to_faces[e]:
                    face_counts[f] += 1
        return selected, face_counts

    best_selected = None
    best_count = 0
    best_face_counts = None
    
    # Run multiple greedy restarts
    while time.time() - start_time < 5.0:
        selected, face_counts = greedy_init()
        count = sum(selected)
        if count > best_count:
            best_count = count
            best_selected = selected
            best_face_counts = face_counts
    
    if best_selected is None:
        best_selected, best_face_counts = greedy_init()
        best_count = sum(best_selected)
    
    selected = best_selected
    face_counts = best_face_counts
    current_count = best_count
    
    # Local Search
    while time.time() - start_time < time_limit:
        # 1. Try to add an edge
        added = False
        for e in range(num_edges):
            if selected[e]:
                continue
            can_add = True
            for f in edge_to_faces[e]:
                if face_counts[f] >= 3:
                    can_add = False
                    break
            if can_add:
                selected[e] = True
                current_count += 1
                for f in edge_to_faces[e]:
                    face_counts[f] += 1
                added = True
        
        if not added:
            # Stuck. Perform a swap.
            unselected = [e for e in range(num_edges) if not selected[e]]
            if not unselected:
                break
            
            # Try to find a productive swap
            # Pick a random unselected edge
            e_in = random.choice(unselected)
            blocking_faces = [f for f in edge_to_faces[e_in] if face_counts[f] == 3]
            
            if blocking_faces:
                f0 = blocking_faces[0]
                candidates = [e for e in face_edges[f0] if selected[e]]
                
                swap_done = False
                for e_out in candidates:
                    valid_swap = True
                    for f in blocking_faces[1:]:
                        if e_out not in face_edges[f]:
                            valid_swap = False
                            break
                    if valid_swap:
                        # Perform swap
                        selected[e_out] = False
                        selected[e_in] = True
                        for f in edge_to_faces[e_out]:
                            face_counts[f] -= 1
                        for f in edge_to_faces[e_in]:
                            face_counts[f] += 1
                        swap_done = True
                        break
                
                if swap_done:
                    continue # Restart add loop
            
            # If no productive swap, do a random valid swap to explore
            # This is optional but helps escape local minima
            # For simplicity and speed, we might skip this if time is tight,
            # but it's good for finding better solutions.
            
            # Try a few random swaps
            for _ in range(10):
                e_in_rand = random.choice(unselected)
                # Find blocking faces
                bf = [f for f in edge_to_faces[e_in_rand] if face_counts[f] == 3]
                if not bf:
                    # Should be addable
                    continue
                
                # Try to find e_out
                # Use the same logic as above
                f0 = bf[0]
                candidates = [e for e in face_edges[f0] if selected[e]]
                for e_out in candidates:
                    valid_swap = True
                    for f in bf[1:]:
                        if e_out not in face_edges[f]:
                            valid_swap = False
                            break
                    if valid_swap:
                        selected[e_out] = False
                        selected[e_in_rand] = True
                        for f in edge_to_faces[e_out]:
                            face_counts[f] -= 1
                        for f in edge_to_faces[e_in_rand]:
                            face_counts[f] += 1
                        break

    result = []
    for e in range(num_edges):
        if selected[e]:
            u, v = edges[e]
            result.append([u, v])
            
    return result
