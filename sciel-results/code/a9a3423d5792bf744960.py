import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    d = 7
    n = 2 ** d
    total_edges = d * (1 << (d - 1)) # 448
    
    # Precompute edges and faces
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y, i))
                edge_to_idx[(x, y)] = idx
    
    # Precompute faces
    # A face is defined by a vertex x and two dimensions i, j
    # Edges are:
    # e1: x -> x^i
    # e2: x -> x^j
    # e3: x^i -> x^i^j
    # e4: x^j -> x^i^j
    faces = []
    edge_to_faces = [[] for _ in range(total_edges)]
    
    for x in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                # Define the 4 edges
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face = (idx1, idx2, idx3, idx4)
                faces.append(face)
                
                edge_to_faces[idx1].append(len(faces) - 1)
                edge_to_faces[idx2].append(len(faces) - 1)
                edge_to_faces[idx3].append(len(faces) - 1)
                edge_to_faces[idx4].append(len(faces) - 1)
    
    # Helper to get the other 3 edges in a face
    def get_other_edges(face_idx, edge_idx):
        f = faces[face_idx]
        # f is a tuple of 4 edge indices
        others = []
        for e in f:
            if e != edge_idx:
                others.append(e)
        return others
    
    def greedy_init():
        rng = random.Random(42)
        S = set()
        block_count = [0] * total_edges
        order = list(range(total_edges))
        rng.shuffle(order)
        
        for idx in order:
            if block_count[idx] == 0:
                S.add(idx)
                # Update block counts
                for f_idx in edge_to_faces[idx]:
                    f = faces[f_idx]
                    others = get_other_edges(f_idx, idx)
                    # Check if others are in S
                    # Since we are adding idx, we need to check if the other 3 are in S
                    # If so, this face is now "blocked" for the 4th edge?
                    # No, the constraint is "No 4 edges".
                    # If we add idx, and the other 3 are in S, we have a violation.
                    # But we checked block_count[idx] == 0, which means for ALL faces containing idx,
                    # the other 3 were NOT all in S.
                    # So adding idx is safe.
                    
                    # Now, does adding idx affect block_count of others?
                    # Yes. For each other edge 'o' in this face, if the other 2 (besides idx and o) are in S,
                    # then this face contributes to block_count[o].
                    
                    # Let's say face is {idx, o1, o2, o3}.
                    # For o1: others are idx, o2, o3.
                    # If o2, o3 in S, and idx is now in S, then block_count[o1] should increase?
                    # block_count[o1] counts faces where the OTHER 3 are in S.
                    # Before adding idx: idx not in S. So this face didn't count for o1.
                    # After adding idx: If o2, o3 in S, then the other 3 (idx, o2, o3) are in S.
                    # So block_count[o1] increases by 1.
                    
                    o1, o2, o3 = others
                    if o2 in S and o3 in S:
                        block_count[o1] += 1
                    if o1 in S and o3 in S:
                        block_count[o2] += 1
                    if o1 in S and o2 in S:
                        block_count[o3] += 1
                        
        return S, block_count

    def local_search(S, block_count, time_limit):
        rng = random.Random()
        current_size = len(S)
        
        while time.time() - start_time < time_limit:
            # Try to add an edge (k=1)
            # Find random unselected edge
            unselected = [i for i in range(total_edges) if i not in S]
            if not unselected:
                break
            
            # Try a few random additions
            found_add = False
            for _ in range(10):
                idx = rng.choice(unselected)
                if block_count[idx] == 0:
                    S.add(idx)
                    current_size += 1
                    found_add = True
                    # Update block counts
                    for f_idx in edge_to_faces[idx]:
                        f = faces[f_idx]
                        others = get_other_edges(f_idx, idx)
                        o1, o2, o3 = others
                        if o2 in S and o3 in S:
                            block_count[o1] += 1
                        if o1 in S and o3 in S:
                            block_count[o2] += 1
                        if o1 in S and o2 in S:
                            block_count[o3] += 1
                    break
            
            if found_add:
                continue
                
            # Try 1-out, 2-in (k=3)
            # Pick random edge to remove
            idx_remove = rng.choice(list(S))
            S.discard(idx_remove)
            current_size -= 1
            
            # Update block counts for removal
            for f_idx in edge_to_faces[idx_remove]:
                f = faces[f_idx]
                others = get_other_edges(f_idx, idx_remove)
                o1, o2, o3 = others
                # If o2, o3 in S, then this face was contributing to block_count[o1]
                # Now idx_remove is gone, so it no longer contributes
                if o2 in S and o3 in S:
                    block_count[o1] -= 1
                if o1 in S and o3 in S:
                    block_count[o2] -= 1
                if o1 in S and o2 in S:
                    block_count[o3] -= 1
            
            # Find addable edges
            addable = [i for i in range(total_edges) if i not in S and block_count[i] == 0]
            
            if len(addable) >= 2:
                # Try to add 2
                # Pick 2 random
                v1, v2 = rng.sample(addable, 2)
                
                # Check compatibility
                # We need to ensure adding v1 doesn't block v2 and vice versa
                # Simple check: simulate adding v1, check block_count[v2]
                
                # Simulate add v1
                S.add(v1)
                # Temporarily update block counts for v1
                temp_counts = {}
                for f_idx in edge_to_faces[v1]:
                    f = faces[f_idx]
                    others = get_other_edges(f_idx, v1)
                    o1, o2, o3 = others
                    if o2 in S and o3 in S:
                        temp_counts[o1] = temp_counts.get(o1, 0) + 1
                    if o1 in S and o3 in S:
                        temp_counts[o2] = temp_counts.get(o2, 0) + 1
                    if o1 in S and o2 in S:
                        temp_counts[o3] = temp_counts.get(o3, 0) + 1
                
                # Check if v2 is still addable
                if block_count[v2] + temp_counts.get(v2, 0) == 0:
                    # Compatible
                    # Commit v1
                    for k, val in temp_counts.items():
                        block_count[k] += val
                    # Add v2
                    S.add(v2)
                    current_size += 2
                    # Update block counts for v2
                    for f_idx in edge_to_faces[v2]:
                        f = faces[f_idx]
                        others = get_other_edges(f_idx, v2)
                        o1, o2, o3 = others
                        if o2 in S and o3 in S:
                            block_count[o1] += 1
                        if o1 in S and o3 in S:
                            block_count[o2] += 1
                        if o1 in S and o2 in S:
                            block_count[o3] += 1
                else:
                    # Not compatible, add only v1
                    for k, val in temp_counts.items():
                        block_count[k] += val
                    current_size += 1
            elif len(addable) == 1:
                v1 = addable[0]
                S.add(v1)
                current_size += 1
                for f_idx in edge_to_faces[v1]:
                    f = faces[f_idx]
                    others = get_other_edges(f_idx, v1)
                    o1, o2, o3 = others
                    if o2 in S and o3 in S:
                        block_count[o1] += 1
                    if o1 in S and o3 in S:
                        block_count[o2] += 1
                    if o1 in S and o2 in S:
                        block_count[o3] += 1
            else:
                # No addable, restore removed edge
                S.add(idx_remove)
                current_size += 1
                # Restore block counts
                for f_idx in edge_to_faces[idx_remove]:
                    f = faces[f_idx]
                    others = get_other_edges(f_idx, idx_remove)
                    o1, o2, o3 = others
                    if o2 in S and o3 in S:
                        block_count[o1] += 1
                    if o1 in S and o3 in S:
                        block_count[o2] += 1
                    if o1 in S and o2 in S:
                        block_count[o3] += 1

        return S, block_count

    S, block_count = greedy_init()
    S, block_count = local_search(S, block_count, time_limit)
    
    result = []
    for idx in sorted(S):
        u, v, i = all_edges[idx]
        result.append([u, v])
        
    return result
