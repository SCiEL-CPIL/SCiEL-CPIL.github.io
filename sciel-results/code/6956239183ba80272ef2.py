import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110.0
    d = 7
    n = 1 << d
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v))
    
    num_edges = len(all_edges)
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute face neighbors
    face_neighbors = [[] for _ in range(num_edges)]
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            face_neighbors[idx].append((edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3]))
    
    # Precompute reverse face neighbors: for each edge, which edges block it
    # edge_blocks[e] = list of (neighbor1, neighbor2, neighbor3) triples
    # Actually, we need: for edge e, the set of other edges that share a face with e
    # If all 3 in a face are selected, e cannot be added.
    # So blocked_count[e] = number of faces where all 3 other edges are selected
    # When we add/remove an edge, we update blocked counts of edges sharing faces with it.
    
    # For each edge, precompute the list of (face_idx, other_edge_in_face) pairs
    # face_neighbors[idx] already has this: list of 3-tuples of the other 3 edges in each face
    
    def greedy_fill(order):
        selected = [False] * num_edges
        blocked = [0] * num_edges  # blocked[e] = number of faces where all 3 others are selected
        
        for idx in order:
            if blocked[idx] == 0:
                selected[idx] = True
                # Update blocked counts for edges that share a face with idx
                for o1, o2, o3 in face_neighbors[idx]:
                    for other in (o1, o2, o3):
                        # Check if adding idx blocks 'other' in any face
                        # 'other' is blocked in a face if all 3 edges in that face (excluding other) are selected
                        # The face containing both idx and other: the 3rd edge in that face
                        # We need to check each face of 'other' that contains idx
                        pass  # We'll handle this differently
        
        # Simpler approach: just check validity directly but cache results
        # Let's use a different strategy: maintain for each unselected edge, how many of its
        # face-mates are selected. If any face has all 3 mates selected, it's blocked.
        
        selected = [False] * num_edges
        # mate_count[e][f] not needed; just count how many of the 3 edges in each face are selected
        # face_selected_count[e][f_idx] = number of the 3 other edges in face f of edge e that are selected
        # This is too complex. Let's just do efficient validity checks.
        
        # Better: for each edge, precompute which faces it belongs to and the other 3 edges
        # face_neighbors[idx] = [(o1, o2, o3), ...] for each face
        
        selected = [False] * num_edges
        for idx in order:
            valid = True
            for o1, o2, o3 in face_neighbors[idx]:
                if selected[o1] and selected[o2] and selected[o3]:
                    valid = False
                    break
            if valid:
                selected[idx] = True
        
        return selected

    def local_search(selected):
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Phase 1: Add any valid edge
            for idx in range(num_edges):
                if not selected[idx]:
                    valid = True
                    for o1, o2, o3 in face_neighbors[idx]:
                        if selected[o1] and selected[o2] and selected[o3]:
                            valid = False
                            break
                    if valid:
                        selected[idx] = True
                        improved = True
            
            if time.time() > deadline - 5.0:
                break

            # Phase 2: Try 1-for-k swaps
            sel_indices = [i for i in range(num_edges) if selected[i]]
            random.shuffle(sel_indices)
            
            for idx in sel_indices:
                if not selected[idx]:
                    continue
                # Temporarily remove
                selected[idx] = False
                
                # Find valid unselected edges
                valid_adds = []
                for jdx in range(num_edges):
                    if not selected[jdx]:
                        valid = True
                        for o1, o2, o3 in face_neighbors[jdx]:
                            if selected[o1] and selected[o2] and selected[o3]:
                                valid = False
                                break
                        if valid:
                            valid_adds.append(jdx)
                
                if len(valid_adds) >= 2:
                    added_count = 0
                    for v_idx in valid_adds:
                        if is_valid_add(selected, v_idx):
                            selected[v_idx] = True
                            added_count += 1
                        if added_count >= 2:
                            break
                    
                    if added_count >= 2:
                        improved = True
                        break
                    else:
                        for v_idx in valid_adds:
                            if selected[v_idx]:
                                selected[v_idx] = False
                        selected[idx] = True
                else:
                    selected[idx] = True
        
        return selected
    
    def is_valid_add(selected, idx):
        for o1, o2, o3 in face_neighbors[idx]:
            if selected[o1] and selected[o2] and selected[o3]:
                return False
        return True

    best_selected = None
    best_count = 0
    
    # Structured initial solution: use a deterministic pattern
    # One known good construction: select all edges except those in a specific pattern
    # For now, use random restarts with the improved local search
    
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        order = list(range(num_edges))
        random.shuffle(order)
        selected = greedy_fill(order)
        selected = local_search(selected)
        count = sum(selected)
        if count > best_count:
            best_count = count
            best_selected = selected[:]
            
    # Final local search on the best
    if best_selected:
        best_selected = local_search(best_selected)
        best_count = sum(best_selected)

    out = []
    for idx in range(num_edges):
        if best_selected[idx]:
            u, v = all_edges[idx]
            out.append([u, v])
    return out
