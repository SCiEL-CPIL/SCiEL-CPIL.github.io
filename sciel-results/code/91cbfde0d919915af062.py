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
    # For edge (u,v) with diff bit i, and other bit j:
    # Face vertices: u, v, u^j, v^j
    # Other edges: (u, u^j), (v, v^j), (u^j, v^j)
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

    def count_edges(selected):
        return sum(selected)

    def is_valid_addition(selected, idx):
        for o1, o2, o3 in face_neighbors[idx]:
            if selected[o1] and selected[o2] and selected[o3]:
                return False
        return True

    def greedy_fill(order):
        selected = [False] * num_edges
        for idx in order:
            if is_valid_addition(selected, idx):
                selected[idx] = True
        return selected

    def local_search(selected):
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # 1. Add any valid edge
            for idx in range(num_edges):
                if not selected[idx] and is_valid_addition(selected, idx):
                    selected[idx] = True
                    improved = True
            
            if time.time() > deadline - 5.0:
                break

            # 2. Try 1-for-k swaps
            # We iterate through selected edges. If removing one allows adding >= 2, we do it.
            # To be efficient, we can just try all selected edges in random order.
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
                    if not selected[jdx] and is_valid_addition(selected, jdx):
                        valid_adds.append(jdx)
                
                if len(valid_adds) >= 2:
                    # Add 2 of them
                    # Greedily pick 2 that don't conflict with each other?
                    # Since they were invalid before, they likely don't conflict with each other
                    # unless they share a face where the 3rd edge is present.
                    # But if they are in the same face, and we add both, do they create a C4?
                    # A C4 requires 4 edges. If we add 2, we have 2 new + 2 old?
                    # Actually, if we add e1 and e2, they might be in the same face.
                    # If e1 and e2 are in the same face, the other 2 edges of that face must be present?
                    # No, the face has 4 edges. If e1 and e2 are in it, the other 2 are already there?
                    # If the other 2 are there, then e1 and e2 would have been invalid before?
                    # Not necessarily.
                    # Safe bet: add them one by one, checking validity.
                    added_count = 0
                    for v_idx in valid_adds:
                        if is_valid_addition(selected, v_idx):
                            selected[v_idx] = True
                            added_count += 1
                        if added_count >= 2:
                            break
                    
                    if added_count >= 2:
                        improved = True
                        # Break and restart loop? Or continue?
                        # Breaking allows re-evaluation of other edges.
                        break
                    else:
                        # Revert if we couldn't add 2
                        # But we already added some? No, we check validity before adding.
                        # If we added 1, we have net 0. We should revert to keep state clean?
                        # Or keep it? Keeping it might help later.
                        # Let's revert if we didn't get net gain.
                        for v_idx in valid_adds:
                            if selected[v_idx]:
                                selected[v_idx] = False
                        selected[idx] = True
                else:
                    # Revert
                    selected[idx] = True
        
        return selected

    best_selected = None
    best_count = 0
    
    # Multiple random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
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
