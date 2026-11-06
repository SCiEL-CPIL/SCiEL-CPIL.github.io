import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
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
                
    # 2. Precompute faces
    face_edge_list = []
    seen_faces = set()
    
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
            
    # 3. Strategy: Greedy Construction with Lookahead + Local Search
    # We want to maximize the number of selected edges such that no face is fully selected.
    
    def greedy_construct(order, rng):
        """
        Construct a valid set of edges by greedily adding edges in a given order,
        skipping an edge if it would complete a face.
        """
        selected = np.zeros(num_edges, dtype=np.uint8)
        face_counts = np.zeros(num_faces, dtype=np.int32)
        
        for e in order:
            # Check if adding e completes any face
            valid = True
            for f in edge_to_faces[e]:
                if face_counts[f] == 3:
                    valid = False
                    break
            
            if valid:
                selected[e] = 1
                for f in edge_to_faces[e]:
                    face_counts[f] += 1
                    
        return selected, face_counts, np.sum(selected)
        
    def local_search(selected, face_counts, rng, max_iters=500):
        """
        Local search: Try to add edges that were previously rejected, or swap edges.
        Here we focus on "add" moves: if an edge was not selected, check if it can be added now
        (this might happen if we removed other edges, but here we only add).
        Actually, since we only add in greedy, the only way to improve is to find a different greedy order
        or to do "swap" moves: remove one edge to allow adding multiple others.
        
        Let's implement "Re-add" phase: 
        The greedy algorithm might reject an edge e because face f had 3 edges.
        If we remove one of those 3 edges, e might become addable.
        But removing one edge loses 1, adding e gains 1. Net 0.
        However, removing that edge might allow adding OTHER edges too.
        
        Simpler: Randomized Greedy + Local "Flip" search.
        Let's try to find a better order.
        
        Alternative Local Search: 
        Start with a valid set.
        Try to add a random unselected edge. If it creates a conflict, remove a random edge from the conflict.
        Check if the net change is positive (unlikely for 1-for-1) or if we can add more.
        
        Let's stick to multiple randomized greedy runs with different "temperatures" or order perturbations.
        And a "repair" step: 
        Given a set, try to add edges. If an edge e is blocked by face f (which has 3 edges), 
        we can remove one of the 3 edges to allow e. 
        Score change: -1 (removed) + 1 (added) = 0. 
        But maybe removing that edge allows adding 2 more edges?
        
        Let's implement a "Swap" local search:
        1. Pick a random unselected edge e.
        2. Find the faces where e is blocked (face_counts[f] == 3).
        3. For each such face, pick a random edge r from the other 3 edges in that face.
        4. Tentatively remove r and add e.
        5. This is valid.
        6. Now, try to greedily add other unselected edges that might have been blocked by r or e.
        7. If the final count is higher, accept.
        """
        current_count = np.sum(selected)
        current_selected = selected.copy()
        current_face_counts = face_counts.copy()
        
        # We will perform a number of iterations
        improved = False
        
        # Heuristic: Try to add edges that are "almost" valid (blocked by only 1 face?)
        # Or just random swaps.
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick a random unselected edge
            unselected_edges = np.where(current_selected == 0)[0]
            if len(unselected_edges) == 0:
                break
                
            e = unselected_edges[rng.integers(0, len(unselected_edges))]
            
            # Check if it's already blocked
            # It is blocked if any face has count 3.
            blocking_faces = []
            for f in edge_to_faces[e]:
                if current_face_counts[f] == 3:
                    blocking_faces.append(f)
                    
            if not blocking_faces:
                # This edge can be added! Greedy missed it? 
                # This can happen if we did swaps that reduced counts.
                current_selected[e] = 1
                for f in edge_to_faces[e]:
                    current_face_counts[f] += 1
                current_count += 1
                improved = True
                continue # Continue to add more if possible
            
            # Pick a random blocking face
            f = blocking_faces[rng.integers(0, len(blocking_faces))]
            
            # The 3 edges in this face currently selected
            face_edges = face_edge_list[f]
            other_edges = [x for x in face_edges if x != e] # 3 edges
            
            # Pick a random edge to remove
            r = other_edges[rng.integers(0, len(other_edges))]
            
            # Check if r is actually selected
            if current_selected[r] != 1:
                continue # Should not happen if logic is right, but safe check
                
            # Tentative move: Remove r, Add e
            # But first, check if removing r opens up other opportunities.
            
            # Save state
            saved_selected = current_selected.copy()
            saved_face_counts = current_face_counts.copy()
            saved_count = current_count
            
            # Apply move
            current_selected[r] = 0
            for f_idx in edge_to_faces[r]:
                current_face_counts[f_idx] -= 1
            current_count -= 1
            
            current_selected[e] = 1
            for f_idx in edge_to_faces[e]:
                current_face_counts[f_idx] += 1
            current_count += 1
            
            # Now try to greedily add other edges
            # We can try adding edges that were blocked by r or e.
            # Simple approach: scan all unselected edges and add if possible.
            # This is O(N * F) which is fast enough for 448 edges.
            
            changed = True
            while changed:
                changed = False
                for u_e in range(num_edges):
                    if current_selected[u_e] == 1:
                        continue
                    # Check if u_e can be added
                    can_add = True
                    for f_idx in edge_to_faces[u_e]:
                        if current_face_counts[f_idx] == 3:
                            can_add = False
                            break
                    if can_add:
                        current_selected[u_e] = 1
                        for f_idx in edge_to_faces[u_e]:
                            current_face_counts[f_idx] += 1
                        current_count += 1
                        changed = True
            
            # If improved, keep. Else revert.
            if current_count > saved_count:
                improved = True
            else:
                # Revert
                current_selected = saved_selected
                current_face_counts = saved_face_counts
                current_count = saved_count
                
        return current_selected, current_face_counts, current_count, improved

    best_state = None
    best_count = 0
    best_fc = None
    
    rng = np.random.default_rng(123)
    
    # Phase 1: Randomized Greedy
    # Try different random orders
    num_greedy_runs = 100
    for i in range(num_greedy_runs):
        if time.time() > deadline - 30:
            break
            
        order = np.arange(num_edges)
        rng.shuffle(order)
        
        # Perturb the order: Put some edges first?
        # Or just random is fine.
        
        sel, fc, count = greedy_construct(order, rng)
        
        if count > best_count:
            best_count = count
            best_state = sel
            best_fc = fc
            
    # Phase 2: Local Search on the best state
    if best_state is not None:
        sel, fc, count, improved = local_search(best_state, best_fc, rng, max_iters=2000)
        
        if count > best_count:
            best_count = count
            best_state = sel
            best_fc = fc
            
    # Phase 3: Try more diverse greedy starts if time permits
    # Maybe "Smart" Greedy: Start with edges that are in fewer faces? All are in 6 faces.
    # Start with edges that are "harder" to satisfy? 
    # Let's try: Sort edges by "random weight" and add in that order.
    
    if time.time() < deadline - 10:
        # Try 50 more runs with "biased" orders
        for i in range(50):
            if time.time() > deadline - 10:
                break
            
            # Biased order: Random weights
            weights = rng.random(num_edges)
            order = np.argsort(weights)
            
            sel, fc, count = greedy_construct(order, rng)
            
            # Immediate local search?
            # Only if it's close to best or better
            if count >= best_count - 2:
                sel2, fc2, count2, _ = local_search(sel, fc, rng, max_iters=200)
                if count2 > best_count:
                    best_count = count2
                    best_state = sel2
                    best_fc = fc2

    # Final Polish: One more local search on the absolute best
    if best_state is not None and time.time() < deadline - 2:
        sel, fc, count, _ = local_search(best_state, best_fc, rng, max_iters=1000)
        if count > best_count:
            best_count = count
            best_state = sel

    # Convert to list of edges
    result = []
    for i in range(num_edges):
        if best_state[i] == 1:
            result.append(list(idx_to_edge[i]))
            
    return result
