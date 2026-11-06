import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # Generate all edges
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
    
    # Precompute faces and edge-face incidence
    face_edges = []
    edge_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                face_id = len(face_edges)
                face_edges.append([e1, e2, e3, e4])
                
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
    
    n_faces = len(face_edges)
    
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    def greedy_construct(seed_order=None):
        """Greedy removal with optional random tie-breaking."""
        selected = [True] * n_edges
        face_count = [4] * n_faces
        violated = list(range(n_faces))
        
        if seed_order is not None:
            random.shuffle(violated)
        
        while violated:
            if time.time() > deadline:
                break
            # Count how many violated faces each edge is in
            edge_violation_count = [0] * n_edges
            for f in violated:
                for e in face_edges[f]:
                    edge_violation_count[e] += 1
            # Pick edge with max violation count (random among ties)
            max_count = max(edge_violation_count)
            candidates = [e for e in range(n_edges) if edge_violation_count[e] == max_count]
            best_e = random.choice(candidates)
            selected[best_e] = False
            for f in edge_faces[best_e]:
                face_count[f] -= 1
            # Update violated list
            violated = [f for f in violated if face_count[f] == 4]
        
        return selected, face_count
    
    def try_add_edges(selected, face_count):
        """Try to add back edges that don't create violations."""
        unselected = [e for e in range(n_edges) if not selected[e]]
        random.shuffle(unselected)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for e in unselected:
                if selected[e]:
                    continue
                can_add = True
                for f in edge_faces[e]:
                    if face_count[f] >= 3:
                        can_add = False
                        break
                if can_add:
                    selected[e] = True
                    for f in edge_faces[e]:
                        face_count[f] += 1
                    improved = True
        return selected, face_count
    
    def swap_local_search(selected, face_count, max_rounds=50):
        """Local search with swap moves: remove one edge, add another."""
        best_count = sum(selected)
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 2:
                break
            
            # Try all possible swaps
            unselected = [e for e in range(n_edges) if not selected[e]]
            random.shuffle(unselected)
            
            found_swap = False
            for e_add in unselected:
                if time.time() > deadline - 2:
                    break
                if selected[e_add]:
                    continue
                
                # Check if e_add can be added (ignoring current state)
                # We need to find an edge to remove first
                # e_add is addable if after removing some edge, all its faces have count < 3
                
                # For each face containing e_add, if face_count[f] >= 3,
                # we need to remove an edge from that face (not e_add itself)
                blocking_faces = [f for f in edge_faces[e_add] if face_count[f] >= 3]
                
                if not blocking_faces:
                    # Can add directly
                    selected[e_add] = True
                    for f in edge_faces[e_add]:
                        face_count[f] += 1
                    best_count += 1
                    found_swap = True
                    break
                
                # Need to remove an edge from at least one blocking face
                # Find edges that appear in multiple blocking faces
                candidate_removals = {}
                for f in blocking_faces:
                    for e in face_edges[f]:
                        if e != e_add:
                            candidate_removals[e] = candidate_removals.get(e, 0) + 1
                
                if not candidate_removals:
                    continue
                
                # Try removing the most beneficial edge
                # Sort by number of blocking faces it's in
                sorted_removals = sorted(candidate_removals.items(), key=lambda x: -x[1])
                
                for e_rem, _ in sorted_removals[:10]:  # Try top 10 candidates
                    if selected[e_rem] is False:
                        continue
                    
                    # Check: after removing e_rem, can we add e_add?
                    # After removal, face_count for faces of e_rem decreases by 1
                    # Then check if e_add is addable
                    
                    new_face_count = face_count.copy()
                    for f in edge_faces[e_rem]:
                        new_face_count[f] -= 1
                    
                    # Check if e_add is now addable
                    can_add = True
                    for f in edge_faces[e_add]:
                        if new_face_count[f] >= 3:
                            can_add = False
                            break
                    
                    if can_add:
                        # Perform the swap
                        selected[e_rem] = False
                        selected[e_add] = True
                        for f in edge_faces[e_rem]:
                            face_count[f] -= 1
                        for f in edge_faces[e_add]:
                            face_count[f] += 1
                        best_count += 0  # Net zero change in count
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                # No more swaps possible, try add-only
                # (already handled by try_add_edges)
                break
        
        return selected, face_count, best_count
    
    # Main search loop
    best_count = 0
    best_selected = None
    best_face_count = None
    
    num_restarts = 0
    while time.time() < deadline - 3:
        num_restarts += 1
        
        # Greedy construction with randomization
        selected, face_count = greedy_construct()
        
        # Try to add edges back
        selected, face_count = try_add_edges(selected, face_count)
        
        # Swap local search
        selected, face_count, cur_count = swap_local_search(selected, face_count)
        
        if cur_count > best_count:
            best_count = cur_count
            best_selected = selected.copy()
            best_face_count = face_count.copy()
        
        # If we've found a good solution, do more refined search
        if best_count >= 300 and time.time() < deadline - 5:
            # Try more aggressive swap search on the best solution
            sel2 = best_selected.copy()
            fc2 = best_face_count.copy()
            sel2, fc2, cnt2 = swap_local_search(sel2, fc2, max_rounds=100)
            if cnt2 > best_count:
                best_count = cnt2
                best_selected = sel2.copy()
                best_face_count = fc2.copy()
    
    # Build result
    result = []
    for i, (u, v) in enumerate(edges):
        if best_selected[i]:
            result.append([u, v])
    
    return result
