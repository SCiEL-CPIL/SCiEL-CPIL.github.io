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
    MAX_EDGES = 308 # Upper bound
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    def greedy_construct(randomized=False, seed=None):
        """Greedy vertex cover on the hypergraph of C4 faces."""
        if seed is not None:
            random.seed(seed)
        
        selected = [True] * n_edges
        face_count = [4] * n_faces
        violated = set(range(n_faces))
        
        while violated:
            edge_viol_count = [0] * n_edges
            for f in violated:
                for e in face_edges[f]:
                    edge_viol_count[e] += 1
            
            if max(edge_viol_count) == 0:
                break
            
            if randomized:
                max_c = max(edge_viol_count)
                candidates = [e for e in range(n_edges) if edge_viol_count[e] == max_c]
                best_e = random.choice(candidates)
            else:
                best_e = max(range(n_edges), key=lambda e: edge_viol_count[e])
            
            selected[best_e] = False
            for f in edge_faces[best_e]:
                face_count[f] -= 1
                if face_count[f] == 4:
                    violated.add(f)
                else:
                    violated.discard(f)
        
        return selected, face_count
    
    def try_add_edges(selected, face_count):
        """Try to add back removed edges greedily."""
        unselected = [e for e in range(n_edges) if not selected[e]]
        random.shuffle(unselected)
        added = 0
        for e in unselected:
            can_add = True
            for f in edge_faces[e]:
                if face_count[f] >= 3:
                    can_add = False
                    break
            if can_add:
                selected[e] = True
                for f in edge_faces[e]:
                    face_count[f] += 1
                added += 1
        return added
    
    def simple_local_search(selected, face_count, time_limit):
        """Simple local search: repeatedly try single adds until no improvement."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            unselected = [e for e in range(n_edges) if not selected[e]]
            random.shuffle(unselected)
            for e in unselected:
                if time.time() > time_limit:
                    break
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
        return improved
    
    def swap_local_search(selected, face_count, time_limit):
        """Swap-based local search: 1-for-2 and 2-for-3 swaps."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1-for-2 Swap
            selected_edges = [e for e in range(n_edges) if selected[e]]
            random.shuffle(selected_edges)
            
            for e_remove in selected_edges:
                if time.time() > time_limit:
                    break
                
                # Temporarily remove
                selected[e_remove] = False
                for f in edge_faces[e_remove]:
                    face_count[f] -= 1
                
                # Try to add back 2 edges
                added_back = []
                unselected_now = [e for e in range(n_edges) if not selected[e]]
                random.shuffle(unselected_now)
                
                for e_add in unselected_now:
                    if len(added_back) >= 2:
                        break
                    can_add = True
                    for f in edge_faces[e_add]:
                        if face_count[f] >= 3:
                            can_add = False
                            break
                    if can_add:
                        selected[e_add] = True
                        for f in edge_faces[e_add]:
                            face_count[f] += 1
                        added_back.append(e_add)
                
                if len(added_back) < 2:
                    # Revert
                    selected[e_remove] = True
                    for f in edge_faces[e_remove]:
                        face_count[f] += 1
                    for e_add in added_back:
                        selected[e_add] = False
                        for f in edge_faces[e_add]:
                            face_count[f] -= 1
                else:
                    improved = True
                    break  # Found an improvement, restart loop
            
            if improved:
                continue
                
            # 2-for-3 Swap
            if time.time() < time_limit:
                selected_edges = [e for e in range(n_edges) if selected[e]]
                random.shuffle(selected_edges)
                
                # Try pairs of edges to remove
                # To keep it fast, we only try a limited number of pairs
                for i in range(min(50, len(selected_edges))):
                    if time.time() > time_limit:
                        break
                    e1 = selected_edges[i]
                    for j in range(i + 1, min(50, len(selected_edges))):
                        e2 = selected_edges[j]
                        
                        # Temporarily remove e1, e2
                        selected[e1] = False
                        selected[e2] = False
                        for f in edge_faces[e1]:
                            face_count[f] -= 1
                        for f in edge_faces[e2]:
                            face_count[f] -= 1
                        
                        # Try to add back 3 edges
                        added_back = []
                        unselected_now = [e for e in range(n_edges) if not selected[e]]
                        random.shuffle(unselected_now)
                        
                        for e_add in unselected_now:
                            if len(added_back) >= 3:
                                break
                            can_add = True
                            for f in edge_faces[e_add]:
                                if face_count[f] >= 3:
                                    can_add = False
                                    break
                            if can_add:
                                selected[e_add] = True
                                for f in edge_faces[e_add]:
                                    face_count[f] += 1
                                added_back.append(e_add)
                        
                        if len(added_back) < 3:
                            # Revert
                            selected[e1] = True
                            selected[e2] = True
                            for f in edge_faces[e1]:
                                face_count[f] += 1
                            for f in edge_faces[e2]:
                                face_count[f] += 1
                            for e_add in added_back:
                                selected[e_add] = False
                                for f in edge_faces[e_add]:
                                    face_count[f] -= 1
                        else:
                            improved = True
                            break
                    if improved:
                        break
        return improved
    
    # Phase 1: Initial greedy (deterministic)
    selected, face_count = greedy_construct(randomized=False)
    try_add_edges(selected, face_count)
    best_count = sum(selected)
    best_selected = selected.copy()
    best_face_count = face_count.copy()
    
    if best_count >= MAX_EDGES:
        return [[edges[i][0], edges[i][1]] for i in range(n_edges) if best_selected[i]]
        
    # Phase 2: Local search on initial solution
    simple_local_search(selected, face_count, deadline)
    current_count = sum(selected)
    if current_count > best_count:
        best_count = current_count
        best_selected = selected.copy()
        best_face_count = face_count.copy()
        if best_count >= MAX_EDGES:
            return [[edges[i][0], edges[i][1]] for i in range(n_edges) if best_selected[i]]
            
    swap_local_search(selected, face_count, deadline)
    current_count = sum(selected)
    if current_count > best_count:
        best_count = current_count
        best_selected = selected.copy()
        best_face_count = face_count.copy()
        if best_count >= MAX_EDGES:
            return [[edges[i][0], edges[i][1]] for i in range(n_edges) if best_selected[i]]
    
    # Phase 3: Multiple randomized restarts
    for restart in range(500):
        if time.time() > deadline - 2:
            break
        
        seed = random.randint(0, 2**31 - 1)
        selected, face_count = greedy_construct(randomized=True, seed=seed)
        try_add_edges(selected, face_count)
        simple_local_search(selected, face_count, deadline - 1)
        swap_local_search(selected, face_count, deadline - 1)
        
        current_count = sum(selected)
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            best_face_count = face_count.copy()
            if best_count >= MAX_EDGES:
                return [[edges[i][0], edges[i][1]] for i in range(n_edges) if best_selected[i]]
    
    # Build result from best solution
    result = []
    for i, (u, v) in enumerate(edges):
        if best_selected[i]:
            result.append([u, v])
    
    return result
