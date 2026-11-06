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
    deadline = start_time + 145  # 5s margin before 150s kill
    
    def greedy_construct(strategy=0, seed=None):
        """Greedy vertex cover on the hypergraph of C4 faces.
        strategy: 0=deterministic max, 1=randomized max, 2=weighted (max + tie-break by degree)
        """
        if seed is not None:
            random.seed(seed)
        
        selected = [True] * n_edges
        face_count = [4] * n_faces
        violated = set(range(n_faces))
        
        # Precompute edge degree (number of faces each edge is in)
        edge_degree = [len(edge_faces[e]) for e in range(n_edges)]
        
        while violated:
            edge_viol_count = [0] * n_edges
            for f in violated:
                for e in face_edges[f]:
                    edge_viol_count[e] += 1
            
            max_c = max(edge_viol_count)
            if max_c == 0:
                break
            
            if strategy == 0:
                # Deterministic: pick edge with max violation count, tie-break by degree
                best_e = -1
                best_key = (-1, -1)
                for e in range(n_edges):
                    key = (edge_viol_count[e], edge_degree[e])
                    if key > best_key:
                        best_key = key
                        best_e = e
            elif strategy == 1:
                # Randomized: pick randomly among max
                candidates = [e for e in range(n_edges) if edge_viol_count[e] == max_c]
                best_e = random.choice(candidates)
            else:
                # Weighted: pick randomly among top 3 violation counts
                sorted_edges = sorted(range(n_edges), key=lambda e: (-edge_viol_count[e], -edge_degree[e]))
                top_k = min(3, len(sorted_edges))
                best_e = random.choice(sorted_edges[:top_k])
            
            selected[best_e] = False
            for f in edge_faces[best_e]:
                face_count[f] -= 1
                if face_count[f] == 4:
                    violated.add(f)
                else:
                    violated.discard(f)
        
        return selected, face_count
    
    def try_add_edges(selected, face_count, deterministic=False):
        """Try to add back removed edges greedily."""
        unselected = [e for e in range(n_edges) if not selected[e]]
        if not deterministic:
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
    
    def local_search(selected, face_count, time_limit, swap_size=1):
        """Local search with swap moves.
        swap_size: 1 for 1-for-2, 2 for 2-for-3
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Single add pass
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
            
            if not improved and time.time() < time_limit:
                # Swap moves
                if swap_size == 1:
                    # 1-for-2: remove one selected edge, try to add two unselected
                    selected_edges = [e for e in range(n_edges) if selected[e]]
                    random.shuffle(selected_edges)
                    
                    for e_remove in selected_edges[:100]:
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
                            break
                else:
                    # 2-for-3: remove two selected edges, try to add three unselected
                    selected_edges = [e for e in range(n_edges) if selected[e]]
                    random.shuffle(selected_edges)
                    
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
    
    # Phase 1: Multiple greedy strategies
    best_count = 0
    best_selected = None
    best_face_count = None
    
    for strategy in [0, 1, 2]:
        if time.time() > deadline - 5:
            break
        selected, face_count = greedy_construct(strategy=strategy)
        try_add_edges(selected, face_count, deterministic=True)
        current_count = sum(selected)
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            best_face_count = face_count.copy()
    
    # Phase 2: Local search on best initial solution
    if best_selected is not None:
        selected = best_selected.copy()
        face_count = best_face_count.copy()
        local_search(selected, face_count, deadline - 10, swap_size=1)
        current_count = sum(selected)
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            best_face_count = face_count.copy()
        
        if time.time() < deadline - 5:
            local_search(selected, face_count, deadline - 5, swap_size=2)
            current_count = sum(selected)
            if current_count > best_count:
                best_count = current_count
                best_selected = selected.copy()
                best_face_count = face_count.copy()
    
    # Phase 3: Multiple randomized restarts
    restart = 0
    while time.time() < deadline - 3:
        restart += 1
        seed = random.randint(0, 2**31 - 1)
        strategy = random.choice([1, 2])
        selected, face_count = greedy_construct(strategy=strategy, seed=seed)
        try_add_edges(selected, face_count, deterministic=False)
        
        # Quick local search
        local_search(selected, face_count, deadline - 3, swap_size=1)
        current_count = sum(selected)
        
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            best_face_count = face_count.copy()
        
        # If we have time, do a second pass with larger swaps
        if time.time() < deadline - 1 and current_count >= best_count - 2:
            local_search(selected, face_count, deadline - 1, swap_size=2)
            current_count = sum(selected)
            if current_count > best_count:
                best_count = current_count
                best_selected = selected.copy()
                best_face_count = face_count.copy()
    
    # Build result from best solution
    result = []
    for i, (u, v) in enumerate(edges):
        if best_selected[i]:
            result.append([u, v])
    
    return result
