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

    def greedy_construct(randomized=True, seed=None):
        """Greedy edge selection avoiding full faces."""
        if seed is not None:
            random.seed(seed)
        
        selected = [False] * n_edges
        face_count = [0] * n_faces
        
        # Order edges randomly if randomized
        edge_indices = list(range(n_edges))
        if randomized:
            random.shuffle(edge_indices)
        
        for e in edge_indices:
            # Check if adding this edge completes any face
            can_add = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    can_add = False
                    break
            if can_add:
                selected[e] = True
                for f in edge_faces[e]:
                    face_count[f] += 1
        
        return selected, face_count

    def try_add_edges(selected, face_count):
        """Try to add back removed edges greedily."""
        improved = True
        while improved:
            improved = False
            unselected = [e for e in range(n_edges) if not selected[e]]
            random.shuffle(unselected)
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
                    improved = True
        return improved

    def local_search(selected, face_count, time_limit):
        """Local search with 1-for-2 swaps."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
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
        
        return improved

    # Phase 1: Multiple randomized restarts
    best_count = 0
    best_selected = None
    best_face_count = None
    
    # Initial deterministic run
    selected, face_count = greedy_construct(randomized=False)
    try_add_edges(selected, face_count)
    best_count = sum(selected)
    best_selected = selected.copy()
    best_face_count = face_count.copy()
    
    if best_count >= MAX_EDGES:
        return [[edges[i][0], edges[i][1]] for i in range(n_edges) if best_selected[i]]

    # Local search on initial
    local_search(selected, face_count, deadline - 10)
    current_count = sum(selected)
    if current_count > best_count:
        best_count = current_count
        best_selected = selected.copy()
        best_face_count = face_count.copy()
        if best_count >= MAX_EDGES:
            return [[edges[i][0], edges[i][1]] for i in range(n_edges) if best_selected[i]]

    # Phase 2: Randomized restarts
    while time.time() < deadline - 5:
        seed = random.randint(0, 2**31 - 1)
        selected, face_count = greedy_construct(randomized=True, seed=seed)
        try_add_edges(selected, face_count)
        
        # Local search with limited time
        local_search(selected, face_count, deadline - 2)
        
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
