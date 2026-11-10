import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    d = 7
    n_vertices = 1 << d  # 128
    n_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    all_edges = []
    edge_idx = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
                edge_idx[(x, y)] = len(all_edges) - 1
    
    # Precompute faces: for each face, the 4 edges
    # A face is determined by vertex x and two distinct bit positions i < j
    # Vertices: x, x^(1<<i), x^(1<<j), x^(1<<(i|j))
    # Edges: (x, x^i), (x^i, x^(i|j)), (x^(i|j), x^j), (x^j, x)
    faces = []
    for x in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                vi = x ^ (1 << i)
                vj = x ^ (1 << j)
                vij = x ^ (1 << i) ^ (1 << j)
                e1 = (min(x, vi), max(x, vi))
                e2 = (min(vi, vij), max(vi, vij))
                e3 = (min(vij, vj), max(vij, vj))
                e4 = (min(x, vj), max(x, vj))
                faces.append((edge_idx[e1], edge_idx[e2], edge_idx[e3], edge_idx[e4]))
    
    n_faces = len(faces)  # C(7,2) * 128 / 4 = 21 * 32 = 672... let me compute
    # Actually: for each x (128), for each pair (i,j) (21 pairs), but each face is counted 4 times
    # So n_faces = 128 * 21 / 4 = 672
    
    # For each edge, precompute which faces contain it
    edge_faces = [[] for _ in range(n_edges)]
    for fi, face in enumerate(faces):
        for ei in face:
            edge_faces[ei].append(fi)
    
    def count_edges(chosen):
        return sum(chosen)
    
    def is_valid(chosen):
        # Check if any face has all 4 edges selected
        for face in faces:
            if all(chosen[e] for e in face):
                return False
        return True
    
    def greedy_random(rng):
        """Randomized greedy: shuffle edges, add if doesn't create C4."""
        order = list(range(n_edges))
        rng.shuffle(order)
        chosen = [False] * n_edges
        face_counts = [0] * n_faces  # how many edges of each face are selected
        
        for ei in order:
            # Check if adding this edge would complete any face
            can_add = True
            for fi in edge_faces[ei]:
                if face_counts[fi] == 3:
                    can_add = False
                    break
            if can_add:
                chosen[ei] = True
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        return chosen, face_counts
    
    def local_search(chosen, face_counts, rng, time_limit):
        """Local search: try single-edge additions and 1-for-1 swaps."""
        n_selected = sum(chosen)
        
        # Try additions first
        while time.time() < time_limit:
            improved = False
            # Try adding a random edge
            for _ in range(500):
                ei = rng.randint(0, n_edges - 1)
                if chosen[ei]:
                    continue
                # Check if adding ei would complete a face
                can_add = True
                for fi in edge_faces[ei]:
                    if face_counts[fi] == 3:
                        can_add = False
                        break
                if can_add:
                    chosen[ei] = True
                    for fi in edge_faces[ei]:
                        face_counts[fi] += 1
                    n_selected += 1
                    improved = True
                    break
            if not improved:
                break
        
        # Try 1-for-1 swaps: remove one edge, add another
        while time.time() < time_limit:
            improved = False
            for _ in range(200):
                # Pick a random selected edge to remove
                ei_remove = rng.randint(0, n_edges - 1)
                if not chosen[ei_remove]:
                    continue
                
                # Temporarily remove it
                chosen[ei_remove] = False
                for fi in edge_faces[ei_remove]:
                    face_counts[fi] -= 1
                
                # Try to add a random unselected edge
                ei_add = rng.randint(0, n_edges - 1)
                if chosen[ei_add]:
                    # Restore and try again
                    chosen[ei_remove] = True
                    for fi in edge_faces[ei_remove]:
                        face_counts[fi] += 1
                    continue
                
                can_add = True
                for fi in edge_faces[ei_add]:
                    if face_counts[fi] == 3:
                        can_add = False
                        break
                
                if can_add:
                    chosen[ei_add] = True
                    for fi in edge_faces[ei_add]:
                        face_counts[fi] += 1
                    # Keep the removal, accept the swap
                    improved = True
                    break
                else:
                    # Restore
                    chosen[ei_remove] = True
                    for fi in edge_faces[ei_remove]:
                        face_counts[fi] += 1
            
            if not improved:
                break
        
        return chosen, face_counts, n_selected
    
    best_count = 0
    best_chosen = None
    
    rng = random.Random(42)
    
    # Multiple restarts with randomized greedy + local search
    n_restarts = 0
    while time.time() < deadline - 5.0:  # leave 5s buffer
        restart_start = time.time()
        chosen, face_counts = greedy_random(rng)
        n_selected = sum(chosen)
        
        # Local search with time limit
        remaining = deadline - 5.0 - time.time()
        if remaining > 0.1:
            chosen, face_counts, n_selected = local_search(
                chosen, face_counts, rng, time.time() + min(remaining * 0.5, 2.0)
            )
        
        if n_selected > best_count:
            best_count = n_selected
            best_chosen = chosen[:]
        
        n_restarts += 1
        
        # If we've hit the known best, try to improve with more aggressive local search
        if best_count >= 304 and time.time() < deadline - 10.0:
            # Try more swaps
            for _ in range(1000):
                if time.time() > deadline - 5.0:
                    break
                ei_remove = rng.randint(0, n_edges - 1)
                if not best_chosen[ei_remove]:
                    continue
                
                # Recompute face counts for best_chosen
                fc = [0] * n_faces
                for ei in range(n_edges):
                    if best_chosen[ei]:
                        for fi in edge_faces[ei]:
                            fc[fi] += 1
                
                best_chosen[ei_remove] = False
                for fi in edge_faces[ei_remove]:
                    fc[fi] -= 1
                
                ei_add = rng.randint(0, n_edges - 1)
                if best_chosen[ei_add]:
                    best_chosen[ei_remove] = True
                    for fi in edge_faces[ei_remove]:
                        fc[fi] += 1
                    continue
                
                can_add = True
                for fi in edge_faces[ei_add]:
                    if fc[fi] == 3:
                        can_add = False
                        break
                
                if can_add:
                    best_chosen[ei_add] = True
                    for fi in edge_faces[ei_add]:
                        fc[fi] += 1
                    best_count = sum(best_chosen)
                else:
                    best_chosen[ei_remove] = True
                    for fi in edge_faces[ei_remove]:
                        fc[fi] += 1
    
    # Convert best_chosen to list of edges
    result = []
    for ei in range(n_edges):
        if best_chosen[ei]:
            result.append(list(all_edges[ei]))
    
    return result
