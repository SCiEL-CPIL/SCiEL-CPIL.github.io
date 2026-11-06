import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    d = 7
    n_vertices = 1 << d
    n_edges = n_vertices * d // 2
    
    # Precompute edges
    # Edge representation: (u, v) with u < v
    # We need to map edge index to (u, v) and vice versa
    edges = []
    edge_map = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_map[(u, v)] = idx
                idx += 1
                
    # Precompute faces
    # A face is defined by a base vertex u and two dimensions i, j (i < j)
    # The 4 vertices are u, u^i, u^j, u^i^j
    # The 4 edges are (u, u^i), (u^j, u^i^j), (u, u^j), (u^i, u^i^j)
    # Note: (u, u^i) is same as (u^i, u) etc. We use canonical (min, max).
    
    faces = [] # List of lists of edge indices
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                # Edges:
                # e1: u - v1 (dim i)
                # e2: v2 - v3 (dim i)
                # e3: u - v2 (dim j)
                # e4: v1 - v3 (dim j)
                
                e1 = edge_map[(min(u, v1), max(u, v1))]
                e2 = edge_map[(min(v2, v3), max(v2, v3))]
                e3 = edge_map[(min(u, v2), max(u, v2))]
                e4 = edge_map[(min(v1, v3), max(v1, v3))]
                
                faces.append([e1, e2, e3, e4])
                
    n_faces = len(faces)
    
    # For each edge, which faces does it belong to?
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, f_edges in enumerate(faces):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)
            
    # State
    # selected: boolean array of size n_edges
    # face_counts: array of size n_faces, count of selected edges in that face
    
    def try_add(e_idx):
        # Check if adding e_idx creates a C4
        # A C4 is created if any face containing e_idx already has 3 selected edges
        for f_idx in edge_faces[e_idx]:
            if face_counts[f_idx] == 3:
                return False
        return True
        
    def add_edge(e_idx):
        selected[e_idx] = True
        current_size[0] += 1
        for f_idx in edge_faces[e_idx]:
            face_counts[f_idx] += 1
            
    def remove_edge(e_idx):
        selected[e_idx] = False
        current_size[0] -= 1
        for f_idx in edge_faces[e_idx]:
            face_counts[f_idx] -= 1
            
    best_size = 0
    best_set = [False] * n_edges
    
    # Multi-start local search
    num_starts = 50
    time_per_start = time_limit / num_starts
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Initialize
        selected = [False] * n_edges
        face_counts = [0] * n_faces
        current_size = [0]
        
        # Greedy Init: Random order
        e_order = list(range(n_edges))
        random.shuffle(e_order)
        for e_idx in e_order:
            if try_add(e_idx):
                add_edge(e_idx)
                
        if current_size[0] > best_size:
            best_size = current_size[0]
            best_set = selected[:]
            
        # Local Search
        # We will perform a random walk with bias towards adding edges
        # If stuck, remove edges
        
        steps = 0
        max_steps = 100000 # Hard cap per start, but time is main limit
        last_improve = time.time()
        
        while time.time() - start_time < time_limit - 5 and steps < max_steps:
            steps += 1
            
            # Check if we can add any edge
            # To be fast, we don't scan all edges. We pick random candidates.
            
            # Strategy:
            # 1. Try to add a random unselected edge.
            # 2. If fail, try to remove a random selected edge.
            
            if current_size[0] < n_edges:
                # Pick random unselected
                # Instead of scanning, pick random index and check if selected
                # This is O(1) average
                for _ in range(10): # Try a few times to find an unselected one
                    e_idx = random.randint(0, n_edges - 1)
                    if not selected[e_idx]:
                        if try_add(e_idx):
                            add_edge(e_idx)
                            if current_size[0] > best_size:
                                best_size = current_size[0]
                                best_set = selected[:]
                                last_improve = time.time()
                        break
                else:
                    # Failed to find unselected edge in 10 tries? Unlikely unless full.
                    # If full, we must remove.
                    pass
            
            # If no add happened or we want to explore, remove
            # Probability of remove: increases if we are stuck
            # Simple heuristic: if we didn't improve recently, remove more aggressively
            # Or just: 20% chance to remove, 80% to add
            
            if random.random() < 0.2 or current_size[0] == n_edges:
                # Remove random selected
                # Find a selected edge
                for _ in range(10):
                    e_idx = random.randint(0, n_edges - 1)
                    if selected[e_idx]:
                        remove_edge(e_idx)
                        break
                        
            # Check time
            if time.time() - start_time > time_limit - 10:
                break
                
    # Reconstruct best set
    result = []
    for i in range(n_edges):
        if best_set[i]:
            result.append(list(edges[i]))
            
    return result
