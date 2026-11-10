import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110.0
    d = 7
    n = 1 << d  # 128 vertices
    total_edges = d * (1 << (d - 1))  # 448 edges
    
    # Precompute edges
    edges = []
    edge_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_index[(u, v)] = len(edges)
                edges.append((u, v, i))
    
    num_edges = len(edges)
    
    # Precompute face membership for each edge
    # A face is identified by (i, j, rest) where i < j are the two varying bits
    # and rest is the (d-2)-bit pattern of the other bits
    pair_index = {}
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            pair_index[(i, j)] = idx
            idx += 1
    num_pairs = len(pair_index)
    num_rest = 1 << (d - 2)
    total_faces = num_pairs * num_rest
    
    # For each edge, compute which faces it belongs to
    edge_faces = [[] for _ in range(num_edges)]
    for e_idx, (u, v, i) in enumerate(edges):
        for j in range(d):
            if j == i:
                continue
            if i < j:
                p_idx = pair_index[(i, j)]
            else:
                p_idx = pair_index[(j, i)]
            # Compute compact rest: bits other than i and j
            rest = 0
            bit_pos = 0
            for k in range(d):
                if k == i or k == j:
                    continue
                if u & (1 << k):
                    rest |= (1 << bit_pos)
                bit_pos += 1
            face_id = p_idx * num_rest + rest
            edge_faces[e_idx].append(face_id)
    
    # Count how many edges touch each face
    face_edge_list = [[] for _ in range(total_faces)]
    for e_idx in range(num_edges):
        for fid in edge_faces[e_idx]:
            face_edge_list[fid].append(e_idx)
    
    # State
    face_counts = np.zeros(total_faces, dtype=np.int32)
    selected = np.zeros(num_edges, dtype=bool)
    
    def add_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] += 1
        selected[e_idx] = True
    
    def remove_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] -= 1
        selected[e_idx] = False
    
    def can_add(e_idx):
        for fid in edge_faces[e_idx]:
            if face_counts[fid] >= 3:
                return False
        return True
    
    def greedy_fill(order=None):
        if order is None:
            order = list(range(num_edges))
            random.shuffle(order)
        for e_idx in order:
            if not selected[e_idx] and can_add(e_idx):
                add_edge(e_idx)
    
    def try_add_missing():
        missing = [i for i in range(num_edges) if not selected[i]]
        random.shuffle(missing)
        added_any = False
        for e_idx in missing:
            if can_add(e_idx):
                add_edge(e_idx)
                added_any = True
        return added_any
    
    # Initial greedy fill
    greedy_fill()
    best_size = int(np.sum(selected))
    best_selected = selected.copy()
    
    # Local search with larger kicks
    p_accept_worse = 0.05
    
    while time.time() < deadline:
        if not try_add_missing():
            # Stuck - perform a kick
            K = random.randint(4, 20)
            
            if int(np.sum(selected)) < K:
                # Too small to kick, restart
                selected[:] = False
                face_counts[:] = 0
                greedy_fill()
                cur_size = int(np.sum(selected))
                if cur_size > best_size:
                    best_size = cur_size
                    best_selected = selected.copy()
                continue
            
            # Save state
            old_face_counts = face_counts.copy()
            old_selected = selected.copy()
            old_size = int(np.sum(selected))
            
            # Remove K random selected edges
            selected_indices = np.where(selected)[0]
            to_remove = random.sample(list(selected_indices), K)
            for e_idx in to_remove:
                remove_edge(e_idx)
            
            # Greedy re-add
            missing = [i for i in range(num_edges) if not selected[i]]
            random.shuffle(missing)
            for e_idx in missing:
                if can_add(e_idx):
                    add_edge(e_idx)
            
            new_size = int(np.sum(selected))
            
            if new_size > best_size:
                best_size = new_size
                best_selected = selected.copy()
            
            # Acceptance criterion
            if new_size >= old_size:
                # Accept
                pass
            else:
                # Worse, accept with probability to explore
                if random.random() < p_accept_worse:
                    # Keep new state
                    pass
                else:
                    # Revert
                    selected[:] = old_selected
                    face_counts[:] = old_face_counts
    
    # Restore best solution
    result = []
    for e_idx in range(num_edges):
        if best_selected[e_idx]:
            u, v, _ = edges[e_idx]
            result.append([u, v])
    
    return result
