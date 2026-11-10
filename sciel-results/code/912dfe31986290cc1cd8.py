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
    
    # For each face, list its 4 edges
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
    
    # Deterministic initial construction using a structured approach
    # For each dimension i, select edges based on a parity rule
    # This gives a valid C4-free set as a starting point
    def deterministic_init():
        selected[:] = False
        face_counts[:] = 0
        # For each vertex u and dimension i, decide whether to include edge (u, u^bit_i)
        # Use a rule: include edge in dim i from u if popcount(u & mask_i) % 2 == 0
        # where mask_i is a fixed mask per dimension
        masks = [0b0101010, 0b1010101, 0b0011001, 0b1100110, 0b0101101, 0b1011010, 0b0110101]
        for u in range(n):
            for i in range(d):
                v = u ^ (1 << i)
                if u < v:
                    e_idx = edge_index[(u, v)]
                    # Use parity of bits in u masked by masks[i]
                    if bin(u & masks[i]).count('1') % 2 == 0:
                        add_edge(e_idx)
    
    # Try multiple initializations and keep the best
    best_size = 0
    best_selected = np.zeros(num_edges, dtype=bool)
    
    # First try deterministic init
    deterministic_init()
    greedy_fill()
    cur_size = int(np.sum(selected))
    if cur_size > best_size:
        best_size = cur_size
        best_selected = selected.copy()
    
    # Also try random greedy init
    selected[:] = False
    face_counts[:] = 0
    greedy_fill()
    cur_size = int(np.sum(selected))
    if cur_size > best_size:
        best_size = cur_size
        best_selected = selected.copy()
    
    # Restore best
    selected[:] = best_selected
    face_counts[:] = 0
    for e_idx in range(num_edges):
        if selected[e_idx]:
            for fid in edge_faces[e_idx]:
                face_counts[fid] += 1
    
    # Local search
    p_accept_worse = 0.15
    
    while time.time() < deadline:
        if not try_add_missing():
            # Stuck - perform a kick
            cur_size = int(np.sum(selected))
            if cur_size < 10:
                # Restart with deterministic
                deterministic_init()
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
            
            # Choose kick strategy
            strategy = random.random()
            if strategy < 0.3:
                # Small random kick
                K = random.randint(3, 8)
                selected_indices = np.where(selected)[0]
                to_remove = random.sample(list(selected_indices), min(K, len(selected_indices)))
            elif strategy < 0.6:
                # Targeted kick: remove edges from faces that are "full" (3 edges)
                # Find faces with count == 3, remove one edge from each
                full_faces = [f for f in range(total_faces) if face_counts[f] == 3]
                if not full_faces:
                    K = random.randint(5, 15)
                    selected_indices = np.where(selected)[0]
                    to_remove = random.sample(list(selected_indices), min(K, len(selected_indices)))
                else:
                    random.shuffle(full_faces)
                    to_remove = []
                    for f in full_faces[:20]:
                        fe = face_edge_list[f]
                        # Remove a random edge from this face
                        e = random.choice(fe)
                        if selected[e] and e not in to_remove:
                            to_remove.append(e)
                    if not to_remove:
                        K = random.randint(5, 15)
                        selected_indices = np.where(selected)[0]
                        to_remove = random.sample(list(selected_indices), min(K, len(selected_indices)))
            else:
                # Medium random kick
                K = random.randint(10, 25)
                selected_indices = np.where(selected)[0]
                to_remove = random.sample(list(selected_indices), min(K, len(selected_indices)))
            
            for e_idx in to_remove:
                remove_edge(e_idx)
            
            # Greedy re-add with multiple passes
            for _ in range(2):
                missing = [i for i in range(num_edges) if not selected[i]]
                random.shuffle(missing)
                for e_idx in missing:
                    if can_add(e_idx):
                        add_edge(e_idx)
            
            new_size = int(np.sum(selected))
            
            if new_size > best_size:
                best_size = new_size
                best_selected = selected.copy()
            
            # Acceptance: always accept if better, accept worse with probability
            if new_size >= old_size:
                pass
            else:
                if random.random() < p_accept_worse:
                    pass
                else:
                    selected[:] = old_selected
                    face_counts[:] = old_face_counts
    
    # Restore best solution
    result = []
    for e_idx in range(num_edges):
        if best_selected[e_idx]:
            u, v, _ = edges[e_idx]
            result.append([u, v])
    
    return result
