import time
import random

def construct():
    deadline = time.time() + 110.0
    d = 7
    n = 1 << d
    
    # 1. Precompute edges
    edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v, i))
    num_edges = len(edges)
    
    # 2. Precompute face IDs
    pair_index = {}
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            pair_index[(i, j)] = idx
            idx += 1
            
    def get_compact_rest(u, i, j):
        val = 0
        bit_pos = 0
        for k in range(d):
            if k == i or k == j:
                continue
            if u & (1 << k):
                val |= (1 << bit_pos)
            bit_pos += 1
        return val

    edge_faces = []
    for u, v, i in edges:
        faces = []
        for j in range(d):
            if j == i:
                continue
            if i < j:
                p_idx = pair_index[(i, j)]
            else:
                p_idx = pair_index[(j, i)]
            rest_compact = get_compact_rest(u, i, j)
            face_id = p_idx * (1 << (d - 2)) + rest_compact
            faces.append(face_id)
        edge_faces.append(faces)
        
    total_faces = len(pair_index) * (1 << (d - 2))
    
    face_counts = [0] * total_faces
    selected = set()
    
    def add_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] += 1
        selected.add(e_idx)
        
    def remove_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] -= 1
        selected.remove(e_idx)
        
    def can_add(e_idx):
        for fid in edge_faces[e_idx]:
            if face_counts[fid] >= 3:
                return False
        return True
        
    def greedy_fill():
        order = list(range(num_edges))
        random.shuffle(order)
        for e_idx in order:
            if can_add(e_idx):
                add_edge(e_idx)
                
    def try_add_missing():
        # Get missing edges
        missing = [i for i in range(num_edges) if i not in selected]
        random.shuffle(missing)
        added_any = False
        for e_idx in missing:
            if can_add(e_idx):
                add_edge(e_idx)
                added_any = True
        return added_any
        
    best_size = 0
    best_selected = set()
    
    # Initial greedy
    greedy_fill()
    best_size = len(selected)
    best_selected = set(selected)
    
    # Local search
    while time.time() < deadline:
        if not try_add_missing():
            # Stuck, do a kick
            if len(selected) < 2:
                # If very small, restart
                selected.clear()
                for i in range(total_faces): face_counts[i] = 0
                greedy_fill()
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
                continue
                
            # Save state
            old_face_counts = face_counts[:]
            old_selected = set(selected)
            
            # Remove 2 random edges
            to_remove = random.sample(list(selected), 2)
            for e_idx in to_remove:
                remove_edge(e_idx)
                
            # Greedy re-add
            missing = [i for i in range(num_edges) if i not in selected]
            random.shuffle(missing)
            for e_idx in missing:
                if can_add(e_idx):
                    add_edge(e_idx)
                    
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = set(selected)
            elif len(selected) < len(old_selected):
                # Revert
                selected = old_selected
                face_counts = old_face_counts
            # If equal size, we can keep or revert. Reverting is safer to avoid drift?
            # Actually, if size is equal, the set is different.
            # We might have opened up space for future additions.
            # Let's keep the new state if size is equal, as it might be "looser".
            
        # Periodic restart to escape deep local optima
        if random.random() < 0.01:
            selected.clear()
            for i in range(total_faces): face_counts[i] = 0
            greedy_fill()
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = set(selected)

    # Restore best solution
    selected = best_selected
    # We need to restore face_counts to match best_selected?
    # No, we just need to return the edges.
    # But if we wanted to continue, we'd need to rebuild face_counts.
    # Since we are returning, we just map indices to edges.
    
    result = []
    for e_idx in best_selected:
        u, v, _ = edges[e_idx]
        result.append([u, v])
        
    return result
