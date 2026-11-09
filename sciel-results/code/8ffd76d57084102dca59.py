"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s to leave 5s safety margin
    
    d = 7
    n = 2 ** d
    num_edges = d * 2 ** (d - 1)  # 448
    
    # Generate all edges and map them to indices
    all_edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_index[(x, y)] = idx
    
    # For each edge, find which dimension it's along
    edge_dim = []
    for (x, y) in all_edges:
        diff = x ^ y
        edge_dim.append(diff.bit_length() - 1)
    
    # Precompute faces
    faces = []
    face_to_edges = []
    edge_to_faces = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            mask_ij = mask_i | mask_j
            for base in range(n):
                if (base & mask_i) == 0 and (base & mask_j) == 0:
                    v0 = base
                    v1 = base ^ mask_i
                    v2 = base ^ mask_j
                    v3 = base ^ mask_ij
                    
                    e0 = edge_index.get((min(v0, v1), max(v0, v1)))
                    e1 = edge_index.get((min(v1, v3), max(v1, v3)))
                    e2 = edge_index.get((min(v0, v2), max(v0, v2)))
                    e3 = edge_index.get((min(v2, v3), max(v2, v3)))
                    
                    if e0 is not None and e1 is not None and e2 is not None and e3 is not None:
                        face_idx = len(faces)
                        faces.append((i, j, base))
                        face_edges = [e0, e1, e2, e3]
                        face_to_edges.append(face_edges)
                        for e in face_edges:
                            edge_to_faces[e].append(face_idx)
    
    num_faces = len(faces)
    assert num_faces == 672, f"Expected 672 faces, got {num_faces}"
    
    # Convert to numpy arrays for speed
    face_edges_arr = np.array(face_to_edges, dtype=np.int32)
    
    # Precompute edge_to_faces as numpy arrays
    edge_to_faces_np = []
    for e in range(num_edges):
        edge_to_faces_np.append(np.array(edge_to_faces[e], dtype=np.int32))
    
    def is_valid_add(face_counts, edge_idx):
        """Check if edge can be added using vectorized max"""
        faces = edge_to_faces_np[edge_idx]
        if len(faces) == 0:
            return True
        return np.max(face_counts[faces]) < 3
    
    def add_edge(chosen, face_counts, edge_idx):
        chosen[edge_idx] = True
        faces = edge_to_faces_np[edge_idx]
        face_counts[faces] += 1
    
    def remove_edge(chosen, face_counts, edge_idx):
        chosen[edge_idx] = False
        faces = edge_to_faces_np[edge_idx]
        face_counts[faces] -= 1
    
    def greedy_repair(chosen, face_counts, unselected, deadline):
        """Greedy repair: add any unselected edge that fits"""
        added_any = True
        while added_any:
            added_any = False
            for e_add in unselected:
                if not chosen[e_add]:
                    if is_valid_add(face_counts, e_add):
                        add_edge(chosen, face_counts, e_add)
                        added_any = True
                        if time.time() > deadline:
                            return
            if time.time() > deadline:
                return
    
    # Phase 1: Greedy construction with multiple strategies
    best_count = 0
    best_chosen = None
    best_face_counts = None
    
    random.seed(42)
    
    # Greedy 1: natural order
    chosen = np.zeros(num_edges, dtype=bool)
    face_counts = np.zeros(num_faces, dtype=np.int32)
    for e in range(num_edges):
        if is_valid_add(face_counts, e):
            add_edge(chosen, face_counts, e)
    count = int(chosen.sum())
    if count > best_count:
        best_count = count
        best_chosen = chosen.copy()
        best_face_counts = face_counts.copy()
    
    # Greedy 2: random orders
    for restart in range(80):
        if time.time() > deadline - 60:
            break
        chosen = np.zeros(num_edges, dtype=bool)
        face_counts = np.zeros(num_faces, dtype=np.int32)
        order = list(range(num_edges))
        random.shuffle(order)
        for e in order:
            if is_valid_add(face_counts, e):
                add_edge(chosen, face_counts, e)
        count = int(chosen.sum())
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
            best_face_counts = face_counts.copy()
    
    # Phase 2: Local search with 1-edge removals
    def local_search(chosen, face_counts, max_rounds=100):
        """Local search: try removing edges, greedily repair"""
        current_count = int(chosen.sum())
        unselected = [e for e in range(num_edges) if not chosen[e]]
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 10:
                break
            
            improved = False
            selected = [e for e in range(num_edges) if chosen[e]]
            
            # Try removing 1 edge
            random.shuffle(selected)
            for e_remove in selected[:15]:
                c2 = chosen.copy()
                fc2 = face_counts.copy()
                remove_edge(c2, fc2, e_remove)
                
                greedy_repair(c2, fc2, unselected, deadline - 5)
                
                new_count = int(c2.sum())
                if new_count > current_count:
                    chosen = c2
                    face_counts = fc2
                    current_count = new_count
                    unselected = [e for e in range(num_edges) if not chosen[e]]
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try removing 2 edges
            if len(selected) >= 2:
                pairs = []
                for i in range(min(8, len(selected))):
                    for j in range(i+1, min(8, len(selected))):
                        pairs.append((selected[i], selected[j]))
                random.shuffle(pairs)
                
                for e1, e2 in pairs[:8]:
                    c2 = chosen.copy()
                    fc2 = face_counts.copy()
                    remove_edge(c2, fc2, e1)
                    remove_edge(c2, fc2, e2)
                    
                    greedy_repair(c2, fc2, unselected, deadline - 5)
                    
                    new_count = int(c2.sum())
                    if new_count > current_count:
                        chosen = c2
                        face_counts = fc2
                        current_count = new_count
                        unselected = [e for e in range(num_edges) if not chosen[e]]
                        improved = True
                        break
                        
        return current_count, chosen, face_counts
    
    best_count, best_chosen, best_face_counts = local_search(best_chosen, best_face_counts, max_rounds=300)
    
    # Phase 3: Multi-edge removals (3-5 edges)
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        
        c2 = best_chosen.copy()
        fc2 = best_face_counts.copy()
        unselected = [e for e in range(num_edges) if not c2[e]]
        
        num_remove = random.randint(3, 5)
        candidates = [e for e in range(num_edges) if c2[e]]
        random.shuffle(candidates)
        for e in candidates[:num_remove]:
            remove_edge(c2, fc2, e)
        
        greedy_repair(c2, fc2, unselected, deadline - 5)
        
        new_count = int(c2.sum())
        if new_count > best_count:
            best_count = new_count
            best_chosen = c2
            best_face_counts = fc2
    
    # Phase 4: Simulated annealing-style perturbation
    current_temp = 1.0
    for step in range(200):
        if time.time() > deadline - 5:
            break
        
        c2 = best_chosen.copy()
        fc2 = best_face_counts.copy()
        unselected = [e for e in range(num_edges) if not c2[e]]
        
        # Random perturbation: remove k edges, add k random unselected
        k = random.randint(1, 3)
        selected = [e for e in range(num_edges) if c2[e]]
        random.shuffle(selected)
        for e in selected[:k]:
            remove_edge(c2, fc2, e)
        
        # Add some random unselected edges
        random.shuffle(unselected)
        for e in unselected[:k]:
            if is_valid_add(fc2, e):
                add_edge(c2, fc2, e)
        
        # Greedy repair
        greedy_repair(c2, fc2, unselected, deadline - 5)
        
        new_count = int(c2.sum())
        if new_count >= best_count:
            best_count = new_count
            best_chosen = c2
            best_face_counts = fc2
            current_temp *= 0.95
        else:
            current_temp *= 0.9
    
    # Convert to output
    out = [list(all_edges[idx]) for idx in range(num_edges) if best_chosen[idx]]
    return out
# EVOLVE-BLOCK-END
