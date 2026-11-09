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
    # A face is (i, j, base) where i < j, base has bits i,j = 0
    # For a given (i, j, base), the 4 vertices are:
    #   base, base^(1<<i), base^(1<<j), base^(1<<(i)|1<<(j))
    # The 4 edges are:
    #   (base, base^(1<<i)) - dim i
    #   (base^(1<<j), base^(1<<(i)|1<<(j))) - dim i
    #   (base, base^(1<<j)) - dim j
    #   (base^(1<<i), base^(1<<(i)|1<<(j))) - dim j
    
    faces = []  # list of (i, j, base)
    face_to_edges = []  # list of 4 edge indices per face
    edge_to_faces = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            mask_ij = mask_i | mask_j
            # base has bits i and j = 0
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
    face_edges_arr = np.array(face_to_edges, dtype=np.int32)  # shape (672, 4)
    
    # Precompute edge_to_faces as numpy arrays for vectorized validity check
    edge_to_faces_np = []
    for e in range(num_edges):
        edge_to_faces_np.append(np.array(edge_to_faces[e], dtype=np.int32))
    
    def is_valid_add(chosen, face_counts, edge_idx):
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
    
    # Phase 1: Greedy construction
    chosen = np.zeros(num_edges, dtype=bool)
    face_counts = np.zeros(num_faces, dtype=np.int32)
    
    # Try multiple greedy orderings
    best_count = 0
    best_chosen = None
    best_face_counts = None
    
    random.seed(42)
    
    # Greedy 1: natural order
    for e in range(num_edges):
        if is_valid_add(chosen, face_counts, e):
            add_edge(chosen, face_counts, e)
    
    count = chosen.sum()
    if count > best_count:
        best_count = count
        best_chosen = chosen.copy()
        best_face_counts = face_counts.copy()
    
    # Greedy 2: random orders
    for restart in range(50):
        if time.time() > deadline - 40:
            break
        chosen = np.zeros(num_edges, dtype=bool)
        face_counts = np.zeros(num_faces, dtype=np.int32)
        order = list(range(num_edges))
        random.shuffle(order)
        for e in order:
            if is_valid_add(chosen, face_counts, e):
                add_edge(chosen, face_counts, e)
        count = chosen.sum()
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
            best_face_counts = face_counts.copy()
    
    # Phase 2: Local search - remove one edge, greedily add
    chosen = best_chosen.copy()
    face_counts = best_face_counts.copy()
    
    def local_search_remove_add(chosen, face_counts, max_iters=2000):
        """Try removing one edge and greedily adding back"""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            if time.time() > deadline - 5:
                break
            improved = False
            iter_count += 1
            
            # Try removing each edge
            for e in range(num_edges):
                if not chosen[e]:
                    continue
                # Remove edge e
                remove_edge(chosen, face_counts, e)
                
                # Try to add back edges
                for e2 in range(num_edges):
                    if e2 == e:
                        continue
                    if is_valid_add(chosen, face_counts, e2):
                        add_edge(chosen, face_counts, e2)
                
                new_count = chosen.sum()
                old_count = best_count  # This is wrong, need to track properly
                
                # Check if we improved
                if new_count > chosen.sum() - 1:  # This logic is flawed
                    pass
                
                # Restore
                # Actually, let's just check if new_count > current best
                if new_count > best_count:
                    best_count = new_count
                    best_chosen = chosen.copy()
                    best_face_counts = face_counts.copy()
                    improved = True
                    break
                else:
                    # Undo
                    # This is complex, let's use a different approach
                    pass
        
        return best_count, best_chosen, best_face_counts
    
    # Optimized local search with correct repair logic
    def local_search(chosen, face_counts, max_rounds=100):
        """Local search: try removing edges, greedily repair unselected edges"""
        current_count = int(chosen.sum())
        unselected = [e for e in range(num_edges) if not chosen[e]]
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 10:
                break
            
            improved = False
            selected = [e for e in range(num_edges) if chosen[e]]
            
            # Try removing 1 edge
            random.shuffle(selected)
            for e_remove in selected[:10]:
                c2 = chosen.copy()
                fc2 = face_counts.copy()
                remove_edge(c2, fc2, e_remove)
                
                # Greedy repair: add any unselected edge that fits
                added_any = True
                while added_any:
                    added_any = False
                    for e_add in unselected:
                        if not c2[e_add]:
                            if is_valid_add(c2, fc2, e_add):
                                add_edge(c2, fc2, e_add)
                                added_any = True
                                if time.time() > deadline - 5:
                                    break
                    if time.time() > deadline - 5:
                        break
                
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
                # Pick a few random pairs
                pairs = []
                for i in range(min(5, len(selected))):
                    for j in range(i+1, min(5, len(selected))):
                        pairs.append((selected[i], selected[j]))
                random.shuffle(pairs)
                
                for e1, e2 in pairs[:5]:
                    c2 = chosen.copy()
                    fc2 = face_counts.copy()
                    remove_edge(c2, fc2, e1)
                    remove_edge(c2, fc2, e2)
                    
                    added_any = True
                    while added_any:
                        added_any = False
                        for e_add in unselected:
                            if not c2[e_add]:
                                if is_valid_add(c2, fc2, e_add):
                                    add_edge(c2, fc2, e_add)
                                    added_any = True
                                    if time.time() > deadline - 5:
                                        break
                        if time.time() > deadline - 5:
                            break
                            
                    new_count = int(c2.sum())
                    if new_count > current_count:
                        chosen = c2
                        face_counts = fc2
                        current_count = new_count
                        unselected = [e for e in range(num_edges) if not chosen[e]]
                        improved = True
                        break
                        
        return current_count, chosen, face_counts
    
    best_count, best_chosen, best_face_counts = local_search(best_chosen, best_face_counts, max_rounds=200)
    
    # Phase 3: More aggressive - try random subsets of removals (3-4 edges)
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        c2 = best_chosen.copy()
        fc2 = best_face_counts.copy()
        unselected = [e for e in range(num_edges) if not c2[e]]
        
        # Remove 3-4 random edges
        num_remove = random.randint(3, 4)
        candidates = [e for e in range(num_edges) if c2[e]]
        random.shuffle(candidates)
        for e in candidates[:num_remove]:
            remove_edge(c2, fc2, e)
        
        # Greedily add back
        added_any = True
        while added_any:
            added_any = False
            for e_add in unselected:
                if not c2[e_add]:
                    if is_valid_add(c2, fc2, e_add):
                        add_edge(c2, fc2, e_add)
                        added_any = True
                        if time.time() > deadline - 5:
                            break
            if time.time() > deadline - 5:
                break
        
        new_count = int(c2.sum())
        if new_count > best_count:
            best_count = new_count
            best_chosen = c2
            best_face_counts = fc2
    
    # Convert to output
    out = [list(all_edges[idx]) for idx in range(num_edges) if best_chosen[idx]]
    return out
# EVOLVE-BLOCK-END
