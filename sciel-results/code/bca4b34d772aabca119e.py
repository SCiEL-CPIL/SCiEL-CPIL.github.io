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
    
    # Precompute edge_to_faces as numpy arrays for vectorized validity check
    edge_to_faces_np = [np.array(faces_list, dtype=np.int32) for faces_list in edge_to_faces]
    
    def is_valid_add(face_counts, edge_idx):
        """Check if edge can be added using vectorized max"""
        faces = edge_to_faces_np[edge_idx]
        if len(faces) == 0:
            return True
        return np.max(face_counts[faces]) < 3
    
    def add_edge(chosen, face_counts, edge_idx):
        chosen[edge_idx] = True
        for f in edge_to_faces[edge_idx]:
            face_counts[f] += 1
    
    def remove_edge(chosen, face_counts, edge_idx):
        chosen[edge_idx] = False
        for f in edge_to_faces[edge_idx]:
            face_counts[f] -= 1
    
    # Phase 1: Multiple greedy restarts
    best_count = 0
    best_chosen = None
    best_face_counts = None
    
    random.seed(42)
    
    # Try multiple greedy orderings
    num_greedy = 50
    for restart in range(num_greedy):
        if time.time() > deadline - 30:
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
    
    # Phase 2: Local search with 2-for-1 swaps
    def local_search_swap(chosen, face_counts, max_rounds=1000):
        """Local search: try removing one edge and adding two back (2-for-1 swap)"""
        current_count = int(chosen.sum())
        no_improve_rounds = 0
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            # Get list of chosen edges
            chosen_edges = [e for e in range(num_edges) if chosen[e]]
            if not chosen_edges:
                break
            random.shuffle(chosen_edges)
            
            found_improvement = False
            
            # Try removing one edge and greedily adding back
            for e_remove in chosen_edges[:20]:
                c2 = chosen.copy()
                fc2 = face_counts.copy()
                
                remove_edge(c2, fc2, e_remove)
                
                # Greedily add back all possible edges
                added = []
                for e2 in range(num_edges):
                    if is_valid_add(fc2, e2):
                        add_edge(c2, fc2, e2)
                        added.append(e2)
                
                new_count = int(c2.sum())
                if new_count > current_count:
                    chosen = c2
                    face_counts = fc2
                    current_count = new_count
                    found_improvement = True
                    no_improve_rounds = 0
                    break
            
            if not found_improvement:
                no_improve_rounds += 1
                if no_improve_rounds > 50:
                    break
        
        return current_count, chosen, face_counts
    
    best_count, best_chosen, best_face_counts = local_search_swap(best_chosen, best_face_counts, max_rounds=500)
    
    # Phase 3: Multi-edge removal and repair
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        
        c2 = best_chosen.copy()
        fc2 = best_face_counts.copy()
        
        # Remove 2-4 random edges
        num_remove = random.randint(2, 4)
        candidates = [e for e in range(num_edges) if c2[e]]
        random.shuffle(candidates)
        for e in candidates[:num_remove]:
            remove_edge(c2, fc2, e)
        
        # Greedily add back
        for e in range(num_edges):
            if is_valid_add(fc2, e):
                add_edge(c2, fc2, e)
        
        new_count = int(c2.sum())
        if new_count > best_count:
            best_count = new_count
            best_chosen = c2
            best_face_counts = fc2
    
    # Phase 4: Final local search
    best_count, best_chosen, best_face_counts = local_search_swap(best_chosen, best_face_counts, max_rounds=200)
    
    # Convert to output
    out = [list(all_edges[idx]) for idx in range(num_edges) if best_chosen[idx]]
    return out
# EVOLVE-BLOCK-END
