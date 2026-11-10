import random
import time
import numpy as np

def construct():
    deadline = time.time() + 110
    d = 7
    n = 1 << d
    best_count = 0
    best_chosen = None
    
    # Build all edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
    
    num_edges = len(all_edges)
    
    edge_tuple_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_tuple_to_idx[(u, v)] = idx
    
    # Build faces
    faces = []
    for i in range(d):
        for j in range(i + 1, d):
            for mask in range(1 << d):
                if (mask & (1 << i)) or (mask & (1 << j)):
                    continue
                v1 = mask
                v2 = mask ^ (1 << i)
                v3 = mask ^ (1 << j)
                v4 = mask ^ (1 << i) ^ (1 << j)
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                idx1 = edge_tuple_to_idx[e1]
                idx2 = edge_tuple_to_idx[e2]
                idx3 = edge_tuple_to_idx[e3]
                idx4 = edge_tuple_to_idx[e4]
                faces.append((idx1, idx2, idx3, idx4))
    
    num_faces = len(faces)
    
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_to_faces[e_idx].append(f_idx)
    
    # Precompute face membership as sets for fast "in" checks
    face_sets = [set(face) for face in faces]
    
    def greedy_random(seed):
        rng = random.Random(seed)
        chosen = [False] * num_edges
        count = 0
        face_counts = [0] * num_faces
        
        indices = list(range(num_edges))
        rng.shuffle(indices)
        
        for idx in indices:
            can_add = True
            for f_idx in edge_to_faces[idx]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] += 1
        
        return count, chosen, face_counts
    
    def greedy_degree_based(seed):
        """Order edges by how many faces they're in that are still 'open'"""
        rng = random.Random(seed)
        chosen = [False] * num_edges
        count = 0
        face_counts = [0] * num_faces
        
        # Start with random order, then iteratively pick the edge with fewest blocking faces
        remaining = list(range(num_edges))
        rng.shuffle(remaining)
        
        while remaining and time.time() < deadline - 5:
            # Pick edge with minimum number of faces at count 2 (closest to blocked)
            # Tie-break randomly
            best_idx = -1
            best_score = float('inf')
            # Sample a subset for speed
            sample_size = min(len(remaining), 50)
            sample = rng.sample(remaining, sample_size) if len(remaining) > sample_size else remaining
            
            for idx in sample:
                score = 0
                for f_idx in edge_to_faces[idx]:
                    if face_counts[f_idx] == 2:
                        score += 1
                    elif face_counts[f_idx] == 3:
                        score += 100
                if score < best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx < 0:
                break
            
            can_add = True
            for f_idx in edge_to_faces[best_idx]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            
            if can_add:
                chosen[best_idx] = True
                count += 1
                for f_idx in edge_to_faces[best_idx]:
                    face_counts[f_idx] += 1
            
            remaining.remove(best_idx)
        
        return count, chosen, face_counts
    
    def local_search(chosen, face_counts, count, deadline):
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Try remove-1-add-2 moves
            for r_idx in range(num_edges):
                if not chosen[r_idx]:
                    continue
                
                chosen[r_idx] = False
                count -= 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] -= 1
                
                # Find candidates: unchosen edges sharing a face with r_idx
                candidate_set = set()
                for f_idx in edge_to_faces[r_idx]:
                    for e_idx in faces[f_idx]:
                        if e_idx != r_idx and not chosen[e_idx]:
                            candidate_set.add(e_idx)
                
                candidates = []
                for e_idx in candidate_set:
                    valid = True
                    for f_idx in edge_to_faces[e_idx]:
                        if face_counts[f_idx] == 3:
                            valid = False
                            break
                    if valid:
                        candidates.append(e_idx)
                
                # Try to find two compatible candidates
                added = []
                for i in range(len(candidates)):
                    e1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        e2 = candidates[j]
                        compatible = True
                        for f_idx in edge_to_faces[e1]:
                            if e2 in face_sets[f_idx]:
                                if face_counts[f_idx] >= 2:
                                    compatible = False
                                    break
                        if compatible:
                            added.append(e1)
                            added.append(e2)
                            break
                    if len(added) >= 2:
                        break
                
                if len(added) >= 2:
                    for e_idx in added:
                        chosen[e_idx] = True
                        count += 1
                        for f_idx in edge_to_faces[e_idx]:
                            face_counts[f_idx] += 1
                    improved = True
                    break
                
                # Try remove-1-add-1 (swap) if no add-2 found
                if len(candidates) > 0:
                    # Pick the best single candidate
                    best_e = candidates[0]
                    chosen[best_e] = True
                    count += 1
                    for f_idx in edge_to_faces[best_e]:
                        face_counts[f_idx] += 1
                    improved = True
                    break
                
                # Restore r_idx
                chosen[r_idx] = True
                count += 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] += 1
        
        # Final polish: try to add any single edge that's now valid
        for e_idx in range(num_edges):
            if not chosen[e_idx] and time.time() < deadline - 5:
                valid = True
                for f_idx in edge_to_faces[e_idx]:
                    if face_counts[f_idx] == 3:
                        valid = False
                        break
                if valid:
                    chosen[e_idx] = True
                    count += 1
                    for f_idx in edge_to_faces[e_idx]:
                        face_counts[f_idx] += 1
        
        return count, chosen, face_counts
    
    attempt = 0
    while time.time() < deadline - 15:
        if attempt % 3 == 0:
            count, chosen, face_counts = greedy_random(attempt)
        elif attempt % 3 == 1:
            count, chosen, face_counts = greedy_degree_based(attempt + 1000)
        else:
            count, chosen, face_counts = greedy_random(attempt + 5000)
        
        count, chosen, face_counts = local_search(chosen, face_counts, count, deadline)
        
        if count > best_count:
            best_count = count
            best_chosen = chosen[:]
        attempt += 1
    
    if best_chosen is None:
        # Fallback: return empty or minimal valid set
        return []
    
    out = []
    for idx in range(num_edges):
        if best_chosen[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out
