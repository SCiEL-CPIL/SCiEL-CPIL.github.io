import numpy as np
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                idx_to_edge[idx] = (x, y)
    
    # Build face_to_edges and edge_to_faces
    face_to_edges = []
    edge_to_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_ij = (1 << i) | (1 << j)
            for u in range(n_vertices):
                if (u & mask_ij) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ mask_ij
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    idx1 = edge_to_idx[e1]
                    idx2 = edge_to_idx[e2]
                    idx3 = edge_to_idx[e3]
                    idx4 = edge_to_idx[e4]
                    face_idx = len(face_to_edges)
                    face_to_edges.append([idx1, idx2, idx3, idx4])
                    edge_to_faces[idx1].append(face_idx)
                    edge_to_faces[idx2].append(face_idx)
                    edge_to_faces[idx3].append(face_idx)
                    edge_to_faces[idx4].append(face_idx)
    
    n_faces = len(face_to_edges)
    
    # Fast greedy hitting set
    def greedy_hitting_set():
        removed = np.zeros(n_edges, dtype=bool)
        face_hit_count = np.zeros(n_faces, dtype=np.int8)
        unhit_faces = set(range(n_faces))
        
        # Precompute for each edge, the list of faces it belongs to
        # All edges have degree 6 in the face hypergraph
        
        while unhit_faces and time.time() < deadline - 5:
            # Find edge that covers most unhit faces
            best_edge = -1
            best_hits = 0
            for e in range(n_edges):
                if removed[e]:
                    continue
                hits = 0
                for f in edge_to_faces[e]:
                    if f in unhit_faces:
                        hits += 1
                if hits > best_hits:
                    best_hits = hits
                    best_edge = e
                    if best_hits >= 4:  # Good enough, don't waste time
                        break
            
            if best_edge == -1:
                break
            
            removed[best_edge] = True
            for f in edge_to_faces[best_edge]:
                face_hit_count[f] += 1
                if face_hit_count[f] == 1:
                    unhit_faces.discard(f)
        
        return removed, face_hit_count
    
    # Local search with un-removal and 2-for-1 swaps
    def local_search_hitting_set(removed, face_hit_count, max_time=135.0):
        removed = removed.copy()
        face_hit_count = face_hit_count.copy()
        
        # Phase 1: Simple un-removals
        improved = True
        while improved and time.time() < max_time - 5:
            improved = False
            for e in range(n_edges):
                if removed[e]:
                    can_unremove = True
                    for f in edge_to_faces[e]:
                        if face_hit_count[f] <= 1:
                            can_unremove = False
                            break
                    if can_unremove:
                        removed[e] = False
                        for f in edge_to_faces[e]:
                            face_hit_count[f] -= 1
                        improved = True
        
        # Phase 2: 2-for-1 swaps - remove 2 edges, add 1 that covers all faces they covered
        # This can reduce size by 1 if the added edge covers faces that both removed edges covered
        for _ in range(200):
            if time.time() > max_time - 2:
                break
            
            sel_indices = np.where(removed)[0]
            if len(sel_indices) < 2:
                break
            
            # Try random pairs
            for _ in range(50):
                if time.time() > max_time - 2:
                    break
                
                e1, e2 = random.sample(sel_indices.tolist(), 2)
                
                # Temporarily remove e1 and e2
                for e in [e1, e2]:
                    removed[e] = False
                    for f in edge_to_faces[e]:
                        face_hit_count[f] -= 1
                
                # Find faces that are now uncovered (count 0)
                uncovered = set()
                for e in [e1, e2]:
                    for f in edge_to_faces[e]:
                        if face_hit_count[f] == 0:
                            uncovered.add(f)
                
                if not uncovered:
                    # Both removals were safe - size decreased by 2!
                    continue  # Already applied
                
                # Find an edge that covers all uncovered faces
                best_e = -1
                best_cov = 0
                for e in range(n_edges):
                    if removed[e]:
                        continue
                    cov = 0
                    for f in edge_to_faces[e]:
                        if f in uncovered:
                            cov += 1
                    if cov > best_cov:
                        best_cov = cov
                        best_e = e
                
                if best_cov == len(uncovered):
                    # Adding best_e covers all uncovered faces
                    removed[best_e] = True
                    for f in edge_to_faces[best_e]:
                        face_hit_count[f] += 1
                    # Net change: -2 + 1 = -1, improvement!
                else:
                    # Revert
                    for e in [e1, e2]:
                        removed[e] = True
                        for f in edge_to_faces[e]:
                            face_hit_count[f] += 1
        
        # Phase 3: More un-removals after swaps
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for e in range(n_edges):
                if removed[e]:
                    can_unremove = True
                    for f in edge_to_faces[e]:
                        if face_hit_count[f] <= 1:
                            can_unremove = False
                            break
                    if can_unremove:
                        removed[e] = False
                        for f in edge_to_faces[e]:
                            face_hit_count[f] -= 1
                        improved = True
        
        return removed, face_hit_count
    
    # Run multiple attempts
    best_removed = None
    best_size = n_edges
    
    # First attempt: greedy + local search
    removed, face_hit_count = greedy_hitting_set()
    removed, face_hit_count = local_search_hitting_set(removed, face_hit_count, max_time=deadline - 10)
    size = np.sum(removed)
    if size < best_size:
        best_size = size
        best_removed = removed.copy()
    
    # Additional attempts with random perturbations
    n_attempts = 30
    for attempt in range(n_attempts):
        if time.time() > deadline - 15:
            break
        
        # Start from best solution and perturb
        if best_removed is not None:
            removed = best_removed.copy()
        else:
            removed, face_hit_count = greedy_hitting_set()
        
        # Rebuild face_hit_count
        face_hit_count = np.zeros(n_faces, dtype=np.int8)
        for e in range(n_edges):
            if removed[e]:
                for f in edge_to_faces[e]:
                    face_hit_count[f] += 1
        
        # Perturb: randomly toggle some edges
        n_perturb = random.randint(5, 20)
        for _ in range(n_perturb):
            e = random.randint(0, n_edges - 1)
            removed[e] = not removed[e]
            for f in edge_to_faces[e]:
                if removed[e]:
                    face_hit_count[f] += 1
                else:
                    face_hit_count[f] -= 1
        
        # Fix violations: if any face has count 0, add a random edge from it
        for f in range(n_faces):
            if face_hit_count[f] == 0:
                # Add a random edge from this face
                e = random.choice(face_to_edges[f])
                if not removed[e]:
                    removed[e] = True
                    for f2 in edge_to_faces[e]:
                        face_hit_count[f2] += 1
        
        # Local search
        removed, face_hit_count = local_search_hitting_set(removed, face_hit_count, max_time=deadline - 5 - attempt * 0.2)
        size = np.sum(removed)
        
        if size < best_size:
            best_size = size
            best_removed = removed.copy()
    
    # Final polish on best solution
    if best_removed is not None:
        face_hit_count = np.zeros(n_faces, dtype=np.int8)
        for e in range(n_edges):
            if best_removed[e]:
                for f in edge_to_faces[e]:
                    face_hit_count[f] += 1
        best_removed, face_hit_count = local_search_hitting_set(best_removed, face_hit_count, max_time=deadline - 2)
        best_size = np.sum(best_removed)
    
    # Convert to selected edges
    selected = ~best_removed
    result = []
    for e in range(n_edges):
        if selected[e]:
            u, v = idx_to_edge[e]
            result.append([u, v])
    
    return result
