import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    d = 7
    n = 1 << d
    total_edges = n * d  # 448

    # Precompute all edges and their face memberships
    # Each edge connects x and x^(1<<i)
    # A 2D face is determined by two dimensions (i, j) and a vertex x not in those dims
    # Face edges: x-x^(1<<i), x-x^(1<<j), x^(1<<i)-x^(1<<i|1<<j), x^(1<<j)-x^(1<<i|1<<j)
    
    # Build edge list
    edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                eid = len(edges)
                edges.append((x, y, i))
                edge_index[(x, y)] = eid
                edge_index[(y, x)] = eid
    
    # Build face list: each face is a set of 4 edge IDs
    # A face is determined by (base_vertex, dim_i, dim_j) where base_vertex has bits i,j = 0
    # But we need to enumerate all faces. Each face has 4 vertices.
    # For each pair of dimensions (i, j), and each assignment of the other 5 bits, we get a face.
    
    faces = []
    for i in range(d):
        for j in range(i + 1, d):
            # For each combination of the remaining d-2 bits
            for mask in range(1 << (d - 2)):
                # Reconstruct the base vertex with bits i,j = 0
                x = 0
                bit_pos = 0
                for k in range(d):
                    if k != i and k != j:
                        if mask & (1 << bit_pos):
                            x |= (1 << k)
                        bit_pos += 1
                
                # The four vertices of the face
                v0 = x
                v1 = x ^ (1 << i)
                v2 = x ^ (1 << j)
                v3 = x ^ (1 << i) ^ (1 << j)
                
                # The four edges
                e01 = edge_index.get((min(v0, v1), max(v0, v1)))
                e02 = edge_index.get((min(v0, v2), max(v0, v2)))
                e13 = edge_index.get((min(v1, v3), max(v1, v3)))
                e23 = edge_index.get((min(v2, v3), max(v2, v3)))
                
                if e01 is not None and e02 is not None and e13 is not None and e23 is not None:
                    faces.append((e01, e02, e13, e23))
    
    num_faces = len(faces)
    
    # For each edge, find which faces it belongs to
    edge_to_faces = [[] for _ in range(len(edges))]
    for fid, face in enumerate(faces):
        for eid in face:
            edge_to_faces[eid].append(fid)
    
    # Greedy heuristic: start with all edges, then remove edges to break faces
    # Better: use a randomized greedy with multiple restarts
    
    def greedy_remove(order):
        """Remove edges in given order if it breaks a face, else keep."""
        selected = set(range(len(edges)))
        face_count = [0] * num_faces  # count of selected edges in each face
        
        # Initialize face counts
        for eid in range(len(edges)):
            for fid in edge_to_faces[eid]:
                face_count[fid] += 1
        
        removed = set()
        
        for eid in order:
            if eid in removed:
                continue
            # Check if removing this edge helps (i.e., it's in some face that would still be complete)
            # Actually, we want to remove edges that are part of "tight" faces
            # Better approach: for each face, if all 4 edges are selected, we must remove at least one
            # Greedy: process faces, for each complete face, remove the edge with fewest other face conflicts
            
            # This is getting complex. Let's use a different approach.
        
        return selected
    
    # Simpler approach: local search
    # Start with a random subset, then flip edges to improve
    
    def evaluate(selected):
        """Count number of faces with all 4 edges selected."""
        count = 0
        for face in faces:
            if all(eid in selected for eid in face):
                count += 1
        return count
    
    def local_search(initial_selected, time_limit):
        """Local search to minimize number of violated faces."""
        selected = set(initial_selected)
        current_violations = evaluate(selected)
        best_selected = set(selected)
        best_violations = current_violations
        
        # Precompute for each edge, the faces it's in
        # For flipping an edge, we need to check which faces are affected
        
        # Build face membership for quick updates
        # face_edge_count[fid] = number of selected edges in face fid
        face_edge_count = [0] * num_faces
        for eid in selected:
            for fid in edge_to_faces[eid]:
                face_edge_count[fid] += 1
        
        # For each edge, precompute list of faces
        # When we add/remove an edge, update face_edge_count
        
        # Local search: try flipping each edge, accept if it reduces violations
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for eid in range(len(edges)):
                if time.time() > time_limit:
                    break
                
                if eid in selected:
                    # Try removing
                    # Check which faces would be affected
                    affected_faces = edge_to_faces[eid]
                    # Removing eid decreases face_edge_count for each affected face
                    # A face is violated if count == 4
                    # After removal, count decreases by 1
                    # Violations decrease for faces that had count == 4 (become 3)
                    # Violations increase for faces that had count == 3 (become 2)? No, only 4 is violation
                    
                    # Actually, only faces with all 4 selected are violated
                    # Removing an edge can only reduce violations (faces going from 4 to 3)
                    # It cannot increase violations
                    
                    # So removing always helps or is neutral
                    # But we want to maximize edges, so we should be careful
                    
                    # Let's think differently: we want to maximize |selected| subject to no face has all 4 edges
                    # This is a constraint satisfaction problem
                    
                    pass
                else:
                    # Try adding
                    # Adding eid increases face_edge_count for each affected face
                    # A face becomes violated if count goes from 3 to 4
                    new_violations = 0
                    for fid in affected_faces:
                        if face_edge_count[fid] == 3:
                            new_violations += 1
                    
                    if new_violations == 0:
                        # Safe to add
                        selected.add(eid)
                        for fid in affected_faces:
                            face_edge_count[fid] += 1
                        current_violations += 0  # no new violations
                        improved = True
                        if len(selected) > len(best_selected):
                            best_selected = set(selected)
        
        return best_selected
    
    # Better approach: use a smarter initialization and local search
    
    # Start with all edges, then iteratively remove edges from violated faces
    def iterative_removal():
        selected = set(range(len(edges)))
        
        # Count edges in each face
        face_count = [0] * num_faces
        for eid in range(len(edges)):
            for fid in edge_to_faces[eid]:
                face_count[fid] += 1
        
        # Find violated faces (all 4 edges present)
        # For each violated face, remove the edge that appears in the most other violated faces
        # Or use a greedy: for each face, remove the edge with highest "conflict"
        
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Find all violated faces
            violated = []
            for fid, face in enumerate(faces):
                if all(eid in selected for eid in face):
                    violated.append(fid)
            
            if not violated:
                break
            
            # For each edge in violated faces, count how many violated faces it's in
            edge_conflict = {}
            for fid in violated:
                for eid in faces[fid]:
                    edge_conflict[eid] = edge_conflict.get(eid, 0) + 1
            
            # Remove the edge with highest conflict
            if edge_conflict:
                max_eid = max(edge_conflict, key=edge_conflict.get)
                selected.discard(max_eid)
                for fid in edge_to_faces[max_eid]:
                    face_count[fid] -= 1
        
        return selected
    
    # Try multiple strategies and keep the best
    
    best_result = None
    best_count = 0
    
    # Strategy 1: Iterative removal
    if time.time() < deadline:
        result1 = iterative_removal()
        count1 = len(result1)
        if count1 > best_count:
            best_count = count1
            best_result = set(result1)
    
    # Strategy 2: Randomized local search with multiple restarts
    if time.time() < deadline:
        random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization: start with random subset
            # Or start with all edges and randomly remove some
            selected = set(range(len(edges)))
            
            # Randomly remove 10-20% of edges
            edges_list = list(selected)
            random.shuffle(edges_list)
            num_remove = len(edges_list) // 5
            for eid in edges_list[:num_remove]:
                selected.discard(eid)
            
            # Local search: try adding edges back
            face_count = [0] * num_faces
            for eid in selected:
                for fid in edge_to_faces[eid]:
                    face_count[fid] += 1
            
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for eid in range(len(edges)):
                    if time.time() > deadline - 2:
                        break
                    if eid in selected:
                        continue
                    # Try adding
                    new_violations = 0
                    for fid in edge_to_faces[eid]:
                        if face_count[fid] == 3:
                            new_violations += 1
                    if new_violations == 0:
                        selected.add(eid)
                        for fid in edge_to_faces[eid]:
                            face_count[fid] += 1
                        improved = True
            
            count = len(selected)
            if count > best_count:
                best_count = count
                best_result = set(selected)
    
    # Convert to edge list
    if best_result is None:
        best_result = set(range(len(edges)))
    
    out = []
    for eid in best_result:
        x, y, i = edges[eid]
        out.append([x, y])
    
    return out
