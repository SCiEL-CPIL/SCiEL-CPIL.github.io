"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    d = 7
    n_vertices = 2 ** d
    n_edges = d * 2 ** (d - 1)  # 448
    
    # Build edge list
    edges = []
    edge_to_idx = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    # Build list of all 4-cycles (faces)
    faces = []
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    v0 = x
                    v1 = x ^ (1 << i)
                    v2 = x ^ (1 << j)
                    v3 = x ^ (1 << i) ^ (1 << j)
                    
                    e01 = (min(v0, v1), max(v0, v1))
                    e02 = (min(v0, v2), max(v0, v2))
                    e13 = (min(v1, v3), max(v1, v3))
                    e23 = (min(v2, v3), max(v2, v3))
                    
                    idx01 = edge_to_idx[e01]
                    idx02 = edge_to_idx[e02]
                    idx13 = edge_to_idx[e13]
                    idx23 = edge_to_idx[e23]
                    
                    faces.append([idx01, idx02, idx13, idx23])
    
    n_faces = len(faces)
    
    # For each edge, find which faces it belongs to
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # Build conflict graph: two edges conflict if they share a face
    # For spectral initialization, build adjacency matrix of conflict graph
    # Conflict graph: vertices = edges, edge between two edges if they're in same face
    # This is a 448x448 matrix - manageable
    
    # Actually, for Fiedler vector, we need the Laplacian of the conflict graph
    # But the conflict graph is dense. Let's think differently.
    
    # Alternative: Use the face-edge incidence structure to create a meaningful ordering
    # The key insight: edges that are in many "tight" faces (faces where other 2 edges are likely selected)
    # should be ordered differently.
    
    # Let's use a different spectral approach: 
    # Create a bipartite graph between edges and faces
    # Then use the singular vectors to get an ordering
    
    # Actually, simpler: just use the degree in the conflict graph for ordering
    # Edges in more faces are more constrained, so process them first or last?
    
    # Let's try: sort edges by number of faces they belong to (ascending)
    # Less constrained edges first in greedy
    
    edge_face_count = [len(edge_faces[e]) for e in range(n_edges)]
    
    # For spectral: build a sparse representation
    # Actually, let's just do a good greedy + local search with efficient updates
    
    # Efficient state tracking
    selected = [False] * n_edges
    selected_count = 0
    face_complete = [0] * n_faces  # number of selected edges in each face
    
    def try_add(e_idx):
        """Check if edge e_idx can be added without creating a complete face."""
        for f_idx in edge_faces[e_idx]:
            if face_complete[f_idx] == 3:
                return False
        return True
    
    def add_edge(e_idx):
        nonlocal selected_count
        selected[e_idx] = True
        selected_count += 1
        for f_idx in edge_faces[e_idx]:
            face_complete[f_idx] += 1
    
    def remove_edge(e_idx):
        nonlocal selected_count
        selected[e_idx] = False
        selected_count -= 1
        for f_idx in edge_faces[e_idx]:
            face_complete[f_idx] -= 1
    
    # Phase 1: Spectral-inspired greedy initialization
    # Use face-edge incidence to create a weighted ordering
    # Edges that are in faces with many other edges should be deprioritized
    
    # Create a score for each edge based on face structure
    # For each edge, compute how "central" it is in the face hypergraph
    edge_scores = [0.0] * n_edges
    for f_idx, face in enumerate(faces):
        # Each face contributes to the centrality of its 4 edges
        for e_idx in face:
            edge_scores[e_idx] += 1.0 / len(edge_faces[e_idx])
    
    # Sort by score: lower score = less central = add first
    order = sorted(range(n_edges), key=lambda e: edge_scores[e])
    
    # Add edges in this order
    for e_idx in order:
        if try_add(e_idx):
            add_edge(e_idx)
    
    best_count = selected_count
    best_selected = selected[:]
    best_face_complete = face_complete[:]
    
    # Phase 2: Local search with 2-opt style moves
    # Move types:
    # 1. Add a single edge (if possible)
    # 2. Remove a single edge
    # 3. Swap: remove one, add another
    # 4. 2-opt: remove two, add two
    
    def get_conflicts():
        """Return list of faces that are complete (all 4 edges selected)."""
        conflicts = []
        for f_idx in range(n_faces):
            if face_complete[f_idx] == 4:
                conflicts.append(f_idx)
        return conflicts
    
    # First, resolve any conflicts from greedy (shouldn't have any, but just in case)
    while get_conflicts():
        conflicts = get_conflicts()
        for f_idx in conflicts:
            face = faces[f_idx]
            # Remove the edge with highest centrality score
            max_score_e = max(face, key=lambda e: edge_scores[e])
            remove_edge(max_score_e)
            if not get_conflicts():
                break
    
    best_count = selected_count
    best_selected = selected[:]
    best_face_complete = face_complete[:]
    
    # Local search loop
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
        improved = False
        
        # Try to add any edge that can be added
        for e_idx in range(n_edges):
            if not selected[e_idx] and try_add(e_idx):
                add_edge(e_idx)
                improved = True
                if selected_count > best_count:
                    best_count = selected_count
                    best_selected = selected[:]
                    best_face_complete = face_complete[:]
        
        if not improved:
            # Try 2-opt swaps: remove edge A, add edge B
            # Focus on edges that are in "tight" faces (faces with 3 selected)
            tight_faces = [f for f in range(n_faces) if face_complete[f] == 3]
            
            if tight_faces:
                # For each tight face, consider removing one of its edges
                # and see if we can add a different edge
                for f_idx in tight_faces[:50]:  # Limit to avoid too much work
                    face = faces[f_idx]
                    for e_remove in face:
                        if not selected[e_remove]:
                            continue
                        
                        # Try removing e_remove
                        remove_edge(e_remove)
                        
                        # Now try to add edges that were previously blocked
                        added_any = False
                        for e_add in range(n_edges):
                            if not selected[e_add] and e_add != e_remove and try_add(e_add):
                                add_edge(e_add)
                                added_any = True
                                break
                        
                        if not added_any:
                            # Put e_remove back
                            add_edge(e_remove)
                        else:
                            # Check if we improved
                            if selected_count > best_count:
                                best_count = selected_count
                                best_selected = selected[:]
                                best_face_complete = face_complete[:]
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
        
        # Random restarts if stuck
        if no_improve_count > 500 and time.time() < deadline - 20.0:
            # Reset and try again with different random seed
            random.seed(int(time.time() * 1000) % 100000)
            selected = [False] * n_edges
            selected_count = 0
            face_complete = [0] * n_faces
            
            # New random order
            order = list(range(n_edges))
            random.shuffle(order)
            
            for e_idx in order:
                if try_add(e_idx):
                    add_edge(e_idx)
            
            # Resolve conflicts
            while get_conflicts():
                conflicts = get_conflicts()
                for f_idx in conflicts:
                    face = faces[f_idx]
                    max_score_e = max(face, key=lambda e: edge_scores[e])
                    remove_edge(max_score_e)
                    if not get_conflicts():
                        break
            
            if selected_count > best_count:
                best_count = selected_count
                best_selected = selected[:]
                best_face_complete = face_complete[:]
            
            no_improve_count = 0
    
    # Final: use best solution
    result = []
    for e_idx in range(n_edges):
        if best_selected[e_idx]:
            result.append(list(edges[e_idx]))
    
    return result
# EVOLVE-BLOCK-END
