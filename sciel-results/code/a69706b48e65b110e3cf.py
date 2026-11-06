"""Program to find a large C4-free edge set in Q7."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave buffer before 120s hard limit
    
    d = 7
    n_vertices = 1 << d  # 128
    n_edges = n_vertices * d // 2  # 448
    
    # Precompute all edges
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx
    
    # Precompute for each edge, the set of edges that form a C4 with it
    # Two edges (a,b) and (c,d) form a C4 if they are disjoint and 
    # the other two edges of the face are also present.
    # Actually, two edges conflict (cannot both be in a C4-free set) if they are
    # opposite edges of a 2-face, OR if they are two edges of a face such that
    # adding them would complete a C4 with two other edges.
    # 
    # More precisely: An edge set is C4-free iff no 4-cycle is fully contained.
    # Two edges e1, e2 "conflict" in the sense that they cannot both be present
    # if there exist two other edges e3, e4 such that {e1,e2,e3,e4} forms a C4.
    # But that's not quite right either - they can both be present as long as
    # e3 and e4 are not both present.
    #
    # The correct model: We want to select a subset of edges such that no C4 is
    # fully selected. This is a 4-uniform hypergraph independent set problem.
    # Each C4 (2-face) is a hyperedge of size 4. We want to avoid having all 4.
    #
    # Equivalent: For each 2-face, at least one of its 4 edges must be excluded.
    #
    # Let's use a different approach: local search / simulated annealing.
    # Start with a greedy solution, then try to add edges and remove edges to improve.
    
    # Precompute all 2-faces (C4s). A 2-face is determined by choosing 2 dimensions
    # and a starting vertex, giving 4 edges.
    faces = []  # Each face is a set of 4 edge indices
    for dim1 in range(d):
        for dim2 in range(dim1 + 1, d):
            bit1 = 1 << dim1
            bit2 = 1 << dim2
            for u in range(n_vertices):
                # Only process if u is the "minimum" vertex of this face
                # to avoid duplicates
                if (u & bit1) or (u & bit2):
                    continue
                v1 = u ^ bit1
                v2 = u ^ bit2
                v12 = u ^ bit1 ^ bit2
                e1 = edge_to_idx[(min(u, v1), max(u, v1))]
                e2 = edge_to_idx[(min(v1, v12), max(v1, v12))]
                e3 = edge_to_idx[(min(u, v2), max(u, v2))]
                e4 = edge_to_idx[(min(v2, v12), max(v2, v12))]
                faces.append([e1, e2, e3, e4])
    
    n_faces = len(faces)
    # faces: list of lists of 4 edge indices
    
    # For each edge, precompute which faces contain it
    edge_to_faces = [[] for _ in range(n_edges)]
    for fi, face in enumerate(faces):
        for ei in face:
            edge_to_faces[ei].append(fi)
    
    # Greedy initialization: start empty, add edges in random order
    # An edge can be added if it doesn't complete any C4
    # i.e., for every face containing this edge, at least one other edge in the face is not selected
    
    def greedy_init():
        selected = [False] * n_edges
        count = 0
        # Track for each face how many of its 4 edges are selected
        face_count = [0] * n_faces
        
        order = list(range(n_edges))
        random.shuffle(order)
        
        for ei in order:
            # Check if adding ei would complete any face
            can_add = True
            for fi in edge_to_faces[ei]:
                if face_count[fi] == 3:
                    can_add = False
                    break
            if can_add:
                selected[ei] = True
                count += 1
                for fi in edge_to_faces[ei]:
                    face_count[fi] += 1
        
        return selected, count, face_count
    
    # Local search: try to add an unselected edge, or swap (add one, remove one)
    def local_search(selected, count, face_count, max_iters=100000):
        improved = True
        while improved and (time.time() - start_time) < time_limit - 5.0:
            improved = False
            # Try adding edges
            for ei in range(n_edges):
                if selected[ei]:
                    continue
                can_add = True
                for fi in edge_to_faces[ei]:
                    if face_count[fi] == 3:
                        can_add = False
                        break
                if can_add:
                    selected[ei] = True
                    count += 1
                    for fi in edge_to_faces[ei]:
                        face_count[fi] += 1
                    improved = True
            
            if not improved:
                # Try swaps: remove one edge, then try to add more
                # Find an edge whose removal allows adding 2+ new edges
                for ei in range(n_edges):
                    if not selected[ei]:
                        continue
                    # Try removing ei
                    selected[ei] = False
                    count -= 1
                    for fi in edge_to_faces[ei]:
                        face_count[fi] -= 1
                    
                    added = 0
                    added_edges = []
                    for ej in range(n_edges):
                        if selected[ej]:
                            continue
                        can_add = True
                        for fi in edge_to_faces[ej]:
                            if face_count[fi] == 3:
                                can_add = False
                                break
                        if can_add:
                            selected[ej] = True
                            count += 1
                            added += 1
                            added_edges.append(ej)
                            for fi in edge_to_faces[ej]:
                                face_count[fi] += 1
                    
                    if added > 1:
                        improved = True
                    else:
                        # Revert
                        for ej in added_edges:
                            selected[ej] = False
                            count -= 1
                            for fi in edge_to_faces[ej]:
                                face_count[fi] -= 1
                        selected[ei] = True
                        count += 1
                        for fi in edge_to_faces[ei]:
                            face_count[fi] += 1
        
        return selected, count, face_count
    
    best_count = 0
    best_selected = None
    best_face_count = None
    
    # Multiple random restarts
    n_restarts = 50
    for r in range(n_restarts):
        if (time.time() - start_time) > time_limit - 10.0:
            break
        
        random.seed(r * 12345 + 67890)
        selected, count, face_count = greedy_init()
        
        selected, count, face_count = local_search(selected, count, face_count)
        
        if count > best_count:
            best_count = count
            best_selected = selected[:]
            best_face_count = face_count[:]
    
    # Convert to output format
    result = []
    for ei in range(n_edges):
        if best_selected[ei]:
            u, v = edges[ei]
            result.append([u, v])
    
    return result
