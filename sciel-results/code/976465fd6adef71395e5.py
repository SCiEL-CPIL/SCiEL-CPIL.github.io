"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_faces():
    """Precompute all 2D faces (C4s) and which edges belong to each face."""
    d = 7
    faces = []  # list of lists of edge keys
    edge_to_faces = {}  # edge key -> list of face indices
    for x in range(1 << d):
        for i in range(d):
            for j in range(i + 1, d):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    e = x ^ (1 << i) ^ (1 << j)
                    e0 = _edge_key(a, b)
                    e1 = _edge_key(c, e)
                    e2 = _edge_key(a, c)
                    e3 = _edge_key(b, e)
                    face = [e0, e1, e2, e3]
                    fi = len(faces)
                    faces.append(face)
                    for ek in face:
                        edge_to_faces.setdefault(ek, []).append(fi)
    return faces, edge_to_faces

def _greedy_construct():
    """Original greedy: scan edges in order and keep one unless it completes a four-cycle."""
    d, chosen, out = 7, set(), []
    for x in range(2 ** d):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                ok = True
                for j in range(d):
                    if j != i:
                        c, e = x ^ (1 << j), y ^ (1 << j)
                        if all(p in chosen for p in ((min(x, c), max(x, c)), (min(y, e), max(y, e)), (min(c, e), max(c, e)))):
                            ok = False
                            break
                if ok:
                    chosen.add((x, y))
                    out.append([x, y])
    return out

def _face_violation_count(chosen, faces, edge_to_faces):
    """Count how many faces have all 4 edges selected."""
    count = 0
    for face in faces:
        if all(e in chosen for e in face):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    d = 7
    faces, edge_to_faces = _build_faces()
    
    # Start with greedy solution
    edges = _greedy_construct()
    chosen = set(_edge_key(u, v) for u, v in edges)
    
    # Precompute all edges
    all_edges = []
    for x in range(2 ** d):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append(_edge_key(x, y))
    
    # Track face violation counts
    face_viol = np.zeros(len(faces), dtype=np.int32)
    for ek in chosen:
        for fi in edge_to_faces.get(ek, []):
            face_viol[fi] += 1
    
    # Count violations
    violations = np.sum(face_viol == 4)
    
    # Build adjacency: for each edge, which other edges form a C4 with it
    # Actually, let's use a different approach: for each edge, the faces it belongs to
    # and the other edges in those faces
    
    # Simulated annealing
    best_edges = list(chosen)
    best_count = len(chosen)
    
    # For efficient moves, precompute for each edge the set of edges that would create a C4
    # If we add edge e, it creates a C4 if there's a face where the other 3 edges are all present
    
    # Let's track for each face how many of its 4 edges are in chosen
    # face_viol[fi] = number of edges in face fi that are in chosen
    
    # Move types:
    # 1. Add an unchosen edge (if it doesn't create a new C4, or if we're in annealing mode)
    # 2. Remove a chosen edge
    # 3. Swap: remove one edge, add another
    
    # For adding edge e: it creates a C4 in face fi if face_viol[fi] == 3 (other 3 edges present)
    # For removing edge e: it decreases face_viol[fi] by 1 for each face fi containing e
    
    # Let's use a simpler approach: iterate over edges and try to add them
    # with some randomization and local search
    
    rng = np.random.default_rng(42)
    
    # Precompute for each edge, the list of face indices it belongs to
    edge_faces = {ek: edge_to_faces.get(ek, []) for ek in all_edges}
    
    # For each edge, precompute the "blocking" edges: edges that share a face
    # Actually, for the C4 constraint: edge e1 and e2 are in the same face if they share a face
    # Adding e creates a C4 in face fi if the other 3 edges of fi are all in chosen
    
    # Let's do a local search with simulated annealing
    # State: chosen set of edges
    # Objective: maximize |chosen| subject to no face has all 4 edges
    
    # We'll use a penalty-based approach: minimize -|chosen| + lambda * violations
    # Or better: only allow moves that don't increase violations, and use annealing to escape
    
    # Actually, let's use a different strategy:
    # 1. Start with greedy
    # 2. Find all violated faces
    # 3. For each violated face, remove one of its edges (the one that appears in the most violated faces)
    # 4. Try to add back edges that were removed or other unchosen edges
    # 5. Repeat until no improvement
    
    # Let's implement a more sophisticated local search
    
    # Track which edges are in chosen
    in_chosen = np.zeros(len(all_edges), dtype=bool)
    edge_index = {ek: i for i, ek in enumerate(all_edges)}
    for ek in chosen:
        in_chosen[edge_index[ek]] = True
    
    # face_viol[fi] = number of edges in face fi that are in chosen
    # A face is violated if face_viol[fi] == 4
    
    # For each edge index, the faces it belongs to
    edge_face_list = [edge_faces[all_edges[i]] for i in range(len(all_edges))]
    
    # For each face, the 4 edge indices
    face_edges = []
    for face in faces:
        face_edges.append([edge_index[ek] for ek in face])
    
    # Rebuild face_viol
    face_viol = np.zeros(len(faces), dtype=np.int32)
    for i in range(len(all_edges)):
        if in_chosen[i]:
            for fi in edge_face_list[i]:
                face_viol[fi] += 1
    
    def get_violations():
        return np.sum(face_viol == 4)
    
    current_violations = get_violations()
    current_count = np.sum(in_chosen)
    
    # Local search with simulated annealing
    # Temperature schedule
    T_start = 1.0
    T_end = 0.001
    cooling = 0.999
    
    T = T_start
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and T > T_end:
        # Randomly pick a move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add an unchosen edge
            unchosen_indices = np.where(~in_chosen)[0]
            if len(unchosen_indices) == 0:
                T *= cooling
                continue
            i = unchosen_indices[rng.integers(len(unchosen_indices))]
            
            # Check if adding this edge creates any new C4
            new_violations = 0
            for fi in edge_face_list[i]:
                if face_viol[fi] == 3:  # other 3 edges present, adding this makes 4
                    new_violations += 1
            
            delta_obj = -1 + new_violations * 10  # penalty for violations
            # Accept if delta < 0 or with probability exp(-delta/T)
            if delta_obj < 0 or rng.random() < np.exp(-delta_obj / T):
                in_chosen[i] = True
                for fi in edge_face_list[i]:
                    face_viol[fi] += 1
                current_count += 1
                current_violations += new_violations
                no_improve_count = 0
            else:
                no_improve_count += 1
        elif move_type < 0.7:
            # Try to remove a chosen edge
            chosen_indices = np.where(in_chosen)[0]
            if len(chosen_indices) == 0:
                T *= cooling
                continue
            i = chosen_indices[rng.integers(len(chosen_indices))]
            
            # Removing an edge: always improves or maintains
            in_chosen[i] = False
            for fi in edge_face_list[i]:
                face_viol[fi] -= 1
            current_count -= 1
            # Check if this fixed any violations
            fixed = 0
            for fi in edge_face_list[i]:
                if face_viol[fi] == 3:  # was 4, now 3
                    fixed += 1
            current_violations -= fixed
            no_improve_count = 0
        else:
            # Swap: remove one chosen, add one unchosen
            chosen_indices = np.where(in_chosen)[0]
            unchosen_indices = np.where(~in_chosen)[0]
            if len(chosen_indices) == 0 or len(unchosen_indices) == 0:
                T *= cooling
                continue
            i_rem = chosen_indices[rng.integers(len(chosen_indices))]
            i_add = unchosen_indices[rng.integers(len(unchosen_indices))]
            
            # Remove i_rem
            in_chosen[i_rem] = False
            for fi in edge_face_list[i_rem]:
                face_viol[fi] -= 1
            
            # Add i_add
            new_violations = 0
            for fi in edge_face_list[i_add]:
                if face_viol[fi] == 3:
                    new_violations += 1
            
            delta_obj = -1 + new_violations * 10
            if delta_obj < 0 or rng.random() < np.exp(-delta_obj / T):
                in_chosen[i_add] = True
                for fi in edge_face_list[i_add]:
                    face_viol[fi] += 1
                current_count += 1  # net: -1 + 1 = 0, but we track count
                # Actually count stays same for swap
                current_violations += new_violations
                no_improve_count = 0
            else:
                # Revert
                in_chosen[i_rem] = True
                for fi in edge_face_list[i_rem]:
                    face_viol[fi] += 1
                no_improve_count += 1
        
        # Track best
        if current_count > best_count and current_violations == 0:
            best_count = current_count
            best_edges = [all_edges[i] for i in range(len(all_edges)) if in_chosen[i]]
        
        # Update temperature
        T *= cooling
        
        if no_improve_count > max_no_improve:
            # Restart with greedy
            chosen = set(_edge_key(u, v) for u, v in _greedy_construct())
            in_chosen = np.zeros(len(all_edges), dtype=bool)
            for ek in chosen:
                in_chosen[edge_index[ek]] = True
            face_viol = np.zeros(len(faces), dtype=np.int32)
            for i in range(len(all_edges)):
                if in_chosen[i]:
                    for fi in edge_face_list[i]:
                        face_viol[fi] += 1
            current_violations = get_violations()
            current_count = np.sum(in_chosen)
            no_improve_count = 0
            T = T_start
    
    # Final cleanup: remove edges from violated faces
    # Greedy removal: for each violated face, remove the edge that's in the most violated faces
    improved = True
    while improved and time.time() < deadline:
        improved = False
        violated_faces = np.where(face_viol == 4)[0]
        if len(violated_faces) == 0:
            break
        
        # For each violated face, count how many violated faces each of its edges is in
        edge_viol_count = {}
        for fi in violated_faces:
            for ei in face_edges[fi]:
                edge_viol_count[ei] = edge_viol_count.get(ei, 0) + 1
        
        # Remove the edge with the highest violation count
        best_ei = max(edge_viol_count, key=edge_viol_count.get)
        in_chosen[best_ei] = False
        for fi in edge_face_list[best_ei]:
            face_viol[fi] -= 1
        current_count -= 1
        improved = True
    
    # Now try to add back edges
    for i in range(len(all_edges)):
        if in_chosen[i]:
            continue
        if time.time() > deadline - 5:
            break
        # Check if adding this edge creates a C4
        can_add = True
        for fi in edge_face_list[i]:
            if face_viol[fi] == 3:
                can_add = False
                break
        if can_add:
            in_chosen[i] = True
            for fi in edge_face_list[i]:
                face_viol[fi] += 1
            current_count += 1
    
    # Convert to output format
    result = []
    for i in range(len(all_edges)):
        if in_chosen[i]:
            ek = all_edges[i]
            result.append([ek[0], ek[1]])
    
    return result

# EVOLVE-BLOCK-END
