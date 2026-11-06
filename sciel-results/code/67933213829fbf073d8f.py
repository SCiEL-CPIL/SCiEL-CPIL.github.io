import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448
    
    # 1. Precompute edges
    idx_to_edge = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # 2. Precompute faces
    seen_faces = set()
    face_edge_list = []
    
    for i in range(d):
        for j in range(i+1, d):
            for u in range(n):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idxs = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                key = frozenset(idxs)
                
                if key not in seen_faces:
                    seen_faces.add(key)
                    face_edge_list.append(idxs)
    
    num_faces = len(face_edge_list)
    
    # Precompute edge -> faces mapping
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(face_edge_list):
        for e in face:
            edge_to_faces[e].append(f_idx)
    
    # 3. Greedy hitting set initialization
    def greedy_hit():
        active = np.ones(num_edges, dtype=np.uint8)
        face_counts = np.full(num_faces, 4, dtype=np.int32)
        removed = 0
        while True:
            active_faces = np.where(face_counts == 4)[0]
            if len(active_faces) == 0:
                break
            scores = np.zeros(num_edges, dtype=np.int32)
            for f_idx in active_faces:
                for e in face_edge_list[f_idx]:
                    scores[e] += 1
            max_score = np.max(scores)
            if max_score == 0:
                break
            candidates = np.where(scores == max_score)[0]
            e_to_remove = candidates[0]
            active[e_to_remove] = 0
            removed += 1
            for f_idx in edge_to_faces[e_to_remove]:
                face_counts[f_idx] -= 1
        return active, removed
    
    # 4. Simulated annealing
    def count_violations(active):
        """Count faces with all 4 edges selected."""
        violations = 0
        for face in face_edge_list:
            if all(active[e] == 1 for e in face):
                violations += 1
        return violations
    
    def try_add_edge(active, face_counts, e):
        """Try to add edge e. Returns (success, new_face_counts)."""
        # Check if adding e creates any face with all 4 edges
        for f_idx in edge_to_faces[e]:
            if face_counts[f_idx] == 3:
                # This face would become 4 if we add e
                return False, face_counts
        # Safe to add
        new_counts = face_counts.copy()
        for f_idx in edge_to_faces[e]:
            new_counts[f_idx] += 1
        return True, new_counts
    
    def sa_refine(active, face_counts, rng, time_limit):
        """Simulated annealing to improve the solution."""
        t_start = time.time()
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        current_violations = count_violations(active)
        best_active = active.copy()
        best_violations = current_violations
        best_size = int(np.sum(active))
        
        # Build list of currently removed edges for quick access
        removed_edges = np.where(active == 0)[0]
        
        while time.time() < time_limit:
            if len(removed_edges) == 0:
                break
            
            # Randomly pick a removed edge to try re-adding
            e = removed_edges[rng.integers(0, len(removed_edges))]
            
            success, new_counts = try_add_edge(active, face_counts, e)
            
            if success:
                # Always accept improvement
                active[e] = 1
                face_counts = new_counts
                removed_edges = np.where(active == 0)[0]
                current_violations = count_violations(active)
                current_size = int(np.sum(active))
                
                if current_violations < best_violations or (current_violations == best_violations and current_size > best_size):
                    best_violations = current_violations
                    best_size = current_size
                    best_active = active.copy()
            else:
                # Adding e would create a violation. Try a kick: remove 1-2 edges from the problematic face, then re-add e
                # Find the face that would be violated
                problem_faces = []
                for f_idx in edge_to_faces[e]:
                    if face_counts[f_idx] == 3:
                        problem_faces.append(f_idx)
                
                if len(problem_faces) > 0:
                    # Pick a random problem face
                    f_idx = problem_faces[rng.integers(0, len(problem_faces))]
                    face = face_edge_list[f_idx]
                    
                    # Pick 1-2 edges from this face to remove (not e itself)
                    other_edges = [x for x in face if x != e]
                    num_remove = rng.integers(1, 3)  # 1 or 2
                    edges_to_remove = rng.choice(other_edges, size=min(num_remove, len(other_edges)), replace=False)
                    
                    # Remove those edges
                    for re in edges_to_remove:
                        active[re] = 0
                        for ff in edge_to_faces[re]:
                            face_counts[ff] -= 1
                    
                    # Now try to add e
                    success2, new_counts2 = try_add_edge(active, face_counts, e)
                    if success2:
                        active[e] = 1
                        face_counts = new_counts2
                        removed_edges = np.where(active == 0)[0]
                        current_violations = count_violations(active)
                        current_size = int(np.sum(active))
                        
                        if current_violations < best_violations or (current_violations == best_violations and current_size > best_size):
                            best_violations = current_violations
                            best_size = current_size
                            best_active = active.copy()
                    else:
                        # Revert the removal
                        for re in edges_to_remove:
                            active[re] = 1
                            for ff in edge_to_faces[re]:
                                face_counts[ff] += 1
            
            T *= cooling
            if T < T_min:
                break
        
        return best_active, best_violations, best_size
    
    # 5. Main search loop
    rng = np.random.default_rng(42)
    
    # Start with greedy initialization
    best_active, best_removed = greedy_hit()
    best_size = num_edges - best_removed
    best_violations = count_violations(best_active)
    
    face_counts = np.full(num_faces, 4, dtype=np.int32)
    for e in range(num_edges):
        if best_active[e] == 0:
            for f_idx in edge_to_faces[e]:
                face_counts[f_idx] -= 1
    
    # Run SA with multiple restarts
    num_restarts = 20
    time_per_restart = (deadline - start_time - 5) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # For restarts > 0, add random perturbation to the best solution
        if restart == 0:
            current_active = best_active.copy()
        else:
            current_active = best_active.copy()
            # Randomly remove 5-15 edges to perturb
            num_perturb = rng.integers(5, 16)
            all_edges = np.arange(num_edges)
            selected = rng.choice(all_edges, size=num_perturb, replace=False)
            for e in selected:
                if current_active[e] == 1:
                    current_active[e] = 0
        
        # Recompute face counts
        fc = np.full(num_faces, 4, dtype=np.int32)
        for e in range(num_edges):
            if current_active[e] == 0:
                for f_idx in edge_to_faces[e]:
                    fc[f_idx] -= 1
        
        # Run SA
        sa_time_limit = deadline - 5 - restart * 0.1  # Leave time for remaining restarts
        improved_active, improved_viol, improved_size = sa_refine(current_active, fc, rng, sa_time_limit)
        
        if improved_size > best_size or (improved_size == best_size and improved_viol < best_violations):
            best_size = improved_size
            best_violations = improved_viol
            best_active = improved_active
    
    # Final polish: try to add back any edges that were removed
    if time.time() < deadline - 2:
        fc = np.full(num_faces, 4, dtype=np.int32)
        for e in range(num_edges):
            if best_active[e] == 0:
                for f_idx in edge_to_faces[e]:
                    fc[f_idx] -= 1
        
        # Try each removed edge
        for e in range(num_edges):
            if best_active[e] == 0:
                success, new_fc = try_add_edge(best_active, fc, e)
                if success:
                    best_active[e] = 1
                    fc = new_fc
    
    # Convert to list of edges
    result = []
    for i in range(num_edges):
        if best_active[i] == 1:
            result.append(list(idx_to_edge[i]))
            
    return result
