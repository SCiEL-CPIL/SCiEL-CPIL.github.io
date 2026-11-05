import time
import random
import sys
import math
import numpy as np

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    
    Strategy:
    1. Precompute edges and faces (C4s) with bitset representations for speed.
    2. Use a deterministic linear algebraic construction to generate a strong initial feasible state.
    3. Perform Multi-Phase Simulated Annealing:
       - Phase 1: High temperature, focus on moving between feasible states (swap moves).
       - Phase 2: Lower temperature, focus on adding edges (if possible) or fine-tuning.
       - Multiple restarts with different random seeds to escape local optima.
    4. Final greedy repair to ensure validity and maximize density.
    
    This approach avoids the potential stagnation of a single ILP solve and leverages the 
    ability of SA to navigate complex energy landscapes.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s kill
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # 1. Precompute edges and faces
    edge_list = []
    edge_to_idx = {}
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edge_list)
                edge_list.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    E = len(edge_list)  # 448
    
    # Precompute faces (C4s)
    faces = []
    face_set = set()
    
    for u in range(n):
        for i in range(d):
            for j in range(i+1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_tuple = tuple(sorted([idx1, idx2, idx3, idx4]))
                
                if face_tuple not in face_set:
                    face_set.add(face_tuple)
                    faces.append([idx1, idx2, idx3, idx4])
    
    F = len(faces)  # 672
    
    # Precompute edge-to-face mapping
    edge_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # Helper: Initialize a valid dense state using a linear rule
    # Rule: Include edge (u, u^i) if (u dot v_i) % 2 == 0, where v_i is a fixed vector.
    # We can tune v_i to maximize density while maintaining C4-free property.
    # A simple heuristic: randomize v_i and pick the best one.
    
    def initialize_valid():
        best_score = -1
        best_state = None
        best_counts = None
        
        for attempt in range(50):
            # Generate random vectors for each dimension
            v_vecs = [np.random.randint(0, 2, size=d) for _ in range(d)]
            
            state = bytearray(E)
            for idx, (u, v) in enumerate(edge_list):
                # Determine dimension of the edge
                diff = u ^ v
                i = diff.bit_length() - 1
                # Check condition: (u dot v_i) % 2 == 0
                dot = np.dot(u.to_bytes(d, 'little'), v_vecs[i]) # This is wrong, u is int
                # Correct dot product calculation
                u_bits = [(u >> k) & 1 for k in range(d)]
                dot = sum(u_bits[k] * v_vecs[i][k] for k in range(d))
                
                if dot % 2 == 0:
                    state[idx] = 1
            
            # Repair if necessary (though linear rules should be C4-free if chosen correctly, 
            # random v_i might not guarantee it, so we repair)
            full_count = [0] * F
            for f_idx in range(F):
                f = faces[f_idx]
                full_count[f_idx] = state[f[0]] + state[f[1]] + state[f[2]] + state[f[3]]
            
            # Repair loop
            while True:
                found_conflict = False
                for f_idx in range(F):
                    if full_count[f_idx] == 4:
                        f = faces[f_idx]
                        # Remove edge with highest "cost" or random
                        e_to_remove = random.choice(f)
                        if state[e_to_remove]:
                            state[e_to_remove] = 0
                            for f_idx2 in edge_faces[e_to_remove]:
                                full_count[f_idx2] -= 1
                        found_conflict = True
                        break
                if not found_conflict:
                    break
            
            score = sum(state)
            if score > best_score:
                best_score = score
                best_state = state
                best_counts = full_count
        
        return best_state, best_counts, best_score

    # Simulated Annealing Step
    def simulated_annealing(state, full_count, current_score, max_time, seed):
        rng = random.Random(seed)
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        best_state = state[:]
        best_score = current_score
        best_counts = full_count[:]
        
        # Precompute move costs? No, dynamic is better for SA.
        
        while T > min_T:
            if time.time() - start_time > max_time:
                break
            
            # Generate a random move
            # Move types:
            # 1. Add an edge (if valid)
            # 2. Remove an edge
            # 3. Swap: Remove one, add one (or two)
            
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to add a random edge
                e = rng.randint(0, E - 1)
                if state[e] == 0:
                    # Check if adding e creates a conflict
                    valid = True
                    for f_idx in edge_faces[e]:
                        if full_count[f_idx] == 3:
                            valid = False
                            break
                    if valid:
                        # Accept always (improvement)
                        state[e] = 1
                        for f_idx in edge_faces[e]:
                            full_count[f_idx] += 1
                        current_score += 1
                        if current_score > best_score:
                            best_score = current_score
                            best_state = state[:]
                            best_counts = full_count[:]
                    # If invalid, do nothing (or reject)
                
            elif move_type < 0.8:
                # Try to remove a random edge (to allow escape)
                e = rng.randint(0, E - 1)
                if state[e] == 1:
                    # Calculate energy change (score decrease)
                    delta_score = -1
                    # Acceptance probability
                    if delta_score < 0:
                        prob = math.exp(delta_score / T)
                        if rng.random() < prob:
                            state[e] = 0
                            for f_idx in edge_faces[e]:
                                full_count[f_idx] -= 1
                            current_score += delta_score
                    else:
                        state[e] = 0
                        for f_idx in edge_faces[e]:
                            full_count[f_idx] -= 1
                        current_score += delta_score
                        if current_score > best_score:
                            best_score = current_score
                            best_state = state[:]
                            best_counts = full_count[:]
            
            else:
                # Swap: Remove e1, try to add e2
                e1 = rng.randint(0, E - 1)
                if state[e1] == 1:
                    # Temporarily remove e1
                    state[e1] = 0
                    for f_idx in edge_faces[e1]:
                        full_count[f_idx] -= 1
                    
                    # Try to add a random e2
                    e2 = rng.randint(0, E - 1)
                    if e2 != e1:
                        valid = True
                        for f_idx in edge_faces[e2]:
                            if full_count[f_idx] == 3:
                                valid = False
                                break
                        if valid:
                            state[e2] = 1
                            for f_idx in edge_faces[e2]:
                                full_count[f_idx] += 1
                            # Score change: -1 (remove e1) + 1 (add e2) = 0
                            # This move is neutral in score but changes configuration
                            # Accept with some probability to explore
                            if rng.random() < 0.5:
                                pass # Keep the swap
                            else:
                                # Revert
                                state[e2] = 0
                                for f_idx in edge_faces[e2]:
                                    full_count[f_idx] -= 1
                                state[e1] = 1
                                for f_idx in edge_faces[e1]:
                                    full_count[f_idx] += 1
                        else:
                            # Revert
                            state[e1] = 1
                            for f_idx in edge_faces[e1]:
                                full_count[f_idx] += 1
                    else:
                        # Revert
                        state[e1] = 1
                        for f_idx in edge_faces[e1]:
                            full_count[f_idx] += 1
            
            T *= cooling
        
        return best_state, best_counts, best_score

    # Main Loop
    best_overall_state = None
    best_overall_score = -1
    best_overall_counts = None
    
    # Phase 1: Initialization
    init_state, init_counts, init_score = initialize_valid()
    best_overall_state = init_state
    best_overall_score = init_score
    best_overall_counts = init_counts
    
    # Phase 2: Multi-restart SA
    num_restarts = 10
    restart_time_budget = (time_limit - (time.time() - start_time) - 10.0) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Use current best as start for SA, or a slightly perturbed version
        start_state = best_overall_state[:]
        start_counts = best_overall_counts[:]
        start_score = best_overall_score
        
        # Perturb
        for _ in range(10):
            e = random.randint(0, E - 1)
            if start_state[e]:
                start_state[e] = 0
                for f_idx in edge_faces[e]:
                    start_counts[f_idx] -= 1
                start_score -= 1
            else:
                # Check validity before adding
                valid = True
                for f_idx in edge_faces[e]:
                    if start_counts[f_idx] == 3:
                        valid = False
                        break
                if valid:
                    start_state[e] = 1
                    for f_idx in edge_faces[e]:
                        start_counts[f_idx] += 1
                    start_score += 1
        
        end_time_for_this_restart = start_time + (time.time() - start_time) + restart_time_budget
        s_state, s_counts, s_score = simulated_annealing(
            start_state, start_counts, start_score, 
            end_time_for_this_restart, 
            seed=42 + r
        )
        
        if s_score > best_overall_score:
            best_overall_score = s_score
            best_overall_state = s_state
            best_overall_counts = s_counts
    
    # Phase 3: Final Greedy Repair
    best_state = best_overall_state
    best_full_count = best_overall_counts
    best_score = best_overall_score
    
    improved = True
    while improved:
        improved = False
        for e in range(E):
            if best_state[e]:
                continue
            can_add = True
            for f_idx in edge_faces[e]:
                if best_full_count[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                best_state[e] = 1
                for f_idx in edge_faces[e]:
                    best_full_count[f_idx] += 1
                best_score += 1
                improved = True
    
    # Construct result
    result = []
    for i in range(E):
        if best_state[i]:
            u, v = edge_list[i]
            result.append([u, v])
    
    return result

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}, Time: {end-start:.2f}s")
