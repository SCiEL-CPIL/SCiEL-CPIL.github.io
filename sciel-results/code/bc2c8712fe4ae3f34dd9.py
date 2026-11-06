import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448
    
    # 1. Precompute Edge-Index Mapping
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    assert idx == num_edges
    
    # 2. Precompute Face Constraints
    # For each edge, store the list of faces it belongs to.
    # For each face, store the 4 edge indices.
    # Also, for fast vectorized checks, we can precompute for each edge e,
    # a list of "conflict triples" (the other 3 edges in a face).
    # If all 3 are selected, e cannot be added.
    
    faces = []
    edge_to_faces = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    ie1 = edge_to_idx[e1]
                    ie2 = edge_to_idx[e2]
                    ie3 = edge_to_idx[e3]
                    ie4 = edge_to_idx[e4]
                    
                    face_indices = (ie1, ie2, ie3, ie4)
                    faces.append(face_indices)
                    
                    for ie in face_indices:
                        edge_to_faces[ie].append(len(faces) - 1)
                        
    num_faces = len(faces)
    
    # 3. Strategy: Max-SAT style Local Search (WalkSAT variant)
    # The problem is equivalent to a Max-SAT problem where:
    # - Variables: 448 edges (True = selected)
    # - Clauses: For each face (4 edges), 4 clauses.
    #   Clause 1: NOT (e1 AND e2 AND e3 AND e4)  ->  (!e1 OR !e2 OR !e3 OR !e4)
    #   Wait, the constraint is "No 2-dimensional face may have all four edges selected".
    #   This is exactly one clause per face: (!e1 OR !e2 OR !e3 OR !e4).
    #   We want to maximize the number of True variables.
    #   
    # Standard WalkSAT/Max-SAT local search:
    # 1. Start with random assignment.
    # 2. While not optimal:
    #    a. If a clause is violated, pick one.
    #    b. Flip a variable in that clause to satisfy it.
    #       - With prob p, flip a variable that satisfies the clause (greedy).
    #       - With prob 1-p, flip a random variable in the clause (random walk).
    #    
    # However, we want to MAXIMIZE selected edges, not just satisfy constraints.
    # A better approach for this specific structure (C4-free in Q7):
    # We can view it as finding a maximum independent set in a 4-uniform hypergraph?
    # No, the constraint is on the edges of the hypergraph.
    # Let's stick to the Max-SAT formulation.
    # Objective: Maximize sum(x_i).
    # Constraint: For all faces f, NOT (x_a AND x_b AND x_c AND x_d).
    
    # Let's use a hybrid:
    # 1. Greedy Construction to get a good starting point.
    # 2. Local Search with "Add" and "Swap" moves.
    #    - Add: Try to add a random edge if valid.
    #    - Swap: Remove one edge, then try to add two or more edges? 
    #      Actually, simple 1-for-1 swap is neutral. 
    #      We need to escape local optima where no single add is possible.
    #      This implies we must remove k edges to add k+1 edges.
    #      Common technique: Remove 1 edge, then do a local repair (greedy add) to see if we can recover and gain.
    
    def greedy_construct(seed):
        random.seed(seed)
        edges = list(range(num_edges))
        random.shuffle(edges)
        selected = np.zeros(num_edges, dtype=bool)
        
        # Fast validity check using precomputed faces
        for e in edges:
            valid = True
            for f_idx in edge_to_faces[e]:
                f_edges = faces[f_idx]
                # Check if the other 3 are selected
                if selected[f_edges[0]] and selected[f_edges[1]] and selected[f_edges[2]] and f_edges[3] != e:
                    # This logic is flawed, need to check specifically the 3 others
                    pass
                
            # Let's do a proper check
            for f_idx in edge_to_faces[e]:
                f_edges = faces[f_idx]
                count = 0
                for oe in f_edges:
                    if oe != e and selected[oe]:
                        count += 1
                if count == 3:
                    valid = False
                    break
            if valid:
                selected[e] = True
                
        return selected

    def is_valid_add(selected, e):
        for f_idx in edge_to_faces[e]:
            f_edges = faces[f_idx]
            count = 0
            for oe in f_edges:
                if oe != e and selected[oe]:
                    count += 1
            if count == 3:
                return False
        return True

    def local_search(selected, seed, max_steps=5000):
        random.seed(seed)
        current_count = np.sum(selected)
        best_count = current_count
        best_selected = selected.copy()
        
        T = 5.0
        cooling = 0.9995
        steps = 0
        
        while steps < max_steps and time.time() < deadline:
            # Pick a random move
            # 1. Try to add a random edge
            # 2. Try to remove a random edge (if it allows adding more later, handled by repair)
            # 3. Try a 2-edge swap (remove 1, add 1) - usually neutral, but helps escape
            # Let's try: Remove 1 random selected edge, then Greedy Repair (add all possible valid edges).
            # This is a powerful move for escaping local optima in dense problems.
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a random edge
                e = random.randint(0, num_edges - 1)
                if not selected[e]:
                    if is_valid_add(selected, e):
                        selected[e] = True
                        current_count += 1
                    else:
                        # With some probability, accept a "bad" move? No, can't violate constraint.
                        # So just skip or try another edge.
                        pass
            elif move_type < 0.9:
                # Remove 1 random selected edge, then Repair
                # Find a selected edge
                sel_edges = np.where(selected)[0]
                if len(sel_edges) > 0:
                    e = sel_edges[random.randint(0, len(sel_edges) - 1)]
                    selected[e] = False
                    current_count -= 1
                    
                    # Greedy Repair: Add any edge that is now valid
                    # To make this fast, we don't scan all 448 edges, just a few random ones?
                    # Or scan all? 448 is small.
                    for oe in range(num_edges):
                        if not selected[oe]:
                            if is_valid_add(selected, oe):
                                selected[oe] = True
                                current_count += 1
            else:
                # Random Walk: Flip a random edge (if valid)
                e = random.randint(0, num_edges - 1)
                if selected[e]:
                    # Removing is always valid
                    selected[e] = False
                    current_count -= 1
                else:
                    if is_valid_add(selected, e):
                        selected[e] = True
                        current_count += 1
                    else:
                        # Cannot add, so this move is rejected (no flip)
                        pass
            
            if current_count > best_count:
                best_count = current_count
                best_selected = selected.copy()
                
            T *= cooling
            if T < 0.01:
                T = 0.01
            steps += 1
            
            if steps % 100 == 0 and time.time() > deadline:
                break
                
        return best_selected, best_count

    best_count = 0
    best_selected = np.zeros(num_edges, dtype=bool)
    
    seed = 0
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        # Greedy start
        current_selected = greedy_construct(seed)
        seed += 1
        current_count = np.sum(current_selected)
        
        # Local Search
        current_selected, current_count = local_search(current_selected, seed, max_steps=3000)
        seed += 1
        
        if current_count > best_count:
            best_count = current_count
            best_selected = current_selected.copy()
            
        # Occasionally, if we have time, do a more intensive search on the best found so far
        if seed % 10 == 0 and time.time() < deadline - 20:
             # Intensive search on best
             intensive_selected, intensive_count = local_search(best_selected.copy(), seed, max_steps=10000)
             seed += 1
             if intensive_count > best_count:
                 best_count = intensive_count
                 best_selected = intensive_selected.copy()

    # Convert best_selected to list of edges
    result = []
    for i in range(num_edges):
        if best_selected[i]:
            result.append(list(idx_to_edge[i]))
            
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
