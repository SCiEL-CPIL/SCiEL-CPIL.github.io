import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Generate candidate vectors with norm squared = 4
    # Type 1: +/- 2 e_i  (22 vectors)
    # Type 2: Four +/- 1s (C(11,4) * 2^4 = 330 * 16 = 5280 vectors)
    
    vectors = []
    
    # Type 1
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Type 2
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            vectors.append(v)
            
    vectors = np.array(vectors, dtype=np.int64)
    N = len(vectors)
    
    # Compute dot products
    dots = vectors @ vectors.T
    
    # Conflict matrix: conflict[i, j] = 1 if dot[i, j] > 2
    # We want an independent set in this graph
    conflict = (dots > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Greedy coloring to get an upper bound for branch-and-bound
    # This is a heuristic: we color the graph, and the number of colors is an upper bound on clique size,
    # but for independent set, we need to think differently.
    # Actually, for maximum independent set, a common approach is to use a greedy coloring on the complement graph
    # or use a specific heuristic.
    # However, a simpler and effective approach for this size is a greedy algorithm with local search,
    # possibly with multiple restarts and different orderings.
    
    # Let's try a different strategy:
    # 1. Greedy selection with random restarts (as in current program)
    # 2. Local search to improve the solution
    
    # Precompute conflicts as lists for fast lookup
    conflicts = []
    for i in range(N):
        conflicts.append(np.where(conflict[i] == 1)[0])
    
    best_size = 0
    best_indices = []
    
    rng = np.random.default_rng(42)
    
    # Try multiple restarts with different random orderings
    num_restarts = 300  # Adjust based on time budget
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        order = rng.permutation(N)
        selected = []
        selected_set = set()
        
        for idx in order:
            # Check if idx conflicts with any selected
            compatible = True
            for j in conflicts[idx]:
                if j in selected_set:
                    compatible = False
                    break
            
            if compatible:
                selected.append(idx)
                selected_set.add(idx)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_indices = selected[:]
            
        # Local search: try to add more vectors or swap to improve
        # Simple local search: try to add vectors not in selected
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add each unselected vector
            for idx in range(N):
                if idx in selected_set:
                    continue
                # Check if idx can be added
                compatible = True
                for j in conflicts[idx]:
                    if j in selected_set:
                        compatible = False
                        break
                if compatible:
                    selected.append(idx)
                    selected_set.add(idx)
                    improved = True
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_indices = selected[:]
                    
                    # After adding, try to add more
                    break  # Restart the loop to try adding more
            
            # Try swapping: remove one selected, add two unselected
            if time.time() > deadline:
                break
                
            # Simple swap heuristic
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                selected.remove(removed)
                selected_set.remove(removed)
                
                # Try to add vectors that conflict with removed but not with others
                for idx in range(N):
                    if idx in selected_set:
                        continue
                    # Check if idx conflicts with any selected
                    compatible = True
                    for j in conflicts[idx]:
                        if j in selected_set:
                            compatible = False
                            break
                    if compatible:
                        selected.append(idx)
                        selected_set.add(idx)
                        improved = True
                        if len(selected) > best_size:
                            best_size = len(selected)
                            best_indices = selected[:]
                        break
                
                # If we didn't improve, put the removed vector back
                if not improved:
                    selected.append(removed)
                    selected_set.add(removed)
                else:
                    # Keep the improvement, don't put it back
                    pass

    if best_size == 0:
        # Fallback to seed solution
        vectors_seed = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors_seed.append(v)
        return np.array(vectors_seed, dtype=np.int64)
        
    return vectors[best_indices]

# EVOLVE-BLOCK-END
