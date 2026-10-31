import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate a mixed pool of vectors with weights 3, 4, and 5.
    # Weight 3: C(11,3) * 2^3 = 165 * 8 = 1320
    # Weight 4: C(11,4) * 2^4 = 330 * 16 = 5280
    # Weight 5: C(11,5) * 2^5 = 462 * 32 = 14784
    # Total ~ 21384 vectors. This is manageable for pairwise dot products.
    
    indices = range(D)
    pool_list = []
    
    # Helper to generate vectors of a specific weight
    def gen_vectors(weight):
        for comb in itertools.combinations(indices, weight):
            for signs in itertools.product([1, -1], repeat=weight):
                v = [0] * D
                for idx, sign in zip(comb, signs):
                    v[idx] = sign
                pool_list.append(v)
                
    gen_vectors(3)
    gen_vectors(4)
    gen_vectors(5)
    
    pool = np.array(pool_list, dtype=np.int64)
    n_pool = len(pool)
    
    # Precompute norms squared
    norms_sq = np.sum(pool * pool, axis=1)
    
    # Precompute dot products
    # n_pool is ~21k. 21k x 21k matrix is ~441M floats (8 bytes) = ~3.5GB. Too large for RAM?
    # Let's check. 21000 * 21000 * 8 bytes = 3.528 GB.
    # Standard RAM might be 8-16GB, so it fits, but computing it takes time.
    # 21k x 11 @ 11 x 21k. 
    # Matrix multiplication is optimized.
    # Let's try to do it in chunks if needed, or just do it if memory allows.
    # Actually, we don't need the full matrix for Simulated Annealing if we can compute
    # conflicts on the fly or use a smarter data structure.
    # However, for SA, we need to check if adding/removing a vector violates constraints
    # with the *currently selected* set.
    
    # Alternative Strategy: 
    # Since we want a large independent set, and the graph is sparse (most vectors don't conflict),
    # we can maintain the selected set as a list of indices.
    # To check if a new vector v can be added, we check its dot product with all selected vectors.
    # If |dot| > threshold, it conflicts.
    # Threshold depends on norms.
    # Condition: ||u - v||^2 >= max(||u||^2, ||v||^2) is NOT the condition.
    # Condition: min pairwise distance >= max vector norm.
    # Let R = max norm in the set.
    # For any pair u, v in set: ||u - v|| >= R.
    # This is a dynamic constraint because R changes as we add larger vectors.
    
    # This makes SA tricky because the "validity" of the whole set depends on the global max norm.
    # Simplification: If we restrict to a single weight, the norm is constant.
    # If we mix weights, the max norm is determined by the largest weight present.
    # If we include weight 5 (norm sqrt(5) ~ 2.23), R = sqrt(5).
    # Then for any pair u, v: ||u-v||^2 >= 5.
    # If u, v are weight 3 (norm^2=3): ||u-v||^2 = 3+3-2dot = 6-2dot.
    # 6-2dot >= 5 => 2dot <= 1 => dot <= 0.5 => dot <= 0 (integer).
    # If u is weight 3, v is weight 5 (norm^2=5): ||u-v||^2 = 3+5-2dot = 8-2dot.
    # 8-2dot >= 5 => 2dot <= 3 => dot <= 1.5 => dot <= 1.
    # If u, v are weight 5: ||u-v||^2 = 5+5-2dot = 10-2dot.
    # 10-2dot >= 5 => 2dot <= 5 => dot <= 2.5 => dot <= 2.
    
    # So the conflict condition depends on the weights of the pair.
    # This is complex for a fast SA loop.
    
    # Let's stick to a simpler, more robust structural change:
    # **Randomized Greedy with Multiple Restarts and Dynamic Pool Filtering**
    # But the prompt asks for a DIFFERENT strategy. The current is Min-Degree Greedy.
    # Let's try **Simulated Annealing on a Fixed-Norm Pool (Weight 4)** but with a better move set and cooling schedule, 
    # OR **Greedy Randomized Adaptive Search Procedure (GRASP)**.
    
    # Actually, the "Inspiration 3" suggests SA on mixed norms. 
    # Let's implement a simpler version: **Local Search (Hill Climbing) with Random Restarts** on the Weight-4 pool, 
    # but instead of just picking min-degree, we allow "swaps" to improve the set.
    # Current: Pick v, remove neighbors.
    # New: Start with a random valid set. Try to add a vector from the pool. If it conflicts, try to remove one of the conflicting vectors and add the new one if it improves the set size (or maintains it but allows future growth).
    
    # Let's go with **Simulated Annealing** on the Weight-4 pool. 
    # State: Set of selected indices.
    # Objective: Maximize size.
    # Moves: 
    # 1. Add a random unselected vector (if valid).
    # 2. Remove a random selected vector.
    # 3. Swap a selected vector with an unselected one (if the new set is valid).
    
    # To make it fast, we need efficient validity checks.
    # We precompute the conflict matrix for Weight-4 vectors.
    # n = 5280. Matrix 5280x5280 bools.
    
    pool = []
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            pool.append(v)
            
    pool = np.array(pool, dtype=np.int64)
    n_pool = len(pool)
    
    # Conflict: dot > 2
    dots = pool @ pool.T
    conflict_mask = dots > 2
    
    # We want a large independent set.
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.99
    max_iters = 50000 # Adjust based on time
    
    best_set = np.array([], dtype=np.int64).reshape(0, D)
    best_len = 0
    
    # Initial solution: Random greedy
    def random_greedy():
        active = np.ones(n_pool, dtype=bool)
        selected = []
        while True:
            if time.time() > deadline - 5:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            # Pick random from active
            v = np.random.choice(active_indices)
            selected.append(v)
            # Remove v and neighbors
            active[v] = False
            neighbors = np.where(conflict_mask[v] & active)[0]
            active[neighbors] = False
        return selected
    
    current_set = random_greedy()
    current_len = len(current_set)
    
    if current_len > best_len:
        best_len = current_len
        best_set = pool[current_set]
        
    # Maintain a boolean array for selected status for fast lookup
    selected_bool = np.zeros(n_pool, dtype=bool)
    for v in current_set:
        selected_bool[v] = True
        
    # Count conflicts for each selected vertex? 
    # If we add v, we need to check if it conflicts with ANY selected vertex.
    # conflict_mask[v] & selected_bool -> if any True, it's invalid.
    
    T = T_start
    iters = 0
    while T > T_end and iters < max_iters:
        if time.time() > deadline - 5:
            break
            
        iters += 1
        move_type = np.random.choice([0, 1, 2], p=[0.3, 0.2, 0.5]) # Add, Remove, Swap
        
        if move_type == 0: # Add
            # Pick a random unselected vertex
            unselected = np.where(~selected_bool)[0]
            if len(unselected) == 0:
                continue
            v = np.random.choice(unselected)
            # Check validity
            if not np.any(conflict_mask[v] & selected_bool):
                # Valid move, always accept
                selected_bool[v] = True
                current_len += 1
                if current_len > best_len:
                    best_len = current_len
                    best_set = pool[np.where(selected_bool)[0]]
            else:
                # Invalid move. Accept with probability exp(-delta/T)
                # Delta = -1 (size decrease? No, size stays same if we don't add, but we "want" to add)
                # In SA for maximization, we accept worse solutions (smaller sets) to escape local optima.
                # Here, adding an invalid vector is "worse" in terms of validity, but we can't have invalid sets.
                # So "Add" move is only valid if it doesn't conflict. If it conflicts, we can't do it directly.
                # We must use "Swap" to replace a conflicting vector.
                pass # Skip invalid adds, rely on swaps
        
        elif move_type == 1: # Remove
            if current_len == 0:
                continue
            # Pick a random selected vertex
            selected_indices = np.where(selected_bool)[0]
            v = np.random.choice(selected_indices)
            # Remove it. This is always valid (reduces size).
            # Accept with probability 1? No, this makes the set smaller. 
            # We accept it to allow future swaps.
            selected_bool[v] = False
            current_len -= 1
            
        else: # Swap
            if current_len == 0:
                continue
            # Pick a selected vertex to remove
            selected_indices = np.where(selected_bool)[0]
            v_old = np.random.choice(selected_indices)
            # Pick an unselected vertex to add
            unselected = np.where(~selected_bool)[0]
            if len(unselected) == 0:
                continue
            v_new = np.random.choice(unselected)
            
            # Check if v_new conflicts with any selected vertex EXCEPT v_old
            # We can temporarily remove v_old from the check mask
            # Or check: (conflict_mask[v_new] & selected_bool).sum() 
            # If the only conflict is v_old, then after removing v_old, it's valid.
            conflicts_with_new = np.where(conflict_mask[v_new] & selected_bool)[0]
            
            # Valid swap if the only conflict is v_old (or no conflicts)
            if len(conflicts_with_new) == 0 or (len(conflicts_with_new) == 1 and conflicts_with_new[0] == v_old):
                # Perform swap
                selected_bool[v_old] = False
                selected_bool[v_new] = True
                # Size remains same. 
                # Do we accept? Yes, to explore the space.
                # We might want to bias towards swaps that "unlock" future additions, 
                # but simple random walk is a start.
            else:
                # Invalid swap. We could accept it as a "penalty" state, but our state space 
                # is defined by valid sets. So we reject.
                pass
        
        T *= cooling_rate
        
    return best_set

# EVOLVE-BLOCK-START
# The code above is the implementation. 
# Note: The SA loop might be slow if `np.where` is called frequently on large arrays.
# Optimization: Keep a list of selected indices and a set of unselected?
# For 5280 elements, `np.where` on a bool array is ~microseconds. 
# 50k iterations * few `np.where` calls should be fast enough.
# Let's add a restart mechanism if SA stagnates.
# EVOLVE-BLOCK-END

# Final check: The code above is inside the construct function? 
# I need to wrap it properly.

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    D = 11
    indices = range(D)
    pool = []
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            pool.append(v)
            
    pool = np.array(pool, dtype=np.int64)
    n_pool = len(pool)
    
    dots = pool @ pool.T
    conflict_mask = dots > 2
    
    best_set = np.array([], dtype=np.int64).reshape(0, D)
    best_len = 0
    
    # Multiple restarts of SA
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initial solution: Random greedy
        active = np.ones(n_pool, dtype=bool)
        current_set = []
        while True:
            if time.time() > deadline - 5:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            v = np.random.choice(active_indices)
            current_set.append(v)
            active[v] = False
            neighbors = np.where(conflict_mask[v] & active)[0]
            active[neighbors] = False
        
        current_len = len(current_set)
        if current_len > best_len:
            best_len = current_len
            best_set = pool[current_set]
            
        selected_bool = np.zeros(n_pool, dtype=bool)
        for v in current_set:
            selected_bool[v] = True
            
        T = 1.0
        T_end = 0.01
        cooling_rate = 0.99
        max_iters = 20000
        
        iters = 0
        while T > T_end and iters < max_iters:
            if time.time() > deadline - 5:
                break
            iters += 1
            
            move_type = np.random.choice([0, 1, 2], p=[0.2, 0.2, 0.6]) # More swaps
            
            if move_type == 0: # Add
                unselected = np.where(~selected_bool)[0]
                if len(unselected) == 0:
                    continue
                v = np.random.choice(unselected)
                if not np.any(conflict_mask[v] & selected_bool):
                    selected_bool[v] = True
                    current_len += 1
                    if current_len > best_len:
                        best_len = current_len
                        best_set = pool[np.where(selected_bool)[0]]
                        
            elif move_type == 1: # Remove
                if current_len == 0:
                    continue
                selected_indices = np.where(selected_bool)[0]
                v = np.random.choice(selected_indices)
                selected_bool[v] = False
                current_len -= 1
                
            else: # Swap
                if current_len == 0:
                    continue
                selected_indices = np.where(selected_bool)[0]
                v_old = np.random.choice(selected_indices)
                unselected = np.where(~selected_bool)[0]
                if len(unselected) == 0:
                    continue
                v_new = np.random.choice(unselected)
                
                conflicts_with_new = np.where(conflict_mask[v_new] & selected_bool)[0]
                if len(conflicts_with_new) == 0 or (len(conflicts_with_new) == 1 and conflicts_with_new[0] == v_old):
                    selected_bool[v_old] = False
                    selected_bool[v_new] = True
                    
            T *= cooling_rate
            
    return best_set
