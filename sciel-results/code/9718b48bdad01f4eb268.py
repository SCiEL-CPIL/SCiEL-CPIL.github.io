"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Generate candidate pool: All weight-4 vectors in R^11
    # Number of candidates: C(11, 4) * 2^4 = 330 * 16 = 5280
    # Norm^2 = 4 for all.
    # Condition for valid kissing config: min pairwise dist >= max norm.
    # Since all norms are 4, we need dist^2 >= 4.
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v = 8 - 2 u.v.
    # 8 - 2 u.v >= 4  =>  4 >= 2 u.v  =>  u.v <= 2.
    # Max possible dot product for distinct weight-4 vectors is 4 (if identical).
    # If they share 3 indices with same signs, dot = 3. (Conflict)
    # If they share 3 indices, one sign flipped, dot = 1. (OK)
    # If they share 2 indices, dot = 2. (OK)
    # So we want to maximize the number of vectors such that no pair has dot product > 2.
    
    candidates = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            candidates.append(v)
            
    C = np.array(candidates, dtype=np.int64)
    N = len(C)
    
    # Precompute Gram matrix G where G[i,j] = C[i] . C[j]
    # N=5280, 5280x5280 matrix is ~27M elements. int64 is 8 bytes -> ~216MB. 
    # This might be tight on memory but usually acceptable. 
    # To be safer, we can compute in chunks or use int32 if possible.
    # Dot products range from -4 to 4. int8 is enough!
    # Let's use int8 for the Gram matrix to save memory and potentially speed up cache.
    
    # Compute Gram matrix
    # C is int64. C @ C.T will be int64.
    # We can cast to int8 after? 
    # Actually, let's just use standard float32 or int32 for speed, or stick to int64 if memory allows.
    # 5280^2 * 8 bytes = 223 MB. Should be fine for most evaluators.
    G = C @ C.T
    G = G.astype(np.int8) # Save memory
    
    # Conflict mask: G[i,j] > 2
    # Note: G[i,i] = 4 > 2. We don't care about self-conflicts for the mask logic if we handle indices carefully, 
    # but for the "forbidden" sets, we need to know which OTHER vectors conflict with i.
    
    # 2. Initialize a valid subset using Greedy Min-Degree
    # We want to start with a large independent set in the conflict graph.
    # Heuristic: Pick vertex with lowest degree in the remaining graph? 
    # Or just greedy pick?
    # Let's do a simple greedy: 
    # 1. Compute degrees in the full conflict graph.
    # 2. Pick vertex with min degree, add to set, remove it and its neighbors from pool.
    # 3. Repeat.
    
    # Degree of i = number of j != i such that G[i,j] > 2.
    # Let's compute degrees.
    # conflict_mask = G > 2
    # deg = np.sum(conflict_mask, axis=1) - 1 (exclude self)
    
    # This is O(N^2). 27M ops. Fast.
    conflict_mask = G > 2
    # Self-conflicts are True. Subtract 1.
    deg = np.sum(conflict_mask, axis=1) - 1
    
    # Greedy selection
    available = np.ones(N, dtype=bool)
    selected = []
    
    # To speed up greedy, we can just pick the first available with min degree?
    # Or just iterate.
    # Since N=5280, even O(N^2) greedy is ~27M ops, which is fast in Python if vectorized, 
    # but the loop structure is tricky.
    # Let's use a simple loop.
    
    # Actually, a better initial solution for Kissing Number is often found by random greedy or simulated annealing.
    # Let's use a "First Fit Decreasing" style or just random restarts of greedy.
    
    # Let's try a simple greedy:
    # Sort vertices by degree (ascending).
    # Iterate through sorted list. If vertex is still available and doesn't conflict with already selected, add it.
    
    order = np.argsort(deg)
    
    # We need to check conflicts with already selected.
    # selected_set = set()
    # for i in order:
    #    if i not in selected_set:
    #       check if any j in selected_set has G[i,j] > 2.
    #       if not, add i.
    
    # This is O(N * K) where K is final size. K is likely ~200-300.
    # 5280 * 300 = 1.5M ops. Very fast.
    
    selected = []
    selected_set = set()
    
    for i in order:
        if i in selected_set:
            continue
        # Check conflict with selected
        conflict = False
        for j in selected:
            if G[i, j] > 2:
                conflict = True
                break
        if not conflict:
            selected.append(i)
            selected_set.add(i)
            
    best_size = len(selected)
    best_solution = selected.copy()
    
    # 3. Local Search (Simulated Annealing / Hill Climbing)
    # Objective: Maximize size of independent set.
    # Moves:
    # A. Add a vertex v from (All - Selected) if it doesn't conflict with any in Selected.
    # B. Remove a vertex u from Selected.
    # C. Swap: Remove u, Add v.
    
    # To make it efficient, we maintain a "blocked" set for each selected vertex?
    # Or just check on the fly.
    # N=5280. Selected size ~ 200.
    # Checking if v can be added: Check G[v, selected] <= 2 for all selected.
    # This is O(|Selected|).
    
    # Let's use a time-budgeted random restart local search.
    # We will perform many small steps.
    
    current_selected = best_solution.copy()
    current_set = set(current_selected)
    current_size = len(current_selected)
    
    # Precompute list of all indices
    all_indices = list(range(N))
    
    # Helper to check if vertex v can be added to a set S (represented as list/set)
    # Returns True if no conflict.
    def can_add(v, S_list, G_matrix):
        for u in S_list:
            if G_matrix[v, u] > 2:
                return False
        return True
        
    # Helper to get conflicts of v within S_list
    def get_conflicts(v, S_list, G_matrix):
        conflicts = []
        for u in S_list:
            if G_matrix[v, u] > 2:
                conflicts.append(u)
        return conflicts

    # We will run a loop until deadline.
    # Each iteration: try a random move.
    
    # To speed up, we can precompute the "neighborhood" of each vertex in the conflict graph?
    # Conflict graph is sparse? 
    # Each vertex conflicts with those sharing 3 indices.
    # For a specific vector v (indices i,j,k,l), how many vectors share 3 indices?
    # Fix 3 indices, vary the 4th.
    # There are 4 choices for which index is dropped.
    # For each dropped index, the remaining 3 are fixed. The 4th index can be any of the other 7 dimensions.
    # And the signs...
    # It's better to just use the G matrix lookups.
    
    # Optimization: Maintain a "degree" array for the induced subgraph?
    # Too complex. Let's rely on the small size of Selected (~200-300).
    
    # Random Restart Logic:
    # If we get stuck, restart from a random subset of current best?
    # Or just keep swapping.
    
    # Let's use a simple hill climbing with random restarts if no improvement for X steps.
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart: Keep top 50% of current selected (randomly) and re-greedy?
            # Or just reset to best_solution and randomize a bit?
            # Let's try: Remove 10 random elements from current, then re-add greedily.
            if len(current_selected) > 10:
                remove_indices = random.sample(range(len(current_selected)), min(10, len(current_selected)))
                for idx in remove_indices:
                    v = current_selected[idx]
                    current_selected.remove(v)
                    current_set.remove(v)
                current_size = len(current_selected)
                # Re-greedy add
                # Sort available by degree?
                # Just try to add back random ones?
                # Let's just reset to best and break out to restart loop?
                # Simpler: Reset to best_solution.
                current_selected = best_solution.copy()
                current_set = set(current_selected)
                current_size = len(current_selected)
                no_improve_count = 0
                max_no_improve = 5000
            else:
                break
                
        # Random Move
        # 1. Try to Add
        # 2. Try to Swap (Remove one, Add one)
        # 3. Try to Remove (rarely useful for maximization, but helps escape local optima if combined with Add)
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Try Add
            # Pick random candidate from all not in current_set
            # To avoid picking from a huge list, we can maintain a list of "available" candidates?
            # Or just pick random index and check if it's in set.
            # Picking random index from 0..N-1.
            # If it's in set, skip.
            # If not, check conflict.
            
            v = random.randint(0, N - 1)
            if v in current_set:
                continue
                
            if can_add(v, current_selected, G):
                # Accept
                current_selected.append(v)
                current_set.add(v)
                current_size += 1
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_solution = current_selected.copy()
            else:
                no_improve_count += 1
                
        elif move_type < 0.9:
            # Try Swap
            # Pick u to remove, v to add.
            if not current_selected:
                continue
            u = random.choice(current_selected)
            v = random.randint(0, N - 1)
            if v in current_set:
                continue
                
            # Check if v can be added if u is removed.
            # v must not conflict with (current_selected - {u}).
            # i.e., for all w in current_selected, if w != u, G[v,w] <= 2.
            
            valid = True
            for w in current_selected:
                if w == u:
                    continue
                if G[v, w] > 2:
                    valid = False
                    break
                    
            if valid:
                # Execute swap
                current_selected.remove(u)
                current_set.remove(u)
                current_selected.append(v)
                current_set.add(v)
                # Size unchanged, but structure changed.
                # We don't update best_size here as size is same.
                # But we reset no_improve? 
                # Swaps are neutral. Let's count them as "activity" but not "improvement".
                # To prevent infinite loops of neutral swaps, we can have a separate counter.
                # For simplicity, we don't reset no_improve_count on swap, 
                # but we do reset it if we find an Add later.
                # Actually, if we get stuck in a local optimum where no Add works, 
                # Swaps might allow a different configuration where an Add works.
                # So we should NOT increment no_improve_count on a valid swap.
                # But we also shouldn't reset it fully? 
                # Let's just reset it to 0 to give the search a chance.
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Try Remove (to escape deep local minima)
            if current_selected:
                u = random.choice(current_selected)
                current_selected.remove(u)
                current_set.remove(u)
                current_size -= 1
                no_improve_count += 1 # Removal is a step away from goal
                
    # Return best found
    result = C[best_solution]
    return result

# EVOLVE-BLOCK-END
