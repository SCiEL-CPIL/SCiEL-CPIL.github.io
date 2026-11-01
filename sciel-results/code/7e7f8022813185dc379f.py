"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix for W4
    # Conflict if u.v > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Compute degrees
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    rng = np.random.default_rng(42)
    
    def greedy_mis(conflicts, deadline, rng, degrees):
        """
        Greedy Maximum Independent Set with degree-based ordering.
        """
        N = len(conflicts)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        selected_indices = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
                
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
                
            # Find min degree among alive
            min_deg = np.min(deg[alive_indices])
            candidates = alive_indices[deg[alive_indices] == min_deg]
            
            # Random tie-breaking
            chosen = candidates[rng.integers(len(candidates))]
            selected_indices.append(chosen)
            
            # Find neighbors to remove
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            
            # Update alive
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            remaining_alive = np.where(alive)[0]
            if len(remaining_alive) > 0:
                removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
                deg[remaining_alive] -= removed_conflicts

        return selected_indices
    
    def compute_conflict_count(conflicts, selected_mask, exclude_node=None):
        """For each node not in selected, count how many selected nodes it conflicts with."""
        N = len(conflicts)
        # Number of conflicts with selected set for each node
        conflict_counts = np.zeros(N, dtype=np.int32)
        selected_indices = np.where(selected_mask)[0]
        if len(selected_indices) > 0:
            conflict_counts = conflicts[:, selected_indices].sum(axis=1)
        if exclude_node is not None:
            # Subtract the conflict with exclude_node if it was in selected
            if selected_mask[exclude_node]:
                conflict_counts -= conflicts[exclude_node]
        return conflict_counts
    
    def local_search_mis(conflicts, deadline, rng, initial_set):
        """
        Local Search for Maximum Independent Set with 1-swap, 2-for-1, and 2-for-2 moves.
        """
        N = len(conflicts)
        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = set(initial_set)
        best_size = current_size
        
        current_mask = np.zeros(N, dtype=bool)
        for v in current_set:
            current_mask[v] = True
            
        # Precompute conflict lists
        conflict_lists = [np.where(conflicts[i])[0] for i in range(N)]
        
        no_improve_count = 0
        max_no_improve = 10000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            
            # Try to add a vertex (if possible)
            if current_size < N:
                # Find vertices that conflict with 0 selected vertices
                conflict_counts = compute_conflict_count(conflicts, current_mask)
                addable = np.where((~current_mask) & (conflict_counts == 0))[0]
                if len(addable) > 0:
                    v_add = addable[rng.integers(len(addable))]
                    current_set.add(v_add)
                    current_mask[v_add] = True
                    current_size += 1
                    if current_size > best_size:
                        best_set = set(current_set)
                        best_size = current_size
                    no_improve_count = 0
                    improved = True
                    continue
            
            # Try 2-for-1: remove one vertex, add two
            if current_size >= 1:
                v_remove = list(current_set)[rng.integers(len(current_set))]
                
                # After removing v_remove, which vertices can be added?
                # They must not conflict with any other vertex in current_set - {v_remove}
                temp_mask = current_mask.copy()
                temp_mask[v_remove] = False
                
                conflict_counts = compute_conflict_count(conflicts, temp_mask)
                addable = np.where((~temp_mask) & (conflict_counts == 0))[0]
                
                if len(addable) >= 2:
                    # Try all pairs from addable (limited)
                    n_addable = min(len(addable), 20)
                    sample_idx = rng.choice(len(addable), size=n_addable, replace=False)
                    addable_sample = addable[sample_idx]
                    
                    found_pair = False
                    for i in range(n_addable):
                        if found_pair:
                            break
                        for j in range(i + 1, n_addable):
                            a, b = addable_sample[i], addable_sample[j]
                            if not conflicts[a, b]:
                                # Valid 2-for-1 move
                                current_set.remove(v_remove)
                                current_set.add(a)
                                current_set.add(b)
                                current_mask[v_remove] = False
                                current_mask[a] = True
                                current_mask[b] = True
                                current_size += 1
                                if current_size > best_size:
                                    best_set = set(current_set)
                                    best_size = current_size
                                no_improve_count = 0
                                improved = True
                                found_pair = True
                                break
            
            # Try 1-swap: remove one, add one (for exploration)
            if not improved and current_size > 0 and rng.random() < 0.3:
                v_remove = list(current_set)[rng.integers(len(current_set))]
                
                temp_mask = current_mask.copy()
                temp_mask[v_remove] = False
                
                conflict_counts = compute_conflict_count(conflicts, temp_mask)
                addable = np.where((~temp_mask) & (conflict_counts == 0))[0]
                
                if len(addable) > 0:
                    v_add = addable[rng.integers(len(addable))]
                    current_set.remove(v_remove)
                    current_set.add(v_add)
                    current_mask[v_remove] = False
                    current_mask[v_add] = True
                    no_improve_count += 1
                    improved = True
            
            if not improved:
                no_improve_count += 1
        
        return list(best_set)
    
    def branch_and_bound_mis(conflicts, deadline, rng, degrees, current_set, remaining_mask):
        """
        Branch and bound for Maximum Independent Set.
        Uses greedy upper bound based on remaining vertices.
        """
        N = len(conflicts)
        current_size = len(current_set)
        
        if time.time() > deadline:
            return list(current_set)
        
        # If no remaining vertices, return current
        if not np.any(remaining_mask):
            return list(current_set)
        
        # Upper bound: current_size + number of remaining vertices
        # (crude but fast)
        remaining_count = np.sum(remaining_mask)
        
        # Try to greedily extend
        alive = remaining_mask.copy()
        deg = degrees.copy()
        
        # Greedy extension
        greedy_extension = []
        temp_alive = alive.copy()
        temp_deg = deg.copy()
        
        while np.any(temp_alive):
            if time.time() > deadline:
                break
            alive_indices = np.where(temp_alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(temp_deg[alive_indices])
            candidates = alive_indices[temp_deg[alive_indices] == min_deg]
            chosen = candidates[rng.integers(len(candidates))]
            greedy_extension.append(chosen)
            
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            temp_alive[to_remove] = False
            
            remaining_alive = np.where(temp_alive)[0]
            if len(remaining_alive) > 0:
                removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
                temp_deg[remaining_alive] -= removed_conflicts
        
        # The greedy extension gives a lower bound on what we can achieve
        # If current_size + len(greedy_extension) is not better than best, prune
        # For now, just return the greedy extension as a heuristic
        
        return list(current_set) + greedy_extension
    
    # Run multiple greedy restarts to get good initial solutions
    best_greedy = []
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() > deadline - 40:
            break
        # Vary the greedy strategy
        if i % 3 == 0:
            greedy_result = greedy_mis(conflicts, min(deadline - 40, time.time() + 8), rng, degrees)
        elif i % 3 == 1:
            # Max degree greedy
            alive = np.ones(N4, dtype=bool)
            deg = degrees.copy()
            selected = []
            while np.any(alive):
                if time.time() > deadline - 40:
                    break
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                max_deg = np.max(deg[alive_indices])
                candidates = alive_indices[deg[alive_indices] == max_deg]
                chosen = candidates[rng.integers(len(candidates))]
                selected.append(chosen)
                neighbors = np.where(conflicts[chosen])[0]
                to_remove = np.unique(np.concatenate(([chosen], neighbors)))
                alive[to_remove] = False
                remaining_alive = np.where(alive)[0]
                if len(remaining_alive) > 0:
                    removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
                    deg[remaining_alive] -= removed_conflicts
            greedy_result = selected
        else:
            # Random greedy
            alive = np.ones(N4, dtype=bool)
            selected = []
            while np.any(alive):
                if time.time() > deadline - 40:
                    break
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                chosen = alive_indices[rng.integers(len(alive_indices))]
                selected.append(chosen)
                neighbors = np.where(conflicts[chosen])[0]
                to_remove = np.unique(np.concatenate(([chosen], neighbors)))
                alive[to_remove] = False
            greedy_result = selected
        
        if len(greedy_result) > len(best_greedy):
            best_greedy = greedy_result
    
    # Use local search to improve
    if best_greedy:
        ls_deadline = deadline - 5
        ls_result = local_search_mis(conflicts, ls_deadline, rng, best_greedy)
        if len(ls_result) > len(best_greedy):
            best_greedy = ls_result
    
    if not best_greedy:
        return vectors_v1
    
    result_v4 = vectors_v4[best_greedy]
    
    # Combine with W1
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
