"""Rewritten program for Kissing Number in dimension 11."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # Compute conflict matrix: conflict if dot > 2
    # Using matrix multiplication for speed
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to neighbor lists for faster local search updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_in_set = np.zeros(N4, dtype=bool)
    best_size = 0

    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def greedy_fill(in_set, counts):
        """
        Add all possible nodes to in_set, updating counts.
        Returns number of nodes added.
        """
        added = 0
        candidates = np.where((counts == 0) & (~in_set))[0]
        
        for v in candidates:
            if counts[v] == 0 and not in_set[v]:
                in_set[v] = True
                added += 1
                for u in neighbors[v]:
                    counts[u] += 1
        return added

    def local_search(in_set):
        """
        Simulated Annealing based local search with 1-swap and 2-swap moves.
        """
        in_set = in_set.copy()
        size = int(in_set.sum())
        
        best_in_set = in_set.copy()
        best_size = size
        
        # Precompute conflict counts for each node
        conflict_counts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_counts[neighbors[i]] += 1
        
        T = 1.0
        cooling = 0.9998
        rng = np.random.default_rng(42)
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 10 and T > 0.01 and no_improve_count < max_no_improve:
            move_type = rng.integers(3)  # 0: add, 1: remove, 2: swap
            
            if move_type == 0:  # Add a node
                # Find nodes with 0 conflicts that are not in set
                candidates = np.where((conflict_counts == 0) & (~in_set))[0]
                if len(candidates) == 0:
                    no_improve_count += 1
                    continue
                v = candidates[rng.integers(len(candidates))]
                if not in_set[v] and conflict_counts[v] == 0:
                    in_set[v] = True
                    size += 1
                    conflict_counts[neighbors[v]] += 1
                    no_improve_count = 0
                    if size > best_size:
                        best_size = size
                        best_in_set = in_set.copy()
                        
            elif move_type == 1:  # Remove a node
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    no_improve_count += 1
                    continue
                v = in_set_indices[rng.integers(len(in_set_indices))]
                if in_set[v]:
                    # Check if removing helps (allows adding more neighbors)
                    freed = np.where((conflict_counts[neighbors[v]] == 1) & (~in_set[neighbors[v]]))[0]
                    if len(freed) > 0:
                        in_set[v] = False
                        size -= 1
                        conflict_counts[neighbors[v]] -= 1
                        # Try to add freed nodes
                        for u in freed:
                            if conflict_counts[u] == 0 and not in_set[u]:
                                in_set[u] = True
                                size += 1
                                conflict_counts[neighbors[u]] += 1
                        no_improve_count = 0
                        if size > best_size:
                            best_size = size
                            best_in_set = in_set.copy()
                    else:
                        no_improve_count += 1
                        
            else:  # 2-swap: remove one, add another
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    no_improve_count += 1
                    continue
                v = in_set_indices[rng.integers(len(in_set_indices))]
                # Find a node u not in set that conflicts with v but would be compatible after removal
                # Simple heuristic: pick random non-in-set node
                out_set_indices = np.where(~in_set)[0]
                if len(out_set_indices) == 0:
                    no_improve_count += 1
                    continue
                u = out_set_indices[rng.integers(len(out_set_indices))]
                
                # Check if swap is valid: u must not conflict with any in_set node except possibly v
                u_conflicts_in_set = np.count_nonzero(in_set[neighbors[u]])
                # After removing v, u's conflicts would be u_conflicts_in_set - (1 if v in neighbors[u] else 0)
                v_is_neighbor = v in neighbors[u]
                new_conflicts = u_conflicts_in_set - (1 if v_is_neighbor else 0)
                
                if new_conflicts == 0:
                    # Valid swap
                    in_set[v] = False
                    conflict_counts[neighbors[v]] -= 1
                    in_set[u] = True
                    conflict_counts[neighbors[u]] += 1
                    no_improve_count = 0
                    if size > best_size:
                        best_size = size
                        best_in_set = in_set.copy()
                else:
                    no_improve_count += 1
            
            T *= cooling
        
        # Final Polish: Greedy fill
        conflict_counts = np.zeros(N4, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_counts[neighbors[i]] += 1
            
        candidates = np.where((conflict_counts == 0) & (~best_in_set))[0]
        for v in candidates:
            if conflict_counts[v] == 0 and not best_in_set[v]:
                best_in_set[v] = True
                for u in neighbors[v]:
                    conflict_counts[u] += 1
        best_size = int(best_in_set.sum())

        return best_in_set

    # Main Loop: Multiple Restarts
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        in_set = local_search(in_set)
        
        current_size = int(in_set.sum())
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 3. Optional: Try a random perturbation of the best set found so far
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            # Remove 3-4 random elements for more aggressive perturbation
            indices = np.where(perturbed)[0]
            if len(indices) > 4:
                n_remove = rng.integers(3, 5)
                to_remove = rng.choice(indices, size=n_remove, replace=False)
                perturbed[to_remove] = False
            elif len(indices) > 2:
                to_remove = rng.choice(indices, size=2, replace=False)
                perturbed[to_remove] = False
            perturbed = local_search(perturbed)
            current_size = int(perturbed.sum())
            if current_size > best_size:
                best_size = current_size
                best_in_set = perturbed.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Add W1 vectors (norm 2, weight 1)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
