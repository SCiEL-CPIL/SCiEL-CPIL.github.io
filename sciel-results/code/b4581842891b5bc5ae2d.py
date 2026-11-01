"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

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
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
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
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
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
    
    def local_search(in_set_init, max_time=10.0):
        """Local search with add, swap, and 2-for-1 moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        ls_deadline = time.time() + max_time
        
        T = 2.0
        cooling = 0.999
        min_T = 0.05
        stuck_counter = 0
        
        while time.time() < ls_deadline:
            in_set_indices = np.where(in_set)[0]
            not_in_set_indices = np.where(~in_set)[0]
            
            if len(not_in_set_indices) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.5:
                # ADD move: add u, remove all its conflicts in in_set
                # Choose u with minimum conflict count
                counts = count_conflicts[not_in_set_indices]
                min_count = np.min(counts)
                min_cands = not_in_set_indices[counts == min_count]
                u = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(conflicts[u] & in_set)[0]
                    # Update count_conflicts for removed vertices
                    for w in to_remove:
                        in_set[w] = False
                        for x in neighbors[w]:
                            count_conflicts[x] -= 1
                    # Add u
                    in_set[u] = True
                    for x in neighbors[u]:
                        count_conflicts[x] += 1
                    current_size += delta
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.8 and len(in_set_indices) > 0:
                # SWAP move: remove v from in_set, add u from not_in_set
                # Only if u's conflicts in in_set are subset of {v}
                v = in_set_indices[rng.integers(len(in_set_indices))]
                u = not_in_set_indices[rng.integers(len(not_in_set_indices))]
                
                # Check if u conflicts with any in_set vertex other than v
                u_conflicts_in_set = np.where(conflicts[u] & in_set)[0]
                other_conflicts = u_conflicts_in_set[u_conflicts_in_set != v]
                
                if len(other_conflicts) == 0:
                    # Safe swap
                    in_set[v] = False
                    in_set[u] = True
                    for w in neighbors[v]:
                        count_conflicts[w] -= 1
                    for w in neighbors[u]:
                        count_conflicts[w] += 1
                    # current_size unchanged
                    stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            else:
                # 2-for-1 move: remove v1, v2 from in_set, add u from not_in_set
                # Net change: -2 + 1 = -1 (worse), but may enable future additions
                # Only do this with low probability and at higher T
                if len(in_set_indices) >= 2 and T > 0.3:
                    v1 = in_set_indices[rng.integers(len(in_set_indices))]
                    remaining = np.delete(in_set_indices, np.where(in_set_indices == v1)[0][0])
                    v2 = remaining[rng.integers(len(remaining))]
                    u = not_in_set_indices[rng.integers(len(not_in_set_indices))]
                    
                    # Check if u conflicts with in_set \ {v1, v2}
                    in_set_copy = in_set.copy()
                    in_set_copy[v1] = False
                    in_set_copy[v2] = False
                    u_conflicts_remaining = np.where(conflicts[u] & in_set_copy)[0]
                    
                    if len(u_conflicts_remaining) == 0:
                        # Safe 2-for-1
                        in_set[v1] = False
                        in_set[v2] = False
                        in_set[u] = True
                        for w in neighbors[v1]:
                            count_conflicts[w] -= 1
                        for w in neighbors[v2]:
                            count_conflicts[w] -= 1
                        for w in neighbors[u]:
                            count_conflicts[w] += 1
                        current_size -= 1
                        
                        # Accept with probability based on T
                        if rng.random() < np.exp(-1.0 / T):
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()
                                stuck_counter = 0
                            else:
                                stuck_counter += 1
                        else:
                            # Revert
                            in_set[u] = False
                            in_set[v1] = True
                            in_set[v2] = True
                            for w in neighbors[u]:
                                count_conflicts[w] -= 1
                            for w in neighbors[v1]:
                                count_conflicts[w] += 1
                            for w in neighbors[v2]:
                                count_conflicts[w] += 1
                            current_size += 1
                            stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if stuck_counter > 500:
                # Perturbation: randomly remove 3-5 vertices and try to re-add
                n_remove = rng.integers(3, 6)
                if len(in_set_indices) > n_remove:
                    to_remove = rng.choice(in_set_indices, size=n_remove, replace=False)
                    for w in to_remove:
                        in_set[w] = False
                        for x in neighbors[w]:
                            count_conflicts[x] -= 1
                    current_size -= n_remove
                    stuck_counter = 0
                    T = max(T, 1.5)
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts with varied seeds
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Vary the random seed for diversity
        rng = np.random.default_rng(n_restarts * 7919 + 1)
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        ls_time = min(remaining_time * 0.5, 8.0)
        if ls_time < 2:
            ls_time = 2.0
            
        ls_result, ls_size = local_search(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
    
    # Final polish: try all single-vertex additions and simple swaps
    polish_deadline = deadline - 2
    in_set = best_in_set.copy()
    count_conflicts = np.zeros(N4, dtype=np.int32)
    for i in np.where(in_set)[0]:
        count_conflicts[neighbors[i]] += 1
    
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        not_in_set = np.where(~in_set)[0]
        for v in not_in_set:
            if time.time() > polish_deadline:
                break
            k = count_conflicts[v]
            if k == 0:
                # Can add v for free
                in_set[v] = True
                for w in neighbors[v]:
                    count_conflicts[w] += 1
                improved = True
    
    if int(in_set.sum()) > best_size:
        best_size = int(in_set.sum())
        best_in_set = in_set
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
