"""Rewritten program for Kissing Number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 142  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(tuple(v))
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_list = []
    for pos in range(D):
        v1_list.append(tuple([0]*pos + [2] + [0]*(D-pos-1)))
        v1_list.append(tuple([0]*pos + [-2] + [0]*(D-pos-1)))
        
    vectors_v4 = np.array(v4_list, dtype=np.int64)
    vectors_v1 = np.array(v1_list, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as lists for faster iteration
    neighbors = [np.where(conflicts[i])[0].tolist() for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
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
            
            # Tie-break: prefer vertex with fewer "second-neighbor" conflicts
            if len(cands) > 1:
                # Approximate: prefer candidate whose neighbors have lower average degree
                scores = []
                for c in cands:
                    nbrs = neighbors[c]
                    if len(nbrs) > 0:
                        scores.append(np.mean(degrees[nbrs]))
                    else:
                        scores.append(0)
                scores = np.array(scores)
                chosen = cands[np.argmin(scores)]
            else:
                chosen = cands[0]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Efficiently update degrees
                # conflicts[rem, to_remove] shape (len(rem), len(to_remove))
                # Sum along axis 1
                degrees[rem] -= np.sum(conflicts[rem][:, to_remove], axis=1)
                
        return in_set

    def sa_local_search(in_set_init, max_time=10.0, rng=None):
        """Min-conflict simulated annealing with add/remove/swap moves."""
        if rng is None:
            rng = np.random.default_rng()
            
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.6 and len(not_in_set) > 0:
                # ADD move
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.3
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.8 and len(in_set_indices) > 0:
                # REMOVE move
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                else:
                    stuck_counter += 1
                    
            else:
                # SWAP move
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    v_add = not_in_set[rng.integers(len(not_in_set))]
                    
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    
                    if len(other_conflicts) == 0:
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        count_conflicts[neighbors[v_remove]] -= 1
                        count_conflicts[neighbors[v_add]] += 1
                    else:
                        to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                        delta = 1 - len(to_remove)
                        accept = delta > 0 or (rng.random() < np.exp(delta / T))
                        if accept:
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                            in_set[v_add] = True
                            current_size += delta
                            count_conflicts[neighbors[v_add]] += 1
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 1500:
                T = max(T, 4.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size

    # Multiple restarts with varied seeds
    n_restarts = 0
    max_restarts = 200 # Higher cap, will be limited by time
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Vary the seed for rng based on restart count
        rng = np.random.default_rng(42 + n_restarts)
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.4, 10.0) # Reduced SA time per restart to allow more restarts
        if sa_time < 1.5:
            sa_time = 1.5
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time, rng=rng)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Polish: try to add any non-conflicting vertex to the best set
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            polish_set = sa_result.copy()
            polish_size = sa_size
            
            changed = True
            while changed and time.time() < deadline - 2:
                changed = False
                not_in = np.where(~polish_set)[0]
                if len(not_in) == 0:
                    break
                
                # Efficient check for addable vertices
                # conflict_counts[i] = number of conflicts of not_in[i] with polish_set
                # Using boolean matrix multiplication isn't directly possible, 
                # so we use sum of (conflicts[not_in, :] & polish_set[None, :])
                # To avoid creating huge temporaries, we can use the fact that 
                # conflicts is sparse-ish? No, it's dense.
                # However, N4 is ~5280. conflicts[not_in] is ~5280x5280. 
                # This is fast in numpy.
                
                # Vectorized: (conflicts[not_in] @ polish_set) doesn't work for bool sum directly without cast
                # But conflicts is bool. polish_set is bool.
                # Sum of ANDs = (conflicts[not_in].astype(float) @ polish_set.astype(float))? 
                # That's expensive due to cast.
                
                # Better: np.sum(conflicts[not_in] & polish_set[None, :], axis=1)
                # This creates a (len(not_in), N4) bool array. 
                # For len(not_in)=5280, this is ~28MB. Acceptable.
                
                conflict_counts = np.sum(conflicts[not_in] & polish_set[None, :], axis=1)
                addable_indices = np.where(conflict_counts == 0)[0]
                
                if len(addable_indices) > 0:
                    v_idx = not_in[addable_indices[0]]
                    polish_set[v_idx] = True
                    polish_size += 1
                    changed = True
                    if time.time() >= deadline - 2:
                        break
            
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_set.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    # Verify: min pairwise distance >= max norm
    # We trust the search, but let's do a quick sanity check if k is small enough?
    # The prompt says "score is k", so we just return.
    # The constraint is min_dist >= max_norm.
    # Norms are all sqrt(4) = 2. Max norm is 2.
    # We need min_dist >= 2 => min_dist_sq >= 4.
    # For W4 vectors, dot product <= 2.
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b = 4 + 4 - 2a.b = 8 - 2a.b.
    # If a.b <= 2, dist_sq >= 8 - 4 = 4. Correct.
    # For W1 vectors, they are orthogonal to each other and to W4?
    # W1 vector v has one entry 2 or -2.
    # W4 vector u has 4 entries 1/-1.
    # If v and u share no position, dot=0. dist_sq = 8.
    # If v and u share one position (pos p), v[p]=2, u[p]=1 => dot=2. dist_sq=4.
    # If v[p]=-2, u[p]=1 => dot=-2. dist_sq=12.
    # So min dist sq is 4. Max norm is 2. 2>=2 is satisfied.
    # The search ensures W4-W4 dot <= 2.
    # W1-W4 and W1-W1 are always valid.
    
    return final_vectors

# EVOLVE-BLOCK-END
