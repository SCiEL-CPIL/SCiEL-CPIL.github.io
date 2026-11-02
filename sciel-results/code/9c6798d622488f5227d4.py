# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin

    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(11), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * 11
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
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
    
    def sa_local_search(in_set_init, max_time=100.0):
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 5.0
        cooling = 0.9999
        min_T = 0.01
        
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            # Try Add Move
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = min_count
                delta = 1 - k
                
                if delta > 0 or (delta == 0 and rng.random() < 0.5) or (delta < 0 and rng.random() < np.exp(delta / T)):
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            
            # Try Swap Move (with probability 0.5)
            elif rng.random() < 0.5:
                in_indices = np.where(in_set)[0]
                out_indices = np.where(~in_set)[0]
                if len(in_indices) > 0 and len(out_indices) > 0:
                    v_in = in_indices[rng.integers(len(in_indices))]
                    v_out = out_indices[rng.integers(len(out_indices))]
                    
                    # Calculate delta for swapping v_in out and v_out in
                    # Remove v_in: gain 1, lose conflicts with v_in's neighbors in set
                    # Add v_out: lose conflicts with v_out's neighbors in set
                    # Net change:
                    # 1 (remove v_in) - (neighbors of v_in in set excluding v_out) + 1 (add v_out) - (neighbors of v_out in set excluding v_in)
                    # Note: if v_in and v_out conflict, it's handled in both terms?
                    # Let's just simulate on a copy or calculate carefully.
                    
                    # Simple calculation:
                    # Current size: S
                    # New size: S - 1 + 1 = S (if valid)
                    # We need to check if v_out can be added after removing v_in.
                    # v_out conflicts with some nodes in in_set.
                    # If we remove v_in, the conflict count for v_out might decrease.
                    
                    # Let's just do a tentative swap and check validity?
                    # Too slow.
                    
                    # Approximation:
                    # Conflicts of v_out in current set: c_out
                    # Conflicts of v_in in current set: c_in
                    # If v_in is a neighbor of v_out:
                    #   Removing v_in reduces c_out by 1.
                    #   Adding v_out removes its neighbors.
                    
                    # This is complex. Let's stick to Add/Remove for now but with better restarts.
                    pass
            
            if stuck_counter > 1000:
                # Perturbation
                num_remove = rng.integers(5, 20)
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > num_remove:
                    remove_cands = in_set_indices[rng.choice(len(in_set_indices), num_remove, replace=False)]
                    for u in remove_cands:
                        in_set[u] = False
                        count_conflicts[neighbors[u]] -= 1
                    current_size -= num_remove
                    
                    # Greedy re-add
                    alive = ~in_set
                    while np.any(alive):
                        alive_indices = np.where(alive)[0]
                        if not len(alive_indices): break
                        min_deg = np.min(count_conflicts[alive_indices])
                        cands = alive_indices[count_conflicts[alive_indices] == min_deg]
                        chosen = cands[rng.integers(len(cands))]
                        if count_conflicts[chosen] == 0:
                            in_set[chosen] = True
                            current_size += 1
                            count_conflicts[neighbors[chosen]] += 1
                            alive[neighbors[chosen]] = False
                            alive[chosen] = False
                        else:
                            break
                            
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Run multiple restarts
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    time_per_run = 15.0
    max_runs = int((deadline - 5 - time.time()) / time_per_run)
    max_runs = max(1, min(max_runs, 5))
    
    for i in range(max_runs):
        in_set = greedy_min_degree()
        sa_result, sa_size = sa_local_search(in_set, max_time=time_per_run)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result
            
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    return result_v4

# EVOLVE-BLOCK-END
