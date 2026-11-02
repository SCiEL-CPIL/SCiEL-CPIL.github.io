"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

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

    # Group W4 vectors by their support (tuple of nonzero positions)
    # For a given support, all 16 sign variants are mutually compatible
    # (dot product between any two distinct sign variants on same support is 0)
    support_to_indices = {}
    for i in range(N4):
        support = tuple(np.nonzero(vectors_v4[i])[0])
        if support not in support_to_indices:
            support_to_indices[support] = []
        support_to_indices[support].append(i)
    
    supports = list(support_to_indices.keys())
    n_supports = len(supports)
    
    # Build support-level conflict matrix:
    # Two supports conflict if they share >= 3 positions
    # (if they share 3 positions, there exist sign variants with dot >= 3 > 2)
    # If they share <= 2 positions, all sign variants are compatible
    support_conflicts = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            if len(si & sj) >= 3:
                support_conflicts[i, j] = True
                support_conflicts[j, i] = True
    
    # Each support contributes 16 vectors if selected
    vectors_per_support = 16
    
    # Also need to check cross-compatibility with W1 vectors
    # W1 vector at position p with value ±2:
    # Compatible with W4 vector if W4 has 0 at position p (dot=0) or |dot|<=2
    # W4 has entries ±1 at 4 positions, so dot with W1 at p is ±1 if p in support, else 0
    # |±1| = 1 <= 2, so all W4 vectors are compatible with all W1 vectors!
    # Great, no need to filter.
    
    # Now solve independent set on support graph
    # This is a much smaller problem: ~330 nodes vs 3328
    
    neighbors_sup = [np.where(support_conflicts[i])[0] for i in range(n_supports)]
    neighbor_counts_sup = np.array([len(n) for n in neighbors_sup], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_support_set = np.zeros(n_supports, dtype=bool)
    
    def greedy_min_degree_supports():
        """Greedy independent set with min-degree heuristic on support graph."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def zero_conflict_polish(in_set, count_conflicts=None):
        """Deterministically add all vertices with zero conflicts, iteratively."""
        in_set = in_set.copy()
        if count_conflicts is None:
            count_conflicts = np.zeros(n_supports, dtype=np.int32)
            for i in np.where(in_set)[0]:
                count_conflicts[neighbors_sup[i]] += 1
        else:
            count_conflicts = count_conflicts.copy()
        
        changed = True
        while changed:
            changed = False
            # Find all vertices not in set with zero conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            zero_conf = not_in_set[count_conflicts[not_in_set] == 0]
            if len(zero_conf) == 0:
                break
            # Add all zero-conflict vertices at once
            for s in zero_conf:
                in_set[s] = True
                count_conflicts[neighbors_sup[s]] += 1
            changed = True
        
        return in_set, in_set.sum()
    
    def sa_local_search_supports(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing with insert, 1-for-1 swap, and 2-for-1 swap moves."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        # count_conflicts[s] = number of neighbors of s that are in in_set
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors_sup[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        # Precompute conflict masks as boolean arrays for fast operations
        # support_conflicts is already n_supports x n_supports boolean
        
        while time.time() < sa_deadline:
            # Choose move type: 50% insert, 25% 1-for-1 swap, 25% 2-for-1 swap
            move_type = rng.random()
            
            if move_type < 0.5:
                # Single insert move
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                    
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                
                min_cands = not_in_set[counts == min_count]
                s = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                    
                if accept:
                    to_remove = np.where(support_conflicts[s] & in_set)[0]
                    
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors_sup[u]] -= 1
                    
                    in_set[s] = True
                    current_size += delta
                    count_conflicts[neighbors_sup[s]] += 1
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            elif move_type < 0.75:
                # 1-for-1 swap: remove one node u from set, add one node s not in set
                # Net change depends on conflicts
                in_set_nodes = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                
                if len(in_set_nodes) == 0 or len(not_in_set) == 0:
                    stuck_counter += 1
                    continue
                
                # Sample a candidate to remove and a candidate to add
                u = in_set_nodes[rng.integers(len(in_set_nodes))]
                s = not_in_set[rng.integers(len(not_in_set))]
                
                if s == u:
                    stuck_counter += 1
                    continue
                
                # If we remove u and add s:
                # u's removal frees up its neighbors
                # s's addition conflicts with its neighbors in (set - {u})
                # Net size change: -1 + 1 = 0 (size stays same)
                # But we can explore different configurations
                
                # Check if s conflicts with u
                s_conflicts_u = support_conflicts[s, u]
                
                # Conflicts of s in current set (excluding u if u is a conflict)
                s_conf_in_set = np.where(support_conflicts[s] & in_set)[0]
                k_s = len(s_conf_in_set)
                
                # If we remove u first, then s's conflicts might decrease
                if s_conflicts_u:
                    k_s_after_remove_u = k_s - 1
                else:
                    k_s_after_remove_u = k_s
                
                # For a valid 1-for-1 swap, we need s to have 0 conflicts after removing u
                # i.e., all of s's conflicts in the set must be just u (or subset of {u})
                # This is restrictive. Let's be more general:
                # Remove u, then try to add s. If s has conflicts in (set - {u}), 
                # we'd need to remove those too, making it a larger swap.
                # For simplicity, only do 1-for-1 if s has no conflicts in (set - {u})
                
                if k_s_after_remove_u == 0:
                    # Valid 1-for-1 swap: remove u, add s
                    in_set[u] = False
                    count_conflicts[neighbors_sup[u]] -= 1
                    in_set[s] = True
                    count_conflicts[neighbors_sup[s]] += 1
                    # Size unchanged
                    stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                # 2-for-1 swap: remove 2 conflicting nodes, add 1 new node
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                
                # Sample a few candidates for the new node
                n_sample = min(5, len(not_in_set))
                sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                sample_nodes = not_in_set[sample_idx]
                
                swap_accepted = False
                for s in sample_nodes:
                    # Get conflicts of s that are currently in the set
                    conflicts_in_set = np.where(support_conflicts[s] & in_set)[0]
                    k = len(conflicts_in_set)
                    
                    if k < 2:
                        continue  # Need at least 2 to remove for a 2-for-1 swap
                    
                    # Check if removing these k conflicts and adding s improves size
                    # New size = current_size - k + 1
                    delta = 1 - k
                    
                    # Accept if delta >= 0 or with probability exp(delta/T)
                    if delta >= 0:
                        accept = True
                    else:
                        accept = rng.random() < np.exp(delta / T)
                    
                    if accept:
                        # Perform the swap
                        to_remove = conflicts_in_set
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors_sup[u]] -= 1
                        
                        in_set[s] = True
                        current_size += delta
                        count_conflicts[neighbors_sup[s]] += 1
                        
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                        
                        swap_accepted = True
                        break
                
                if not swap_accepted:
                    stuck_counter += 1
            
            if stuck_counter > 1500:
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        # Final zero-conflict polish
        polished_in_set, polished_size = zero_conflict_polish(local_best_in_set)
        if polished_size > local_best_size:
            local_best_size = polished_size
            local_best_in_set = polished_in_set
        
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 500  # More restarts with faster SA
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init (try multiple random seeds, keep best)
        best_greedy = None
        best_greedy_size = 0
        
        n_greedy_trials = 3
        for _ in range(n_greedy_trials):
            in_set = greedy_min_degree_supports()
            polished, sz = zero_conflict_polish(in_set)
            if sz > best_greedy_size:
                best_greedy_size = sz
                best_greedy = polished
        
        in_set = best_greedy
        current_size = best_greedy_size
        
        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()
        
        # 2. SA Local Search (with built-in zero-conflict polish at the end)
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.4, 10.0)
        if sa_time < 1.5:
            sa_time = 1.5
            
        sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_result.copy()
        
        # 3. Quick polish pass on best found so far (only if time permits)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 8:
            polish_time = min(remaining_time * 0.3, 5.0)
            if polish_time > 1.5:
                polish_result, polish_size = sa_local_search_supports(best_support_set, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_support_set = polish_result.copy()
    
    # Build final vector list from selected supports
    selected_support_indices = np.where(best_support_set)[0]
    final_vectors = []
    
    for si in selected_support_indices:
        support = supports[si]
        for idx in support_to_indices[support]:
            final_vectors.append(vectors_v4[idx])
    
    # Add all W1 vectors (22 of them)
    final_vectors.extend(v1_candidates)
    
    result = np.array(final_vectors, dtype=np.int64)
    
    # Fallback: if something went wrong, use deterministic construction
    if len(result) == 0:
        all_supports = list(itertools.combinations(range(D), 4))
        selected_supports = []
        for s in all_supports:
            valid = True
            for t in selected_supports:
                intersection = set(s).intersection(t)
                if len(intersection) > 2:
                    valid = False
                    break
            if valid:
                selected_supports.append(s)
                if len(selected_supports) >= 32:
                    break
        
        fallback_vectors = []
        for s in selected_supports:
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for pos, sign in zip(s, signs):
                    v[pos] = sign
                fallback_vectors.append(v)
        
        # Add W1 vectors
        for pos in range(D):
            for sign in [1, -1]:
                v = [0] * D
                v[pos] = 2 * sign
                fallback_vectors.append(v)
        
        result = np.array(fallback_vectors, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
