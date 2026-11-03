"""Kissing number in dimension 11: Simulated Annealing on expanded candidate set."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def time_left():
        return deadline - time.time()

    # Generate candidate vectors
    # We'll include vectors with entries in {-2, -1, 0, 1, 2}
    # Focus on vectors with small L2 norms to keep the problem tractable
    # 
    # Key insight: if we use vectors of different norms, the max norm sets the threshold.
    # Strategy: try different "base norm" levels and find the best configuration.
    #
    # For a set S of vectors, the constraint is:
    # min_{u,v in S, u!=v} ||u-v|| >= max_{w in S} ||w||
    #
    # Let R = max ||w||. Then all pairs must have distance >= R.
    #
    # Approach: enumerate candidate vectors by norm level, then for each possible
    # max-norm level R, find the largest subset with pairwise distance >= R.

    # Generate all integer vectors with entries in {-2,-1,0,1,2} and small norm
    # To keep candidate set manageable, limit to vectors with squared norm <= 8
    candidates = []
    
    # Weight 1: single non-zero entry
    for i in range(D):
        for val in (-2, -1, 1, 2):
            v = np.zeros(D, dtype=np.int8)
            v[i] = val
            candidates.append(v)
    
    # Weight 2: two non-zero entries
    for i, j in itertools.combinations(range(D), 2):
        for vi in (-2, -1, 1, 2):
            for vj in (-2, -1, 1, 2):
                v = np.zeros(D, dtype=np.int8)
                v[i] = vi
                v[j] = vj
                candidates.append(v)
    
    # Weight 3: three non-zero entries (only with entries ±1 to keep norm small)
    for i, j, k in itertools.combinations(range(D), 3):
        for vi in (-1, 1):
            for vj in (-1, 1):
                for vk in (-1, 1):
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = vi
                    v[j] = vj
                    v[k] = vk
                    candidates.append(v)
    
    # Weight 4: four non-zero entries with ±1
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product((-1, 1), repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, val in zip(support, signs):
                v[idx] = val
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int8)
    n_cand = len(candidates)
    
    # Compute squared norms
    sq_norms = np.sum(candidates.astype(np.int16) ** 2, axis=1)
    
    # Compute all pairwise squared distances
    # dist_sq[i,j] = ||cand_i - cand_j||^2
    # = sq_norms[i] + sq_norms[j] - 2 * dot(cand_i, cand_j)
    dots = candidates.astype(np.int16) @ candidates.astype(np.int16).T  # (n_cand, n_cand)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * dots
    np.fill_diagonal(dist_sq, 0)
    
    # Now, for a given threshold R_sq (squared max norm), we need:
    # For all selected pairs i,j: dist_sq[i,j] >= R_sq
    # And: max sq_norms[i] for i in selected <= R_sq
    #
    # The constraint is: dist_sq[i,j] >= max(sq_norms[i], sq_norms[j]) for all pairs
    # Actually the constraint is: dist_sq[i,j] >= R_sq where R_sq = max sq_norms in set
    #
    # This is a maximum independent set problem on a graph where edges connect
    # incompatible pairs. But the graph depends on R_sq.
    #
    # Alternative: for each candidate vector v, define its "required threshold" as sq_norms[v].
    # If we select a set S, the threshold is max sq_norms in S.
    # All pairs in S must have dist_sq >= threshold.
    #
    # Let's try a different approach: sort candidates by norm, and for each possible
    # threshold level, solve the MIS problem.
    
    # Get unique norm levels
    unique_norms = np.unique(sq_norms)
    
    # For each threshold, build the conflict graph and solve MIS greedily + local search
    best_count = 0
    best_set = []
    
    # We'll focus on thresholds that are actually achieved by some candidate
    # For each threshold R_sq, eligible candidates are those with sq_norm <= R_sq
    # and pairwise dist_sq >= R_sq among selected
    
    # Process thresholds in order of likelihood of being optimal
    # The D11 roots have norm_sq = 2, giving 220 vectors
    # Let's start with norm_sq = 2 and see if we can do better with higher norms
    
    # For efficiency, let's precompute for each threshold which pairs are compatible
    # But that's too many thresholds. Instead, let's use a smarter approach.
    
    # Key observation: if all selected vectors have the same norm r, then we need
    # ||u-v|| >= r, i.e., ||u-v||^2 >= r^2
    # ||u-v||^2 = 2*r^2 - 2*dot(u,v) >= r^2
    # => dot(u,v) <= r^2/2
    #
    # For r^2 = 2 (D11 roots): dot <= 1. Max dot for distinct roots is 1. All 220 work.
    # For r^2 = 4 (weight-4 ±1 or 2e_i): dot <= 2. 
    # For r^2 = 3 (weight-3 ±1): dot <= 1.5, so dot <= 1.
    
    # Let's try a multi-phase approach:
    # Phase 1: All vectors with same norm r. Find max subset.
    # Phase 2: Mixed norms. Use simulated annealing.
    
    # Phase 1: Same-norm configurations
    # Group candidates by squared norm
    norm_groups = {}
    for idx in range(n_cand):
        ns = int(sq_norms[idx])
        if ns not in norm_groups:
            norm_groups[ns] = []
        norm_groups[ns].append(idx)
    
    for norm_sq_val, group in norm_groups.items():
        if time_left() < 5:
            break
        group = np.array(group)
        n_group = len(group)
        if n_group == 0:
            continue
        
        # For this group, all have norm_sq = norm_sq_val
        # Need dist_sq >= norm_sq_val for all pairs
        # dist_sq[i,j] = 2*norm_sq_val - 2*dot(i,j) >= norm_sq_val
        # => dot(i,j) <= norm_sq_val/2
        
        max_dot = norm_sq_val / 2.0
        
        # Build conflict matrix within group
        group_dots = dots[group][:, group]  # (n_group, n_group)
        conflict = (group_dots > max_dot)
        np.fill_diagonal(conflict, False)
        
        # Greedy + local search for MIS
        best_local = _solve_mis(conflict, n_group, time_budget=min(20, time_left()))
        if len(best_local) > best_count:
            best_count = len(best_local)
            best_set = [group[i] for i in best_local]
    
    # Phase 2: Simulated annealing on full candidate set
    if time_left() > 10:
        # Initialize with best from Phase 1
        if best_count > 0:
            current_set = set(best_set)
        else:
            # Random initialization
            current_set = set(random.sample(range(n_cand), min(50, n_cand)))
        
        # Compute current threshold
        def get_threshold(s):
            if not s:
                return 0
            return max(int(sq_norms[i]) for i in s)
        
        def is_valid(s):
            if not s:
                return True
            thresh = get_threshold(s)
            s_list = list(s)
            for i_idx in range(len(s_list)):
                for j_idx in range(i_idx + 1, len(s_list)):
                    if dist_sq[s_list[i_idx], s_list[j_idx]] < thresh:
                        return False
            return True
        
        # Simulated annealing
        current_temp = 100.0
        cooling_rate = 0.995
        min_temp = 0.1
        
        best_sa_count = len(current_set)
        best_sa_set = set(current_set)
        
        while current_temp > min_temp and time_left() > 5:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random candidate
                cand = random.randint(0, n_cand - 1)
                if cand not in current_set:
                    new_set = current_set | {cand}
                    new_thresh = max(get_threshold(current_set), int(sq_norms[cand]))
                    # Check if cand is compatible with all in current_set
                    compatible = True
                    for c in current_set:
                        if dist_sq[cand, c] < new_thresh:
                            compatible = False
                            break
                    if compatible:
                        current_set = new_set
                        if len(current_set) > best_sa_count:
                            best_sa_count = len(current_set)
                            best_sa_set = set(current_set)
                    else:
                        # Accept with probability based on temperature
                        # Penalty: number of violated constraints
                        violations = sum(1 for c in current_set if dist_sq[cand, c] < new_thresh)
                        delta = -violations
                        if random.random() < np.exp(delta / current_temp):
                            current_set = new_set
                            if len(current_set) > best_sa_count:
                                best_sa_count = len(current_set)
                                best_sa_set = set(current_set)
            
            elif move_type < 0.7:
                # Remove a random element
                if current_set:
                    to_remove = random.choice(list(current_set))
                    current_set.discard(to_remove)
            
            else:
                # Swap: remove one, add another
                if current_set:
                    to_remove = random.choice(list(current_set))
                    cand = random.randint(0, n_cand - 1)
                    if cand not in current_set:
                        new_set = (current_set - {to_remove}) | {cand}
                        new_thresh = get_threshold(new_set)
                        # Check compatibility of cand with new_set - {cand}
                        compatible = True
                        for c in new_set:
                            if c != cand and dist_sq[cand, c] < new_thresh:
                                compatible = False
                                break
                        if compatible:
                            current_set = new_set
                            if len(current_set) > best_sa_count:
                                best_sa_count = len(current_set)
                                best_sa_set = set(current_set)
                        else:
                            violations = sum(1 for c in new_set if c != cand and dist_sq[cand, c] < new_thresh)
                            delta = -violations
                            if random.random() < np.exp(delta / current_temp):
                                current_set = new_set
                                if len(current_set) > best_sa_count:
                                    best_sa_count = len(current_set)
                                    best_sa_set = set(current_set)
            
            current_temp *= cooling_rate
        
        if best_sa_count > best_count:
            best_count = best_sa_count
            best_set = list(best_sa_set)
    
    # Final verification and return
    if best_count == 0:
        # Fallback: return D11 roots
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = candidates[best_set].astype(np.int64)
    
    # Verify the result
    if len(result) > 1:
        max_norm_sq = np.max(np.sum(result ** 2, axis=1))
        # Check all pairwise distances
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d_sq = np.sum((result[i] - result[j]) ** 2)
                if d_sq < max_norm_sq:
                    # Invalid, fall back
                    vectors = []
                    for i2, j2 in itertools.combinations(range(D), 2):
                        for si in (-1, 1):
                            for sj in (-1, 1):
                                v = np.zeros(D, dtype=np.int64)
                                v[i2] = si
                                v[j2] = sj
                                vectors.append(v)
                    return np.array(vectors, dtype=np.int64)
    
    return result


def _solve_mis(conflict, n, time_budget):
    """Solve maximum independent set using greedy + local search."""
    start = time.time()
    deadline = start + time_budget
    
    best_set = set()
    
    # Multiple random restarts
    for restart in range(100):
        if time.time() > deadline:
            break
        
        np.random.seed(restart)
        perm = np.random.permutation(n)
        
        current = set()
        for idx in perm:
            if time.time() > deadline:
                break
            # Check if idx is compatible with all in current
            if not conflict[idx, list(current)].any():
                current.add(idx)
        
        if len(current) > len(best_set):
            best_set = set(current)
        
        # Local search: try swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_list = list(current)
            for out_idx in current_list:
                if time.time() > deadline:
                    break
                remaining = current - {out_idx}
                # Find candidates compatible with remaining
                if remaining:
                    rem_arr = np.array(list(remaining))
                    compat = ~conflict[rem_arr].any(axis=0)
                else:
                    compat = np.ones(n, dtype=bool)
                in_current = np.zeros(n, dtype=bool)
                in_current[list(current)] = True
                compat[in_current] = False
                candidates = np.where(compat)[0]
                
                for cand in candidates:
                    if time.time() > deadline:
                        break
                    new_set = remaining | {cand}
                    if len(new_set) > len(current):
                        current = new_set
                        improved = True
                        break
                if improved:
                    break
        
        if len(current) > len(best_set):
            best_set = set(current)
    
    return list(best_set)
# EVOLVE-BLOCK-END
