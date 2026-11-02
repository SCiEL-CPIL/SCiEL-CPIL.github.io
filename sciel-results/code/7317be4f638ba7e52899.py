"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate all W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    # and W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    # Key insight: Two W4 vectors conflict iff their supports share exactly 3 coordinates
    # (max dot product = 3 > 2). W1 vectors never conflict with W4 vectors.
    # So we need max independent set on the 330-vertex graph where edges = pairs of
    # 4-subsets sharing exactly 3 elements.
    
    # Generate all 4-subsets (supports)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Build conflict graph at support level: edge if |intersection| == 3
    # Two 4-subsets share exactly 3 elements iff they differ by swapping one element
    # For each support, find all supports sharing exactly 3 elements
    support_conflict = [set() for _ in range(n_supports)]
    
    for i in range(n_supports):
        s_i = set(supports[i])
        for j in range(i + 1, n_supports):
            s_j = set(supports[j])
            inter = len(s_i & s_j)
            if inter == 3:
                support_conflict[i].add(j)
                support_conflict[j].add(i)
    
    # Convert to numpy for speed
    support_conflicts = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        for j in support_conflict[i]:
            support_conflicts[i, j] = True
    
    # Find maximum independent set using greedy + local search on the 330-vertex graph
    rng = np.random.default_rng(42)
    
    # Precompute neighbor lists
    support_neighbors = [np.where(support_conflicts[i])[0] for i in range(n_supports)]
    support_degrees = np.array([len(n) for n in support_neighbors], dtype=np.int32)
    
    def greedy_support_init():
        """Greedy min-degree independent set on support graph."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = support_degrees.copy().astype(np.float64)
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            to_remove = np.unique(np.concatenate(([chosen], support_neighbors[chosen])))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def local_search_support(in_set_init, max_time=10.0):
        """Local search on support graph."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[support_neighbors[i]] += 1
        
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        ls_deadline = time.time() + max_time
        
        T = 2.0
        cooling = 0.999
        min_T = 0.01
        stuck_counter = 0
        
        while time.time() < ls_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.50:
                # ADD
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                delta = 1 - min_count
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.3
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(support_conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[support_neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[support_neighbors[v]] += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.80 and len(in_set_indices) > 0:
                # SWAP
                counts_add = count_conflicts[not_in_set]
                min_count_add = np.min(counts_add)
                cands_add = not_in_set[counts_add == min_count_add]
                v_add = cands_add[rng.integers(len(cands_add))]
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                
                conflicts_with = np.where(support_conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]
                
                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[support_neighbors[v_remove]] -= 1
                    count_conflicts[support_neighbors[v_add]] += 1
                else:
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[support_neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[support_neighbors[v_add]] += 1
                    else:
                        stuck_counter += 1
                        
            else:
                # REMOVE
                if len(in_set_indices) > 0:
                    v = in_set_indices[rng.integers(len(in_set_indices))]
                    accept = rng.random() < np.exp(-1.0 / T)
                    if accept:
                        in_set[v] = False
                        current_size -= 1
                        count_conflicts[support_neighbors[v]] -= 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 500:
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    def polish_support(in_set_init, max_time=3.0):
        """Greedily add non-conflicting supports."""
        polish_set = in_set_init.copy()
        polish_size = int(polish_set.sum())
        polish_deadline = time.time() + max_time
        
        for trial in range(10):
            if time.time() >= polish_deadline:
                break
            not_in = np.where(~polish_set)[0]
            if len(not_in) == 0:
                break
            order = rng.permutation(len(not_in))
            for idx in order:
                if time.time() >= polish_deadline:
                    break
                v = not_in[idx]
                if np.any(support_conflicts[v] & polish_set):
                    continue
                polish_set[v] = True
                polish_size += 1
        
        return polish_set, polish_size
    
    # Multiple restarts on the smaller 330-vertex graph
    best_support_set = np.zeros(n_supports, dtype=bool)
    best_support_size = 0
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        
        if n_restarts % 3 == 0:
            # Randomized greedy
            in_set = np.zeros(n_supports, dtype=bool)
            alive = np.ones(n_supports, dtype=bool)
            while np.any(alive) and time.time() < deadline - 3:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                chosen = alive_indices[rng.integers(len(alive_indices))]
                in_set[chosen] = True
                to_remove = np.unique(np.concatenate(([chosen], support_neighbors[chosen])))
                alive[to_remove] = False
        else:
            in_set = greedy_support_init()
        
        current_size = int(in_set.sum())
        if current_size > best_support_size:
            best_support_size = current_size
            best_support_set = in_set.copy()
        
        remaining = deadline - 5 - time.time()
        if remaining < 1:
            break
        
        ls_time = min(remaining * 0.4, 8.0)
        if ls_time < 1:
            ls_time = 1.0
            
        ls_result, ls_size = local_search_support(in_set, max_time=ls_time)
        if ls_size > best_support_size:
            best_support_size = ls_size
            best_support_set = ls_result.copy()
        
        remaining2 = deadline - 3 - time.time()
        if remaining2 > 1:
            polish_result, polish_size = polish_support(ls_result, max_time=min(remaining2 * 0.3, 3.0))
            if polish_size > best_support_size:
                best_support_size = polish_size
                best_support_set = polish_result.copy()
    
    # Construct final vector set:
    # For each support in the independent set, include all 16 sign variants
    # Plus all 22 W1 vectors
    selected_supports = np.where(best_support_set)[0]
    
    all_vectors = []
    for s_idx in selected_supports:
        positions = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            all_vectors.append(v)
    
    # Add W1 vectors
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            all_vectors.append(v)
    
    result = np.array(all_vectors, dtype=np.int64)
    
    # Verify: check that min pairwise distance >= max norm
    # For efficiency, spot check
    if len(result) > 100:
        norms_sq = np.sum(result ** 2, axis=1)
        max_norm = np.sqrt(np.max(norms_sq))
        # Check a random sample of pairs
        n_check = min(10000, len(result) * 50)
        idx1 = rng.integers(0, len(result), size=n_check)
        idx2 = rng.integers(0, len(result), size=n_check)
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]
        dists_sq = np.sum((result[idx1] - result[idx2]) ** 2, axis=1)
        min_dist = np.sqrt(np.min(dists_sq))
        # If verification fails, fall back to just W1 vectors
        if min_dist < max_norm - 1e-9:
            # Fallback: return only W1 vectors (always valid)
            v1 = []
            for pos in range(D):
                for sign in [1, -1]:
                    v = [0] * D
                    v[pos] = 2 * sign
                    v1.append(v)
            return np.array(v1, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
