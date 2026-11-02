"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def construct():
    deadline = time.time() + 110  # 10s safety margin
    rng = np.random.default_rng(42)
    
    # Generate all candidate integer vectors with squared norm = 4
    # Type A: (±1, ±1, 0, ..., 0) — 2 nonzero entries
    # Type B: (±1, ±1, ±1, ±1, 0, ..., 0) — 4 nonzero entries
    # Type C: (±2, 0, ..., 0) — 1 nonzero entry with value ±2
    
    candidates = []
    
    # Type A: two ±1's
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Type B: four ±1's
    for combo in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    
    # Type C: ±2 in one coordinate
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
    
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cand = len(cand_arr)
    
    # Precompute squared norms (all should be 4)
    cand_sq_norms = np.sum(cand_arr ** 2, axis=1)
    
    # Precompute pairwise squared distances between candidates (n_cand x n_cand)
    # dist^2 = |a|^2 + |b|^2 - 2*a.b = 4 + 4 - 2*a.b = 8 - 2*a.b
    # So dist >= 2 (norm) means dist^2 >= 4, i.e., 8 - 2*a.b >= 4, i.e., a.b <= 2
    # Equivalently: we need inner product <= 2 for all pairs
    
    # Compute all pairwise inner products
    dot_products = cand_arr @ cand_arr.T  # (n_cand, n_cand)
    # A pair (i,j) is compatible if dot_products[i,j] <= 2
    # (since both have sq norm 4, dist^2 = 4+4-2*dot = 8-2*dot >= 4 iff dot <= 2)
    
    # Build adjacency: compatible[i] = set of j where j != i and dot <= 2
    # For efficiency, use a boolean matrix
    compatible = (dot_products <= 2)
    np.fill_diagonal(compatible, False)
    
    # Start with a greedy maximal independent set on the "incompatible" graph
    # i.e., a set of mutually compatible candidates
    # Greedy: pick candidates in order of decreasing degree (fewest conflicts)
    
    # Degree = number of incompatible neighbors = n_cand - 1 - sum(compatible[i])
    incompatible_count = n_cand - 1 - compatible.sum(axis=1)
    
    # Greedy: repeatedly pick the candidate with fewest conflicts among remaining
    remaining = set(range(n_cand))
    selected = []
    selected_set = set()
    
    # Use a priority approach: sort by incompatible_count initially
    order = np.argsort(incompatible_count)
    
    for idx in order:
        if time.time() > deadline - 30:
            break
        idx = int(idx)
        if idx not in remaining:
            continue
        # Check if idx is compatible with all already selected
        can_add = True
        for s in selected:
            if not compatible[idx, s]:
                can_add = False
                break
        if can_add:
            selected.append(idx)
            selected_set.add(idx)
    
    # Now apply simulated annealing to try to add more
    # State: selected set of candidate indices
    # Moves: add one candidate, remove one candidate, swap (remove one, add one)
    # Energy = -|selected| (we want to maximize size)
    
    best_selected = list(selected)
    best_size = len(best_selected)
    current_selected = list(selected)
    current_size = len(current_selected)
    
    # Precompute for fast compatibility checks
    # comp_mask[i] = boolean array of which candidates are compatible with i
    
    t_start = time.time()
    T = 1.0  # initial temperature
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline - 10:
        # Random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add a random candidate
            if current_size >= n_cand:
                continue
            # Pick a random unselected candidate
            unselected = [i for i in range(n_cand) if i not in set(current_selected)]
            if not unselected:
                continue
            new_cand = unselected[rng.integers(len(unselected))]
            # Check compatibility
            if all(compatible[new_cand, s] for s in current_selected):
                current_selected.append(new_cand)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_selected = list(current_selected)
            else:
                # Count how many conflicts
                conflicts = sum(1 for s in current_selected if not compatible[new_cand, s])
                delta_E = -(1) + conflicts  # energy change: gain 1, but conflicts penalize
                # Accept with probability exp(-delta_E / T)
                if rng.random() < np.exp(-delta_E / T):
                    # Swap: remove a random conflicting candidate, add new_cand
                    conflicting = [s for s in current_selected if not compatible[new_cand, s]]
                    if conflicting:
                        remove_idx = conflicting[rng.integers(len(conflicting))]
                        current_selected.remove(remove_idx)
                        current_selected.append(new_cand)
                        # current_size stays the same
                        if current_size > best_size:
                            best_size = current_size
                            best_selected = list(current_selected)
        elif move_type < 0.7:
            # Try to remove a random candidate
            if current_size > 1:
                remove_idx = current_selected[rng.integers(current_size)]
                current_selected.remove(remove_idx)
                current_size -= 1
                # This is a worsening move; accept with probability exp(1/T)
                if rng.random() < np.exp(1.0 / T):
                    pass  # keep removal
                else:
                    current_selected.append(remove_idx)
                    current_size += 1
        else:
            # Swap: remove one, add one
            if current_size > 0:
                unselected = [i for i in range(n_cand) if i not in set(current_selected)]
                if unselected:
                    new_cand = unselected[rng.integers(len(unselected))]
                    remove_idx = current_selected[rng.integers(current_size)]
                    # Check if new_cand is compatible with all except remove_idx
                    compatible_with_rest = all(
                        compatible[new_cand, s] for s in current_selected if s != remove_idx
                    )
                    if compatible_with_rest:
                        current_selected.remove(remove_idx)
                        current_selected.append(new_cand)
                        # current_size stays the same
                        if current_size > best_size:
                            best_size = current_size
                            best_selected = list(current_selected)
                    else:
                        # Count conflicts with rest (excluding remove_idx)
                        conflicts = sum(
                            1 for s in current_selected 
                            if s != remove_idx and not compatible[new_cand, s]
                        )
                        delta_E = conflicts
                        if rng.random() < np.exp(-delta_E / T):
                            current_selected.remove(remove_idx)
                            current_selected.append(new_cand)
                            if current_size > best_size:
                                best_size = current_size
                                best_selected = list(current_selected)
        
        T = max(T * cooling, T_min)
    
    # Use best found
    final_indices = best_selected
    result = cand_arr[final_indices]
    
    # Verify
    if len(result) > 0:
        max_norm = np.max(np.linalg.norm(result, axis=1))
        from scipy.spatial.distance import cdist
        dists = cdist(result, result)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        if min_dist >= max_norm - 1e-9:
            return result
    
    # Fallback to greedy result
    greedy_result = cand_arr[selected]
    if len(greedy_result) > 0:
        max_norm = np.max(np.linalg.norm(greedy_result, axis=1))
        from scipy.spatial.distance import cdist
        dists = cdist(greedy_result, greedy_result)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        if min_dist >= max_norm - 1e-9:
            return greedy_result
    
    # Ultimate fallback: just Type A vectors
    vecs2 = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vecs2.append(v)
    return np.array(vecs2, dtype=np.int64)

# EVOLVE-BLOCK-END
