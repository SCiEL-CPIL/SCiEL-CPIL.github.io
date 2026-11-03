import itertools
import numpy as np
import time

D = 11
DEADLINE = time.time() + 140  # safety margin before 150s kill


def _remaining_time():
    return max(0.0, DEADLINE - time.time())


def generate_candidates():
    """
    Generate candidate integer vectors:
    - Type A: 2 non-zero entries (±1), norm^2 = 2
    - Type B: 4 non-zero entries (±1), norm^2 = 4
    """
    candidates = []

    # Type A: ±e_i ± e_j for i < j (norm^2 = 2)
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)

    # Type B: 4 non-zero entries (±1) (norm^2 = 4)
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)

    return np.array(candidates, dtype=np.int64)


def build_conflict(candidates):
    """
    Build conflict matrix: conflict[i,j] = True if vectors i and j CANNOT coexist.
    Two vectors can coexist if ||vi - vj||^2 >= max(||vi||^2, ||vj||^2).
    Equivalently, they conflict if ||vi - vj||^2 < max(||vi||^2, ||vj||^2).
    
    ||vi - vj||^2 = ||vi||^2 + ||vj||^2 - 2*vi.vj
    So conflict if: ||vi||^2 + ||vj||^2 - 2*vi.vj < max(||vi||^2, ||vj||^2)
    Which simplifies to: 2*vi.vj > min(||vi||^2, ||vj||^2)
    """
    n = len(candidates)
    norms_sq = np.sum(candidates * candidates, axis=1)
    dots = candidates @ candidates.T
    
    # Conflict condition: 2*dot > min(norm_i^2, norm_j^2)
    min_norms_sq = np.minimum(norms_sq[:, None], norms_sq[None, :])
    conflict = (2 * dots > min_norms_sq)
    np.fill_diagonal(conflict, False)
    return conflict


def min_degree_greedy_fast(conflict, rng, time_limit=None):
    """
    Fast min-degree greedy with random tie-breaking.
    """
    n = len(conflict)
    active = np.ones(n, dtype=bool)
    deg = conflict.sum(axis=1).astype(np.int64)
    selected = []
    
    start = time.time()
    
    while active.any():
        if time_limit and (time.time() - start) > time_limit:
            break
        if _remaining_time() < 5:
            break
        
        # Find active nodes with minimum degree
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
        
        active_deg = deg[active_indices]
        min_deg = np.min(active_deg)
        
        # Find all active nodes with min degree
        candidates_for_pick = active_indices[active_deg == min_deg]
        
        # Pick one randomly
        idx = int(rng.choice(candidates_for_pick))
        
        selected.append(idx)
        
        # Remove idx and its conflicting neighbors
        neighbors_mask = conflict[idx].astype(bool)
        removed_mask = np.zeros(n, dtype=bool)
        removed_mask[idx] = True
        removed_mask[neighbors_mask & active] = True
        
        # Update degrees
        removed_indices = np.where(removed_mask)[0]
        if len(removed_indices) > 0:
            removal_counts = conflict[removed_indices, :].sum(axis=0).astype(np.int64)
            deg[active] -= removal_counts[active]
        
        # Update active mask
        active[removed_mask] = False
    
    return selected


def local_search_swap(conflict, selected, rng, time_limit=None, max_rounds=20):
    """
    Swap-based local search: try to improve the independent set by
    removing one selected vertex and adding two or more unselected vertices.
    """
    n = len(conflict)
    selected_set = set(selected)
    unselected = set(range(n)) - selected_set
    
    improved = True
    rounds = 0
    
    start = time.time()
    
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        
        if time_limit and (time.time() - start) > time_limit:
            break
        if _remaining_time() < 5:
            break
        
        # Try to add any compatible unselected vertex first
        for u in list(unselected):
            if all(not conflict[u, s] for s in selected_set):
                selected_set.add(u)
                unselected.discard(u)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove s, add 2+ vertices
        sel_list = list(selected_set)
        rng.shuffle(sel_list)
        
        for s in sel_list:
            if _remaining_time() < 5:
                return list(selected_set)
            
            # Find unselected vertices that are compatible with (selected - {s})
            new_sel = set(selected_set) - {s}
            candidates = []
            for u in unselected:
                if all(not conflict[u, t] for t in new_sel):
                    candidates.append(u)
            
            if len(candidates) >= 2:
                # Greedily add as many as possible
                count = 0
                for u in candidates:
                    if all(not conflict[u, t] for t in new_sel):
                        new_sel.add(u)
                        count += 1
                if len(new_sel) > len(selected_set):
                    selected_set = new_sel
                    unselected = set(range(n)) - selected_set
                    improved = True
                    break
    
    return list(selected_set)


def construct():
    """
    Construct a kissing configuration using a mixed-norm candidate pool
    and min-degree greedy with random restarts + local search.
    """
    candidates = generate_candidates()
    n = len(candidates)
    
    conflict = build_conflict(candidates)
    
    best_set = []
    best_size = 0
    
    # Run multiple random restarts of min-degree greedy
    n_restarts = 50
    
    for seed in range(n_restarts):
        if _remaining_time() < 10:
            break
        
        rng = np.random.default_rng(seed)
        selected = min_degree_greedy_fast(conflict, rng)
        
        # Apply local search
        if _remaining_time() > 15:
            selected = local_search_swap(conflict, selected, rng)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
    
    if len(best_set) == 0:
        # Fallback: return D11 roots
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[np.array(best_set), :]


if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} vectors")
    if len(result) > 0:
        norms_sq = np.sum(result * result, axis=1)
        max_norm_sq = np.max(norms_sq)
        print(f"Max norm^2: {max_norm_sq}")
        # Check min distance
        n = len(result)
        min_d2 = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                d2 = np.sum((result[i] - result[j]) ** 2)
                min_d2 = min(min_d2, d2)
        print(f"Min dist^2: {min_d2}")
        print(f"Valid: {min_d2 >= max_norm_sq}")
