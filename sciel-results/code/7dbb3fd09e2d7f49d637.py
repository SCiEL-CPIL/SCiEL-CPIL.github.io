import numpy as np
import itertools
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    D = 11
    
    # Generate candidates: norm squared 4
    # Type A: weight 4, entries in {-1, 1}
    # Type B: weight 1, entries in {-2, 2}
    
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    all_candidates = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute conflict matrix
    # Conflict if dot > 2 (since norm^2 = 4, distance^2 >= 4 implies dot <= 2)
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to list of neighbor indices for fast access
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def greedy_init(seed=0):
        """Greedy min-degree initialization with random tie-breaking."""
        local_rng = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        # Use a priority queue for min-degree? 
        # Or just iterate and pick min.
        # For N=5300, simple iteration is fine.
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            chosen = cands[local_rng.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Update degrees of remaining vertices
                # deg[rem] -= sum(conflicts[rem, to_remove], axis=1)
                # This is O(N^2) per step if not careful.
                # But `to_remove` is small on average?
                # Actually, `conflicts[rem, to_remove]` is a matrix multiplication/slicing.
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search(in_set, max_time=5.0, seed=0):
        """Local search with 1-add, 1-1 swap, and 1-2 swap."""
        local_rng = np.random.default_rng(seed)
        in_set = in_set.copy()
        current_size = in_set.sum()
        best_size = current_size
        best_in_set = in_set.copy()
        
        # conflict_count[i]: number of neighbors of i that are in in_set
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        ls_deadline = time.time() + max_time
        stuck = 0
        no_improve_counter = 0
        
        while time.time() < ls_deadline:
            improved = False
            
            # 1. Try to add a vertex with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = conflict_count[not_in_set]
                zero_conf = not_in_set[counts == 0]
                if len(zero_conf) > 0:
                    v = zero_conf[local_rng.integers(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                    stuck = 0
                    no_improve_counter = 0
                    continue
            
            # If we can't add, try swaps
            # 2. Try 1-2 Swap (Remove 1, Add 2) -> Size increases by 1
            # This is the most valuable move.
            if not improved:
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 0:
                    # Find vertices not in set with conflict_count == 1
                    # These are "blocked" by exactly one vertex in the set
                    cands_v = np.where((~in_set) & (conflict_count == 1))[0]
                    if len(cands_v) >= 2:
                        # For each u in set, check if we can find two v's blocked only by u
                        # and not conflicting with each other.
                        # To be fast, we can sample a few u's.
                        sample_size = min(10, len(in_set_indices))
                        sampled_u = in_set_indices[local_rng.choice(len(in_set_indices), size=sample_size, replace=False)]
                        
                        for u in sampled_u:
                            # Find v's in cands_v whose only conflict is u
                            # conflict_count[v] == 1 and conflicts[v, u] is True
                            # We need to check if the blocker is indeed u.
                            # We can precompute blocker? No, dynamic.
                            # Check: for v in cands_v, is neighbors[v] intersect in_set == {u}?
                            # This is expensive.
                            # Alternative: 
                            # If v is in cands_v, it has 1 conflict. 
                            # We can check if conflicts[v, u] is True.
                            # If yes, and we assume it's the only one (since count is 1), then u is the blocker.
                            potential_vs = cands_v[conflicts[cands_v, u]]
                            if len(potential_vs) >= 2:
                                # Pick 2 from potential_vs that don't conflict with each other
                                # Check conflicts among potential_vs
                                # potential_vs is small usually
                                for i in range(len(potential_vs)):
                                    for j in range(i+1, len(potential_vs)):
                                        v1 = potential_vs[i]
                                        v2 = potential_vs[j]
                                        if not conflicts[v1, v2]:
                                            # Valid 1-2 swap
                                            in_set[u] = False
                                            in_set[v1] = True
                                            in_set[v2] = True
                                            
                                            conflict_count[neighbors[u]] -= 1
                                            conflict_count[neighbors[v1]] += 1
                                            conflict_count[neighbors[v2]] += 1
                                            
                                            current_size += 1
                                            improved = True
                                            stuck = 0
                                            no_improve_counter = 0
                                            if current_size > best_size:
                                                best_size = current_size
                                                best_in_set = in_set.copy()
                                            break
                                    if improved:
                                        break
                            if improved:
                                break
            if improved:
                continue

            # 3. Try 1-1 Swap (Keep size, improve structure)
            if not improved and stuck > 5:
                cands_v = np.where((~in_set) & (conflict_count == 1))[0]
                if len(cands_v) > 0:
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        u = in_set_indices[local_rng.integers(len(in_set_indices))]
                        # Find v blocked by u
                        potential_vs = cands_v[conflicts[cands_v, u]]
                        if len(potential_vs) > 0:
                            v = potential_vs[0]
                            # Swap u and v
                            in_set[u] = False
                            in_set[v] = True
                            conflict_count[neighbors[u]] -= 1
                            conflict_count[neighbors[v]] += 1
                            stuck = 0
                            improved = True
                            # Size unchanged, so best_size might not update, but structure might improve

            if not improved:
                stuck += 1
                no_improve_counter += 1
                # Perturbation if stuck too long
                if stuck > 50:
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        # Remove a random vertex to escape local optimum
                        u = in_set_indices[local_rng.integers(len(in_set_indices))]
                        in_set[u] = False
                        current_size -= 1
                        conflict_count[neighbors[u]] -= 1
                        stuck = 0

        return best_in_set, best_size

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    # Run multiple restarts
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        seed = 1000 + n_restarts
        in_set = greedy_init(seed)
        size = in_set.sum()
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
            
        ls_time = min(remaining_time * 0.4, 10.0)
        ls_result, ls_size = local_search(in_set, max_time=ls_time, seed=seed)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    # Final polish: try to add any remaining vertices with 0 conflicts
    if time.time() < deadline - 2:
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1
        
        not_in_set = np.where(~best_in_set)[0]
        if len(not_in_set) > 0:
            counts = conflict_count[not_in_set]
            zero_conf = not_in_set[counts == 0]
            for v in zero_conf:
                best_in_set[v] = True
                conflict_count[neighbors[v]] += 1
                # Update counts for other not-in-set vertices
                remaining = np.where(~best_in_set)[0]
                if len(remaining) > 0:
                    conflict_count[remaining] += conflicts[remaining, v].astype(np.int32)
    
    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]
    return result_vectors
