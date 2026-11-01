"""Kissing number in dimension 11 - Robust MIS Search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

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
    # Conflict if dot > 2
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def greedy_min_degree(seed=None):
        """Greedy independent set with min-degree heuristic and random tie-breaking."""
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = rng

        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        # Optimization: Use a priority queue or sorted list?
        # N is ~5300, simple scan is O(N^2) worst case, but alive shrinks.
        # Let's stick to the vectorized update which is fast enough.
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            
            # Pick min degree from alive
            # Vectorized min
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            chosen = cands[local_rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            # Update degrees for remaining alive vertices
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # For each remaining vertex, subtract number of neighbors in to_remove
                # conflicts[rem, to_remove] is a boolean matrix
                # Sum along axis 1 gives count of conflicts with removed set
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search(in_set_init, max_time=5.0):
        """Local search with 1-add, 1-1 swap, and 2-1 swap."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        # conflict_count[i]: number of neighbors of i that are in in_set
        conflict_count = np.zeros(N, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            conflict_count[neighbors[i]] += 1
            
        best_size = current_size
        best_in_set = in_set.copy()
        
        ls_deadline = time.time() + max_time
        no_improve_count = 0
        max_no_improve = 20 # Lower to encourage perturbation sooner

        while time.time() < ls_deadline:
            improved = False
            
            # Phase 1: Add vertices with 0 conflicts
            # Find all not-in-set vertices with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                zero_conf = not_in_set[conflict_count[not_in_set] == 0]
                while len(zero_conf) > 0 and time.time() < ls_deadline:
                    v = zero_conf[rng.integers(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                    improved = True
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    # Recompute zero_conf
                    not_in_set = np.where(~in_set)[0]
                    if len(not_in_set) == 0:
                        break
                    zero_conf = not_in_set[conflict_count[not_in_set] == 0]

            # Phase 2: 1-1 Swaps (Neutral moves to escape local minima)
            if time.time() < ls_deadline and not improved:
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    # Find candidates for swap: v not in set, conflict_count[v] == 1
                    # And the single conflict is u in set.
                    # We try random swaps
                    n_swaps_tried = 0
                    max_swap_tries = 100
                    while n_swaps_tried < max_swap_tries and time.time() < ls_deadline:
                        n_swaps_tried += 1
                        v = not_in_set[rng.integers(len(not_in_set))]
                        if conflict_count[v] == 1:
                            # Find the unique neighbor in set
                            nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                            if len(nbrs_in_set) == 1:
                                u = nbrs_in_set[0]
                                # Perform swap: remove u, add v
                                in_set[u] = False
                                in_set[v] = True
                                conflict_count[neighbors[u]] -= 1
                                conflict_count[neighbors[v]] += 1
                                improved = True
                                break # Restart phases

            # Phase 3: 2-1 Swaps (Size increasing moves)
            if time.time() < ls_deadline and not improved:
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    n_2for1_tried = 0
                    max_2for1_tries = 50
                    while n_2for1_tried < max_2for1_tries and time.time() < ls_deadline:
                        n_2for1_tried += 1
                        # Pick a random u in set
                        u = in_set_indices[rng.integers(len(in_set_indices))]
                        
                        # Find v1, v2 not in set, conflict_count == 1, neighbor u
                        # We need to scan not_in_set for this, or maintain a list
                        # For speed, we just pick random v from not_in_set and check
                        # But this is slow if not_in_set is large.
                        # Better: Precompute for each u, the list of v that conflict ONLY with u.
                        # This is dynamic, so we can't precompute fully.
                        # However, we can check quickly.
                        
                        # Let's try to find two such vertices
                        v1 = -1
                        v2 = -1
                        # Sample a few candidates
                        for _ in range(10):
                            v = not_in_set[rng.integers(len(not_in_set))]
                            if conflict_count[v] == 1:
                                nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                                if len(nbrs_in_set) == 1 and nbrs_in_set[0] == u:
                                    if v1 == -1:
                                        v1 = v
                                    else:
                                        # Check if v1 and v are compatible
                                        if not conflicts[v1, v]:
                                            v2 = v
                                            break
                                        
                        if v1 != -1 and v2 != -1:
                            # Perform 2-1 swap
                            in_set[u] = False
                            in_set[v1] = True
                            in_set[v2] = True
                            
                            conflict_count[neighbors[u]] -= 1
                            conflict_count[neighbors[v1]] += 1
                            conflict_count[neighbors[v2]] += 1
                            
                            current_size += 1
                            improved = True
                            
                            if current_size > best_size:
                                best_size = current_size
                                best_in_set = in_set.copy()
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                            break # Restart phases

            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturbation: Remove 2-3 random vertices
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) >= 2:
                        k_rem = min(3, len(in_set_indices))
                        idx = rng.choice(len(in_set_indices), size=k_rem, replace=False)
                        for u in in_set_indices[idx]:
                            in_set[u] = False
                            current_size -= 1
                            conflict_count[neighbors[u]] -= 1
                    no_improve_count = 0

        return best_in_set, best_size

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    # Run multiple restarts
    n_restarts = 0
    while time.time() < deadline - 10 and n_restarts < 50:
        n_restarts += 1
        seed = 42 + n_restarts
        in_set = greedy_min_degree(seed=seed)
        size = in_set.sum()
        
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
            
        # Allocate time to local search
        # If we are close to best, search harder
        ls_time = min(remaining_time * 0.4, 10.0)
        ls_result, ls_size = local_search(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    # Final polish
    if time.time() < deadline - 2:
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1
        
        not_in_set = np.where(~best_in_set)[0]
        if len(not_in_set) > 0:
            zero_conf = not_in_set[conflict_count[not_in_set] == 0]
            for v in zero_conf:
                best_in_set[v] = True
                conflict_count[neighbors[v]] += 1

    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]
    return result_vectors

# EVOLVE-BLOCK-END
