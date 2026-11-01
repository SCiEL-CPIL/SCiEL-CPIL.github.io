"""Kissing number in dimension 11 - Optimized Local Search + Simulated Annealing."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates: norm squared 4
    # 1. Weight 4 vectors (entries in {-1, 0, 1})
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # 2. Weight 1 vectors (entries in {-2, 0, 2})
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
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors as lists for fast access in some contexts, 
    # but we will rely on the boolean matrix for vectorized ops where possible.
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def greedy_init(seed=42):
        """Greedy independent set with min-degree heuristic."""
        rng_local = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            chosen = cands[rng_local.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search_grouped(in_set, max_time=10.0, seed=42):
        """
        Optimized local search.
        Replaces slow Python loops for swap detection with vectorized NumPy ops.
        """
        rng_local = np.random.default_rng(seed)
        in_set = in_set.copy()
        current_size = int(in_set.sum())
        best_size = current_size
        best_in_set = in_set.copy()
        
        # conflict_count[v] = number of neighbors of v that are in in_set
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        ls_deadline = time.time() + max_time
        stuck = 0
        
        while time.time() < ls_deadline:
            improved = False
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break

            counts = conflict_count[not_in_set]
            
            # 1. Add 0-conflict
            zero_conf = not_in_set[counts == 0]
            if len(zero_conf) > 0:
                v = zero_conf[rng_local.integers(len(zero_conf))]
                in_set[v] = True
                current_size += 1
                conflict_count[neighbors[v]] += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
                stuck = 0
                continue

            # 2. 1-2 Swaps (Remove 1, Add 2)
            # We look for vertices u in in_set such that there are at least 2 vertices v1, v2 not in in_set
            # that conflict ONLY with u (among in_set vertices) and do not conflict with each other.
            
            # For each v not in set with count 1, identify its unique neighbor u in set.
            one_conf = not_in_set[counts == 1]
            if len(one_conf) > 0:
                # Vectorized mapping: for each v in one_conf, find its neighbor in in_set
                # conflicts[v] & in_set gives a boolean vector. Since count is 1, there is exactly one True.
                # We can use argmax on the boolean array to find the index.
                # But we need to be careful with memory.
                
                # Let's group by the neighbor u.
                # Instead of a defaultdict, we can create an array of u's for each v.
                # However, iterating is still O(N). Let's try to optimize the inner loop.
                
                # Map v -> u
                # Create a temporary array to hold the unique neighbor for each v in one_conf
                # conflicts[one_conf, :] is a matrix. 
                # We want the column index where it is True.
                # Since sum is 1, we can do:
                # unique_neighbors = np.argmax(conflicts[one_conf, :] * in_set, axis=1)
                # But we must ensure it's actually in in_set.
                
                mask = conflicts[one_conf, :] & in_set
                # If a row has no True (shouldn't happen if count==1 and we computed count correctly), argmax is 0.
                # We check if mask.sum(axis=1) == 1
                valid_mask = mask.sum(axis=1) == 1
                if np.any(valid_mask):
                    valid_one_conf = one_conf[valid_mask]
                    u_indices = np.argmax(mask[valid_mask], axis=1)
                    
                    # Now we have pairs (v, u). We need to group by u.
                    # Sort by u to group them
                    order = np.argsort(u_indices)
                    sorted_u = u_indices[order]
                    sorted_v = valid_one_conf[order]
                    
                    # Find boundaries where u changes
                    # np.diff returns array of size M-1
                    changes = np.diff(sorted_u)
                    boundaries = np.where(changes != 0)[0] + 1
                    start_indices = np.concatenate(([0], boundaries))
                    end_indices = np.concatenate((boundaries, [len(sorted_u)]))
                    
                    # Iterate through groups
                    # To avoid Python loop overhead, we can check if any group has size >= 2
                    # and then check for non-conflict within that group.
                    
                    found_swap = False
                    for start, end in zip(start_indices, end_indices):
                        if end - start < 2:
                            continue
                        
                        u = sorted_u[start]
                        vs = sorted_v[start:end]
                        
                        # Check if any pair in vs is non-conflicting
                        # conflicts[vs[:, None], vs[None, :]]
                        # This creates a dense matrix for the group.
                        # If the group is large, this is expensive.
                        # But groups are usually small.
                        
                        # We can check if there is any 0 in the off-diagonal of conflicts[vs, vs]
                        sub_conf = conflicts[vs[:, None], vs[None, :]]
                        # sub_conf[i, j] is True if vs[i] conflicts with vs[j]
                        # We need a pair (i, j) with i != j such that sub_conf[i, j] is False.
                        
                        # If the group size is small, we can just check.
                        if len(vs) < 50:
                            # Check all pairs
                            # We can use np.triu_indices to get upper triangle
                            # Or simply check if the number of conflicts is less than total pairs
                            total_pairs = len(vs) * (len(vs) - 1) // 2
                            conflict_pairs = np.sum(sub_conf) // 2
                            if conflict_pairs < total_pairs:
                                # Found a non-conflicting pair
                                # Find the specific pair
                                # We can iterate or use logic
                                # For speed, let's just pick the first non-conflicting pair found
                                # Or just pick two random ones from the group if we know one exists?
                                # No, we need the specific pair.
                                
                                # Let's find one
                                # We can use np.where on the flattened matrix?
                                # Or just loop since we only do this on success
                                for i in range(len(vs)):
                                    found = False
                                    for j in range(i+1, len(vs)):
                                        if not sub_conf[i, j]:
                                            v1, v2 = vs[i], vs[j]
                                            # Perform swap
                                            in_set[u] = False
                                            in_set[v1] = True
                                            in_set[v2] = True
                                            
                                            conflict_count[neighbors[u]] -= 1
                                            conflict_count[neighbors[v1]] += 1
                                            conflict_count[neighbors[v2]] += 1
                                            
                                            current_size += 1
                                            improved = True
                                            found_swap = True
                                            if current_size > best_size:
                                                best_size = current_size
                                                best_in_set = in_set.copy()
                                            found = True
                                            break
                                    if found:
                                        break
                        else:
                            # Large group: random sampling
                            for _ in range(10):
                                i = rng_local.integers(len(vs))
                                j = rng_local.integers(len(vs))
                                if i == j: continue
                                if not conflicts[vs[i], vs[j]]:
                                    v1, v2 = vs[i], vs[j]
                                    in_set[u] = False
                                    in_set[v1] = True
                                    in_set[v2] = True
                                    
                                    conflict_count[neighbors[u]] -= 1
                                    conflict_count[neighbors[v1]] += 1
                                    conflict_count[neighbors[v2]] += 1
                                    
                                    current_size += 1
                                    improved = True
                                    found_swap = True
                                    if current_size > best_size:
                                        best_size = current_size
                                        best_in_set = in_set.copy()
                                    break
                        
                        if found_swap:
                            break
                    
                    if found_swap:
                        stuck = 0
                        continue

            # 3. 1-3 Swaps
            if not improved and len(one_conf) > 0:
                # Reuse the grouping logic from above?
                # The grouping is expensive. Let's only do this if stuck.
                if stuck > 2:
                     mask = conflicts[one_conf, :] & in_set
                     valid_mask = mask.sum(axis=1) == 1
                     if np.any(valid_mask):
                         valid_one_conf = one_conf[valid_mask]
                         u_indices = np.argmax(mask[valid_mask], axis=1)
                         
                         order = np.argsort(u_indices)
                         sorted_u = u_indices[order]
                         sorted_v = valid_one_conf[order]
                         
                         changes = np.diff(sorted_u)
                         boundaries = np.where(changes != 0)[0] + 1
                         start_indices = np.concatenate(([0], boundaries))
                         end_indices = np.concatenate((boundaries, [len(sorted_u)]))
                         
                         found_swap = False
                         for start, end in zip(start_indices, end_indices):
                             if end - start < 3:
                                 continue
                             
                             u = sorted_u[start]
                             vs = sorted_v[start:end]
                             
                             # Find 3 mutually non-conflicting
                             chosen_vs = []
                             for v in vs:
                                 if all(not conflicts[v, cv] for cv in chosen_vs):
                                     chosen_vs.append(v)
                                     if len(chosen_vs) == 3:
                                         break
                             if len(chosen_vs) == 3:
                                 in_set[u] = False
                                 for v in chosen_vs:
                                     in_set[v] = True
                                 conflict_count[neighbors[u]] -= 1
                                 for v in chosen_vs:
                                     conflict_count[neighbors[v]] += 1
                                 
                                 current_size += 2
                                 improved = True
                                 found_swap = True
                                 if current_size > best_size:
                                     best_size = current_size
                                     best_in_set = in_set.copy()
                                 break
                             
                             if found_swap:
                                 break
                         
                         if found_swap:
                             stuck = 0
                             continue

            # 4. 2-2 Swaps
            if not improved and stuck > 5:
                two_conf = not_in_set[counts == 2]
                if len(two_conf) > 0:
                    # Group by pair of neighbors
                    # This is harder to vectorize efficiently without a dict.
                    # We'll stick to a simpler heuristic: random 2-2 swap attempt
                    # Pick 2 random in_set vertices, see if we can replace them with 2 non-conflicting out_set vertices.
                    
                    # Actually, let's just do a random perturbation if stuck too long
                    pass 

            if not improved:
                stuck += 1
                if stuck > 15:
                    # Perturbation
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        n_remove = rng_local.integers(2, 5)
                        to_remove = rng_local.choice(in_set_indices, size=min(n_remove, len(in_set_indices)), replace=False)
                        for u in to_remove:
                            in_set[u] = False
                            current_size -= 1
                            conflict_count[neighbors[u]] -= 1
                        stuck = 0

        return best_in_set, best_size

    def sa_local_search(in_set_init, max_time=8.0, seed=42):
        """Simulated annealing local search."""
        rng_local = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 5.0
        cooling = 0.9995
        min_T = 0.05

        count_conflicts = np.zeros(N, dtype=np.int32)
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

            move_type = rng_local.random()

            if move_type < 0.5 and len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng_local.integers(len(min_cands))]
                k = min_count
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng_local.random() < 0.3
                else:
                    accept = rng_local.random() < np.exp(delta / T)

                if accept:
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

            elif move_type < 0.7 and len(in_set_indices) > 0:
                v = in_set_indices[rng_local.integers(len(in_set_indices))]
                delta = -1
                accept = rng_local.random() < np.exp(delta / T)

                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                    stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.85 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                v_remove = in_set_indices[rng_local.integers(len(in_set_indices))]
                v_add_candidates = not_in_set[(count_conflicts[not_in_set] <= 1)]
                if len(v_add_candidates) == 0:
                    v_add_candidates = not_in_set
                v_add = v_add_candidates[rng_local.integers(len(v_add_candidates))]

                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]

                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                    stuck_counter += 1
                else:
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng_local.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1

                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
            else:
                # 2-for-1
                if len(in_set_indices) >= 2 and len(not_in_set) > 0:
                    v_add = not_in_set[rng_local.integers(len(not_in_set))]
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    if len(conflicts_with) == 2:
                        to_remove = conflicts_with
                        delta = 1 - 2
                        accept = rng_local.random() < np.exp(delta / T)
                        if accept:
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                            in_set[v_add] = True
                            current_size += delta
                            count_conflicts[neighbors[v_add]] += 1
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()
                                stuck_counter = 0
                            else:
                                stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            T = max(min_T, T * cooling)
            
            if stuck_counter > 50:
                T = 5.0
                stuck_counter = 0

        return local_best_in_set, local_best_size

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    in_set = greedy_init()
    size = in_set.sum()
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
        
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 2:
        ls_time = min(remaining_time * 0.4, 15.0)
        ls_result, ls_size = local_search_grouped(best_in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            sa_time = min(remaining_time * 0.5, 10.0)
            sa_result, sa_size = sa_local_search(best_in_set, max_time=sa_time)
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()

    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 30:
        n_restarts += 1
        in_set = greedy_init(seed=42 + n_restarts)
        size = in_set.sum()
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
            
        if n_restarts % 2 == 0:
            ls_time = min(remaining_time * 0.3, 8.0)
            ls_result, ls_size = local_search_grouped(in_set, max_time=ls_time)
            if ls_size > best_size:
                best_size = ls_size
                best_in_set = ls_result.copy()
        else:
            sa_time = min(remaining_time * 0.3, 8.0)
            sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()

    # Final polish: add any 0-conflict vertices
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
                remaining = np.where(~best_in_set)[0]
                if len(remaining) > 0:
                    conflict_count[remaining] += conflicts[remaining, v].astype(np.int32)
    
    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]
    return result_vectors

# EVOLVE-BLOCK-END
