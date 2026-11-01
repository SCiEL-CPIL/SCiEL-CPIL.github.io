"""Rewritten program using model-and-optimize approach for kissing number in d=11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate all candidate vectors with norm^2 = 4
    # Type A: weight 4, entries in {-1,0,1}, norm^2 = 4
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Type B: weight 1, entry ±2, norm^2 = 4
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

    # Compute dot product matrix
    dots = all_candidates @ all_candidates.T
    # All vectors have norm^2 = 4, so ||u||*||v|| = 4, threshold = 4/2 = 2
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists for fast access if needed, but we will mostly use matrix
    # neighbor_counts = np.sum(conflicts, axis=1)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    def greedy_min_degree():
        """Greedy independent set using min-degree heuristic, vectorized where possible."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        # degrees[i] is the number of neighbors of i that are currently alive
        # Initially, it's the degree in the full graph
        degrees = np.sum(conflicts, axis=1)

        while np.any(alive) and time.time() < deadline - 5:
            # Find alive vertices
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            # Find min degree among alive vertices
            min_deg = np.min(degrees[alive_indices])
            # Get all vertices with this min degree
            cands = alive_indices[degrees[alive_indices] == min_deg]
            
            # Randomly pick one from candidates
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            alive[chosen] = False
            
            # Remove neighbors of chosen from alive
            nbrs = np.where(conflicts[chosen])[0]
            alive[nbrs] = False
            
            # Update degrees of remaining alive vertices
            # A vertex i's degree decreases by the number of its neighbors that are in {chosen} U nbrs
            # But since we just removed nbrs from alive, we only care about alive vertices.
            # For each alive vertex i, deg_new[i] = deg_old[i] - (number of neighbors of i that are in nbrs)
            # Note: chosen is not a neighbor of i if i is alive (since nbrs are removed), 
            # wait, if i is alive, i is not in nbrs and i != chosen.
            # So we just subtract the overlap between neighbors[i] and nbrs.
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Vectorized update: 
                # We need to count for each i in rem, how many j in nbrs are neighbors of i.
                # This is summing conflicts[i, nbrs] for i in rem.
                # conflicts is NxN boolean. 
                # We can do: degrees[rem] -= conflicts[rem][:, nbrs].sum(axis=1)
                # This creates a temporary array of size len(rem) x len(nbrs).
                # If nbrs is large, this is expensive. 
                # Alternative: loop over nbrs and update degrees? No, loop over rem is better if rem is small?
                # Actually, conflicts[rem][:, nbrs].sum(axis=1) is quite efficient in NumPy.
                degrees[rem] -= conflicts[rem][:, nbrs].sum(axis=1)

        return in_set

    def local_search_sa(in_set_init, max_time=3.0):
        """
        Simulated Annealing style local search.
        Objective: Maximize size of independent set.
        Moves: 
        1. Add vertex v if conflict_count[v] == 0. (Always accept)
        2. Swap: Remove u from set, Add v to set. 
           Gain = 1 - 1 = 0 in size, but might allow future additions.
           Actually, let's define energy as -size. 
           Or just track size.
           Let's use a temperature based acceptance for swaps that don't increase size immediately 
           but might lead to better configurations.
           However, a simpler and effective approach for independent sets is:
           - Try to add v if 0 conflicts.
           - If stuck, try 1-for-1 swap: remove u, add v. 
             This is valid if v's only conflict in the set is u.
             This keeps size constant but changes the set.
             We accept if it leads to a state with more "potential" (e.g. lower sum of degrees of new set members? 
             Or just random acceptance with cooling).
        """
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        # Precompute conflict counts
        # conflict_count[i] = number of neighbors of i that are in in_set
        in_set_indices = np.where(in_set)[0]
        if len(in_set_indices) > 0:
            conflict_count = np.zeros(N, dtype=np.int32)
            # Vectorized: sum conflicts of set members
            # conflict_count = conflicts[in_set_indices].sum(axis=0)
            # This sums the rows of conflicts corresponding to in_set members.
            # Result is a vector of length N where entry i is the number of in_set members adjacent to i.
            conflict_count = conflicts[in_set_indices].sum(axis=0)
        else:
            conflict_count = np.zeros(N, dtype=np.int32)

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        ls_deadline = time.time() + max_time
        no_improve = 0
        
        while time.time() < ls_deadline and T > T_min:
            improved = False
            
            # Phase 1: Add vertices with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                zero_conf = not_in_set[conflict_count[not_in_set] == 0]
                if len(zero_conf) > 0:
                    # Add all zero conflict vertices? Or one by one?
                    # Adding all is safe and fast.
                    in_set[zero_conf] = True
                    current_size += len(zero_conf)
                    improved = True
                    
                    # Update conflict counts for neighbors of added vertices
                    # conflict_count[neighbors[v]] += 1 for each v in zero_conf
                    # Vectorized:
                    # We need to add 1 to conflict_count for all neighbors of all v in zero_conf.
                    # This is equivalent to adding the sum of rows of conflicts for zero_conf.
                    if len(zero_conf) > 0:
                        delta = conflicts[zero_conf].sum(axis=0)
                        conflict_count += delta
                        
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                        
                    # If we added vertices, we might have new zero conflict vertices?
                    # No, adding vertices only increases conflict counts for others.
                    # So the set of zero conflict vertices can only shrink.
                    # We can stop adding if no more zero conflicts.
                    
            # Phase 2: Swaps
            if time.time() < ls_deadline:
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    # Try random swaps
                    # We want to find a pair (u, v) such that:
                    # 1. u in set, v not in set
                    # 2. v conflicts with u
                    # 3. v does NOT conflict with any other u' in set
                    # If such a pair exists, swapping u for v keeps size same.
                    # Why do it? To change the "structure" of the set to allow more additions later.
                    
                    # Optimization: 
                    # For each v in not_in_set, let Conf_v = set of neighbors of v in in_set.
                    # If len(Conf_v) == 1, let u be the unique element.
                    # Then swapping u and v is a valid 1-for-1 swap.
                    
                    # We can find all such v efficiently.
                    # conflict_count[v] is the number of neighbors of v in in_set.
                    # Candidates for v: not_in_set[conflict_count[not_in_set] == 1]
                    
                    swap_candidates = not_in_set[conflict_count[not_in_set] == 1]
                    
                    if len(swap_candidates) > 0:
                        # Pick a random candidate v
                        v = swap_candidates[rng.integers(len(swap_candidates))]
                        
                        # Find the unique neighbor u in set
                        # u is the only index in in_set such that conflicts[v, u] is True
                        # We can find it by:
                        # u_idx = np.where(conflicts[v] & in_set)[0]
                        # len(u_idx) should be 1
                        u_indices = np.where(conflicts[v] & in_set)[0]
                        if len(u_indices) == 1:
                            u = u_indices[0]
                            
                            # Perform swap
                            in_set[u] = False
                            in_set[v] = True
                            
                            # Update conflict_count
                            # Removing u: decrement count for neighbors of u
                            # Adding v: increment count for neighbors of v
                            # Note: v's neighbors include u, and u's neighbors include v.
                            # Also, v's other neighbors had their count incremented by 1 (from v being added)
                            # and u's other neighbors had their count decremented by 1 (from u being removed).
                            
                            # delta = -conflicts[u] + conflicts[v]
                            # conflict_count += delta
                            
                            # But we must be careful: 
                            # conflict_count[i] = sum_{j in in_set} conflicts[i, j]
                            # New in_set = (old in_set \ {u}) U {v}
                            # New conflict_count[i] = old conflict_count[i] - conflicts[i, u] + conflicts[i, v]
                            
                            conflict_count -= conflicts[u]
                            conflict_count += conflicts[v]
                            
                            # Acceptance: 
                            # Size is unchanged. 
                            # We accept based on "quality" metric or random.
                            # A good metric: sum of degrees of set members? 
                            # Or just accept with probability exp(-delta_energy / T).
                            # Let's define energy as -local_density or similar.
                            # Simple: Accept with probability 0.5 or based on cooling.
                            # Since size is same, we use a heuristic: 
                            # Does this swap reduce the "average conflict count" of the set?
                            # Or just random accept to explore.
                            
                            # Let's accept if rng.random() < T
                            if rng.random() < T:
                                improved = True
                                # Update best if size increased? No, size is same.
                                # But it might lead to better size later.
                            else:
                                # Revert
                                in_set[u] = True
                                in_set[v] = False
                                conflict_count -= conflicts[v]
                                conflict_count += conflicts[u]
                                
            if not improved:
                no_improve += 1
                if no_improve > 50:
                    # Perturb: remove 2-3 random vertices to allow larger rearrangement
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) >= 2:
                        n_remove = min(3, len(in_set_indices))
                        to_remove = in_set_indices[rng.choice(len(in_set_indices), size=n_remove, replace=False)]
                        for u in to_remove:
                            in_set[u] = False
                            current_size -= 1
                            conflict_count -= conflicts[u]
                        no_improve = 0
                    else:
                        break
            else:
                no_improve = 0
                
            T *= cooling_rate

        # Final polish: add any remaining zero-conflict vertices
        not_in_set = np.where(~in_set)[0]
        if len(not_in_set) > 0:
            zero_conf = not_in_set[conflict_count[not_in_set] == 0]
            if len(zero_conf) > 0:
                in_set[zero_conf] = True
                current_size += len(zero_conf)
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()

        return local_best_in_set, local_best_size

    # Multiple restarts
    n_restarts = 0
    max_restarts = 500

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        # Greedy initialization
        in_set = greedy_min_degree()
        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # Local search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 1.5:
            break

        ls_time = min(remaining_time * 0.4, 3.0)
        if ls_time < 0.5:
            ls_time = 0.5

        ls_result, ls_size = local_search_sa(in_set, max_time=ls_time)

        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

        if time.time() > deadline - 10:
            break

    # Final result
    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]

    return result_vectors

# EVOLVE-BLOCK-END
