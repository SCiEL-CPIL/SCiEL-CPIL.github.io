import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors (weight 4, entries in {-1, 0, 1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2, 0, 2}, norm^2=4)
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

    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor bitsets
    # Python integers can hold up to 2^N-1. N=5280 is fine.
    neighbor_bitsets = [0] * N4
    for i in range(N4):
        nbrs = np.where(conflicts[i])[0]
        mask = 0
        for j in nbrs:
            mask |= (1 << j)
        neighbor_bitsets[i] = mask

    rng = np.random.default_rng(42)

    best_size = 0
    best_mask = 0

    def greedy_random():
        """Greedy independent set with random selection using bitsets."""
        in_set_mask = 0
        not_in_set_mask = (1 << N4) - 1
        size = 0

        while not_in_set_mask != 0 and time.time() < deadline - 5:
            # Get list of candidates
            candidates = []
            temp_mask = not_in_set_mask
            while temp_mask:
                # Find lowest set bit
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                candidates.append(idx)
                temp_mask ^= lsb
                if len(candidates) > 1000: # Limit list size for speed
                    break
            
            if not candidates:
                break

            # Pick random candidate
            v = candidates[rng.integers(len(candidates))]
            
            # Add v
            in_set_mask |= (1 << v)
            size += 1
            
            # Remove v and its neighbors from not_in_set
            to_remove = (1 << v) | neighbor_bitsets[v]
            not_in_set_mask &= ~to_remove

        return in_set_mask, size

    def local_search_bitset(init_mask, max_time=10.0):
        """SA using bitsets."""
        in_set_mask = init_mask
        not_in_set_mask = ((1 << N4) - 1) & ~in_set_mask
        current_size = bin(init_mask).count('1')
        
        T = 2.0
        min_T = 0.01
        cooling = 0.999
        stuck_counter = 0
        local_best_size = current_size
        local_best_mask = init_mask
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            if not_in_set_mask == 0:
                break
            
            # Sample candidates from not_in_set_mask
            # To speed up, we can pick a random subset if mask is large
            # But bit operations are fast. Let's just pick a few random candidates.
            
            # Get a few random candidates
            # Efficient way to pick random set bits:
            # Use a random number to jump? 
            # Simpler: extract a few bits.
            
            candidates = []
            temp_mask = not_in_set_mask
            # Pick up to 50 candidates
            while temp_mask and len(candidates) < 50:
                # Random jump is hard with standard int ops. 
                # Just take first 50? No, that's biased.
                # Let's use a different strategy: 
                # Pick a random index and check if it's in mask?
                # N=5280. 
                # Let's just extract all candidates if not too many, or sample.
                pass
            
            # Actually, for N=5280, extracting all candidates every step is expensive.
            # Let's maintain a list of not_in_set indices? 
            # Bitset is good for updates, bad for random sampling.
            # Hybrid: Maintain a list `not_in_list` and a bitset `not_in_mask`.
            
            # Let's restart with Hybrid approach for better sampling.
            break 

        return local_best_mask, local_best_size

    # Hybrid Approach: Bitset for conflicts, List for sampling.
    
    def local_search_hybrid(init_mask, max_time=10.0):
        in_set_mask = init_mask
        not_in_set_mask = ((1 << N4) - 1) & ~in_set_mask
        current_size = bin(init_mask).count('1')
        
        # Build list from mask
        not_in_list = []
        temp_mask = not_in_set_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            not_in_list.append(idx)
            temp_mask ^= lsb
            
        # pos_in_list for O(1) removal
        pos_in_list = {v: i for i, v in enumerate(not_in_list)}
        
        T = 2.0
        min_T = 0.01
        cooling = 0.999
        stuck_counter = 0
        local_best_size = current_size
        local_best_mask = in_set_mask
        sa_deadline = time.time() + max_time

        def remove_from_list(v):
            if v in pos_in_list:
                idx = pos_in_list[v]
                last = not_in_list[-1]
                not_in_list[idx] = last
                pos_in_list[last] = idx
                not_in_list.pop()
                del pos_in_list[v]

        def add_to_list(v):
            if v not in pos_in_list:
                pos_in_list[v] = len(not_in_list)
                not_in_list.append(v)

        while time.time() < sa_deadline:
            if len(not_in_list) == 0:
                break
            
            # Sample candidates
            sample_size = min(50, len(not_in_list))
            sample_indices = rng.choice(len(not_in_list), size=sample_size, replace=False)
            sample = [not_in_list[i] for i in sample_indices]
            
            # Calculate "cost" (number of conflicts with in_set) for sample
            # cost(v) = popcount(neighbor_bitsets[v] & in_set_mask)
            costs = []
            for v in sample:
                c = bin(neighbor_bitsets[v] & in_set_mask).count('1')
                costs.append(c)
            
            min_cost = min(costs)
            
            if min_cost == 0:
                # Add a zero-cost node
                zero_cands = [sample[i] for i, c in enumerate(costs) if c == 0]
                v = zero_cands[rng.integers(len(zero_cands))]
                
                in_set_mask |= (1 << v)
                current_size += 1
                
                # Update not_in_set
                to_remove_mask = (1 << v) | neighbor_bitsets[v]
                # Remove from list
                # Iterate through neighbors and remove from list if present
                # This is O(degree)
                nbrs = np.where(conflicts[v])[0]
                for u in nbrs:
                    if u in pos_in_list:
                        remove_from_list(int(u))
                remove_from_list(v)
                
                not_in_set_mask &= ~to_remove_mask
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_mask = in_set_mask
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue
            
            # SA Step
            # Pick node with min cost or weighted
            # If min_cost > 0, we must remove some nodes.
            # Delta = 1 - min_cost
            # Acceptance prob
            
            # Pick the node with min cost (greedy) or random among low cost?
            # Let's pick random among sample
            v = sample[rng.integers(len(sample))]
            cost = costs[sample.index(v)] # Inefficient, but sample is small
            
            delta = 1 - cost
            
            accept = False
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                if delta < -20 * T:
                    accept = False
                else:
                    accept = rng.random() < np.exp(delta / T)
            
            if accept:
                # Remove conflicting nodes
                conflict_mask = neighbor_bitsets[v] & in_set_mask
                # Get indices of conflicting nodes
                temp_mask = conflict_mask
                to_remove_list = []
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    idx = lsb.bit_length() - 1
                    to_remove_list.append(idx)
                    temp_mask ^= lsb
                
                for u in to_remove_list:
                    in_set_mask &= ~(1 << u)
                    current_size -= 1
                    add_to_list(u)
                
                in_set_mask |= (1 << v)
                current_size += 1
                
                # Update not_in_set for v's neighbors
                nbrs = np.where(conflicts[v])[0]
                for u in nbrs:
                    if u in pos_in_list:
                        remove_from_list(int(u))
                remove_from_list(v)
                
                to_remove_mask = (1 << v) | neighbor_bitsets[v]
                not_in_set_mask &= ~to_remove_mask
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_mask = in_set_mask
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
                
            # Perturbation
            if stuck_counter > 1000:
                # Randomly remove some nodes from in_set
                # Get list of in_set nodes
                in_list = []
                temp_mask = in_set_mask
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    idx = lsb.bit_length() - 1
                    in_list.append(idx)
                    temp_mask ^= lsb
                    if len(in_list) > 1000: break
                
                if in_list:
                    n_remove = min(5, len(in_list))
                    to_perturb = rng.choice(in_list, size=n_remove, replace=False)
                    for u in to_perturb:
                        in_set_mask &= ~(1 << u)
                        current_size -= 1
                        add_to_list(int(u))
                        # Update mask
                        not_in_set_mask |= (1 << u)
                        # Note: neighbors of u might become available? 
                        # No, we only remove from in_set. 
                        # The "not_in_set" mask should include u.
                        # But what about neighbors of u that were removed because of u?
                        # They are already in not_in_set_list? 
                        # Actually, if u was in set, its neighbors were NOT in set (independent set).
                        # So neighbors are already in not_in_set.
                        # So we just add u to not_in_set.
                    
                    stuck_counter = 0
                    T = max(T, 2.0)
            
            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_mask, local_best_size

    # Main Loop
    n_restarts = 0
    max_restarts = 50

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1

        # Greedy Init
        in_set_mask, current_size = greedy_random()
        
        if current_size > best_size:
            best_size = current_size
            best_mask = in_set_mask

        # Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break

        ls_time = min(remaining_time * 0.5, 12.0)
        if ls_time < 1:
            ls_time = 1.0

        ls_mask, ls_size = local_search_hybrid(in_set_mask, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_mask = ls_mask

    # Final Polish
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1 and best_size > 0:
        polish_time = min(remaining_time * 0.8, 5.0)
        if polish_time > 0.5:
            final_mask, final_size = local_search_hybrid(best_mask, max_time=polish_time)
            if final_size > best_size:
                best_size = final_size
                best_mask = final_mask

    # Construct final array
    # Extract indices from best_mask
    final_indices = []
    temp_mask = best_mask
    while temp_mask:
        lsb = temp_mask & -temp_mask
        idx = lsb.bit_length() - 1
        final_indices.append(idx)
        temp_mask ^= lsb

    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
