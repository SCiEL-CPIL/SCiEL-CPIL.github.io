import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave margin for return overhead

    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Map each block to the indices of t-subsets it covers
    # And map each t-subset to the blocks that cover it
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_indices = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_indices.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_indices)
        
    # Convert to numpy arrays for faster operations where possible
    # But lists of lists are often faster for membership checks in small sets
    # Let's keep block_to_ts as list of lists of ints
    # ts_to_blocks as list of lists of ints

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def optimize_redundancy(blocks):
        """Remove blocks that are not uniquely covering any t-subset."""
        if not blocks:
            return []
        
        # Count coverage
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        valid_indices = []
        
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_indices.append(idx)
        
        n_blocks = len(block_indices)
        if n_blocks == 0:
            return []

        # Initial cover count
        for bidx in block_indices:
            for ts_idx in block_to_ts[bidx]:
                cover_count[ts_idx] += 1

        changed = True
        while changed:
            if time.time() > deadline - 2.0:
                break
            changed = False
            new_block_indices = []
            for bidx in block_indices:
                # Check if this block covers any t-subset with count 1
                is_unique = False
                for ts_idx in block_to_ts[bidx]:
                    if cover_count[ts_idx] == 1:
                        is_unique = True
                        break
                if is_unique:
                    new_block_indices.append(bidx)
                else:
                    # Remove this block
                    for ts_idx in block_to_ts[bidx]:
                        cover_count[ts_idx] -= 1
                    changed = True
            
            block_indices = new_block_indices
            if not block_indices:
                break

        return [list(all_k_blocks[idx]) for idx in block_indices]

    def greedy(seed=0, explore=0.0):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        # Pre-compute block covers as arrays for speed in gain update?
        # The current gain update is O(num_t_subsets_in_block * num_blocks_covering_those_ts)
        # Which is roughly O(56 * 100) per block added.
        
        while True:
            if time.time() > deadline - 8.0:
                return None
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            if explore > 0 and random.random() < explore:
                min_gain = max(0, max_gain - 3)
                candidates = np.where((gain >= min_gain) & (~is_selected))[0]
            else:
                candidates = np.where((gain == max_gain) & (~is_selected))[0]
            
            if len(candidates) == 0:
                break
                
            b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            newly_covered = [ts_idx for ts_idx in block_to_ts[b_idx] if is_uncovered[ts_idx]]
            if len(newly_covered) > 0:
                for ts_idx in newly_covered:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
        return sol

    def local_search_1swap(blocks, max_rounds=20):
        """Try to replace one block with another to maintain coverage."""
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
        
        improved = False
        n = len(blocks)
        if n == 0:
            return blocks, False
            
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            round_improved = False
            order = list(range(n))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 3.0:
                    break
                old_b_idx = get_block_idx(blocks[i])
                if old_b_idx < 0:
                    continue
                
                # Find t-subsets uniquely covered by this block
                unique_ts = [ts_idx for ts_idx in block_to_ts[old_b_idx] if cover_count[ts_idx] == 1]
                if not unique_ts:
                    # This block is redundant, handled by optimize_redundancy
                    continue
                
                # Find candidates that cover ALL unique_ts
                # Start with the set of blocks covering the first unique ts
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Exclude current solution blocks
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                if not valid_candidates:
                    continue
                
                # Pick the best candidate (max new coverage)
                best_new = None
                best_gain = -1
                # Limit candidates to speed up
                if len(valid_candidates) > 20:
                    valid_candidates = random.sample(valid_candidates, 20)
                
                for c in valid_candidates:
                    # Gain = number of t-subsets in c that are currently uncovered (count 0)
                    gain_val = 0
                    for ts_idx in block_to_ts[c]:
                        if cover_count[ts_idx] == 0:
                            gain_val += 1
                    if gain_val > best_gain:
                        best_gain = gain_val
                        best_new = c
                
                if best_new is None:
                    continue
                    
                # Check validity: Does best_new cover all unique_ts? 
                # Yes, by construction.
                # Does it maintain coverage for all other ts in old_b_idx?
                # If old_b_idx had count > 1 for some ts, removing it makes count >= 1.
                # If we add best_new, counts increase.
                # So coverage is maintained.
                
                # Apply swap
                for ts_idx in block_to_ts[old_b_idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[best_new]:
                    cover_count[ts_idx] += 1
                    
                sol_block_indices.discard(old_b_idx)
                sol_block_indices.add(best_new)
                blocks[i] = list(all_k_blocks[best_new])
                round_improved = True
                improved = True
                
            if not round_improved:
                break
                
        return blocks, improved

    def local_search_2exchange(blocks):
        """Try to replace two blocks with one block."""
        if len(blocks) < 2:
            return blocks, False
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
                    
        improved = False
        n = len(blocks)
        
        for attempt in range(50):
            if time.time() > deadline - 2.0:
                break
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
                
            old_i_idx = get_block_idx(blocks[i])
            old_j_idx = get_block_idx(blocks[j])
            if old_i_idx < 0 or old_j_idx < 0:
                continue
                
            # Union of t-subsets uniquely covered by i or j
            unique_ts = []
            for ts_idx in block_to_ts[old_i_idx]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
            for ts_idx in block_to_ts[old_j_idx]:
                if cover_count[ts_idx] == 1 and ts_idx not in unique_ts:
                    unique_ts.append(ts_idx)
                    
            if not unique_ts:
                continue
                
            # Find a single block that covers all these unique_ts
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
                    
            if not candidates:
                continue
                
            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            if not valid_candidates:
                continue
                
            best_new = random.choice(valid_candidates)
            
            # Perform swap
            for ts_idx in block_to_ts[old_i_idx]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[old_j_idx]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[best_new]:
                cover_count[ts_idx] += 1
                
            sol_block_indices.discard(old_i_idx)
            sol_block_indices.discard(old_j_idx)
            sol_block_indices.add(best_new)
            
            # Update blocks list
            if j > i:
                blocks.pop(j)
                blocks[i] = list(all_k_blocks[best_new])
            else:
                blocks.pop(i)
                blocks[j] = list(all_k_blocks[best_new])
            n -= 1
            improved = True
            break # Restart or continue? Restart is safer for state consistency
            
        return blocks, improved

    def perturb(blocks, num_perturb=2):
        """Randomly swap some blocks to escape local minima."""
        blocks = [b[:] for b in blocks]
        n = len(blocks)
        if n == 0:
            return blocks
            
        for _ in range(num_perturb):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            # Swap i and j
            blocks[i], blocks[j] = blocks[j], blocks[i]
            
        return blocks

    # Phase 1: Initial Greedy
    best_solution = greedy(seed=0, explore=0.0)
    best_len = len(best_solution)
    best_solution = optimize_redundancy(best_solution)
    best_len = len(best_solution)

    # Phase 2: Local Search Improvement
    improved = True
    while improved and time.time() < deadline - 20.0:
        improved = False
        best_solution, imp1 = local_search_1swap(best_solution, max_rounds=10)
        if imp1:
            improved = True
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        
        if time.time() < deadline - 20.0:
            best_solution, imp2 = local_search_2exchange(best_solution)
            if imp2:
                improved = True
                best_solution = optimize_redundancy(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)

    # Phase 3: Iterated Local Search with Perturbation
    # If stuck, perturb and re-optimize
    stall_count = 0
    max_stalls = 50
    
    while time.time() < deadline - 10.0 and stall_count < max_stalls:
        # Perturb
        perturbed = perturb(best_solution, num_perturb=2)
        perturbed = optimize_redundancy(perturbed)
        
        # Local Search on perturbed
        p_improved = True
        while p_improved and time.time() < deadline - 10.0:
            p_improved = False
            perturbed, imp1 = local_search_1swap(perturbed, max_rounds=5)
            if imp1:
                p_improved = True
                perturbed = optimize_redundancy(perturbed)
                
            if time.time() < deadline - 10.0:
                perturbed, imp2 = local_search_2exchange(perturbed)
                if imp2:
                    p_improved = True
                    perturbed = optimize_redundancy(perturbed)
                    
        if len(perturbed) < best_len:
            best_len = len(perturbed)
            best_solution = perturbed
            stall_count = 0
        else:
            stall_count += 1
            
    # Final cleanup
    best_solution = optimize_redundancy(best_solution)
    
    return best_solution
