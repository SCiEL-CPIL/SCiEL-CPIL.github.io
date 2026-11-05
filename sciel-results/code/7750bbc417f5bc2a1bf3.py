import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # 1. Precomputation
    # Generate all triplets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage matrix: (n_blocks, n_triplets) of uint8
    # A[b, i] = 1 if block b covers triplet i
    # This is 34668 * 2024 bytes ~ 70MB. Acceptable.
    A = np.zeros((n_blocks, n_triplets), dtype=np.uint8)
    for b_idx, block in enumerate(all_blocks):
        # Iterate over triplets in this block
        for triplet in itertools.combinations(block, t):
            A[b_idx, triplet_to_id[triplet]] = 1
            
    # Precompute number of triplets covered by each block (for initial greedy)
    block_sizes = A.sum(axis=1)
    
    # Helper to check time
    def time_check():
        return time.time() > deadline - 10.0

    # 2. Initial Greedy Cover
    def greedy_cover():
        counts = np.zeros(n_triplets, dtype=np.int16)
        selected = []
        # We need to update counts and check which are uncovered
        # Uncovered mask: counts == 0
        # Gain of block b = sum(A[b, counts==0])
        
        # To speed up, we can maintain a list of uncovered triplets?
        # Or just compute gain = A[b] @ (counts == 0)
        
        while True:
            if time_check():
                break
            uncovered_mask = (counts == 0)
            if not np.any(uncovered_mask):
                break
                
            # Calculate gain for all blocks
            # gain[b] = sum(A[b, uncovered_mask])
            # This is A[:, uncovered_mask].sum(axis=1)
            # This is expensive if done from scratch every time.
            # But n_blocks is 34k, n_triplets 2k.
            # A is (34k, 2k).
            # A[:, uncovered_mask] is a view/slice? No, boolean indexing creates a copy.
            # 34k * (num_uncovered) operations.
            # Initially num_uncovered is 2024.
            # 34k * 2024 ~ 70M ops. NumPy handles this fast.
            
            gains = A[:, uncovered_mask].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Pick block with max gain (random tie-break)
            candidates = np.where(gains == max_gain)[0]
            b_idx = random.choice(candidates)
            
            selected.append(b_idx)
            counts += A[b_idx]
            
        return selected

    initial_cover = greedy_cover()
    
    # 3. Local Search / Simulated Annealing
    # Objective: Minimize |S| subject to coverage.
    # Penalty method: Cost = |S| + P * (number of uncovered triplets)
    P = 10000 # Large penalty to ensure validity is prioritized over size in early stages
    
    def get_cost(selected_indices, counts):
        uncovered = np.sum(counts == 0)
        return len(selected_indices) + P * uncovered

    # Initialize state
    current_selected = list(initial_cover)
    current_counts = np.zeros(n_triplets, dtype=np.int16)
    for b in current_selected:
        current_counts += A[b]
    
    current_cost = get_cost(current_selected, current_counts)
    best_selected = list(current_selected)
    best_cost = current_cost
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.999 # Adjust based on time
    # Estimate steps: We have ~100s. 
    # Each step involves O(2024) ops for counts update and O(2024) for cost check.
    # ~4000 ops. 100s / (4000 * 1e-7 s/op) ? 
    # Python overhead is higher. Let's just loop until time runs out.
    
    rng = random.Random(42)
    
    # Precompute indices for random selection
    all_block_indices = list(range(n_blocks))
    
    while time.time() < deadline - 20.0: # Leave 20s for final polish
        T = T_start * (T_end / T_start) ** ((time.time() - start_time) / (deadline - start_time))
        if T < T_end:
            T = T_end
            
        # Choose a random move
        move_type = rng.choice(['add', 'remove', 'swap'])
        
        new_selected = list(current_selected)
        new_counts = current_counts.copy()
        
        if move_type == 'add':
            # Add a random block
            b = rng.choice(all_block_indices)
            if b in new_selected:
                continue
            new_selected.append(b)
            new_counts += A[b]
            
        elif move_type == 'remove':
            if not current_selected:
                continue
            # Remove a random block
            idx = rng.randrange(len(current_selected))
            b = current_selected[idx]
            new_selected.pop(idx)
            new_counts -= A[b]
            
        else: # swap
            if not current_selected:
                continue
            # Remove one, add one
            idx = rng.randrange(len(current_selected))
            b_old = current_selected[idx]
            new_selected.pop(idx)
            new_counts -= A[b_old]
            
            b_new = rng.choice(all_block_indices)
            if b_new in new_selected:
                # Revert
                new_selected.insert(idx, b_old)
                new_counts += A[b_old]
                continue
                
            new_selected.append(b_new)
            new_counts += A[b_new]
            
        new_cost = get_cost(new_selected, new_counts)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_selected = new_selected
            current_counts = new_counts
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_selected = list(current_selected)
                
    # 4. Final Polish: Deterministic Local Search
    # Ensure validity first
    current_selected = list(best_selected)
    current_counts = np.zeros(n_triplets, dtype=np.int16)
    for b in current_selected:
        current_counts += A[b]
        
    # Greedy repair if invalid
    while np.any(current_counts == 0):
        if time_check():
            break
        uncovered_mask = (current_counts == 0)
        gains = A[:, uncovered_mask].sum(axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        b = rng.choice(candidates)
        current_selected.append(b)
        current_counts += A[b]
        
    # Greedy removal of redundant blocks
    improved = True
    while improved and not time_check():
        improved = False
        # Check each block if it's redundant
        # A block is redundant if removing it doesn't change the coverage of any triplet
        # i.e., for all t where A[b,t]==1, current_counts[t] > 1
        
        # To be efficient, iterate in random order
        indices = list(range(len(current_selected)))
        rng.shuffle(indices)
        
        for i in indices:
            b = current_selected[i]
            # Check if redundant
            # ones = np.where(A[b] == 1)[0]
            # if np.all(current_counts[ones] > 1):
            #    remove
            
            # Vectorized check:
            # mask = A[b] == 1
            # if np.all(current_counts[mask] > 1):
            
            mask = A[b]
            if np.all(current_counts[mask] > 1):
                # Remove
                current_selected.pop(i)
                current_counts -= A[b]
                improved = True
                break # Restart loop as indices changed
                
    # 2-for-1 Swap Local Search
    # Try to replace 2 blocks with 1
    # This is O(N^2 * N) if done naively.
    # N is ~70. 70^3 = 343k.
    # For each pair (i, j), compute deficit D = max(0, 1 - (counts - A[i] - A[j]))
    # Then find if any block b covers D.
    
    # We do this a few times or until no improvement
    
    def try_2_for_1_swaps(selected, counts):
        n = len(selected)
        if n < 2:
            return selected, counts, False
        
        # Precompute column sums? No, counts is the column sum.
        # For pair (i, j):
        # counts_without = counts - A[selected[i]] - A[selected[j]]
        # deficit = (counts_without == 0)
        # We need a block b such that A[b] covers all 1s in deficit.
        # i.e., A[b] & deficit == deficit
        
        # To speed up, we can iterate over pairs.
        # Randomize order.
        indices = list(range(n))
        rng.shuffle(indices)
        
        improved = False
        for idx_i in indices:
            if time_check():
                break
            for idx_j in indices:
                if idx_j <= idx_i:
                    continue
                if time_check():
                    break
                    
                b_i = selected[idx_i]
                b_j = selected[idx_j]
                
                # Calculate deficit
                # counts_without_ij = counts - A[b_i] - A[b_j]
                # deficit_mask = (counts_without_ij == 0)
                # Note: counts is int16. A is uint8.
                # We need to be careful with types.
                
                # Let's use a temporary array to avoid modifying counts
                # temp_counts = counts - A[b_i] - A[b_j]
                # deficit_indices = np.where(temp_counts == 0)[0]
                
                # If deficit is empty, both are redundant (handled by previous step ideally)
                # If not, we need a block covering deficit_indices.
                
                # Optimization:
                # The union of triplets covered by b_i and b_j that are NOT covered by others
                # is the deficit.
                
                # Let's just compute the mask of triplets that would be uncovered.
                # uncovered_mask = (counts - A[b_i] - A[b_j] == 0)
                
                # We need to find if there is a block b such that A[b, uncovered_mask] are all 1s.
                # i.e., sum(A[b, uncovered_mask]) == sum(uncovered_mask)
                
                # This is expensive to check for all b.
                # However, the number of uncovered triplets is small (usually).
                # If the number of uncovered triplets is 0, we can just remove both.
                
                temp_counts = counts - A[b_i] - A[b_j]
                deficit_mask = (temp_counts == 0)
                
                if not np.any(deficit_mask):
                    # Both redundant
                    # Remove both
                    keep = [x for x in range(n) if x != idx_i and x != idx_j]
                    new_selected = [selected[x] for x in keep]
                    new_counts = counts - A[b_i] - A[b_j]
                    return new_selected, new_counts, True
                    
                # Find block covering deficit
                # deficit_indices = np.where(deficit_mask)[0]
                # target_size = len(deficit_indices)
                
                # Gains for all blocks on deficit
                # gains = A[:, deficit_mask].sum(axis=1)
                # valid_blocks = np.where(gains == target_size)[0]
                
                # This is O(N_blocks * deficit_size).
                # deficit_size is likely small.
                
                gains = A[:, deficit_mask].sum(axis=1)
                target_size = np.sum(deficit_mask)
                
                valid_blocks = np.where(gains == target_size)[0]
                
                # Exclude current blocks?
                # If b_i or b_j is in valid_blocks, it's not a "new" block, but we are removing them.
                # So we just need ANY block from valid_blocks that is not b_i or b_j?
                # Actually, if b_i covers the deficit, it means b_i covers all triplets that become uncovered.
                # But b_i is being removed. So b_i cannot cover them after removal.
                # Wait, the deficit is defined as triplets uncovered by (all \ {i, j}).
                # If b_i covers a triplet t in deficit, it means b_i covers t.
                # But we are removing b_i. So t is uncovered.
                # So we need a block b (not i, not j) that covers all t in deficit.
                
                # Filter valid_blocks to exclude b_i and b_j
                valid_blocks = [b for b in valid_blocks if b != b_i and b != b_j]
                
                if valid_blocks:
                    # Pick one
                    b_new = valid_blocks[0]
                    
                    # Apply swap
                    keep = [x for x in range(n) if x != idx_i and x != idx_j]
                    new_selected = [selected[x] for x in keep]
                    new_selected.append(b_new)
                    
                    new_counts = counts - A[b_i] - A[b_j] + A[b_new]
                    
                    return new_selected, new_counts, True
                    
        return selected, counts, False

    # Run 2-for-1 swaps until no improvement
    improved = True
    while improved and not time_check():
        current_selected, current_counts, improved = try_2_for_1_swaps(current_selected, current_counts)

    # Final redundancy check
    improved = True
    while improved and not time_check():
        improved = False
        indices = list(range(len(current_selected)))
        rng.shuffle(indices)
        for i in indices:
            b = current_selected[i]
            mask = A[b]
            if np.all(current_counts[mask] > 1):
                current_selected.pop(i)
                current_counts -= A[b]
                improved = True
                break

    # Convert to list of lists
    result = [list(all_blocks[b_idx]) for b_idx in current_selected]
    
    # Final verification
    covered = set()
    for block in result:
        for triplet in itertools.combinations(block, t):
            covered.add(triplet)
            
    if len(covered) < n_triplets:
        # Fallback to initial cover if something went wrong
        return [list(all_blocks[b_idx]) for b_idx in initial_cover]
        
    return result
