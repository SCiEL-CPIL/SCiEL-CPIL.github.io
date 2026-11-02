import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute all k-subsets and their t-subset indices
    all_k_subsets = list(itertools.combinations(range(v), k))
    k_to_t_indices = {}
    for ks in all_k_subsets:
        t_indices = [t_index_map[ts] for ts in itertools.combinations(ks, t)]
        k_to_t_indices[ks] = t_indices
        
    # Initialize with a greedy solution to get a good starting point
    # Greedy: Pick block covering most uncovered t-subsets
    uncovered = [1] * num_t_subsets # 1 if uncovered, 0 if covered
    blocks = []
    
    # Greedy Phase
    while any(uncovered):
        best_block = None
        max_covered = 0
        # To speed up, we could sample, but for v=19, 50k blocks is okay for a few iterations
        # However, doing this 150 times is 7.5M checks.
        # Let's do a simpler greedy: pick a random uncovered t-subset, 
        # find a block covering it that covers the most other uncovered t-subsets.
        
        # Find an uncovered t-subset
        # Using list comprehension to find first uncovered is O(N)
        # We can maintain a set of uncovered indices for O(1) lookup if needed, 
        # but 'any' and 'next' are okay.
        
        # Optimization: Instead of checking all blocks, check blocks containing a random uncovered t-subset.
        # Pick random uncovered t-subset
        uncov_indices = [i for i in range(num_t_subsets) if uncovered[i]]
        if not uncov_indices:
            break
        target_t_idx = random.choice(uncov_indices)
        target_t_subset = all_t_subsets[target_t_idx]
        
        # Find all blocks containing this t-subset
        # We can precompute a reverse map: t_subset -> list of blocks containing it
        # But memory might be high. 
        # Alternative: Just iterate all blocks and check if they contain the t-subset?
        # Better: Precompute for each t-subset, which k-subsets contain it.
        
        # Let's precompute t_to_blocks
        t_to_blocks = [[] for _ in range(num_t_subsets)]
        for ks, t_indices in k_to_t_indices.items():
            for t_idx in t_indices:
                t_to_blocks[t_idx].append(ks)
                
        candidates = t_to_blocks[target_t_idx]
        best_count = 0
        for b in candidates:
            count = 0
            for t_idx in k_to_t_indices[b]:
                if uncovered[t_idx]:
                    count += 1
            if count > best_count:
                best_count = count
                best_block = b
                
        if best_block:
            blocks.append(best_block)
            for t_idx in k_to_t_indices[best_block]:
                uncovered[t_idx] = 0
                
    # Now we have a valid covering. 
    # Use Simulated Annealing to minimize block count while keeping validity (uncovered == 0).
    # Objective: Minimize len(blocks). Constraint: uncovered count == 0.
    # We will allow temporary invalid states (uncovered > 0) with a penalty.
    
    # State
    current_blocks = blocks[:]
    coverage_counts = [0] * num_t_subsets
    for b in current_blocks:
        for t_idx in k_to_t_indices[b]:
            coverage_counts[t_idx] += 1
            
    def get_uncovered_count(counts):
        # Counts of 0 mean uncovered
        return sum(1 for c in counts if c == 0)
        
    current_uncovered = get_uncovered_count(coverage_counts)
    current_cost = current_uncovered * 10000 + len(current_blocks)
    
    # SA Parameters
    start_temp = 100.0
    end_temp = 0.1
    cooling_rate = 0.995
    time_limit = 110.0 # Leave 10s buffer
    start_time = time.time()
    
    best_blocks = current_blocks[:]
    best_cost = current_cost
    
    # To speed up move evaluation, we need to update coverage counts efficiently.
    
    while time.time() - start_time < time_limit:
        if start_temp < end_temp:
            break
        start_temp *= cooling_rate
        
        # Choose a move
        # 1. Delete a block
        # 2. Add a block
        # 3. Swap a block
        
        move_type = random.random()
        
        if move_type < 0.4 and current_blocks:
            # Delete
            idx = random.randrange(len(current_blocks))
            b = current_blocks[idx]
            t_indices = k_to_t_indices[b]
            
            # Calculate delta uncovered
            delta_unc = 0
            for t_idx in t_indices:
                if coverage_counts[t_idx] == 1:
                    delta_unc += 1
            
            new_uncovered = current_uncovered + delta_unc
            new_cost = new_uncovered * 10000 + (len(current_blocks) - 1)
            
            if new_cost < current_cost or random.random() < pow(2.718, -(new_cost - current_cost) / start_temp):
                # Apply move
                for t_idx in t_indices:
                    coverage_counts[t_idx] -= 1
                current_blocks.pop(idx)
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = current_blocks[:]
                    
        elif move_type < 0.8:
            # Add
            # Pick a random block that covers at least one uncovered t-subset? 
            # Or just random block. Random block is better for exploration.
            b = random.choice(all_k_subsets)
            t_indices = k_to_t_indices[b]
            
            # Check if already in blocks? 
            # It's possible to add duplicate blocks if we are not careful, but 
            # "blocks must be distinct" is a constraint. 
            # However, in SA, we can allow duplicates temporarily or just check.
            # If we add a duplicate, it doesn't change coverage but increases cost. 
            # So it will likely be rejected.
            
            # Calculate delta uncovered (negative if it covers new ones)
            delta_unc = 0
            for t_idx in t_indices:
                if coverage_counts[t_idx] == 0:
                    delta_unc -= 1
            
            new_uncovered = current_uncovered + delta_unc
            new_cost = new_uncovered * 10000 + (len(current_blocks) + 1)
            
            if new_cost < current_cost or random.random() < pow(2.718, -(new_cost - current_cost) / start_temp):
                # Apply move
                for t_idx in t_indices:
                    coverage_counts[t_idx] += 1
                current_blocks.append(b)
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = current_blocks[:]
                    
        else:
            # Swap
            if not current_blocks:
                continue
            idx = random.randrange(len(current_blocks))
            b_old = current_blocks[idx]
            b_new = random.choice(all_k_subsets)
            
            if b_old == b_new:
                continue
            
            t_indices_old = k_to_t_indices[b_old]
            t_indices_new = k_to_t_indices[b_new]
            
            # Calculate delta
            # Remove b_old, Add b_new
            # This is complex to calculate delta directly without applying.
            # Let's apply tentatively.
            
            old_counts = coverage_counts[:] # Copy is expensive. 
            # Better: calculate delta manually.
            
            # t_idx in b_old but not in b_new: count decreases
            # t_idx in b_new but not in b_old: count increases
            # t_idx in both: count unchanged
            
            # We need to know which t_indices are in old but not new, and vice versa.
            # Since t=4, k=7, each block has 35 t-subsets.
            # Intersection can be computed.
            
            set_old = set(t_indices_old)
            set_new = set(t_indices_new)
            
            delta_unc = 0
            
            # Decreases
            for t_idx in set_old - set_new:
                if coverage_counts[t_idx] == 1:
                    delta_unc += 1
            
            # Increases
            for t_idx in set_new - set_old:
                if coverage_counts[t_idx] == 0:
                    delta_unc -= 1
            
            # Note: If a t_idx is in both, count is unchanged.
            # If a t_idx is in neither, count is unchanged.
            
            new_uncovered = current_uncovered + delta_unc
            new_cost = new_uncovered * 10000 + len(current_blocks) # Length unchanged
            
            if new_cost < current_cost or random.random() < pow(2.718, -(new_cost - current_cost) / start_temp):
                # Apply move
                for t_idx in t_indices_old:
                    coverage_counts[t_idx] -= 1
                for t_idx in t_indices_new:
                    coverage_counts[t_idx] += 1
                    
                current_blocks[idx] = b_new
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = current_blocks[:]

    # Return the best valid solution found
    # Ensure best_blocks is valid (uncovered == 0)
    # If best solution is invalid (shouldn't happen if we started valid and only accepted valid improvements 
    # or if penalty is high enough), we might need to repair.
    # But with penalty 10000, it's very unlikely to keep an invalid solution if a valid one is found.
    # However, if we start valid, and only accept moves that improve cost, 
    # and cost includes heavy penalty for uncovered, we should stay valid or become valid quickly.
    # If best_blocks is invalid, we can just return the initial greedy solution which was valid.
    
    # Verify best_blocks
    final_counts = [0] * num_t_subsets
    for b in best_blocks:
        for t_idx in k_to_t_indices[b]:
            final_counts[t_idx] += 1
            
    if any(c == 0 for c in final_counts):
        # Fallback to greedy solution
        return blocks
    else:
        return best_blocks
