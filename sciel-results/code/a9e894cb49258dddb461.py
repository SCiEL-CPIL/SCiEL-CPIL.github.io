import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Solve the covering design problem C(v, k, t) using Simulated Annealing.
    
    Strategy:
    1. Initialize a valid covering using a fast greedy heuristic.
    2. Precompute coverage relationships using bitsets (integers) for speed.
    3. Run Simulated Annealing to minimize the number of blocks.
       - Moves: Swap one block in the set with a random block from the universe.
       - Acceptance: Accept if valid (coverage maintained) and size reduced, 
         or with probability exp(-delta/T) if size increased but still valid? 
         Actually, standard SA minimizes cost. Here cost = number of blocks.
         If we swap and maintain coverage, cost is 0 change (size same) or 
         we can try to remove redundant blocks after swaps.
         
    Refined Strategy for SA:
    State: Set of block indices.
    Objective: Minimize |State| subject to Coverage(Union(State)) == Universe.
    
    Neighborhood:
    1. Remove a block: If coverage remains valid, accept (improvement).
    2. Add a block: If it helps remove multiple others? Hard to see directly.
    3. Swap: Replace block A with block B. If Coverage(State \ {A} U {B}) is valid, accept.
       This maintains size but explores the space of valid coverings of fixed size.
       Once we find a valid covering of size N, we try to reduce to N-1.
       
    Algorithm:
    1. Greedy init -> Size S0.
    2. Loop:
       a. Try to reduce size by 1:
          - Pick a random block to remove. If valid, keep it.
          - If not, try swapping the removed block with others to see if we can 
            maintain coverage with size S0-1? No, that's complex.
          
    Alternative SA formulation:
    Cost function: 
    If valid: Cost = -|State| (we want to maximize size? No, minimize size. So Cost = |State|)
    If invalid: Cost = |State| + Penalty * (Uncovered_Count)
    
    We want to find a state with low Cost.
    Start with valid state (Penalty=0).
    Moves:
    - Add random block: Size increases, Cost increases (unless it allows removal later).
    - Remove random block: Size decreases. If invalid, Cost jumps up. SA might accept invalid states to escape local minima, then repair.
    - Swap: Size constant.
    
    Let's use a penalty-based SA.
    State: Set of blocks.
    Energy E(S) = |S| + Lambda * Uncovered(S)
    Lambda is large (e.g., 1000).
    
    Initialization: Greedy covering.
    
    Moves:
    1. Add a random block from Universe.
    2. Remove a random block from State.
    3. Swap: Remove random from State, Add random from Universe.
    
    If new E < old E: Accept.
    Else: Accept with prob exp(-(E_new - E_old) / T).
    
    This allows the algorithm to temporarily increase size or become invalid to find a better configuration.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage mask for each block as a Python integer (bitset)
    # This is fast for bit operations.
    block_masks = [0] * num_blocks
    # Also store the list of t-subset indices for each block for easier counting if needed?
    # Bitsets are enough for "is covered" checks.
    
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    full_mask = (1 << num_t_subsets) - 1
    
    # Helper: Calculate uncovered count?
    # It's faster to just check if (OR of masks) == full_mask.
    # For penalty, we need count of uncovered.
    # Python ints are big integers. Bit counting is fast in C.
    # uncovered = full_mask ^ union_mask
    # count = bin(uncovered).count('1')
    
    # 1. Greedy Initialization
    def greedy_init(seed):
        random.seed(seed)
        uncovered_mask = full_mask
        selected = []
        # We need to pick blocks that reduce uncovered_mask most.
        # To do this fast, we can't scan all blocks every time.
        # Heuristic: Pick a random uncovered bit, find a block covering it, pick best among those.
        # We need reverse index: t_idx -> list of block indices.
        # Building this is O(num_blocks * C(k,t)).
        
        # Let's build reverse index once globally? 
        # It's memory heavy but we are in a function. 
        # Actually, for v=24, num_blocks=346k. 
        # Let's just use a simple greedy that scans all blocks? 
        # 346k * 70 steps = 24M operations. Might be too slow in Python loop.
        
        # Faster Greedy:
        # Maintain a list of uncovered t-subsets.
        # Pick one randomly.
        # Iterate all blocks? No.
        
        # Let's use the reverse index approach for initialization only, or just use a fixed seed and hope.
        # Actually, let's just do a simple random greedy:
        # Pick random block. If it covers new stuff, add it.
        # This is very slow to converge.
        
        # Better: 
        # Precompute reverse index.
        t_to_blocks = [[] for _ in range(num_t_subsets)]
        for i, b in enumerate(all_blocks):
            for ts in itertools.combinations(b, t):
                idx = t_subset_to_idx[ts]
                t_to_blocks[idx].append(i)
        
        uncovered_set = set(range(num_t_subsets))
        selected_indices = []
        
        while uncovered_set:
            if time.time() > deadline - 5.0:
                break
            # Pick random uncovered t-subset
            target_t = random.choice(list(uncovered_set))
            candidates = t_to_blocks[target_t]
            
            best_block = -1
            best_gain = -1
            # Evaluate candidates
            for b_idx in candidates:
                # Gain = number of bits in block_masks[b_idx] that are in uncovered_mask
                # We don't have uncovered_mask as int here, we have set.
                # Let's convert uncovered_set to int mask once per step? 
                # Or just count intersections with set?
                # Set intersection is fast in C.
                # But we need to map t_idx to bit? No, we can just count how many of the block's t-subsets are in uncovered_set.
                # block_covered_t_indices[b_idx] would be useful.
                pass # This is getting complicated. Let's use a simpler init.
            
            # Simpler Init: Randomized Greedy with full scan of blocks?
            # If we just pick the block with max gain from ALL blocks.
            # To speed this up, we can precompute block_covered_t_indices.
            
            # Let's assume we have block_covered_t_indices.
            # Let's build it.
            
            # Wait, I can just use a known good construction or a very fast greedy.
            # Let's try: 
            # 1. Pick random block.
            # 2. Calculate gain.
            # 3. Keep best.
            # To make it faster, limit candidates to those covering a random uncovered point.
            
            # Let's stick to the reverse index logic but implement it carefully.
            
            # Re-implementing Greedy properly:
            if not selected_indices: # First block
                 selected_indices.append(random.randint(0, num_blocks-1))
                 # Update uncovered
                 m = block_masks[selected_indices[0]]
                 # Remove covered from uncovered_set
                 # This is O(num_t_subsets) to check each bit?
                 # Let's use bit masks for the greedy too.
            
            # Let's restart Greedy using Bitmasks.
            # uncovered_mask = full_mask
            # selected = []
            # while uncovered_mask:
            #    best_gain = -1
            #    best_b = -1
            #    # Optimization: Only check blocks that cover a specific uncovered bit
            #    # Find an uncovered bit
            #    # lsb = uncovered_mask & -uncovered_mask
            #    # idx = lsb.bit_length() - 1
            #    # candidates = t_to_blocks[idx]
            #    for b in candidates:
            #        gain = (block_masks[b] & uncovered_mask).bit_count()
            #        if gain > best_gain:
            #            best_gain = gain
            #            best_b = b
            #    if best_b == -1: break
            #    selected.append(best_b)
            #    uncovered_mask &= ~block_masks[best_b]
            
            # This requires t_to_blocks. Let's build it.
            
            # Actually, let's just use a standard greedy that scans all blocks if the set is small enough? 
            # 346k blocks is too many for 70 iterations in Python if we do bit_count each time?
            # 346k * 70 = 24M bit_counts. Python bit_count is C-speed. 24M ops is ~1-2 seconds.
            # This is acceptable for initialization!
            
            # So:
            # Greedy:
            #   uncovered = full_mask
            #   selected = []
            #   while uncovered:
            #      best_gain = -1
            #      best_b = -1
            #      for b in range(num_blocks):
            #          gain = (block_masks[b] & uncovered).bit_count()
            #          if gain > best_gain:
            #             best_gain = gain
            #             best_b = b
            #             if gain == 0: break # No point checking others if max is 0? No, we need to find ANY block that covers something. If gain is 0 for all, we are stuck (shouldn't happen).
            #      if best_b == -1: break
            #      selected.append(best_b)
            #      uncovered &= ~block_masks[best_b]
            #   return selected
            
            # This is O(num_blocks * num_iterations). 
            # 346k * 70 = 24M. 
            # Let's use this. It's simple and robust.
            
            # But wait, we need to randomize it to get different starts.
            # Add noise to gain? Or pick random among top K?
            # Let's pick random among top 10 gains.
            
            # Revised Greedy with Randomization:
            #   uncovered = full_mask
            #   selected = []
            #   while uncovered:
            #      gains = []
            #      for b in range(num_blocks):
            #          gain = (block_masks[b] & uncovered).bit_count()
            #          if gain > 0:
            #             gains.append((gain, b))
            #      if not gains: break
            #      # Sort by gain desc
            #      gains.sort(reverse=True)
            #      # Pick from top 10
            #      top_k = min(10, len(gains))
            #      choice = random.choice(gains[:top_k])
            #      selected.append(choice[1])
            #      uncovered &= ~block_masks[choice[1]]
            
            # This is 70 * (346k * bit_count + sort).
            # Sort is 346k log 346k ~ 346k * 19 ~ 6M ops.
            # Total ~ 70 * (24M + 6M) = 2.1B ops. Too slow.
            
            # Optimization: Don't sort all. Just find top 10.
            # Or, use the reverse index heuristic.
            # Let's go back to reverse index. It's O(1) to get candidates for a specific uncovered triple.
            # Candidates size: C(21, 4) = 5985 blocks cover a specific triple.
            # 70 iterations * 5985 candidates * bit_count = 70 * 6k = 420k ops. Very fast.
            
            # So we MUST build reverse index.
            
            # Building reverse index:
            # t_to_blocks = [[] for _ in range(num_t_subsets)]
            # for i, b in enumerate(all_blocks):
            #    for ts in itertools.combinations(b, t):
            #        idx = t_subset_to_idx[ts]
            #        t_to_blocks[idx].append(i)
            
            # This takes ~12M appends. ~1-2 seconds. Acceptable.
            
            # Let's do this.
            
            # Since I'm inside the function, I'll build it here.
            # Note: This is expensive. If we call construct multiple times, it's wasted.
            # But the problem says "The evaluator passes this task's parameters".
            # It implies one call.
            
            # So:
            t_to_blocks = [[] for _ in range(num_t_subsets)]
            for i, b in enumerate(all_blocks):
                for ts in itertools.combinations(b, t):
                    idx = t_subset_to_idx[ts]
                    t_to_blocks[idx].append(i)
            
            # Now Greedy:
            uncovered_mask = full_mask
            selected_indices = []
            
            while uncovered_mask:
                if time.time() > deadline - 5.0:
                    break
                # Find an uncovered triple index
                # Get a random uncovered bit
                # We can't pick random bit from int easily.
                # We can pick a random index from 0..num_t_subsets and check if it's uncovered?
                # That's O(1) check but might miss if many are covered.
                # Better: Maintain a list of uncovered indices? 
                # Or just pick the lowest set bit? (Deterministic)
                # To randomize, we can pick a random index and if covered, try again?
                # Or, precompute a list of all indices and filter?
                
                # Let's just pick the lowest set bit for determinism in greedy, 
                # but randomize the choice of block among candidates.
                
                lsb = uncovered_mask & -uncovered_mask
                target_idx = lsb.bit_length() - 1
                
                candidates = t_to_blocks[target_idx]
                if not candidates:
                    break # Should not happen
                
                best_gain = -1
                best_block = -1
                # Randomized tie breaking
                # Just pick max gain. Randomness comes from... nowhere?
                # If we always pick max gain, it's deterministic.
                # To get diversity, we can pick randomly from top 5 gains.
                
                gains = []
                for b in candidates:
                    gain = (block_masks[b] & uncovered_mask).bit_count()
                    gains.append((gain, b))
                
                if not gains:
                    break
                    
                # Sort by gain
                gains.sort(reverse=True)
                top_k = min(5, len(gains))
                choice = random.choice(gains[:top_k])
                selected_indices.append(choice[1])
                uncovered_mask &= ~block_masks[choice[1]]
            
            return selected_indices

    # 2. Simulated Annealing
    def simulated_annealing(initial_blocks):
        if not initial_blocks:
            return []
            
        current_blocks = list(set(initial_blocks)) # Ensure unique
        current_mask = 0
        for b in current_blocks:
            current_mask |= block_masks[b]
            
        # Check validity
        if current_mask != full_mask:
            # Repair? Just return as is if invalid? 
            # Greedy should produce valid.
            pass
            
        current_size = len(current_blocks)
        best_blocks = list(current_blocks)
        best_size = current_size
        
        # SA Parameters
        T = 10.0
        cooling = 0.99
        min_T = 0.1
        
        # We need to track which blocks are in the set for fast add/remove?
        # Set of indices is O(1) lookup.
        current_set = set(current_blocks)
        
        while T > min_T:
            if time.time() > deadline - 5.0:
                break
                
            # Generate Neighbor
            move_type = random.random()
            new_set = set(current_set)
            
            if move_type < 0.3:
                # Remove a block
                if not current_set:
                    continue
                b_to_remove = random.choice(list(current_set))
                new_set.discard(b_to_remove)
            elif move_type < 0.6:
                # Add a block
                b_to_add = random.randint(0, num_blocks - 1)
                new_set.add(b_to_add)
            else:
                # Swap
                if not current_set:
                    continue
                b_to_remove = random.choice(list(current_set))
                b_to_add = random.randint(0, num_blocks - 1)
                if b_to_remove == b_to_add:
                    continue
                new_set.discard(b_to_remove)
                new_set.add(b_to_add)
            
            # Calculate Energy
            # E = size + Lambda * uncovered_count
            # Lambda = 1000
            
            # Calculate new mask
            # This is O(|new_set|) bit ORs.
            # If set is ~70, 70 ORs is fast.
            new_mask = 0
            for b in new_set:
                new_mask |= block_masks[b]
                
            new_size = len(new_set)
            uncovered_bits = full_mask ^ new_mask
            # Count bits
            # bin(uncovered_bits).count('1') is slow for large ints?
            # Python 3.8+ has int.bit_count()
            uncovered_count = uncovered_bits.bit_count()
            
            new_energy = new_size + 1000 * uncovered_count
            
            # Old Energy
            # We can calculate old energy similarly or store it.
            # Let's store it.
            # But we need to update it if we accept.
            
            # For now, calculate old energy each time? 
            # It's 70 ORs + bit_count. Fast enough.
            old_mask = 0
            for b in current_set:
                old_mask |= block_masks[b]
            old_size = len(current_set)
            old_uncovered = (full_mask ^ old_mask).bit_count()
            old_energy = old_size + 1000 * old_uncovered
            
            delta_E = new_energy - old_energy
            
            if delta_E < 0 or random.random() < pow(2.718281828, -delta_E / T):
                # Accept
                current_set = new_set
                current_size = new_size
                current_mask = new_mask # Store for next? 
                # Actually, we recalculate mask every step. 
                # To save time, we could update incrementally.
                # But for simplicity, recalc is fine if fast.
                
                if new_size < best_size and uncovered_count == 0:
                    best_size = new_size
                    best_blocks = list(new_set)
            
            T *= cooling
            
        return best_blocks

    # Run
    # 1. Greedy Init
    # We need to seed it.
    # Let's try a few seeds.
    
    best_overall = []
    best_size_overall = float('inf')
    
    # Time budget allocation
    # Precomputation (blocks, masks, reverse index) takes ~2-3s.
    # Greedy takes ~0.1s.
    # SA takes remaining time.
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_masks = [0] * num_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    full_mask = (1 << num_t_subsets) - 1
    
    # Build reverse index
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for i, b in enumerate(all_blocks):
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            t_to_blocks[idx].append(i)
            
    # Greedy Function (using precomputed data)
    def run_greedy(seed):
        random.seed(seed)
        uncovered_mask = full_mask
        selected_indices = []
        
        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break
            lsb = uncovered_mask & -uncovered_mask
            target_idx = lsb.bit_length() - 1
            candidates = t_to_blocks[target_idx]
            
            if not candidates:
                break
                
            gains = []
            for b in candidates:
                gain = (block_masks[b] & uncovered_mask).bit_count()
                gains.append((gain, b))
                
            if not gains:
                break
                
            gains.sort(reverse=True)
            top_k = min(5, len(gains))
            choice = random.choice(gains[:top_k])
            selected_indices.append(choice[1])
            uncovered_mask &= ~block_masks[choice[1]]
            
        return selected_indices

    # SA Function
    def run_sa(initial_blocks, time_limit):
        if not initial_blocks:
            return []
            
        current_set = set(initial_blocks)
        best_blocks = list(current_set)
        best_size = len(current_set)
        
        # Calculate initial best mask to ensure validity
        init_mask = 0
        for b in current_set:
            init_mask |= block_masks[b]
        if init_mask != full_mask:
            # Invalid init? Should not happen with greedy.
            # But if it does, SA will try to fix it.
            pass
            
        T = 5.0
        cooling = 0.995
        min_T = 0.01
        
        start = time.time()
        
        while T > min_T:
            if time.time() > start + time_limit:
                break
                
            move_type = random.random()
            new_set = set(current_set)
            
            if move_type < 0.4:
                # Remove
                if not current_set:
                    continue
                b_to_remove = random.choice(list(current_set))
                new_set.discard(b_to_remove)
            elif move_type < 0.7:
                # Add
                b_to_add = random.randint(0, num_blocks - 1)
                new_set.add(b_to_add)
            else:
                # Swap
                if not current_set:
                    continue
                b_to_remove = random.choice(list(current_set))
                b_to_add = random.randint(0, num_blocks - 1)
                if b_to_remove == b_to_add:
                    continue
                new_set.discard(b_to_remove)
                new_set.add(b_to_add)
                
            # Calculate New Energy
            new_mask = 0
            for b in new_set:
                new_mask |= block_masks[b]
            new_size = len(new_set)
            new_uncovered = (full_mask ^ new_mask).bit_count()
            new_energy = new_size + 1000 * new_uncovered
            
            # Calculate Old Energy (cached? or calc)
            # To save time, we can maintain current_mask and current_size.
            # But we need old_energy for comparison.
            # Let's maintain current_mask.
            
            # Wait, I didn't maintain current_mask in the loop.
            # Let's refactor to maintain state.
            
            # Actually, recalculating is safer and 70 ORs is fast.
            # But we do it twice (old and new).
            # Let's just calculate new_energy and compare to a stored current_energy.
            
            # I will refactor the SA loop to be cleaner.
            pass # Placeholder
            
        return best_blocks

    # Let's implement a cleaner SA loop with state maintenance.
    
    def optimized_sa(initial_blocks, time_limit):
        if not initial_blocks:
            return []
            
        current_set = set(initial_blocks)
        current_mask = 0
        for b in current_set:
            current_mask |= block_masks[b]
        current_size = len(current_set)
        current_uncovered = (full_mask ^ current_mask).bit_count()
        current_energy = current_size + 1000 * current_uncovered
        
        best_blocks = list(current_set)
        best_size = current_size
        best_mask = current_mask
        
        T = 5.0
        cooling = 0.995
        min_T = 0.01
        
        start = time.time()
        
        while T > min_T:
            if time.time() > start + time_limit:
                break
                
            move_type = random.random()
            new_set = set(current_set)
            new_mask = current_mask
            new_size = current_size
            
            # Perform Move on Mask and Set
            # This is tricky because we need to update mask based on set changes.
            # Add: new_mask |= block_masks[b]
            # Remove: new_mask = (OR of all others). Hard to update mask by removal without recalculating.
            # So, if we remove, we must recalculate mask from scratch.
            # If we add, we can just OR.
            # If we swap, we recalculate.
            
            # To make it fast, we can try Add first (cheap update), 
            # and if we want to Remove, we recalc.
            
            if move_type < 0.4:
                # Remove
                if not current_set:
                    continue
                b_to_remove = random.choice(list(current_set))
                new_set.discard(b_to_remove)
                new_size = len(new_set)
                # Recalc mask
                new_mask = 0
                for b in new_set:
                    new_mask |= block_masks[b]
            elif move_type < 0.7:
                # Add
                b_to_add = random.randint(0, num_blocks - 1)
                new_set.add(b_to_add)
                new_size = len(new_set)
                new_mask = current_mask | block_masks[b_to_add]
            else:
                # Swap
                if not current_set:
                    continue
                b_to_remove = random.choice(list(current_set))
                b_to_add = random.randint(0, num_blocks - 1)
                if b_to_remove == b_to_add:
                    continue
                new_set.discard(b_to_remove)
                new_set.add(b_to_add)
                new_size = len(new_set)
                # Recalc mask
                new_mask = 0
                for b in new_set:
                    new_mask |= block_masks[b]
            
            new_uncovered = (full_mask ^ new_mask).bit_count()
            new_energy = new_size + 1000 * new_uncovered
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or random.random() < pow(2.718281828, -delta_E / T):
                current_set = new_set
                current_mask = new_mask
                current_size = new_size
                current_uncovered = new_uncovered
                current_energy = new_energy
                
                if new_size < best_size and new_uncovered == 0:
                    best_size = new_size
                    best_blocks = list(new_set)
                    best_mask = new_mask
            
            T *= cooling
            
        return best_blocks

    # Main Execution
    # 1. Greedy Init
    # Use multiple seeds for greedy to get diverse starts?
    # Or just one good one and let SA do the work.
    
    # Let's try 3 greedy inits.
    candidates = []
    seeds = [1, 2, 3]
    
    for seed in seeds:
        if time.time() > deadline - 10.0:
            break
        g_blocks = run_greedy(seed)
        if g_blocks:
            candidates.append(g_blocks)
            
    if not candidates:
        # Fallback
        return []
        
    # 2. SA on the best candidate (or all?)
    # SA is slow. Let's run it on the smallest greedy result first.
    # Or run it on one for a while, then switch?
    
    # Let's pick the smallest greedy result.
    candidates.sort(key=len)
    best_start = candidates[0]
    
    # Run SA with remaining time
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 0:
        sa_result = optimized_sa(best_start, remaining_time)
        if sa_result and len(sa_result) < len(best_start):
            best_start = sa_result
            
    # 3. Final Cleanup: Remove redundant blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        blocks = list(set(blocks))
        # Calculate coverage counts
        # Use a dict or array of counts?
        # Array of size num_t_subsets (2024) is fine.
        counts = [0] * num_t_subsets
        for b in blocks:
            mask = block_masks[b]
            # Iterate bits? 
            # Or just check if removing b leaves mask unchanged?
            # If (current_mask ^ block_masks[b]) == current_mask, then b is redundant?
            # No, that means b adds nothing.
            # But if b is the ONLY one covering a triple, removing it makes that triple uncovered.
            # So b is redundant if for all triples in b, another block covers it.
            # This is equivalent to: (Union of all blocks except b) covers (Union of all blocks).
            # i.e., Union_All == Union_All_without_b.
            
            # Efficient check:
            # Calculate Union_All.
            # For each b, calculate Union_All_without_b.
            # If they are equal, b is redundant.
            
            # Calculating Union_All_without_b for each b is O(N^2) bit ORs.
            # N~70. 70*70 = 4900 ORs. Fast.
            
            union_all = 0
            for b in blocks:
                union_all |= block_masks[b]
                
            new_blocks = []
            for b in blocks:
                union_without = 0
                for other in blocks:
                    if other != b:
                        union_without |= block_masks[other]
                
                if union_without == union_all:
                    # b is redundant
                    continue
                else:
                    new_blocks.append(b)
                    
            return new_blocks
            
    final_blocks = remove_redundant(best_start)
    
    # Convert to list of lists
    result = [list(all_blocks[b_idx]) for b_idx in final_blocks]
    
    # Ensure distinctness and sorting
    # all_blocks are tuples, so distinct.
    for b in result:
        b.sort()
        
    return result
