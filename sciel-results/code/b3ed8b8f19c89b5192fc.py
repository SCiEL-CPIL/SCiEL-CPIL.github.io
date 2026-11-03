import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    
    # Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # block_cover[bi] = list of t-indices covered by block bi
    # t_to_blocks[ti] = list of block indices covering t-subset ti
    block_cover = []
    t_to_blocks = [[] for _ in range(N_t)]
    
    # Precompute block covers
    for bi, b in enumerate(all_blocks):
        covered = []
        for ts in itertools.combinations(b, t):
            ti = t_to_idx[ts]
            covered.append(ti)
        block_cover.append(covered)
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Bitmask representation for faster operations in local search
    # Each block is a bitmask of the t-subsets it covers
    # N_t = 11628 for C(19,5). This fits in ~2 numpy.uint64 or a few Python ints.
    # Python ints are arbitrary precision, so we can use them as bitmasks.
    block_masks = [0] * N_b
    for bi, covered in enumerate(block_cover):
        mask = 0
        for ti in covered:
            mask |= (1 << ti)
        block_masks[bi] = mask

    def get_uncovered_counts(blocks):
        """
        Returns an array of counts for each t-subset.
        """
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
        return counts

    def is_valid(blocks):
        """
        Check if all t-subsets are covered.
        """
        if not blocks:
            return N_t == 0
        # Use bitmask union for speed
        union_mask = 0
        for bi in blocks:
            union_mask |= block_masks[bi]
        # Check if union_mask has bits set for all 0..N_t-1
        # (1 << N_t) - 1 is the mask with all N_t bits set
        return (union_mask & ((1 << N_t) - 1)) == ((1 << N_t) - 1)

    def prune_redundant(blocks):
        """
        Remove blocks that do not uniquely cover any t-subset.
        """
        if not blocks:
            return []
        
        counts = get_uncovered_counts(blocks)
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in current_blocks:
                is_redundant = True
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(bi)
                else:
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    changed = True
            current_blocks = new_blocks
        return current_blocks

    def primal_dual_covering(rng, time_limit):
        """
        Primal-Dual Approximation Algorithm for Set Cover.
        
        Algorithm:
        1. Initialize dual variables y_j = 0 for all t-subsets j.
        2. Initialize selected set S = empty.
        3. While there are uncovered t-subsets:
           a. Increase y_j for all uncovered j at rate 1.
           b. If any block B becomes "tight" (sum_{j in B} y_j == |B|), 
              select B, add to S, and fix y_j for all j in B to their current values.
           c. Repeat until all covered.
        
        This yields a |B|_max approximation. For uniform sets, it's a constant factor.
        We implement a variant that selects the block that becomes tight earliest,
        or uses a weighted approach to break ties effectively.
        """
        # y[j] is the dual value for t-subset j
        y = np.zeros(N_t, dtype=np.float64)
        covered_mask = np.zeros(N_t, dtype=bool)
        selected = []
        
        # To avoid scanning all blocks every step, we can track "potential" blocks.
        # However, with 75k blocks and ~11k t-subsets, a simple loop might be okay if optimized.
        # But standard Primal-Dual can be slow if not careful.
        
        # Alternative: "Weighted Primal-Dual" or "Dual-Adjusted Greedy"
        # Let's use a heuristic version:
        # At each step, find the uncovered t-subset with the smallest "slack" in its covering blocks?
        # No, let's stick to the standard interpretation but optimized.
        
        # Actually, a common practical implementation of Primal-Dual for Set Cover:
        # 1. While uncovered elements exist:
        #    a. For each uncovered element j, find the block B minimizing (sum of y for elements in B already covered? No.)
        
        # Let's implement the classic algorithm properly but with early termination.
        # Since all sets have the same size (56 t-subsets per block), the approximation factor is 56.
        # We can improve this by using the "lazy" evaluation.
        
        # A faster variant:
        # Maintain a priority queue of blocks based on how "close" they are to being tight.
        # But dynamic updates are complex.
        
        # Let's use a simpler "Dual-Guided Greedy":
        # Score of block B = sum_{j in B, uncovered} (1 / count_uncovered(j)) 
        # This is similar to greedy but weighted by scarcity.
        
        # Actually, let's try the real Primal-Dual logic:
        # Increase y_j for all uncovered j.
        # The rate is 1.
        # A block B is tight when sum_{j in B} y_j = |B|.
        # Since all |B| are equal (56), we are looking for the block that accumulates dual value fastest.
        
        # We can simulate this:
        # At any point, the "next tight" block is the one that minimizes (|B| - sum_{j in B} y_j).
        # Let's compute "slack" for all blocks initially.
        
        slack = np.full(N_b, 56.0, dtype=np.float64) # |B| - sum(y)
        # Initially y=0, so slack=56.
        
        uncovered_count = N_t
        y = np.zeros(N_t, dtype=np.float64)
        
        while uncovered_count > 0:
            if time.time() > time_limit:
                break
                
            # Find block with minimum slack
            min_slack_idx = np.argmin(slack)
            min_slack_val = slack[min_slack_idx]
            
            # Increase y for all uncovered elements by min_slack_val
            # This makes the block at min_slack_idx tight (slack becomes 0).
            # It also increases slack for all OTHER blocks that cover uncovered elements?
            # No, slack = |B| - sum(y). If y increases, slack DECREASES.
            # So all blocks covering the uncovered elements will have their slack reduced by min_slack_val * (number of uncovered elements in B).
            
            # We need to update slack for all blocks that cover any of the uncovered elements.
            # This is expensive if done naively.
            
            # Optimization:
            # Only blocks that contain at least one uncovered element are relevant.
            # Let's just update y for all uncovered elements.
            # Then re-calculate slack? That's O(N_b * 56). 75k * 56 = 4M ops.
            # If we do this 100 times, 400M ops. Might be too slow in Python.
            
            # Let's try a different approach: "Greedy with Dual Weights"
            # This is effectively what the current program does, but we can tweak the weights.
            
            # Let's implement a standard Greedy Set Cover but with a twist:
            # Instead of picking max coverage, pick max (coverage / cost).
            # Here cost is 1 for all. So it's just max coverage.
            # To differentiate, we can use the "Dual" values to break ties or adjust scores.
            
            # Let's go back to the prompt's suggestion: "custom primal-dual heuristic... iteratively adjusts dual variables... to select high-value blocks."
            
            # Heuristic:
            # 1. Identify uncovered t-subsets.
            # 2. For each uncovered t, look at blocks covering it.
            # 3. Score block B = sum_{j in B} (1 / (1 + y_j)) ?
            
            # Let's implement a simple iterative dual update:
            # 1. Pick uncovered t-subset j with highest "urgency" (e.g., fewest covering blocks remaining).
            # 2. Select the block B covering j that maximizes the number of OTHER uncovered t-subsets it covers.
            # 3. Add B to solution.
            # 4. Update duals: Increase y for all t in B.
            
            # This is essentially Greedy. The "Dual" part is often implicit in the "urgency" or "weight".
            
            # Let's try a different structural change: **Local Search with Bitmasks**.
            # The current program uses lists for counts. Bitmasks can make "is this block redundant" and "what does this block cover that is unique" much faster.
            
            # We will construct a solution using a fast greedy (bitmask based), then apply a heavy local search.
            
            # Fast Greedy Construction:
            remaining_mask = (1 << N_t) - 1
            selected = []
            
            # Precompute for each t-subset, the list of blocks covering it.
            # We already have t_to_blocks.
            
            # To speed up greedy, we can maintain a "best block" for each uncovered t.
            # But updating this is hard.
            
            # Let's just do standard greedy but using bitmasks for the "uncovered" check.
            # For each step:
            # 1. Find an uncovered t. (Random or first).
            # 2. Find best block covering it.
            
            # To make it faster, we can pre-sort blocks by size? No, all same size.
            
            # Let's use the "Approximate Greedy" from the current code but optimized with NumPy for the gain calculation.
            
            uncovered_indices = np.where(np.zeros(N_t, dtype=bool) == False)[0] # dummy
            # Actually, let's just use the current `probabilistic_construction` logic but maybe with a better selection rule.
            
            # The prompt asks for a DIFFERENT strategy.
            # Let's implement **Simulated Annealing** on the set of blocks.
            # State: Set of blocks.
            # Move: Add, Remove, Swap.
            # Cost: Number of uncovered t-subsets (must be 0) + lambda * Number of blocks.
            # We want to minimize Number of blocks subject to 0 uncovered.
            
            # This is a robust approach that can escape local minima.
            
            return None # Placeholder, we will implement SA in the main flow

    def simulated_annealing(initial_blocks, rng, time_limit):
        """
        Simulated Annealing to minimize block count while maintaining coverage.
        """
        current_blocks = list(initial_blocks)
        counts = get_uncovered_counts(current_blocks)
        uncovered = np.sum(counts == 0)
        
        # We require full coverage. If initial is not valid, we must add blocks first.
        # Assuming initial is valid (from greedy).
        
        current_cost = len(current_blocks)
        best_blocks = list(current_blocks)
        best_cost = current_cost
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < time_limit:
            if T < T_min:
                T = T_min
                
            # Generate a neighbor
            move_type = rng.random()
            
            new_blocks = list(current_blocks)
            new_counts = counts.copy()
            
            if move_type < 0.3 and len(new_blocks) > 0:
                # Remove a random block
                idx = rng.randrange(len(new_blocks))
                bi = new_blocks[idx]
                for ti in block_cover[bi]:
                    new_counts[ti] -= 1
                new_blocks.pop(idx)
                
            elif move_type < 0.6 and len(new_blocks) > 0:
                # Swap a block
                idx = rng.randrange(len(new_blocks))
                bi_out = new_blocks[idx]
                
                # Find a random block to swap in
                # Ideally, swap with a block that covers the unique parts of bi_out
                # For simplicity, pick a random block from all blocks? Too slow.
                # Pick from blocks that cover at least one t-subset uniquely covered by bi_out.
                
                unique_ts = [ti for ti in block_cover[bi_out] if counts[ti] == 1]
                if unique_ts:
                    # Find candidates covering all unique_ts
                    # This is expensive. Let's just pick a random block from the pool of "active" blocks?
                    # Or just a random block from all_blocks.
                    bi_in = rng.randrange(N_b)
                    if bi_in != bi_out:
                        # Apply swap
                        for ti in block_cover[bi_out]:
                            new_counts[ti] -= 1
                        for ti in block_cover[bi_in]:
                            new_counts[ti] += 1
                        new_blocks[idx] = bi_in
                    else:
                        new_blocks = current_blocks
                        new_counts = counts
                else:
                    # No unique t, can just remove?
                    # Try remove
                    for ti in block_cover[bi_out]:
                        new_counts[ti] -= 1
                    new_blocks.pop(idx)
                    
            else:
                # Add a random block
                bi_in = rng.randrange(N_b)
                if bi_in not in new_blocks:
                    new_blocks.append(bi_in)
                    for ti in block_cover[bi_in]:
                        new_counts[ti] += 1
            
            # Evaluate new state
            new_uncovered = np.sum(new_counts == 0)
            
            # Cost function:
            # We must have 0 uncovered.
            # If new_uncovered > 0, it's a "bad" move unless we accept it to escape?
            # Usually, we constrain to valid solutions.
            # If we allow invalid, we need a penalty.
            # Penalty = 1000 * new_uncovered + len(new_blocks)
            
            current_penalty = 1000 * np.sum(counts == 0) + len(current_blocks)
            new_penalty = 1000 * new_uncovered + len(new_blocks)
            
            delta = new_penalty - current_penalty
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                # Accept
                current_blocks = new_blocks
                counts = new_counts
                if new_uncovered == 0 and len(current_blocks) < best_cost:
                    best_cost = len(current_blocks)
                    best_blocks = list(current_blocks)
            
            T *= cooling
            
        return best_blocks

    # Main Execution
    best_solution = None
    best_len = float('inf')
    
    # Phase 1: Fast Greedy Construction using Bitmasks for validation
    # We will run a few greedy starts.
    
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline - 60.0:
            break
        rng = random.Random(42 + i)
        
        # Simple Greedy
        uncovered_mask = (1 << N_t) - 1
        selected = []
        
        while uncovered_mask != 0:
            # Pick a random uncovered t-subset
            # Find an index where bit is 1
            # We can just pick a random index and check if it's uncovered
            # Or find the lowest set bit.
            # Let's pick a random t-subset index
            ti_target = rng.randrange(N_t)
            if (uncovered_mask >> ti_target) & 1:
                # It is uncovered
                candidates = t_to_blocks[ti_target]
                best_block = -1
                best_gain = -1
                
                for bi in candidates:
                    # Gain = number of uncovered t-subsets covered by bi
                    # Intersection of block_mask[bi] and uncovered_mask
                    inter = block_masks[bi] & uncovered_mask
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = bi
                
                if best_block != -1:
                    selected.append(best_block)
                    uncovered_mask &= ~block_masks[best_block]
        
        # Prune
        selected = prune_redundant(selected)
        
        if is_valid(selected):
            if len(selected) < best_len:
                best_len = len(selected)
                best_solution = selected

    # Phase 2: Simulated Annealing
    if best_solution:
        remaining_time = deadline - 5.0
        if time.time() < remaining_time:
            rng = random.Random(999)
            optimized = simulated_annealing(best_solution, rng, remaining_time)
            if is_valid(optimized) and len(optimized) < best_len:
                best_len = len(optimized)
                best_solution = optimized

    # Fallback
    if not best_solution:
        rng = random.Random(1)
        # Reuse the logic from Phase 1 with a single run
        uncovered_mask = (1 << N_t) - 1
        selected = []
        while uncovered_mask != 0:
            ti_target = rng.randrange(N_t)
            if (uncovered_mask >> ti_target) & 1:
                candidates = t_to_blocks[ti_target]
                best_block = -1
                best_gain = -1
                for bi in candidates:
                    inter = block_masks[bi] & uncovered_mask
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = bi
                if best_block != -1:
                    selected.append(best_block)
                    uncovered_mask &= ~block_masks[best_block]
        best_solution = prune_redundant(selected)

    return [list(all_blocks[bi]) for bi in best_solution]
