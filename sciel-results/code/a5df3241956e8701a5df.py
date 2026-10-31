import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Precompute all t-subsets as bitmasks
    all_t_subsets = []
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_t_subsets.append(mask)
    
    uncovered = set(all_t_subsets)
    blocks = []
    
    # Point load tracking: how many blocks contain each point
    point_load = [0] * v
    
    # Greedy phase with deficiency-ratio scoring
    while uncovered:
        if time.time() > deadline - 30:  # Leave 30s for local search
            break
        
        # Find best block using deficiency ratio heuristic
        # Score = newly_covered_t_subsets / (1 + sum(point_load[p] for p in block))
        # This balances coverage against point saturation
        
        best_block = None
        best_score = -1
        best_gain = 0
        
        # Candidate generation: pick several uncovered t-subsets and try to complete them
        # For efficiency, we sample a few uncovered subsets and greedily extend
        # But for better quality, we also try random completions
        
        # Strategy: try multiple candidate blocks and pick the best by deficiency ratio
        
        # First, try completing the most "deficient" uncovered subset
        # Pick an uncovered subset with low point load
        candidates_to_try = []
        
        # Try a few different uncovered subsets
        uncovered_list = list(uncovered)
        if len(uncovered_list) <= 20:
            candidates_to_try = uncovered_list
        else:
            # Pick subsets with low total point load
            scored = []
            for mask in uncovered_list:
                load = 0
                for i in range(v):
                    if mask & (1 << i):
                        load += point_load[i]
                scored.append((load, mask))
            scored.sort()
            candidates_to_try = [m for _, m in scored[:15]]
        
        for seed_mask in candidates_to_try:
            # Build initial block from seed_mask
            block = []
            for i in range(v):
                if seed_mask & (1 << i):
                    block.append(i)
            
            # Greedily extend to size k using deficiency-aware gain
            while len(block) < k:
                best_x = -1
                best_gain = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Calculate gain: number of uncovered t-subsets that would be covered
                    # by adding x to block
                    gain = 0
                    if len(block) >= t - 1:
                        for p_comb in itertools.combinations(block, t - 1):
                            p_mask = 0
                            for i in p_comb:
                                p_mask |= (1 << i)
                            s_mask = p_mask | (1 << x)
                            if s_mask in uncovered:
                                gain += 1
                    
                    # Deficiency ratio: gain / (1 + point_load[x])
                    # This penalizes adding points that are already heavily used
                    score = gain / (1.0 + point_load[x])
                    
                    if score > best_gain or (score == best_gain and gain > best_gain):
                        best_gain = score
                        best_x = x
                
                if best_x == -1:
                    break
                block.append(best_x)
            
            if len(block) != k:
                continue
            
            # Calculate actual gain and deficiency score for this complete block
            block_set = set(block)
            gain = 0
            for comb in itertools.combinations(block, t):
                mask = 0
                for i in comb:
                    mask |= (1 << i)
                if mask in uncovered:
                    gain += 1
            
            # Deficiency ratio score
            load_penalty = 1.0 + sum(point_load[p] for p in block)
            score = gain / load_penalty
            
            if score > best_score:
                best_score = score
                best_block = block
                best_gain = gain
        
        # If no valid block found, fallback to simple greedy
        if best_block is None:
            # Pick first uncovered and extend arbitrarily
            first_uncovered = min(uncovered)
            best_block = []
            for i in range(v):
                if first_uncovered & (1 << i):
                    best_block.append(i)
            while len(best_block) < k:
                for x in range(v):
                    if x not in best_block:
                        best_block.append(x)
                        break
            best_gain = 0
        
        best_block.sort()
        blocks.append(best_block)
        
        # Update uncovered and point loads
        for comb in itertools.combinations(best_block, t):
            mask = 0
            for i in comb:
                mask |= (1 << i)
            if mask in uncovered:
                uncovered.discard(mask)
        
        for p in best_block:
            point_load[p] += 1
    
    # Local search phase: redundancy removal + 1-for-1 swaps
    # This is where we can escape greedy local minima
    
    local_search_start = time.time()
    local_search_deadline = deadline - 2
    
    # Precompute: for each block, which t-subsets it covers
    # And for each t-subset, which blocks cover it
    
    def compute_block_tsubsets(block):
        """Return list of masks of t-subsets contained in block."""
        result = []
        for comb in itertools.combinations(block, t):
            mask = 0
            for i in comb:
                mask |= (1 << i)
            result.append(mask)
        return result
    
    # Map each t-subset mask to list of block indices that cover it
    tsub_to_blocks = {}
    for idx, block in enumerate(blocks):
        for mask in compute_block_tsubsets(block):
            if mask not in tsub_to_blocks:
                tsub_to_blocks[mask] = []
            tsub_to_blocks[mask].append(idx)
    
    # Phase 1: Remove redundant blocks
    # A block is redundant if every t-subset it covers is also covered by another block
    changed = True
    while changed and time.time() < local_search_deadline:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > local_search_deadline:
                break
            
            block_i = blocks[i]
            redundant = True
            for mask in compute_block_tsubsets(block_i):
                covering = tsub_to_blocks.get(mask, [])
                # Check if any other block covers this mask
                other_covers = False
                for j in covering:
                    if j != i:
                        other_covers = True
                        break
                if not other_covers:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                del blocks[i]
                changed = True
                # Rebuild tsub_to_blocks (expensive but necessary)
                tsub_to_blocks = {}
                for idx, block in enumerate(blocks):
                    for mask in compute_block_tsubsets(block):
                        if mask not in tsub_to_blocks:
                            tsub_to_blocks[mask] = []
                        tsub_to_blocks[mask].append(idx)
                break  # Restart scan after removal
    
    # Phase 2: 1-for-1 block swaps
    # Try replacing block i with a new block that covers all unique t-subsets of block i
    # and possibly more, reducing total count or improving spread
    
    # For efficiency, limit number of swap attempts
    max_swap_attempts = 500
    swap_attempts = 0
    
    # Identify "unique" t-subsets: those covered by exactly one block
    # If we remove a block, its unique t-subsets must be covered by the replacement
    
    while swap_attempts < max_swap_attempts and time.time() < local_search_deadline:
        swap_attempts += 1
        
        # Pick a random block to try to replace
        i = random.randint(0, len(blocks) - 1)
        block_i = blocks[i]
        
        # Find t-subsets uniquely covered by block i
        unique_masks = []
        for mask in compute_block_tsubsets(block_i):
            covering = tsub_to_blocks.get(mask, [])
            if len(covering) == 1:
                unique_masks.append(mask)
        
        if not unique_masks:
            # Block i is fully redundant (all its t-subsets are covered elsewhere)
            # Just remove it
            del blocks[i]
            # Rebuild tsub_to_blocks
            tsub_to_blocks = {}
            for idx, block in enumerate(blocks):
                for mask in compute_block_tsubsets(block):
                    if mask not in tsub_to_blocks:
                        tsub_to_blocks[mask] = []
                    tsub_to_blocks[mask].append(idx)
            continue
        
        # Try to find a replacement block that covers all unique_masks
        # and has good deficiency ratio
        
        best_replacement = None
        best_score = -1
        
        # Generate candidate replacements: random k-subsets that contain at least one element from each unique mask
        # For efficiency, try a limited number of random candidates
        
        num_candidates = 50
        for _ in range(num_candidates):
            # Start with a random point from the first unique mask
            first_mask = unique_masks[0]
            first_points = [p for p in range(v) if first_mask & (1 << p)]
            if not first_points:
                continue
            start_point = random.choice(first_points)
            
            # Build a candidate block
            candidate = [start_point]
            # Add random points to reach size k
            remaining = [p for p in range(v) if p != start_point]
            random.shuffle(remaining)
            candidate.extend(remaining[:k - 1])
            
            # Check if candidate covers all unique masks
            candidate_set = set(candidate)
            covers_all = True
            for mask in unique_masks:
                # Check if mask is subset of candidate
                for p in range(v):
                    if mask & (1 << p) and p not in candidate_set:
                        covers_all = False
                        break
                if not covers_all:
                    break
            
            if not covers_all:
                continue
            
            # Calculate deficiency score
            gain = 0
            for comb in itertools.combinations(candidate, t):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                if mask in uncovered or mask in tsub_to_blocks:
                    # Count as potential gain if it covers something new or helps
                    pass
            
            # Simpler: just check if it covers unique masks and has low load
            load = sum(point_load[p] for p in candidate)
            score = -load  # Prefer low-load replacements
            
            if score > best_score:
                best_score = score
                best_replacement = candidate
        
        if best_replacement is not None:
            # Replace block i with best_replacement
            best_replacement.sort()
            blocks[i] = best_replacement
            # Rebuild tsub_to_blocks
            tsub_to_blocks = {}
            for idx, block in enumerate(blocks):
                for mask in compute_block_tsubsets(block):
                    if mask not in tsub_to_blocks:
                        tsub_to_blocks[mask] = []
                    tsub_to_blocks[mask].append(idx)
    
    # Final validation: ensure all t-subsets are covered
    # If not, fall back to adding blocks for uncovered subsets
    if uncovered and time.time() < local_search_deadline:
        # Rebuild uncovered set
        covered_masks = set()
        for block in blocks:
            for comb in itertools.combinations(block, t):
                mask = 0
                for i in comb:
                    mask |= (1 << i)
                covered_masks.add(mask)
        
        uncovered = set(all_t_subsets) - covered_masks
        
        # Add blocks for remaining uncovered subsets
        while uncovered and time.time() < local_search_deadline:
            first_uncovered = min(uncovered)
            block = []
            for i in range(v):
                if first_uncovered & (1 << i):
                    block.append(i)
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            block.sort()
            blocks.append(block)
            
            for comb in itertools.combinations(block, t):
                mask = 0
                for i in comb:
                    mask |= (1 << i)
                if mask in uncovered:
                    uncovered.discard(mask)
    
    return blocks
