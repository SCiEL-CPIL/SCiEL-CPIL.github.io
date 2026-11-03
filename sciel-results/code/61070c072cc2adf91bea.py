import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) with as few blocks as possible.
    Uses Greedy initialization followed by Simulated Annealing local search.
    """
    # For this specific task, v=24, k=7, t=3
    # But we keep it general as per interface
    
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for Python overhead and return
    
    # --- Bitmask Encoding for t-subsets ---
    # Encode a t-subset {a, b, c} with a < b < c as a single integer
    # For v=24, we need 5 bits per element. 5*3 = 15 bits. Fits in int.
    def encode_subset(subset):
        # subset is a sorted tuple of length t
        code = 0
        for i, val in enumerate(subset):
            code |= (val << (i * 5)) # 5 bits per value, enough for v<=32
        return code

    def decode_subset(code):
        # Not strictly needed if we just use codes as keys
        pass

    # Precompute all t-subsets and their codes
    all_t_subsets = []
    code_to_subset = {}
    for comb in itertools.combinations(range(v), t):
        code = encode_subset(comb)
        all_t_subsets.append(code)
        code_to_subset[code] = comb
        
    total_t_subsets = len(all_t_subsets)
    
    # --- Greedy Initialization ---
    def greedy_cover():
        uncovered = set(all_t_subsets)
        blocks = []
        # Precompute t-subsets for potential blocks? No, too many.
        # Just iterate.
        
        while uncovered:
            # Pick a random uncovered subset to seed the block
            seed_code = rng.choice(list(uncovered))
            seed_subset = code_to_subset[seed_code]
            
            # Start with the seed subset
            block = list(seed_subset)
            rng.shuffle(block)
            
            # Greedily add points to maximize coverage of uncovered subsets
            while len(block) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in block]
                rng.shuffle(cands)
                
                for x in cands:
                    gain = 0
                    # Check how many new t-subsets are covered by adding x
                    for rest in itertools.combinations(block, t - 1):
                        # Create new subset
                        new_sub = list(rest) + [x]
                        new_sub.sort()
                        code = encode_subset(tuple(new_sub))
                        if code in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x is None or best_gain == 0:
                    # If no gain, just pick a random one to fill the block
                    # This ensures we get a full block, even if it doesn't help immediately
                    # But usually, there is some gain.
                    if best_x is None:
                        best_x = rng.choice(cands)
                        
                block.append(best_x)
                
            block.sort()
            b_tuple = tuple(block)
            
            # Add block and update uncovered
            if b_tuple not in blocks:
                blocks.append(b_tuple)
                for c in itertools.combinations(block, t):
                    uncovered.discard(encode_subset(c))
                    
        return blocks

    # --- Local Search ---
    
    def run_local_search(initial_blocks, time_budget):
        blocks = list(initial_blocks)
        
        # Data structures for fast updates
        # counts[code] = number of blocks covering this t-subset
        counts = {}
        for b in blocks:
            for c in itertools.combinations(b, t):
                code = encode_subset(c)
                counts[code] = counts.get(code, 0) + 1
                
        # Track number of subsets with count == 1 (critical)
        # and number of subsets with count == 0 (uncovered, should be 0 in valid cover)
        num_critical = sum(1 for c in counts.values() if c == 1)
        num_uncovered = 0 # Should be 0
        
        best_blocks = list(blocks)
        best_count = len(blocks)
        
        # Precompute t-subsets for each block to speed up swap evaluation
        # block_subs[i] = list of codes for block i
        block_subs = []
        for b in blocks:
            subs = [encode_subset(c) for c in itertools.combinations(b, t)]
            block_subs.append(subs)
            
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            
            # 1. Try to remove a redundant block
            removed_any = False
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                # Check if block i is redundant
                redundant = True
                for code in block_subs[i]:
                    if counts[code] <= 1:
                        redundant = False
                        break
                        
                if redundant:
                    # Remove block i
                    for code in block_subs[i]:
                        counts[code] -= 1
                        if counts[code] == 1:
                            num_critical += 1
                        elif counts[code] == 2:
                            num_critical -= 1
                        # If it becomes 0, we have a problem (shouldn't happen if redundant check passed)
                        if counts[code] == 0:
                            del counts[code]
                            num_uncovered += 1 # Should not happen in valid state
                
                    blocks.pop(i)
                    block_subs.pop(i)
                    
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                        
                    removed_any = True
                    T = max(T_min, T * 0.9) # Cool down on improvement
                    break
                    
            if removed_any:
                continue
                
            # 2. Try a swap
            if len(blocks) > 0:
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                old_subs = block_subs[i]
                
                # Identify critical subsets of old_block
                critical_codes = [code for code in old_subs if counts[code] == 1]
                
                if not critical_codes:
                    # No critical subsets, swapping is neutral regarding validity
                    # We can still swap to improve redundancy profile, but let's skip for now
                    # or do a random swap
                    continue
                
                # Points in critical subsets
                points_needed = set()
                for code in critical_codes:
                    points_needed.update(code_to_subset[code])
                    
                if len(points_needed) > k:
                    continue
                    
                # Generate candidate new block
                # Must contain points_needed
                # Fill rest with random points
                remaining_slots = k - len(points_needed)
                cands = [x for x in range(v) if x not in points_needed]
                
                # Try a few candidates
                best_new_block = None
                best_new_subs = None
                best_delta_critical = 0
                best_delta_total = 0
                
                # Heuristic: try to cover other critical subsets too?
                # For now, just random completion
                for _ in range(20):
                    if not cands:
                        break
                    fill = rng.sample(cands, min(remaining_slots, len(cands)))
                    new_block_set = set(points_needed)
                    new_block_set.update(fill)
                    
                    if len(new_block_set) < k:
                        # Fill with any
                        for x in cands:
                            if len(new_block_set) >= k: break
                            new_block_set.add(x)
                            
                    new_block = tuple(sorted(new_block_set))
                    if new_block == old_block:
                        continue
                    if new_block in blocks:
                        continue
                        
                    # Evaluate swap
                    # Remove old_block
                    for code in old_subs:
                        counts[code] -= 1
                        if counts[code] == 1:
                            num_critical += 1
                        elif counts[code] == 2:
                            num_critical -= 1
                        if counts[code] == 0:
                            del counts[code]
                            num_uncovered += 1
                            
                    # Add new_block
                    new_subs_list = [encode_subset(c) for c in itertools.combinations(new_block, t)]
                    for code in new_subs_list:
                        old_c = counts.get(code, 0)
                        new_c = old_c + 1
                        counts[code] = new_c
                        if old_c == 1:
                            num_critical -= 1
                        elif old_c == 2:
                            num_critical += 1
                        if old_c == 0:
                            num_uncovered -= 1
                            
                    delta_critical = num_critical - (num_critical - (num_critical - num_critical)) # This logic is messy
                    # Let's just track the change directly
                    # Actually, I modified the global counters. I need to revert if I don't accept.
                    # Let's store the state before modification.
                    
                    # This is getting complicated with in-place updates.
                    # Let's use a "dry run" evaluation.
                    
                    # REVERT DRY RUN
                    for code in new_subs_list:
                        old_c = counts[code] - 1
                        counts[code] = old_c
                        if old_c == 1:
                            num_critical += 1
                        elif old_c == 2:
                            num_critical -= 1
                        if old_c == 0:
                            num_uncovered += 1
                            
                    for code in old_subs:
                        old_c = counts[code] + 1
                        counts[code] = old_c
                        if old_c == 1:
                            num_critical -= 1
                        elif old_c == 2:
                            num_critical += 1
                        if old_c == 0:
                            num_uncovered -= 1
                            
                    # Okay, the dry run revert is done.
                    # Now, how to evaluate?
                    # We want to minimize num_critical.
                    # Let's calculate the change in num_critical for this swap.
                    
                    delta_crit = 0
                    # Removing old_block
                    for code in old_subs:
                        c = counts[code]
                        if c == 2:
                            # Becomes 1, so critical increases by 1
                            delta_crit += 1
                        elif c == 1:
                            # Becomes 0, critical decreases by 1 (but uncovered increases)
                            delta_crit -= 1
                            
                    # Adding new_block
                    for code in new_subs_list:
                        c = counts.get(code, 0)
                        if c == 1:
                            # Becomes 2, critical decreases by 1
                            delta_crit -= 1
                        elif c == 2:
                            # Becomes 3, critical increases by 1
                            delta_crit += 1
                            
                    # We accept if delta_crit < 0 (improvement)
                    # or with prob exp(-delta_crit / T) if delta_crit > 0
                    
                    if delta_crit < best_delta_critical or (delta_crit == best_delta_critical and rng.random() < 0.5):
                        best_delta_critical = delta_crit
                        best_new_block = new_block
                        best_new_subs = new_subs_list
                        
                if best_new_block is not None:
                    # Acceptance
                    accept = False
                    if best_delta_critical <= 0:
                        accept = True
                    else:
                        if rng.random() < math.exp(-best_delta_critical / T):
                            accept = True
                            
                    if accept:
                        # Apply swap
                        # Remove old
                        for code in old_subs:
                            counts[code] -= 1
                            if counts[code] == 1:
                                num_critical += 1
                            elif counts[code] == 2:
                                num_critical -= 1
                            if counts[code] == 0:
                                del counts[code]
                                num_uncovered += 1
                                
                        # Add new
                        for code in best_new_subs:
                            old_c = counts.get(code, 0)
                            counts[code] = old_c + 1
                            if old_c == 1:
                                num_critical -= 1
                            elif old_c == 2:
                                num_critical += 1
                            if old_c == 0:
                                num_uncovered -= 1
                                
                        blocks[i] = best_new_block
                        block_subs[i] = best_new_subs
                        
                        if len(blocks) < best_count:
                            best_count = len(blocks)
                            best_blocks = list(blocks)
                        
            T = max(T_min, T * cooling_rate)
            
        return best_blocks

    # Main Loop
    best_overall = None
    best_overall_count = float('inf')
    
    # Run greedy once to get a baseline
    initial_blocks = greedy_cover()
    best_overall = initial_blocks
    best_overall_count = len(initial_blocks)
    
    # Run local search with multiple restarts if time permits
    # Each restart gets a portion of the time
    num_restarts = 5
    time_per_restart = (time_limit - (time.time() - start_time)) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > start_time + time_limit - 5:
            break
            
        # New greedy start
        if r > 0:
            current_blocks = greedy_cover()
        else:
            current_blocks = initial_blocks
            
        # Local search
        result = run_local_search(current_blocks, time_per_restart)
        
        if len(result) < best_overall_count:
            best_overall_count = len(result)
            best_overall = result
            
    return [list(b) for b in best_overall]

# EVOLVE-BLOCK-END
