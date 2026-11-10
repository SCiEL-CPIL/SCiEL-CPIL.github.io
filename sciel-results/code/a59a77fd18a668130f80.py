"""Strategy: Stochastic Local Search (Simulated Annealing) on a fixed pool of blocks."""
import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # 118s out of 120s limit, leaving 2s safety margin
    
    # 1. Precomputation
    # Map all t-subsets to indices for fast array access
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubsets)
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubsets)}
    
    # Precompute for each point x, which t-subsets contain x?
    # Actually, we need to know which t-subsets are covered by a block.
    # A block B covers t-subsets that are subsets of B.
    # To calculate the "uncovered count" efficiently, we maintain an array cover_count[i]
    # which is the number of blocks covering t-subset i.
    
    # Initialize a random pool of blocks
    # Target size: The best known is 76. We start with a pool of size N=85 to allow some slack
    # and let the optimizer find the most efficient arrangement, then prune.
    # However, if we fix the size, we can't reduce it below the initial size in the SA loop.
    # Strategy: Start with a random set of size N_init. 
    # We want to minimize the number of blocks. 
    # Standard approach for covering designs with SA:
    # 1. Start with a valid covering (e.g., from greedy or random + repair).
    # 2. Allow moves that increase uncovered count if it allows removing a block later?
    #    No, simpler: Start with a pool of size M > optimal. 
    #    Objective: Minimize (number of uncovered t-subsets * large_weight) + (number of blocks).
    #    But we must ensure validity at the end.
    
    # Let's use a different formulation:
    # Start with a greedy solution (like the parent) to get a valid cover.
    # Then, use local search to reduce the number of blocks by finding redundant ones
    # or swapping points to create more redundancy.
    
    # Alternative: Start with random blocks, repair to valid, then optimize.
    
    # Let's try: 
    # 1. Generate a random set of blocks.
    # 2. Use a greedy repair to ensure all t-subsets are covered.
    # 3. Use Local Search to remove redundant blocks or merge/swap to reduce count.
    
    # Actually, the most effective simple change from "Greedy Construct" is "Greedy Construct + Local Search Refinement".
    # The parent already does a basic greedy construct.
    # The inspiration 2 (score 1.0093) tried SA on a fixed pool.
    # Let's try a **Greedy + Local Search (Block Removal & Point Swap)** approach.
    
    # Phase 1: Fast Greedy Construction (similar to parent but slightly more aggressive)
    uncovered = set(all_tsubsets)
    blocks = []
    cover_count = [0] * num_tsubs
    
    # Helper to add a block
    def add_block(b):
        blocks.append(tuple(b))
        for tsub in itertools.combinations(b, t):
            idx = tsub_to_idx[tsub]
            cover_count[idx] += 1
            if cover_count[idx] == 1:
                uncovered.discard(tsub)
                
    # Helper to remove a block
    def remove_block(idx):
        b = blocks[idx]
        del blocks[idx]
        for tsub in itertools.combinations(b, t):
            idx_t = tsub_to_idx[tsub]
            cover_count[idx_t] -= 1
            if cover_count[idx_t] == 0:
                uncovered.add(tsub)

    # Phase 1: Greedy
    while uncovered and time.time() < deadline - 30: # Leave 30s for local search
        # Pick random uncovered
        target = random.choice(list(uncovered))
        block = list(target)
        block_set = set(block)
        
        # Greedily fill block
        while len(block) < k:
            block_combos = list(itertools.combinations(block, t - 1))
            best_gain = -1
            best_point = None
            
            # Optimization: Only check points that appear in uncovered subsets with current block parts
            # For now, simple loop
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in block_combos:
                    subset = tuple(sorted(rest + (x,)))
                    if subset in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None or best_gain <= 0:
                available = [x for x in range(v) if x not in block_set]
                if available:
                    best_point = random.choice(available)
                else:
                    break
            block.append(best_point)
            block_set.add(best_point)
            
        if len(block) == k:
            block.sort()
            add_block(block)

    # Phase 2: Local Search to reduce block count
    # We iterate through blocks and try to remove them.
    # If a block cannot be removed, we try to "move" its coverage responsibility to other blocks
    # by swapping points, or we just keep it.
    # A more powerful move: "Block Swap" or "Point Swap".
    # Let's try: 
    # 1. Try to remove any redundant block.
    # 2. If none, try to modify a block to make it redundant or cover more uncovered (if any).
    # Since we start with a valid cover, uncovered is empty.
    # So the goal is purely to find redundant blocks.
    # A block B is redundant if every t-subset in B is covered by some other block.
    
    # Improvement: Randomized Local Search
    # 1. Randomly select a block B.
    # 2. Try to remove it. If valid, keep it removed.
    # 3. If not valid, identify the "critical" t-subsets in B (those covered only by B).
    # 4. Try to modify B (swap a point) such that it covers fewer critical t-subsets 
    #    OR modify OTHER blocks to cover the critical t-subsets of B.
    
    # Simpler effective strategy: **Iterated Local Search on Block Count**
    # We maintain a valid cover. We try to remove blocks.
    # If we can't remove a block, we perturb the solution (swap points in a few blocks)
    # and try to remove again.
    
    # Let's implement a "Try to Remove" loop with random restarts of the removal order.
    
    # First, do a simple pass to remove obvious redundancies
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            # Check if b is redundant
            redundant = True
            for tsub in itertools.combinations(b, t):
                idx = tsub_to_idx[tsub]
                if cover_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                remove_block(i)
                changed = True
                # Continue loop to check more
    
    # Now, use Local Search to break ties or find hidden redundancies.
    # Move: Pick a block B, pick a point p in B, pick a point q not in B.
    # Swap p and q. 
    # This changes the coverage.
    # We only accept the swap if it leads to a valid cover (uncovered is still empty) 
    # AND it creates a new redundancy (allows removing a block) OR keeps the same count but "shuffles" 
    # to allow future removals.
    # Actually, simpler: Just try swaps that maintain validity. 
    # Then try to remove blocks again.
    
    # To be efficient, we track which t-subsets are "critical" for each block.
    # But recalculating is O(k^t) per block check.
    
    # Let's do a time-boxed random search for block removals.
    # We will randomly select a block, try to remove it.
    # If it fails, we identify the blocking t-subsets.
    # Then we randomly pick one of those t-subsets and try to "fix" it by modifying another block 
    # to cover it, so that our target block becomes redundant.
    
    # This is complex. Let's stick to a simpler heuristic that often works well:
    # **Randomized Removal Order with Point Swaps**
    
    # 1. Shuffle blocks.
    # 2. Iterate. Try to remove.
    # 3. If fail, pick a random point in the block, swap with a random point outside.
    #    Check if validity is maintained.
    #    If valid, keep swap. Try to remove again (maybe the swap made it redundant? Unlikely directly).
    #    Actually, the swap is to help OTHER blocks become redundant or to prepare for removal.
    
    # Let's try a different tactic: **Simulated Annealing on the set of blocks**
    # State: Set of blocks.
    # Energy: Number of uncovered t-subsets.
    # We want Energy = 0.
    # Start with the greedy solution (Energy = 0).
    # Move: Remove a block. Energy increases by (number of t-subsets uniquely covered by that block).
    # If Energy > 0, we must repair.
    # Repair: Add a block that covers the most uncovered t-subsets.
    # This is basically what the greedy does.
    
    # The key to improvement is **Global Optimization**.
    # Let's try: **Start with a random valid cover, then use Local Search to minimize block count.**
    # How to start with a random valid cover?
    # 1. Generate random blocks until valid. (Too slow for v=24, k=7).
    # 2. Use the greedy solution as the start.
    
    # Let's refine the "Local Search" to be more aggressive.
    # We will perform "Block Merges" or "Point Swaps" that might temporarily increase uncovered count,
    # but allow a net reduction in blocks after repair.
    
    # Given time constraints, a simpler robust improvement:
    # **Multiple Greedy Passes with Randomization**
    # The greedy algorithm is sensitive to the order of uncovered subsets.
    # We can run the greedy construction multiple times with different random seeds/orders
    # and keep the best result.
    # This is a "Restart" strategy.
    
    # Let's implement:
    # 1. Run greedy construction 5-10 times (if time permits).
    # 2. For each result, run a local search refinement (remove redundant blocks).
    # 3. Keep the smallest valid cover.
    
    # This is structurally different from the single-pass greedy + post-process.
    # It explores the space of greedy solutions.
    
    best_blocks = None
    best_size = float('inf')
    
    num_restarts = 5
    time_per_restart = (deadline - start_time - 5) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Reset state
        uncovered = set(all_tsubsets)
        blocks = []
        cover_count = [0] * num_tsubs
        
        # Greedy Construction
        # Randomize the choice of target
        while uncovered and time.time() < start_time + r*time_per_restart + time_per_restart*0.8:
            target = random.choice(list(uncovered))
            block = list(target)
            block_set = set(block)
            
            while len(block) < k:
                block_combos = list(itertools.combinations(block, t - 1))
                best_gain = -1
                best_point = None
                
                # Randomize tie-breaking
                candidates = []
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in block_combos:
                        subset = tuple(sorted(rest + (x,)))
                        if subset in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [x]
                    elif gain == best_gain:
                        candidates.append(x)
                
                if candidates:
                    best_point = random.choice(candidates)
                else:
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_point = random.choice(available)
                    else:
                        break
                block.append(best_point)
                block_set.add(best_point)
                
            if len(block) == k:
                block.sort()
                # Add block
                blocks.append(tuple(block))
                for tsub in itertools.combinations(block, t):
                    idx = tsub_to_idx[tsub]
                    cover_count[idx] += 1
                    if cover_count[idx] == 1:
                        uncovered.discard(tsub)
        
        # Local Search Refinement: Remove redundant blocks
        # We do this iteratively until no more can be removed
        changed = True
        while changed and time.time() < start_time + r*time_per_restart + time_per_restart*0.95:
            changed = False
            # Shuffle order to break symmetry
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(blocks):
                    continue
                b = blocks[i]
                redundant = True
                for tsub in itertools.combinations(b, t):
                    idx = tsub_to_idx[tsub]
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove
                    del blocks[i]
                    for tsub in itertools.combinations(b, t):
                        idx_t = tsub_to_idx[tsub]
                        cover_count[idx_t] -= 1
                    changed = True
                    # Restart index loop? No, just continue, but indices are now stale.
                    # Break and restart the shuffle loop for safety
                    break
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    if best_blocks is None:
        # Fallback to the last generated blocks if they were valid
        # But we need to ensure validity.
        # The greedy loop ensures validity if it completes.
        # If it didn't complete, it's invalid.
        # We assume it completes.
        return [] # Should not happen
        
    return best_blocks
