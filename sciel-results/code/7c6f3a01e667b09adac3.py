import time
import random
import itertools
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets and map them to indices for fast lookup
    all_t_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    n_subs = len(all_t_subs)
    
    # Precompute all k-blocks and their t-subsets
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)
    
    # For each block, store the list of indices of t-subsets it covers
    block_subsets = []
    for block in all_k_blocks:
        subs = [sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_subsets.append(subs)
    
    # Reverse index: for each t-subset, list of blocks that cover it
    sub_to_blocks = [[] for _ in range(n_subs)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            sub_to_blocks[s].append(i)
    
    best_blocks = None
    best_len = float('inf')
    
    def greedy_set_cover(seed_val):
        random.seed(seed_val)
        covered = [0] * n_subs  # Count of how many times each subset is covered
        selected_blocks = []
        
        # For efficiency, we maintain a count of how many uncovered subsets each block covers
        # Initially, this is just the number of subsets in the block (56)
        # But we need to update this dynamically.
        # A faster approach:
        # Maintain `uncovered_count` for each block? No, that's expensive to update.
        # Instead, at each step, we can scan all blocks to find the one with max new coverage.
        # n_blocks = 75582. n_steps ~ 300.
        # 75582 * 300 * (cost to check coverage)
        # Cost to check coverage for a block: iterate its 56 subsets, check if covered[s]==0.
        # Total ops: 75582 * 300 * 56 ≈ 1.26 billion. This is too slow in pure Python.
        
        # Alternative: Use the reverse index.
        # When we select a block B, we update covered[s] for s in B.
        # This doesn't directly help find the next best block without scanning.
        
        # Let's try a different heuristic: Randomized Greedy with a smaller candidate pool?
        # Or use the "Greedy Point Addition" but with a different randomization strategy?
        # The current program already does Randomized Greedy Point Addition.
        
        # Let's try a different approach: Local Search / Hill Climbing.
        # Start with a random set of blocks, then iteratively replace a block with another
        # that covers more uncovered subsets.
        
        # Actually, let's try a "Weighted" Greedy Set Cover.
        # Assign random weights to subsets? No, that's not standard.
        
        # Let's try to speed up the Set Cover Greedy using NumPy?
        # But we can't use external libs easily in the inner loop if it's not fast.
        
        # Given the time constraint, let's stick to a faster heuristic.
        # How about "Greedy with Lookahead"?
        # Or "Best-Fit Decreasing" style?
        
        # Let's try a different randomization in Greedy Point Addition:
        # Instead of picking a random start subset, pick the one with the highest "difficulty"
        # (e.g., fewest blocks can cover it? No, all blocks that contain it can cover it).
        
        # Let's try: Simulated Annealing style?
        # Start with a valid covering, then try to remove blocks.
        
        # Strategy:
        # 1. Generate a valid covering using Greedy Point Addition (fast).
        # 2. Try to remove redundant blocks: if a block's subsets are all covered by other blocks, remove it.
        # 3. Repeat removal until no more can be removed.
        # 4. Randomly swap a block with a random other block if it maintains coverage? (Hard to check).
        
        # Let's implement the "Remove Redundant" strategy.
        
        return None # Placeholder
    
    # Let's implement the "Remove Redundant" strategy on top of Greedy Point Addition.
    
    def greedy_point_addition(seed_val):
        random.seed(seed_val)
        uncovered = set(range(n_subs))  # Use indices for speed
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset
            start_idx = random.choice(list(uncovered))
            start_sub = all_t_subs[start_idx]
            block = list(start_sub)
            
            while len(block) < k:
                best_gain = -1
                best_candidates = []
                
                # Precompute combinations of current block of size t-1
                block_combos = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    for rest in block_combos:
                        new_sub = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx.get(new_sub)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if best_gain <= 0:
                    available = [x for x in range(v) if x not in block]
                    if available:
                        x = random.choice(available)
                    else:
                        break
                else:
                    x = random.choice(best_candidates)
                
                block.append(x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                uncovered.discard(idx)
        
        return blocks
    
    def remove_redundant(blocks):
        # Check if a block is redundant: all its t-subsets are covered by other blocks.
        # This is O(N_blocks * C(k,t) * N_blocks) if done naively.
        # We can optimize by maintaining a count of how many blocks cover each subset.
        
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks
        
        # Count coverage for each subset
        cover_count = [0] * n_subs
        block_subsets_list = []
        for block in blocks:
            subs = [sub_to_idx[sub] for sub in itertools.combinations(block, t)]
            block_subsets_list.append(subs)
            for s in subs:
                cover_count[s] += 1
        
        # A block is redundant if for all s in its subsets, cover_count[s] > 1
        # But removing a block changes cover_counts.
        # We can iteratively remove blocks that are "safe" to remove.
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, block in enumerate(blocks):
                redundant = True
                for s in block_subsets_list[i]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for s in block_subsets_list[i]:
                        cover_count[s] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks
            # Re-index block_subsets_list? It's getting complex.
            # Let's just do a simpler version: one pass.
            
        return blocks
    
    # Actually, the iterative removal is tricky with the indices.
    # Let's just do a simple greedy removal:
    # Sort blocks by some heuristic (e.g., number of unique subsets they cover).
    # Try to remove the one that covers the fewest unique subsets.
    
    def greedy_remove(blocks):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks
        
        cover_count = [0] * n_subs
        block_subsets_list = []
        for block in blocks:
            subs = [sub_to_idx[sub] for sub in itertools.combinations(block, t)]
            block_subsets_list.append(subs)
            for s in subs:
                cover_count[s] += 1
        
        # Calculate "unique coverage" for each block: how many subsets it covers that no other block covers.
        unique_counts = [0] * n_blocks
        for i in range(n_blocks):
            for s in block_subsets_list[i]:
                if cover_count[s] == 1:
                    unique_counts[i] += 1
        
        # Sort blocks by unique_counts (ascending)
        order = sorted(range(n_blocks), key=lambda i: unique_counts[i])
        
        new_blocks = []
        new_block_subsets = []
        
        for i in order:
            # Check if we can remove block i
            # It is removable if for all s in block_subsets_list[i], cover_count[s] > 1
            removable = True
            for s in block_subsets_list[i]:
                if cover_count[s] == 1:
                    removable = False
                    break
            
            if removable:
                for s in block_subsets_list[i]:
                    cover_count[s] -= 1
            else:
                new_blocks.append(blocks[i])
                new_block_subsets.append(block_subsets_list[i])
        
        return new_blocks
    
    # Main loop: Generate, then Remove.
    run_id = 0
    while time.time() < deadline:
        # Generate a covering
        blocks = greedy_point_addition(run_id * 1000 + 42)
        
        # Remove redundant blocks
        blocks = greedy_remove(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        run_id += 1
        if time.time() > deadline:
            break
    
    if best_blocks is None:
        return greedy_point_addition(42)
        
    return best_blocks
