import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Strategy: Stochastic Local Search (Simulated Annealing) with Add/Remove/Repair moves.
    
    This approach starts with a random set of blocks and optimizes by:
    1. Adding blocks that cover many uncovered t-subsets.
    2. Removing redundant blocks.
    3. Swapping blocks to improve coverage efficiency.
    
    It avoids the "greedy trap" of always picking the globally best block, 
    allowing for a more diverse exploration of the solution space.
    """
    start_time = time.time()
    timeout = 145  # Safety margin
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all possible blocks and their coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, list of t-subset indices it covers
    block_covered_tsubs = []
    for block in all_blocks:
        covered = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_covered_tsubs.append(covered)
        
    # Inverse index: for each t-subset, which blocks cover it?
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, covered in enumerate(block_covered_tsubs):
        for t_idx in covered:
            tsub_to_blocks[t_idx].append(b_idx)

    best_blocks = None
    best_size = float('inf')
    
    # Simulated Annealing parameters
    # We will run a few "annealing" runs with different random seeds
    num_runs = 5
    max_iterations_per_run = 5000  # Adjust based on speed
    
    for run in range(num_runs):
        if time.time() - start_time > timeout - 5:
            break
            
        random.seed(run * 98765 + 43210)
        
        # Initialize with a random set of blocks
        # Start with a size slightly larger than the known upper bound (151)
        # or estimate based on density. Let's start with 160 blocks.
        initial_size = 160
        current_blocks = set(random.sample(range(n_blocks), initial_size))
        
        # Calculate initial coverage
        # uncovered: set of t-subset indices not covered
        # tsub_count: how many blocks in current_blocks cover this t-subset
        tsub_count = [0] * n_tsubs
        for b_idx in current_blocks:
            for t_idx in block_covered_tsubs[b_idx]:
                tsub_count[t_idx] += 1
        
        uncovered = set(i for i, c in enumerate(tsub_count) if c == 0)
        
        # Annealing schedule
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        for it in range(max_iterations_per_run):
            if time.time() - start_time > timeout - 2:
                break
                
            # Check if we have a valid covering
            if not uncovered:
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks).copy()
                
                # Try to remove redundant blocks to minimize size
                # Find a block that is redundant (removing it doesn't uncover any tsub)
                redundant_found = False
                # Shuffle current blocks to avoid bias
                blocks_list = list(current_blocks)
                random.shuffle(blocks_list)
                
                for b_idx in blocks_list:
                    if time.time() - start_time > timeout - 1:
                        break
                    # Check if b_idx is redundant
                    is_redundant = True
                    for t_idx in block_covered_tsubs[b_idx]:
                        if tsub_count[t_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        # Remove it
                        current_blocks.remove(b_idx)
                        for t_idx in block_covered_tsubs[b_idx]:
                            tsub_count[t_idx] -= 1
                        # If it was the last block covering some tsub, it would have been caught,
                        # so uncovered remains empty.
                        redundant_found = True
                        break
                
                if not redundant_found:
                    # No more simple removals. If we are stuck at a local minimum of valid solutions,
                    # we might want to try swaps or just stop this run if we've hit a good size.
                    # For now, let's try a swap to see if we can get smaller, or just break.
                    # Actually, if no redundant blocks, we are at a minimal covering for this set.
                    # We could try swapping a block for another that allows further reduction,
                    # but that's complex. Let's just accept this solution for this run.
                    pass
            else:
                # We have uncovered tsubs. We need to add a block or swap.
                # Strategy: Pick a random uncovered tsub, and a random block covering it.
                # Or, pick a block that covers the most uncovered tsubs (greedy add).
                # To maintain SA characteristics, let's mix greedy and random.
                
                if random.random() < 0.8:
                    # Greedy add: Pick block covering max uncovered tsubs
                    # This is expensive to compute from scratch. 
                    # Approximation: Pick a random uncovered tsub, then pick a random block covering it.
                    # This is fast.
                    target_tsub_idx = random.choice(list(uncovered))
                    candidate_blocks = tsub_to_blocks[target_tsub_idx]
                    if candidate_blocks:
                        # Among candidates, pick one that covers many uncovered tsubs?
                        # Let's just pick one at random to allow exploration.
                        # Or pick the one with highest "gain" among candidates.
                        # Let's try: pick random candidate, but if we have time, evaluate gain.
                        # For speed, let's just pick random.
                        b_idx = random.choice(candidate_blocks)
                        
                        # Add block
                        current_blocks.add(b_idx)
                        for t_idx in block_covered_tsubs[b_idx]:
                            if tsub_count[t_idx] == 0:
                                uncovered.discard(t_idx)
                            tsub_count[t_idx] += 1
                else:
                    # Random swap: Remove a random block, add a random block.
                    # Only if current size is not too small?
                    if len(current_blocks) > 50:
                        b_remove = random.choice(list(current_blocks))
                        # Find a block to add that covers at least one uncovered tsub?
                        # Or just any block?
                        # Let's pick a block that covers a random uncovered tsub.
                        target_tsub_idx = random.choice(list(uncovered))
                        candidate_blocks = tsub_to_blocks[target_tsub_idx]
                        if candidate_blocks:
                            b_add = random.choice(candidate_blocks)
                            if b_add not in current_blocks:
                                # Perform swap
                                current_blocks.remove(b_remove)
                                for t_idx in block_covered_tsubs[b_remove]:
                                    tsub_count[t_idx] -= 1
                                    if tsub_count[t_idx] == 0:
                                        uncovered.add(t_idx)
                                
                                current_blocks.add(b_add)
                                for t_idx in block_covered_tsubs[b_add]:
                                    if tsub_count[t_idx] == 0:
                                        uncovered.discard(t_idx)
                                    tsub_count[t_idx] += 1
            
            # Cool down
            T *= cooling_rate
            if T < min_T:
                T = min_T
        
        # End of run. Check best.
        if not uncovered:
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks).copy()
        
        # If we haven't found a valid covering, we might still have a good partial one?
        # No, we must return a valid covering.
        # If best_blocks is None, we need a fallback.
        # But SA usually finds a valid covering quickly with 160 blocks.
        
    if best_blocks is None:
        # Fallback: Simple greedy to ensure validity
        # This is a safety net.
        uncovered = set(range(n_tsubs))
        current_blocks = []
        while uncovered:
            target_tsub_idx = random.choice(list(uncovered))
            candidate_blocks = tsub_to_blocks[target_tsub_idx]
            if not candidate_blocks:
                break
            b_idx = random.choice(candidate_blocks)
            current_blocks.append(b_idx)
            for t_idx in block_covered_tsubs[b_idx]:
                if t_idx in uncovered:
                    uncovered.discard(t_idx)
        best_blocks = current_blocks
        best_size = len(current_blocks)
        
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return final_blocks

# No extra code needed
