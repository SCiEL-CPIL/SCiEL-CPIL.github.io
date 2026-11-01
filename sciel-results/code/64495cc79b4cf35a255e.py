import itertools
import random
import time
import math
from collections import Counter

def construct(v, k, t):
    all_t_subsets = list(itertools.combinations(range(v), t))
    all_t_set = set(all_t_subsets)
    
    # 1. Fast Greedy Initialization
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = set(all_t_set)
        blocks = []
        while uncovered:
            # Pick a random uncovered subset to anchor the block
            start_subset = random.choice(list(uncovered))
            block = list(start_subset)
            
            # Complete the block greedily
            while len(block) < k:
                best_point = None
                best_gain = -1
                # To speed up, we only check a subset of points or use a heuristic
                # Here we check all, but optimized
                current_t_minus_1 = list(itertools.combinations(block, t - 1))
                
                # Optimization: Pre-compute potential gains is hard dynamically.
                # Just iterate points.
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in current_t_minus_1:
                        new_subset = tuple(sorted(rest + (x,)))
                        if new_subset in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        best_point = random.choice(candidates)
                    else:
                        break
                block.append(best_point)
            
            block.sort()
            blocks.append(tuple(block))
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        return blocks

    # Generate initial solution
    best_initial = None
    best_len = float('inf')
    start_time = time.time()
    
    # Run a few greedy instances
    for i in range(20):
        if time.time() - start_time > 10:
            break
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_initial = blocks

    if best_initial is None:
        best_initial = greedy_cover(0)

    # 2. Simulated Annealing Optimization
    # State: list of blocks (tuples)
    current_blocks = [tuple(b) for b in best_initial]
    
    # Precompute t-subsets for each block for fast access
    block_subs = [list(itertools.combinations(b, t)) for b in current_blocks]
    
    # Coverage counts
    cov_count = Counter()
    for subs in block_subs:
        for s in subs:
            cov_count[s] += 1
    
    # Helper to check validity and count blocks
    def get_state_metrics(blocks, block_subs, cov_count):
        # Check for uncovered
        uncovered_count = sum(1 for s in all_t_set if cov_count.get(s, 0) == 0)
        return len(blocks), uncovered_count

    # Helper to remove redundant blocks greedily
    def clean_redundant(blocks, block_subs, cov_count):
        changed = True
        while changed:
            changed = False
            new_blocks = []
            new_block_subs = []
            for i, b in enumerate(blocks):
                is_redundant = True
                for s in block_subs[i]:
                    if cov_count.get(s, 0) <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove
                    for s in block_subs[i]:
                        cov_count[s] -= 1
                        if cov_count[s] == 0:
                            del cov_count[s]
                    changed = True
                else:
                    new_blocks.append(b)
                    new_block_subs.append(block_subs[i])
            blocks = new_blocks
            block_subs = new_block_subs
        return blocks, block_subs, cov_count

    # Initial cleanup
    current_blocks, block_subs, cov_count = clean_redundant(current_blocks, block_subs, cov_count)
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    start_time_sa = time.time()
    time_limit_sa = 100 # Leave 10s for final cleanup/checks
    
    current_energy = len(current_blocks)
    
    while time.time() - start_time_sa < time_limit_sa:
        # Generate Move: Swap a random block with a new random block
        if len(current_blocks) == 0:
            break
            
        idx = random.randint(0, len(current_blocks) - 1)
        old_block = current_blocks[idx]
        old_subs = block_subs[idx]
        
        # Generate new block
        # Bias towards covering unique subsets of old_block?
        # For now, random sample, but we can be smarter later.
        new_block = tuple(sorted(random.sample(range(v), k)))
        new_subs = list(itertools.combinations(new_block, t))
        
        # Calculate change
        # 1. Remove old block effects
        for s in old_subs:
            cov_count[s] -= 1
            if cov_count[s] == 0:
                del cov_count[s]
        
        # 2. Add new block effects
        for s in new_subs:
            cov_count[s] += 1
        
        # 3. Check for new redundancies and remove them (Greedy Repair)
        # We simulate the removal to see the new state
        # This is expensive if done naively. 
        # Optimization: Only check blocks that might have become redundant.
        # A block becomes redundant if all its subsets have count > 1.
        # Removing old_block decreased counts. Adding new_block increased counts.
        # So blocks that *were* close to redundant might become redundant if their counts went UP?
        # No, redundancy requires count > 1. 
        # If a block had a subset with count 1, it was NOT redundant.
        # If we ADD a block, counts go UP. A block with count 1 becomes count 2. 
        # So adding a block can make OTHER blocks redundant.
        # Removing a block decreases counts. It cannot make a block redundant (it might make it invalid).
        
        # So, after swap, we should check if any *other* blocks are now redundant.
        # Also, is the new state valid?
        
        # Let's identify candidates for redundancy removal:
        # Any block that shares a subset with new_block might have its counts increased.
        # Actually, just run a fast check on all blocks? 
        # N is ~150. Blocks per block is C(7,4)=35. 
        # Checking redundancy for 150 blocks * 35 subsets = 5250 ops. Very fast.
        
        new_blocks_list = list(current_blocks)
        new_blocks_list[idx] = new_block
        new_block_subs_list = list(block_subs)
        new_block_subs_list[idx] = new_subs
        
        # Perform greedy cleanup on the modified state
        # We need to be careful not to mutate the global cov_count if we reject the move.
        # So we copy cov_count? Counter copy is O(N). N=35*150 ~ 5000. Acceptable.
        temp_cov = cov_count.copy()
        
        # Re-apply changes to temp_cov
        # Wait, we already modified cov_count. 
        # Let's revert cov_count first, then apply to temp?
        # Better: Keep cov_count global, but for move evaluation, use a copy.
        
        # REVERT global cov_count changes for evaluation
        for s in new_subs:
            temp_cov[s] -= 1
            if temp_cov[s] == 0:
                del temp_cov[s]
        for s in old_subs:
            temp_cov[s] += 1
            
        # Now temp_cov is the state BEFORE the move.
        # Apply move to temp_cov
        for s in old_subs:
            temp_cov[s] -= 1
            if temp_cov[s] == 0:
                del temp_cov[s]
        for s in new_subs:
            temp_cov[s] += 1
            
        # Now perform greedy redundancy removal on temp state
        temp_blocks = new_blocks_list
        temp_block_subs = new_block_subs_list
        
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(temp_blocks):
                is_red = True
                for s in temp_block_subs[i]:
                    if temp_cov.get(s, 0) <= 1:
                        is_red = False
                        break
                if is_red:
                    for s in temp_block_subs[i]:
                        temp_cov[s] -= 1
                        if temp_cov[s] == 0:
                            del temp_cov[s]
                    temp_blocks.pop(i)
                    temp_block_subs.pop(i)
                    changed = True
                else:
                    i += 1
        
        # Calculate new energy
        # Check validity
        uncovered = 0
        for s in all_t_set:
            if temp_cov.get(s, 0) == 0:
                uncovered += 1
                if uncovered > 10: # Early exit if very bad
                    break
                    
        new_energy = len(temp_blocks) + (1000 * uncovered if uncovered > 0 else 0)
        
        delta = new_energy - current_energy
        
        # Acceptance
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            current_blocks = temp_blocks
            block_subs = temp_block_subs
            cov_count = temp_cov
            current_energy = new_energy
            # Note: We might have accepted an invalid solution if delta was small enough?
            # The penalty 1000 makes invalid solutions very high energy.
            # If we accept invalid, we will try to fix it in next steps.
            
        # Cool
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final Cleanup to ensure validity and minimal blocks
    final_blocks, final_subs, final_cov = clean_redundant(current_blocks, block_subs, cov_count)
    
    # If still invalid (shouldn't happen with high penalty), try to repair
    # Check validity
    uncovered = [s for s in all_t_set if final_cov.get(s, 0) == 0]
    while uncovered:
        # Add random blocks to cover uncovered
        s = uncovered.pop()
        # Find a block covering s
        # Just add a random block containing s?
        # Better: Add a block that covers s and maximizes other coverage
        best_block = None
        best_gain = -1
        # Sample some blocks
        for _ in range(20):
            # Create block containing s
            rest = random.sample([x for x in range(v) if x not in s], k - t)
            b = tuple(sorted(list(s) + rest))
            # Calculate gain
            gain = 0
            for sub in itertools.combinations(b, t):
                if sub in set(uncovered):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
        if best_block:
            final_blocks.append(best_block)
            final_subs.append(list(itertools.combinations(best_block, t)))
            for s in itertools.combinations(best_block, t):
                final_cov[s] += 1
            # Re-run cleanup
            final_blocks, final_subs, final_cov = clean_redundant(final_blocks, final_subs, final_cov)
            # Recalculate uncovered
            uncovered = [s for s in all_t_set if final_cov.get(s, 0) == 0]
        else:
            break
            
    return [list(b) for b in final_blocks]
