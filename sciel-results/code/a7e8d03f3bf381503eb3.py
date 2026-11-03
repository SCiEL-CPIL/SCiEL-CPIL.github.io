import random
import itertools
import math

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Simulated Annealing Local Search.
    """
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index for faster counting
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    def get_block_t_subsets(block):
        """Return list of indices of t-subsets covered by this block."""
        block_set = set(block)
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[ts])
        return indices

    def random_block():
        return tuple(sorted(random.sample(points, k)))

    # Initial state: Random set of blocks
    # Start with a reasonable number of blocks, maybe slightly above lower bound or greedy estimate
    # Greedy usually gives ~335. Let's start with 350.
    initial_num_blocks = 350
    
    # State
    blocks = [random_block() for _ in range(initial_num_blocks)]
    # Coverage counts
    coverage_counts = [0] * num_t_subsets
    uncovered_count = num_t_subsets
    
    # Initialize coverage
    for b in blocks:
        for idx in get_block_t_subsets(b):
            if coverage_counts[idx] == 0:
                uncovered_count -= 1
            coverage_counts[idx] += 1
    
    best_blocks = blocks[:]
    best_uncovered = uncovered_count
    best_num_blocks = len(blocks)
    
    # Simulated Annealing Parameters
    T = 100.0
    cooling = 0.99
    max_iters = 5000 # Adjust based on time limit
    # Time limit check is tricky inside, but 5000 iterations * 56 ops is fast.
    # Each iteration involves O(1) or O(56) updates.
    # 5000 * 56 = 280k ops. Very fast.
    # We can do more iterations.
    max_iters = 20000
    
    for i in range(max_iters):
        if uncovered_count == 0:
            # If covered, try to reduce blocks
            # Objective is primarily to cover, secondarily to minimize count
            pass
            
        # Generate a move
        move_type = random.random()
        if move_type < 0.33:
            # Add a block
            new_block = random_block()
            # Check if it's a duplicate? For SA, duplicates are bad but allowed in state, 
            # but we should avoid them in final output. 
            # Actually, adding a duplicate block doesn't change coverage but increases count.
            # We can check against current blocks set for efficiency.
            # For now, just add.
            
            # Calculate delta
            delta_uncovered = 0
            added_indices = []
            for idx in get_block_t_subsets(new_block):
                if coverage_counts[idx] == 0:
                    delta_uncovered -= 1 # Uncovered becomes covered
                added_indices.append(idx)
            
            # Objective function: We want to minimize uncovered_count.
            # If uncovered_count > 0, priority is to reduce it.
            # If uncovered_count == 0, priority is to reduce number of blocks.
            
            if uncovered_count > 0:
                # Cost change: -delta_uncovered (since we want to decrease uncovered)
                # We accept if delta_uncovered < 0 (improvement)
                # Or with probability exp(-delta/T) if delta > 0? 
                # Let's define DeltaCost.
                # If we reduce uncovered, DeltaCost is negative.
                delta_cost = -delta_uncovered
            else:
                # Already covered. Adding a block increases block count. Bad.
                # Unless we are swapping.
                delta_cost = 1.0 # Penalize increasing block count
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                blocks.append(new_block)
                for idx in added_indices:
                    coverage_counts[idx] += 1
                if delta_uncovered < 0:
                    uncovered_count += delta_uncovered # delta_uncovered is negative
                # Update best
                if uncovered_count < best_uncovered or (uncovered_count == best_uncovered and len(blocks) < best_num_blocks):
                    best_uncovered = uncovered_count
                    best_num_blocks = len(blocks)
                    best_blocks = blocks[:]
                    
        elif move_type < 0.66:
            # Remove a block
            if len(blocks) <= 1:
                continue
            idx_to_remove = random.randint(0, len(blocks) - 1)
            block_to_remove = blocks[idx_to_remove]
            
            delta_uncovered = 0
            removed_indices = []
            for idx in get_block_t_subsets(block_to_remove):
                if coverage_counts[idx] == 1:
                    delta_uncovered += 1 # Covered becomes uncovered
                removed_indices.append(idx)
                
            if uncovered_count > 0:
                # Removing a block might uncover subsets. Bad.
                delta_cost = delta_uncovered
            else:
                # Already covered. Removing a block is good if it doesn't uncover anything.
                # If it uncovers nothing, delta_uncovered = 0.
                # If it uncovers some, delta_uncovered > 0.
                # We want to minimize block count.
                # So if delta_uncovered == 0, we improve block count.
                delta_cost = -1.0 if delta_uncovered == 0 else delta_uncovered
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                blocks.pop(idx_to_remove)
                for idx in removed_indices:
                    coverage_counts[idx] -= 1
                if delta_uncovered > 0:
                    uncovered_count += delta_uncovered
                # Update best
                if uncovered_count < best_uncovered or (uncovered_count == best_uncovered and len(blocks) < best_num_blocks):
                    best_uncovered = uncovered_count
                    best_num_blocks = len(blocks)
                    best_blocks = blocks[:]
            
        else:
            # Swap a block
            if len(blocks) == 0:
                continue
            idx_to_swap = random.randint(0, len(blocks) - 1)
            old_block = blocks[idx_to_swap]
            new_block = random_block()
            
            if old_block == new_block:
                continue
                
            # Calculate delta
            # Remove old, Add new
            
            delta_uncovered = 0
            old_indices = get_block_t_subsets(old_block)
            new_indices = get_block_t_subsets(new_block)
            
            # Temporary storage for counts to avoid double updating
            # Since we are just evaluating, we can compute the net change.
            # Count change for each index:
            # -1 for old_block, +1 for new_block
            
            # This is slightly complex to do in one pass without modifying global state.
            # Let's just simulate the counts locally for the affected indices.
            
            # Collect all affected indices
            affected = set(old_indices) | set(new_indices)
            
            # Calculate new uncovered status for these
            # Current state: coverage_counts
            # New state: coverage_counts[idx] - (1 if idx in old_indices else 0) + (1 if idx in new_indices else 0)
            
            num_newly_uncovered = 0
            num_newly_covered = 0
            
            for idx in affected:
                c_old = coverage_counts[idx]
                c_new = c_old
                if idx in old_indices:
                    c_new -= 1
                if idx in new_indices:
                    c_new += 1
                
                if c_old == 0 and c_new > 0:
                    num_newly_covered += 1
                elif c_old > 0 and c_new == 0:
                    num_newly_uncovered += 1
            
            delta_uncovered = num_newly_uncovered - num_newly_covered
            
            if uncovered_count > 0:
                delta_cost = delta_uncovered
            else:
                # If covered, we don't care about uncovered change as long as it stays 0?
                # Actually if we swap, we might uncover some.
                # If delta_uncovered > 0, we lose coverage.
                # If delta_uncovered <= 0, we maintain or gain coverage.
                # Since we are swapping, block count stays same.
                # So we only accept if delta_uncovered <= 0?
                # Or if it's a small increase, accept with prob?
                # If we are already covered, we should avoid moves that uncover.
                delta_cost = delta_uncovered
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                # Apply changes
                for idx in old_indices:
                    coverage_counts[idx] -= 1
                    if coverage_counts[idx] == 0:
                        uncovered_count += 1
                for idx in new_indices:
                    if coverage_counts[idx] == 0:
                        uncovered_count -= 1
                    coverage_counts[idx] += 1
                    
                blocks[idx_to_swap] = new_block
                
                # Update best
                if uncovered_count < best_uncovered or (uncovered_count == best_uncovered and len(blocks) < best_num_blocks):
                    best_uncovered = uncovered_count
                    best_num_blocks = len(blocks)
                    best_blocks = blocks[:]

        T *= cooling
        
    # Prune redundant blocks from best_blocks
    # We want the smallest subset of best_blocks that still covers everything.
    # This is the Set Cover problem on the best_blocks found.
    # Use a greedy set cover on the candidate blocks.
    
    if best_uncovered > 0:
        # Fallback: Use a greedy construction to ensure validity
        # This shouldn't happen if SA is good, but just in case.
        # Or return best_blocks if it's the best we have? 
        # The task requires valid covering. 
        # If SA fails to cover, we must repair.
        # Let's implement a simple repair: Add greedy blocks until covered.
        
        # Re-evaluate uncovered subsets for best_blocks
        coverage_counts_repair = [0] * num_t_subsets
        for b in best_blocks:
            for idx in get_block_t_subsets(b):
                coverage_counts_repair[idx] += 1
        
        uncovered_set = set()
        for i in range(num_t_subsets):
            if coverage_counts_repair[i] == 0:
                uncovered_set.add(all_t_subsets[i])
        
        while uncovered_set:
            # Greedy pick
            best_gain = -1
            best_b = None
            # Try random blocks? Or generate new ones?
            # Generate a random block and see gain.
            # Or pick a random uncovered subset and build a block around it.
            # Let's pick a random uncovered subset and add a block containing it.
            start_subset = random.choice(tuple(uncovered_set))
            block = list(start_subset)
            while len(block) < k:
                # Add random point
                remaining = [x for x in points if x not in block]
                if not remaining: break
                block.append(random.choice(remaining))
            block.sort()
            
            # Add this block
            best_blocks.append(block)
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                if ts in uncovered_set:
                    uncovered_set.discard(ts)
                    
        # Prune again
        pass # Pruning logic below applies to any list

    # Prune best_blocks
    # Greedy Set Cover on best_blocks to minimize count
    # We need to cover all_t_subsets using blocks from best_blocks.
    # This is exact set cover on a subset of the universe.
    # Since best_blocks might be large, use greedy.
    
    # However, if best_blocks is already a valid covering, we just need to remove redundant ones.
    # A block is redundant if all its t-subsets are covered by other blocks.
    
    # Efficient pruning:
    # 1. Count how many blocks cover each t-subset.
    # 2. If a t-subset is covered by only 1 block, that block is essential.
    # 3. Remove non-essential blocks one by one if they don't uncover any t-subset.
    
    # Let's do a simple iterative pruning.
    pruned_blocks = best_blocks[:]
    changed = True
    while changed:
        changed = False
        # Count coverage
        counts = [0] * num_t_subsets
        for b in pruned_blocks:
            for idx in get_block_t_subsets(b):
                counts[idx] += 1
        
        # Find blocks that can be removed
        # A block b can be removed if for all t-subsets T in b, counts[T] > 1.
        # If we remove b, counts[T] becomes counts[T]-1.
        # So we need counts[T] > 1 for all T in b.
        
        removable_indices = []
        for i, b in enumerate(pruned_blocks):
            removable = True
            for idx in get_block_t_subsets(b):
                if counts[idx] == 1:
                    removable = False
                    break
            if removable:
                removable_indices.append(i)
                
        if removable_indices:
            # Remove one at a time to be safe? Or all?
            # If we remove all, we might uncover a subset that was covered by 2 blocks, one of which was removed?
            # No, if counts[T] > 1 for all T in b, removing b leaves counts[T] >= 1.
            # So we can remove all such blocks simultaneously.
            # But wait, if two blocks b1 and b2 are both "removable", and they share a subset T with count 2.
            # Removing both leaves count 0 for T.
            # So we must remove them one by one or check carefully.
            # Let's remove one at a time.
            idx_to_remove = removable_indices[0]
            b_to_remove = pruned_blocks[idx_to_remove]
            for idx in get_block_t_subsets(b_to_remove):
                counts[idx] -= 1
            pruned_blocks.pop(idx_to_remove)
            changed = True
            
    return pruned_blocks
