import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Deadline: 150s limit, use 145s for safety
    deadline = time.time() + 145.0
    
    # Helper: get indices of t-subsets in a block
    def get_block_tsub_indices(block):
        return [tsub_index[ts] for ts in itertools.combinations(block, t)]

    # Helper: Greedy construction (fast initializer)
    def greedy_construct(seed):
        random.seed(seed)
        covered = bytearray(num_tsubs)
        uncovered_count = num_tsubs
        blocks = []
        
        while uncovered_count > 0:
            # Pick a random uncovered t-subset to anchor the block
            while True:
                i = random.randint(0, num_tsubs - 1)
                if not covered[i]:
                    start_tsub = all_tsubs[i]
                    break
            
            block = list(start_tsub)
            
            # Extend to k points
            while len(block) < k:
                best_point = None
                best_gain = -1
                # Calculate gain for each candidate point
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    # New t-subsets formed by adding x
                    # They are (t-1)-combinations of current block + x
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_index[ts]
                        if not covered[idx]:
                            gain += 1
                    if gain > best_gain or (gain == best_gain and random.random() < 0.5):
                        best_gain = gain
                        best_point = x
                
                if best_point is None:
                    # Should not happen if v > k
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Mark as covered
            for ts in itertools.combinations(block, t):
                idx = tsub_index[ts]
                if not covered[idx]:
                    covered[idx] = 1
                    uncovered_count -= 1
        
        return blocks

    # Helper: Reduction phase (simple redundancy removal)
    def reduce_blocks(blocks):
        if not blocks:
            return blocks
        
        # Count how many blocks cover each t-subset
        cover_count = [0] * num_tsubs
        
        # Initialize counts
        for b in blocks:
            for ts in itertools.combinations(b, t):
                idx = tsub_index[ts]
                cover_count[idx] += 1
        
        # Iteratively remove redundant blocks
        changed = True
        current_blocks = blocks[:]
        
        while changed:
            changed = False
            new_blocks = []
            for b in current_blocks:
                # Check if any t-subset in b has count == 1
                # If yes, this block is essential (cannot be removed yet)
                # If no, this block is redundant
                is_essential = False
                for ts in itertools.combinations(b, t):
                    idx = tsub_index[ts]
                    if cover_count[idx] == 1:
                        is_essential = True
                        break
                
                if is_essential:
                    new_blocks.append(b)
                else:
                    # Remove this block
                    changed = True
                    # Decrement counts
                    for ts in itertools.combinations(b, t):
                        idx = tsub_index[ts]
                        cover_count[idx] -= 1
            
            current_blocks = new_blocks
        
        return current_blocks

    # Helper: Local Search 1-swap minimization
    # Tries to replace a block with a different one that covers the same "essential" t-subsets
    # or allows removal of another block, thereby reducing total count.
    # Actually, a simpler and effective local search for set cover is:
    # 1. Start with a valid cover.
    # 2. Iterate: try to remove a block. If cover remains valid, keep it removed.
    # 3. If no block can be removed, try to replace a block with a "better" one that
    #    covers more "scarce" elements, potentially allowing future removals.
    # However, direct removal is already done in reduce_blocks.
    # A stronger local search: 1-move (replace one block with another) to minimize cost.
    # Cost = number of blocks.
    # We can try to replace a block B with a block B' such that:
    #   - B' is not in the current set.
    #   - All t-subsets covered only by B are also covered by B' or other blocks.
    #   - This might not reduce count directly, but if we can find a B' that covers
    #     the unique t-subsets of B, we can remove B. Wait, that's just removal if B' is added?
    #   - No, if we replace B with B', the count stays the same.
    #   - To reduce count, we must find a set of blocks that covers everything with fewer blocks.
    #   - Local search 1-swap: Replace a block B with a block B' where B' covers
    #     all t-subsets that B uniquely covers, AND B' allows the removal of another block?
    #     This is complex.
    
    # Alternative Local Search: Simulated Annealing or Hill Climbing on block set.
    # Let's try a simpler heuristic:
    # 1. Get a reduced cover.
    # 2. Randomly pick a block to remove.
    # 3. Find a block that covers the newly uncovered t-subsets.
    # 4. If such a block exists and is not already in the set, add it.
    # 5. This doesn't reduce count.
    
    # Let's stick to a robust "Iterated Local Search" on the block set:
    # 1. Start with reduced greedy.
    # 2. Perturb: randomly remove a block.
    # 3. Repair: greedily add blocks to cover uncovered t-subsets.
    # 4. Reduce: remove redundant blocks.
    # 5. If better, keep.
    
    def repair_cover(blocks, covered, uncovered_count):
        """
        Given a list of blocks and their coverage state, add blocks to cover all t-subsets.
        Returns new list of blocks and updated coverage state.
        """
        # Rebuild coverage state from blocks
        covered = bytearray(num_tsubs)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                idx = tsub_index[ts]
                covered[idx] = 1
        
        uncovered_count = num_tsubs - sum(covered)
        
        # Greedily add blocks to cover uncovered t-subsets
        while uncovered_count > 0:
            # Pick a random uncovered t-subset
            while True:
                i = random.randint(0, num_tsubs - 1)
                if not covered[i]:
                    start_tsub = all_tsubs[i]
                    break
            
            # Find a block containing start_tsub that maximizes new coverage
            best_block = None
            best_gain = -1
            
            # Candidate blocks: all k-subsets containing start_tsub
            # There are C(v-t, k-t) such blocks.
            # For v=19, k=8, t=5: C(14, 3) = 364 blocks.
            # This is small enough to iterate over.
            
            # Generate all blocks containing start_tsub
            remaining_points = [x for x in range(v) if x not in start_tsub]
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(start_tsub + combo))
                
                # Calculate gain
                gain = 0
                for ts in itertools.combinations(candidate_block, t):
                    idx = tsub_index[ts]
                    if not covered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = list(candidate_block)
            
            if best_block is None:
                # Should not happen
                break
                
            blocks.append(best_block)
            
            # Update coverage
            for ts in itertools.combinations(best_block, t):
                idx = tsub_index[ts]
                if not covered[idx]:
                    covered[idx] = 1
                    uncovered_count -= 1
        
        return blocks, covered, uncovered_count

    # Main Loop: Iterated Local Search
    best_blocks = None
    best_len = float('inf')
    
    # Initial solution
    initial_blocks = greedy_construct(0)
    initial_blocks = reduce_blocks(initial_blocks)
    best_blocks = initial_blocks[:]
    best_len = len(initial_blocks)
    
    current_blocks = initial_blocks[:]
    
    while time.time() < deadline - 5.0:
        # Perturb: randomly remove a block
        if not current_blocks:
            break
            
        remove_idx = random.randint(0, len(current_blocks) - 1)
        perturbed_blocks = current_blocks[:]
        perturbed_blocks.pop(remove_idx)
        
        # Repair
        repaired_blocks, covered, uncovered_count = repair_cover(perturbed_blocks, None, None)
        
        # Reduce
        reduced_blocks = reduce_blocks(repaired_blocks)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks[:]
            current_blocks = reduced_blocks[:]
            
        # If we hit a very good solution, we might want to continue or stop
        if best_len <= 320:
            # Keep trying for a bit more
            if time.time() > deadline - 10.0:
                break

    return [list(b) for b in best_blocks]
