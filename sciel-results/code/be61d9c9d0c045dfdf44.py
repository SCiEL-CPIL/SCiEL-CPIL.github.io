import itertools
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Deadline: 150s limit, use 145s for safety
    deadline = time.time() + 145.0
    
    best_blocks = None
    best_len = float('inf')
    
    # Cache for t-subset indices of a block to avoid recomputing in reduction
    # Key: tuple(block), Value: list of indices
    block_tsub_cache = {}
    
    def get_block_tsub_indices(block_tuple):
        if block_tuple in block_tsub_cache:
            return block_tsub_cache[block_tuple]
        idxs = [tsub_index[ts] for ts in itertools.combinations(block_tuple, t)]
        block_tsub_cache[block_tuple] = idxs
        return idxs

    # Helper: Greedy construction
    def greedy_construct(seed):
        random.seed(seed)
        covered = bytearray(num_tsubs)
        uncovered_count = num_tsubs
        blocks = []
        
        # To speed up, we can pre-generate a list of random uncovered indices?
        # No, the set of uncovered changes.
        
        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                return None # Timeout
            
            # Pick a random uncovered t-subset to anchor the block
            # Optimization: Instead of random.randint and checking, 
            # we can maintain a list of uncovered indices? 
            # Removing from a list is O(N). 
            # Random probing is O(1) expected if density is high, O(N) if low.
            # Early on, density is low (all uncovered). Probing is fast.
            # Later on, density is high (mostly covered). Probing is slow.
            # Hybrid: If uncovered_count < num_tsubs / 10, use a list.
            
            if uncovered_count < num_tsubs // 10:
                # Build list of uncovered indices
                uncov_list = [i for i in range(num_tsubs) if not covered[i]]
                if not uncov_list:
                    break
                i = random.choice(uncov_list)
            else:
                while True:
                    i = random.randint(0, num_tsubs - 1)
                    if not covered[i]:
                        break
            
            start_tsub = all_tsubs[i]
            block = list(start_tsub)
            
            # Extend to k points
            # Optimization: Precompute combinations of the current block?
            # The block size changes.
            # We need to calculate gain for each candidate x.
            # gain(x) = number of (t-1)-combinations of block such that {rest, x} is uncovered.
            
            while len(block) < k:
                best_point = None
                best_gain = -1
                
                # Candidate points
                # Optimization: Instead of iterating all v points, 
                # we can iterate all points not in block.
                block_set = set(block)
                
                # Pre-compute the (t-1) combinations of the current block
                # These are the "rest" parts.
                # For a candidate x, the new t-subsets are {rest, x} for rest in C(block, t-1).
                # We need to check if {rest, x} is uncovered.
                
                rest_combos = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # Check each rest combo
                    for rest in rest_combos:
                        # Construct the t-subset tuple
                        # rest is a tuple of t-1 ints. x is an int.
                        # We need to sort them to match tsub_index keys.
                        # Optimization: Since rest is sorted and x is added,
                        # we can merge efficiently.
                        # Or just tuple(sorted(rest + (x,)))
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_index[ts]
                        if not covered[idx]:
                            gain += 1
                            # Early exit if gain > best_gain? 
                            # No, we need exact gain to compare, but if gain > best_gain, 
                            # we can't stop early because another point might have even more.
                            # But we can break if gain is already higher than possible max?
                            # Max gain is len(rest_combos).
                            if gain > best_gain:
                                # We can update best_point here, but we still need to check other x?
                                # Yes, to find the MAX.
                                pass
                    
                    if gain > best_gain or (gain == best_gain and random.random() < 0.5):
                        best_gain = gain
                        best_point = x
                
                if best_point is None:
                    # Should not happen
                    break
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Mark as covered
            block_tuple = tuple(block)
            idxs = get_block_tsub_indices(block_tuple)
            for idx in idxs:
                if not covered[idx]:
                    covered[idx] = 1
                    uncovered_count -= 1
        
        return blocks

    # Helper: Reduction phase
    def reduce_blocks(blocks):
        if not blocks:
            return blocks
        
        # Count how many blocks cover each t-subset
        cover_count = [0] * num_tsubs
        
        # Initialize counts and store block indices
        block_indices = []
        for b in blocks:
            b_tuple = tuple(b)
            idxs = get_block_tsub_indices(b_tuple)
            block_indices.append(idxs)
            for idx in idxs:
                cover_count[idx] += 1
        
        current_blocks = blocks[:]
        current_indices = block_indices[:]
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            new_indices = []
            
            # Identify blocks to keep
            for i, b in enumerate(current_blocks):
                idxs = current_indices[i]
                is_essential = False
                for idx in idxs:
                    if cover_count[idx] == 1:
                        is_essential = True
                        break
                
                if is_essential:
                    new_blocks.append(b)
                    new_indices.append(idxs)
                else:
                    # Remove this block
                    changed = True
                    # Decrement counts
                    for idx in idxs:
                        cover_count[idx] -= 1
            
            current_blocks = new_blocks
            current_indices = new_indices
        
        return current_blocks

    # Main loop: Multiple restarts
    num_restarts = 500 # High number, controlled by time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # 1. Greedy Construction
        blocks = greedy_construct(restart)
        
        if blocks is None:
            continue
            
        # 2. Reduction
        reduced_blocks = reduce_blocks(blocks)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks
            
        # Early exit if we reach a very good solution
        if best_len <= 320:
            if restart > 10:
                break

    if best_blocks is None:
        # Fallback: One deterministic greedy + reduction
        blocks = greedy_construct(0)
        if blocks is None:
             # Emergency fallback: return empty or simple set? 
             # Must be valid.
             # If greedy timed out, we need a valid covering.
             # A trivial covering is all k-subsets? Too many.
             # We should ensure greedy doesn't timeout in the first restart.
             # If it does, we have a problem.
             # Let's force a simpler greedy if needed.
             pass
        best_blocks = reduce_blocks(blocks)
        best_len = len(best_blocks)

    return [list(b) for b in best_blocks]

# EVOLVE-BLOCK-END
