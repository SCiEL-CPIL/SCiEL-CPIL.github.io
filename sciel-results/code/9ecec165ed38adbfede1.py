import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using optimized simulated annealing.
    """
    random.seed(42)
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Helper to get index of a t-subset
    def get_idx(ts):
        return t_subset_to_idx[ts]
        
    # Greedy Initialization with better block construction
    def greedy_covering():
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            # Pick a random uncovered triple to start the block
            target = random.choice(list(uncovered))
            block = list(target)
            while len(block) < k:
                # Find the point that maximizes coverage of uncovered triples
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    best_x = random.choice([x for x in range(v) if x not in block])
                block.append(best_x)
            block.sort()
            blocks.append(block)
            # Update uncovered
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        return blocks

    blocks = greedy_covering()
    
    # Initialize counts
    counts = [0] * n_t_subsets
    for block in blocks:
        for ts in itertools.combinations(block, t):
            idx = get_idx(ts)
            counts[idx] += 1
            
    current_uncovered = sum(1 for c in counts if c == 0)
    n_blocks = len(blocks)
    
    # SA Parameters - tuned for better exploration
    start_time = time.time()
    max_time = 115.0
    T = 5.0
    T_min = 0.001
    cooling_rate = 0.9998
    
    best_blocks = [b[:] for b in blocks]
    best_n_blocks = n_blocks
    best_uncovered = current_uncovered
    
    # Helper to check if a block is redundant
    def is_redundant(block):
        for ts in itertools.combinations(block, t):
            idx = get_idx(ts)
            if counts[idx] <= 1:
                return False
        return True

    # Helper to apply a move and update counts - FIXED
    def apply_point_swap(block_idx, old_point, new_point):
        """
        Returns delta_uncovered if the move is applied.
        Modifies counts and blocks.
        """
        block = blocks[block_idx]
        
        # Identify affected triples: those containing old_point
        old_affected = []
        for ts in itertools.combinations(block, t):
            if old_point in ts:
                old_affected.append(get_idx(ts))
                
        # New block with swap
        new_block = block[:]
        new_block[new_block.index(old_point)] = new_point
        new_block.sort()
        
        # Triples containing new_point in the new block
        new_affected = []
        for ts in itertools.combinations(new_block, t):
            if new_point in ts:
                new_affected.append(get_idx(ts))
                
        # Calculate delta
        delta = 0
        
        # Decrease counts for old_affected (these triples lose coverage)
        for idx in old_affected:
            if counts[idx] == 1:
                delta += 1  # Becomes uncovered
            counts[idx] -= 1
            
        # Increase counts for new_affected (these triples gain coverage)
        for idx in new_affected:
            if counts[idx] == 0:
                delta -= 1  # Becomes covered
            counts[idx] += 1
            
        blocks[block_idx] = new_block
        return delta

    def revert_point_swap(block_idx, old_point, new_point):
        """Revert the swap: new_point was put in, old_point was removed.
        Now we put old_point back and remove new_point."""
        return apply_point_swap(block_idx, new_point, old_point)

    while T > T_min:
        if time.time() - start_time > max_time:
            break
            
        # Choose move with better balance
        move_type = random.random()
        
        if move_type < 0.5 and n_blocks > 0:
            # Move 1: Point Swap - most common
            bi = random.randint(0, n_blocks - 1)
            block = blocks[bi]
            pi = random.randint(0, k - 1)
            old_point = block[pi]
            
            available = [x for x in range(v) if x not in block]
            if not available:
                T *= cooling_rate
                continue
                
            new_point = random.choice(available)
            
            delta = apply_point_swap(bi, old_point, new_point)
            current_uncovered += delta
            
            # Accept/Reject
            if delta <= 0:
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = [b[:] for b in blocks]
            else:
                # Deterioration in coverage - accept with probability
                if random.random() < math.exp(-delta / T):
                    pass  # Already applied
                else:
                    # Revert
                    revert_delta = revert_point_swap(bi, old_point, new_point)
                    current_uncovered += revert_delta

        elif move_type < 0.75 and n_blocks > 1:
            # Move 2: Try to remove a redundant block
            bi = random.randint(0, n_blocks - 1)
            block = blocks[bi]
            
            if is_redundant(block):
                # Remove it
                for ts in itertools.combinations(block, t):
                    idx = get_idx(ts)
                    counts[idx] -= 1
                blocks.pop(bi)
                n_blocks -= 1
                current_uncovered = sum(1 for c in counts if c == 0)
                
                if current_uncovered == 0 and n_blocks < best_n_blocks:
                    best_n_blocks = n_blocks
                    best_blocks = [b[:] for b in blocks]
            else:
                # Not redundant - try to make it redundant by swapping points
                # Pick a random point in the block and try swapping
                if random.random() < 0.3:
                    pi = random.randint(0, k - 1)
                    old_point = block[pi]
                    available = [x for x in range(v) if x not in block]
                    if available:
                        new_point = random.choice(available)
                        delta = apply_point_swap(bi, old_point, new_point)
                        current_uncovered += delta
                        
                        if delta <= 0:
                            if current_uncovered < best_uncovered:
                                best_uncovered = current_uncovered
                                best_blocks = [b[:] for b in blocks]
                        else:
                            if random.random() < math.exp(-delta / T):
                                pass
                            else:
                                revert_delta = revert_point_swap(bi, old_point, new_point)
                                current_uncovered += revert_delta

        else:
            # Move 3: Repair if uncovered > 0
            if current_uncovered > 0:
                # Find uncovered triples
                uncovered_list = [ts for ts in all_t_subsets if counts[get_idx(ts)] == 0]
                if uncovered_list:
                    # Pick a random uncovered triple
                    start_triple = random.choice(uncovered_list)
                    block = list(start_triple)
                    while len(block) < k:
                        best_x = None
                        best_gain = -1
                        for x in range(v):
                            if x in block:
                                continue
                            gain = 0
                            for rest in itertools.combinations(block, t - 1):
                                candidate = tuple(sorted(rest + (x,)))
                                idx = get_idx(candidate)
                                if counts[idx] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            best_x = random.choice([x for x in range(v) if x not in block])
                        block.append(best_x)
                    block.sort()
                    
                    # Add block
                    blocks.append(block)
                    n_blocks += 1
                    for ts in itertools.combinations(block, t):
                        idx = get_idx(ts)
                        counts[idx] += 1
                    current_uncovered = sum(1 for c in counts if c == 0)
                    
                    if current_uncovered == 0 and n_blocks < best_n_blocks:
                        best_n_blocks = n_blocks
                        best_blocks = [b[:] for b in blocks]
                        
        T *= cooling_rate

    # Final cleanup - aggressive redundancy removal
    blocks = best_blocks
    
    # Recount and remove redundant blocks iteratively
    changed = True
    while changed:
        changed = False
        counts = [0] * n_t_subsets
        for block in blocks:
            for ts in itertools.combinations(block, t):
                counts[get_idx(ts)] += 1
                
        for i in range(len(blocks) - 1, -1, -1):
            block = blocks[i]
            redundant = True
            for ts in itertools.combinations(block, t):
                if counts[get_idx(ts)] <= 1:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                changed = True
                break
                
    # Verify coverage
    counts = [0] * n_t_subsets
    for block in blocks:
        for ts in itertools.combinations(block, t):
            counts[get_idx(ts)] += 1
            
    if any(c == 0 for c in counts):
        # Fallback to greedy
        blocks = greedy_covering()
        
    return blocks
