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
        
    # Greedy Initialization
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
                    # Check which uncovered triples are completed by adding x
                    # A triple is completed if it contains x and k-1 other points from the block
                    # Wait, the block is being built.
                    # The new block will be block + [x].
                    # The triples covered by this new block that were previously uncovered
                    # are those triples that are subsets of (block + [x]) and contain x?
                    # No, a triple is a subset of the final block.
                    # If we add x to the current partial block `block`,
                    # any triple that is a subset of (block + [x]) and was in `uncovered` is now covered.
                    # Specifically, triples that contain x and (t-1) points from `block`.
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    # Should not happen if v > k
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
    
    # SA Parameters
    start_time = time.time()
    max_time = 115.0
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    
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

    # Helper to apply a move and update counts
    def apply_point_swap(block_idx, old_point, new_point):
        """
        Returns delta_uncovered if the move is applied.
        Modifies counts and blocks.
        """
        block = blocks[block_idx]
        
        # Identify affected triples
        # Triples containing old_point in the current block
        old_affected = []
        for ts in itertools.combinations(block, t):
            if old_point in ts:
                old_affected.append(get_idx(ts))
                
        # New block
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
        
        # Decrease counts for old_affected
        for idx in old_affected:
            # If new_point is in this triple, it's still covered by the new block?
            # No, we are replacing old_point with new_point.
            # The triple `ts` is a subset of the OLD block.
            # If `new_point` is in `ts`, then `ts` is also a subset of the NEW block?
            # Yes, because `new_block` contains `new_point` and all other points from `block` except `old_point`.
            # If `ts` contains `old_point`, it cannot contain `new_point` (distinct points).
            # So `ts` is NOT in `new_block`.
            # Thus, coverage for `ts` decreases by 1.
            if counts[idx] == 1:
                delta += 1 # Becomes uncovered
            counts[idx] -= 1
            
        # Increase counts for new_affected
        for idx in new_affected:
            # `ts` is a subset of NEW block containing `new_point`.
            # Does it contain `old_point`? No.
            # So it was NOT covered by the OLD block.
            # Thus, coverage for `ts` increases by 1.
            if counts[idx] == 0:
                delta -= 1 # Becomes covered
            counts[idx] += 1
            
        blocks[block_idx] = new_block
        return delta

    def revert_point_swap(block_idx, old_point, new_point):
        """Revert the swap. old_point was replaced by new_point."""
        block = blocks[block_idx]
        
        # Current block has new_point. We want to restore old_point.
        # This is the inverse of apply_point_swap(block_idx, new_point, old_point)
        # But we need to be careful with counts.
        # Actually, it's easier to just call apply_point_swap with the reversed points.
        # However, apply_point_swap modifies `blocks`.
        # Let's just implement a direct revert or call apply.
        # Calling apply is safer if logic is symmetric.
        return apply_point_swap(block_idx, new_point, old_point)

    while T > T_min:
        if time.time() - start_time > max_time:
            break
            
        # Choose move
        move_type = random.random()
        
        if move_type < 0.6 and n_blocks > 0:
            # Move 1: Point Swap
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
                # Deterioration in coverage
                # We allow this if T is high enough
                if random.random() < math.exp(-delta / T):
                    pass # Already applied
                else:
                    # Revert
                    revert_delta = revert_point_swap(bi, old_point, new_point)
                    current_uncovered += revert_delta # Should return to original state

        elif move_type < 0.8 and n_blocks > 1:
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
                # Block is not redundant. 
                # We could try to "break" it to see if we can restructure?
                # For now, just skip or accept a small penalty?
                # Let's just skip to avoid complex state management.
                pass
                
        else:
            # Move 3: Repair if uncovered > 0
            if current_uncovered > 0:
                # Find a block that covers the most uncovered triples
                best_gain = 0
                best_block = None
                # Sample a few random blocks to speed up
                # Or construct a greedy block
                target_triples = [ts for ts in all_t_subsets if counts[get_idx(ts)] == 0]
                if target_triples:
                    # Pick a random uncovered triple
                    start_triple = random.choice(target_triples)
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

    # Final cleanup
    blocks = best_blocks
    # Remove redundant blocks again
    changed = True
    while changed:
        changed = False
        # Recount
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
                
    # Verify
    counts = [0] * n_t_subsets
    for block in blocks:
        for ts in itertools.combinations(block, t):
            counts[get_idx(ts)] += 1
            
    if any(c == 0 for c in counts):
        # Fallback
        blocks = greedy_covering()
        
    return blocks
