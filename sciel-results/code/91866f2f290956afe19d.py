import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # Pre-compute all t-subsets for the universe
    # For v=24, k=7, t=3, this is C(24,3) = 2024 triples.
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    
    # Map triple to index for faster counting
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Helper to get indices of triples in a block
    def get_block_triple_indices(block):
        return [triple_to_idx[tri] for tri in itertools.combinations(block, t)]

    # 1. Fast Greedy Construction to get a valid starting point
    def greedy_construct():
        uncovered = set(range(num_triples))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to start the block
            start_tri_idx = random.choice(list(uncovered))
            start_tri = all_triples[start_tri_idx]
            
            block = list(start_tri)
            
            while len(block) < k:
                # Find candidates that maximize new coverage
                current_block_set = set(block)
                candidates = [x for x in range(v) if x not in current_block_set]
                
                best_gain = -1
                best_candidates = []
                
                for x in candidates:
                    new_block = block + [x]
                    # Calculate gain: number of triples in new_block that are in uncovered
                    # We only need to check triples involving x and pairs from block
                    gain = 0
                    for pair in itertools.combinations(block, t-1):
                        tri = tuple(sorted(pair + (x,)))
                        tri_idx = triple_to_idx[tri]
                        if tri_idx in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                        
                if best_candidates:
                    block.append(random.choice(best_candidates))
                else:
                    # Fallback (should not happen if v > k and t < k)
                    block.append(random.choice(candidates))
                    
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for tri_idx in get_block_triple_indices(block):
                if tri_idx in uncovered:
                    uncovered.remove(tri_idx)
                    
        return blocks

    # Initial solution
    initial_blocks = greedy_construct()
    current_blocks = initial_blocks
    best_blocks = current_blocks[:]
    best_len = len(current_blocks)
    
    # State for SA
    # counts[i] is the number of blocks covering triple i
    counts = [0] * num_triples
    for b in current_blocks:
        for tri_idx in get_block_triple_indices(b):
            counts[tri_idx] += 1
    
    # Verify initial validity (should be valid by construction)
    # min_count = min(counts)
    # if min_count < 1: 
    #    # Fallback if greedy failed somehow
    #    pass
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Pre-compute triple indices for all possible blocks? 
    # C(24,7) is ~346k. Too many to pre-compute fully in memory if not careful, 
    # but we can compute on the fly or cache.
    # Given 115s budget, on-the-fly with lru_cache might be okay, 
    # but let's just compute the list of indices for a block quickly.
    
    def compute_block_indices(block):
        return [triple_to_idx[tri] for tri in itertools.combinations(block, t)]

    # To speed up SA moves, we can pre-compute the triple indices for 
    # a random sample of all possible blocks, or just generate them on the fly.
    # Let's try generating on the fly first, but optimize the inner loop.
    
    # Actually, for SA, we need to evaluate delta quickly.
    # Delta = change in number of uncovered triples.
    # Removing a block: 
    #   For each triple in block, if count drops to 0, delta += 1 (worse)
    # Adding a block:
    #   For each triple in block, if count was 0, delta -= 1 (better)
    # Swapping:
    #   Remove old, add new.
    
    # Let's implement the SA loop.
    # We will try to reduce the number of blocks.
    # Cost function: len(blocks). 
    # But we must maintain validity (min count >= 1).
    # Standard SA allows invalid states. 
    # Let's define cost = len(blocks) + penalty * (number of uncovered triples).
    # If we allow invalid states, we can jump out of local minima.
    
    penalty = 100.0 # High penalty to force validity quickly
    
    def get_cost(blocks, counts):
        uncovered = sum(1 for c in counts if c == 0)
        return len(blocks) + penalty * uncovered

    current_cost = get_cost(current_blocks, counts)
    best_cost = current_cost
    
    # To make "Add Block" move efficient, we need a pool of candidate blocks.
    # We can generate random blocks.
    
    # To make "Swap Block" move efficient, we pick a block from current set 
    # and replace it with a random block.
    
    # Optimization: 
    # Instead of recalculating full counts, we update them incrementally.
    
    # Let's run SA
    steps = 0
    while time.time() < deadline:
        # Choose a move type
        r = random.random()
        
        if r < 0.4:
            # Try Remove
            if len(current_blocks) <= 1:
                continue
            idx = random.randint(0, len(current_blocks) - 1)
            b_remove = current_blocks[idx]
            b_indices = compute_block_indices(b_remove)
            
            # Check if removal creates uncovered triples
            delta_uncovered = 0
            for tri_idx in b_indices:
                if counts[tri_idx] == 1:
                    delta_uncovered += 1
            
            # Apply removal
            for tri_idx in b_indices:
                counts[tri_idx] -= 1
            new_blocks = current_blocks[:idx] + current_blocks[idx+1:]
            
            new_cost = get_cost(new_blocks, counts)
            old_cost = current_cost
            
            # Acceptance
            if new_cost < old_cost or random.random() < math.exp(-(new_cost - old_cost) / T):
                current_blocks = new_blocks
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = current_blocks[:]
            else:
                # Revert
                for tri_idx in b_indices:
                    counts[tri_idx] += 1
                # current_blocks unchanged
            
        elif r < 0.8:
            # Try Add
            # Generate a random block
            new_block = tuple(sorted(random.sample(range(v), k)))
            b_indices = compute_block_indices(new_block)
            
            # Check gain
            delta_uncovered = 0
            for tri_idx in b_indices:
                if counts[tri_idx] == 0:
                    delta_uncovered -= 1 # Improvement
            
            # Apply add
            for tri_idx in b_indices:
                counts[tri_idx] += 1
            new_blocks = current_blocks + [new_block]
            
            new_cost = get_cost(new_blocks, counts)
            old_cost = current_cost
            
            if new_cost < old_cost or random.random() < math.exp(-(new_cost - old_cost) / T):
                current_blocks = new_blocks
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = current_blocks[:]
            else:
                # Revert
                for tri_idx in b_indices:
                    counts[tri_idx] -= 1
                # current_blocks unchanged
                
        else:
            # Try Swap
            if len(current_blocks) == 0:
                continue
            idx = random.randint(0, len(current_blocks) - 1)
            b_old = current_blocks[idx]
            old_indices = compute_block_indices(b_old)
            
            new_block = tuple(sorted(random.sample(range(v), k)))
            # Ensure new block is different
            if new_block == b_old:
                continue
            new_indices = compute_block_indices(new_block)
            
            # Calculate delta
            # Remove old, Add new
            # We need to be careful with the order of count updates for accuracy if we were checking validity,
            # but for cost calculation, we can just compute the new counts state virtually?
            # No, let's just apply, check, and revert if needed.
            
            # Apply Remove Old
            for tri_idx in old_indices:
                counts[tri_idx] -= 1
            # Apply Add New
            for tri_idx in new_indices:
                counts[tri_idx] += 1
            
            new_blocks = current_blocks[:idx] + [new_block] + current_blocks[idx+1:]
            
            new_cost = get_cost(new_blocks, counts)
            old_cost = current_cost
            
            if new_cost < old_cost or random.random() < math.exp(-(new_cost - old_cost) / T):
                current_blocks = new_blocks
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = current_blocks[:]
            else:
                # Revert
                for tri_idx in new_indices:
                    counts[tri_idx] -= 1
                for tri_idx in old_indices:
                    counts[tri_idx] += 1
                # current_blocks unchanged

        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        steps += 1
        
        # Optional: Periodic local search (removing redundant blocks) to clean up
        if steps % 1000 == 0:
            # Quick local optimization: remove any block that is redundant
            # This helps SA settle into valid states
            changed = True
            while changed:
                changed = False
                for i in range(len(current_blocks)):
                    b = current_blocks[i]
                    b_indices = compute_block_indices(b)
                    can_remove = True
                    for tri_idx in b_indices:
                        if counts[tri_idx] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for tri_idx in b_indices:
                            counts[tri_idx] -= 1
                        current_blocks.pop(i)
                        changed = True
                        break # Restart loop as indices changed
                        
            if len(current_blocks) < len(best_blocks):
                best_blocks = current_blocks[:]
                best_cost = current_cost

    # Final cleanup: remove any redundant blocks from best_blocks
    # Rebuild counts for best_blocks to be safe
    best_counts = [0] * num_triples
    for b in best_blocks:
        for tri_idx in compute_block_indices(b):
            best_counts[tri_idx] += 1
            
    final_blocks = []
    for b in best_blocks:
        b_indices = compute_block_indices(b)
        can_remove = True
        for tri_idx in b_indices:
            if best_counts[tri_idx] == 1:
                can_remove = False
                break
        if not can_remove:
            final_blocks.append(b)
            for tri_idx in b_indices:
                best_counts[tri_idx] -= 1
        # If can_remove, we just skip it (don't add to final_blocks)
        
    # Verify final validity
    # min(best_counts) should be >= 1? No, we only kept blocks that were NOT redundant.
    # Wait, if a block was redundant, we removed it. 
    # The remaining blocks must cover everything?
    # Yes, because we only removed blocks if all their triples were covered by OTHERS.
    # So the union of remaining blocks is the same as the union of original blocks.
    # Since original was valid, final is valid.
    
    return [list(b) for b in final_blocks]
