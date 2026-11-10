import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    max_time = 110.0 # Safety margin
    
    # 1. Precomputation
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_triples = list(itertools.combinations(points, t))
    n_triples = len(all_triples)
    triple_index = {trip: i for i, trip in enumerate(all_triples)}
    
    # Map each block to the list of triple indices it covers
    # block_triples[i] is a list of t-subset indices covered by all_blocks[i]
    block_triples = []
    for block in all_blocks:
        covered = []
        for trip in itertools.combinations(block, t):
            covered.append(triple_index[trip])
        block_triples.append(covered)

    # 2. Greedy Initialization
    # We want a valid covering to start SA from.
    # Greedy: Pick block covering most uncovered triples.
    
    # Use a bitmask for uncovered triples for speed in greedy
    # n_triples = 2024, fits in 2024 bits. Python ints are arbitrary precision, so this is fine.
    
    def greedy_init():
        uncovered_mask = (1 << n_triples) - 1
        selected = []
        
        # Precompute mask for each block for greedy
        # This is expensive to compute for all 346k blocks initially? 
        # 346k * 35 ops = ~12M ops. Fast enough.
        block_masks = [0] * n_blocks
        for i, trips in enumerate(block_triples):
            m = 0
            for t_idx in trips:
                m |= (1 << t_idx)
            block_masks[i] = m
            
        while uncovered_mask:
            best_gain = -1
            best_block = -1
            # Optimization: If we have many blocks, scanning all is O(N).
            # 346k blocks * ~70 steps (avg) = 24M checks. Might be slow in Python if not careful.
            # But we only do this once.
            
            # To speed up greedy, we can just iterate.
            for i in range(n_blocks):
                inter = block_masks[i] & uncovered_mask
                if inter:
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
                        if best_gain == 35: # Max possible
                            # Can't do better than covering all triples in a block
                            # But there might be multiple. Pick one.
                            # Actually, if gain is 35, it's max. 
                            pass
            
            if best_block == -1:
                break # Should not happen if covering exists
                
            selected.append(best_block)
            uncovered_mask &= ~block_masks[best_block]
            
        return selected, block_masks

    selected, block_masks = greedy_init()
    
    # 3. Simulated Annealing
    # State: set of block indices
    current_set = set(selected)
    
    # Coverage counts for incremental updates
    cover_count = [0] * n_triples
    uncovered_count = 0
    
    # Initialize cover_count
    for idx in current_set:
        for t_idx in block_triples[idx]:
            cover_count[t_idx] += 1
    
    for i in range(n_triples):
        if cover_count[i] == 0:
            uncovered_count += 1
            
    current_cost = len(current_set) + 100 * uncovered_count
    best_set = set(current_set)
    best_cost = current_cost
    
    # SA Parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # We need to pick random blocks.
    # List of all block indices
    all_indices = list(range(n_blocks))
    
    while time.time() - start_time < max_time:
        if T < min_T:
            break
            
        # Choose a move
        move_type = random.random()
        
        delta_cost = 0
        old_block = -1
        new_block = -1
        
        if move_type < 0.33:
            # Add a random block
            new_block = random.choice(all_indices)
            if new_block in current_set:
                continue # No change
            
            # Calculate delta
            # Add new_block
            for t_idx in block_triples[new_block]:
                if cover_count[t_idx] == 0:
                    delta_cost -= 100 # Uncovered decreases by 1 -> cost decreases by 100
                cover_count[t_idx] += 1
            delta_cost += 1 # Size increases by 1
            
            current_set.add(new_block)
            
        elif move_type < 0.66:
            # Remove a random block from current set
            if not current_set:
                continue
            old_block = random.choice(list(current_set))
            
            # Calculate delta
            for t_idx in block_triples[old_block]:
                cover_count[t_idx] -= 1
                if cover_count[t_idx] == 0:
                    delta_cost += 100 # Uncovered increases by 1 -> cost increases by 100
            delta_cost -= 1 # Size decreases by 1
            
            current_set.remove(old_block)
            
        else:
            # Swap: Remove one, Add one
            if not current_set:
                continue
            old_block = random.choice(list(current_set))
            new_block = random.choice(all_indices)
            if new_block == old_block or new_block in current_set:
                continue
            
            # Remove old
            for t_idx in block_triples[old_block]:
                cover_count[t_idx] -= 1
                if cover_count[t_idx] == 0:
                    delta_cost += 100
            # Add new
            for t_idx in block_triples[new_block]:
                if cover_count[t_idx] == 0:
                    delta_cost -= 100
                cover_count[t_idx] += 1
            # Size change: 0
            
            current_set.remove(old_block)
            current_set.add(new_block)
            
        new_cost = current_cost + delta_cost
        
        # Metropolis Acceptance
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_set = set(current_set)
                # If we found a valid solution (uncovered_count == 0), 
                # we can check if it's smaller than current best valid
                # best_cost includes penalty, so if uncovered==0, best_cost == len(best_set)
        else:
            # Reject: Undo the move
            if move_type < 0.33:
                # Undo Add
                current_set.remove(new_block)
                for t_idx in block_triples[new_block]:
                    cover_count[t_idx] -= 1
            elif move_type < 0.66:
                # Undo Remove
                current_set.add(old_block)
                for t_idx in block_triples[old_block]:
                    cover_count[t_idx] += 1
            else:
                # Undo Swap
                current_set.remove(new_block)
                current_set.add(old_block)
                # Undo coverage changes
                # Re-add old
                for t_idx in block_triples[old_block]:
                    cover_count[t_idx] += 1
                # Re-remove new
                for t_idx in block_triples[new_block]:
                    cover_count[t_idx] -= 1
                    
        T *= cooling_rate
        
    # Return the best set found
    # Convert indices back to blocks
    final_blocks = [all_blocks[i] for i in best_set]
    return [list(b) for b in final_blocks]
