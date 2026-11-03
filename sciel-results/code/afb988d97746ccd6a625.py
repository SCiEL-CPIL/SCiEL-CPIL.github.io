"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(list(uncovered))
        block = list(seed)
        rng.shuffle(block)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in blocks:
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
                
    return blocks

def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Precompute all t-subsets as frozensets for fast lookup
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    total_t = len(all_tsubsets)
    
    # 1. Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    
    # 2. Initialize state variables
    # counts[s] = number of blocks covering s
    counts = {}
    for b in blocks:
        for c in itertools.combinations(b, t):
            f = frozenset(c)
            counts[f] = counts.get(f, 0) + 1
            
    uncovered_count = 0
    for s in all_tsubsets:
        if counts.get(s, 0) == 0:
            uncovered_count += 1
            
    current_blocks = list(blocks)
    current_uncovered = uncovered_count
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Penalty for uncovered sets
    # We want to minimize len(blocks) + alpha * uncovered_count
    # If uncovered_count > 0, we prioritize reducing it.
    # If uncovered_count == 0, we prioritize reducing len(blocks).
    alpha = 50.0 
    
    T = 100.0
    cooling = 0.9995
    
    def get_energy(n_blocks, n_uncovered):
        return n_blocks + alpha * n_uncovered
        
    current_energy = get_energy(len(current_blocks), current_uncovered)
    
    while time.time() - start_time < time_limit and T > 0.1:
        
        # Generate a move
        move_type = rng.random()
        
        # Try to update state
        old_blocks = current_blocks
        old_uncovered = current_uncovered
        old_energy = current_energy
        
        if move_type < 0.3:
            # Remove a random block
            if not current_blocks:
                continue
            idx = rng.randrange(len(current_blocks))
            block_to_remove = current_blocks[idx]
            
            # Calculate delta uncovered
            delta_uncovered = 0
            for c in itertools.combinations(block_to_remove, t):
                f = frozenset(c)
                c_count = counts.get(f, 0)
                if c_count == 1:
                    delta_uncovered += 1
                # Note: we don't update counts yet, just simulate
            # New state
            new_blocks = [b for i, b in enumerate(current_blocks) if i != idx]
            new_uncovered = current_uncovered + delta_uncovered
            new_energy = get_energy(len(new_blocks), new_uncovered)
            
            # Acceptance criterion
            delta_e = new_energy - old_energy
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                # Accept
                # Update counts
                for c in itertools.combinations(block_to_remove, t):
                    f = frozenset(c)
                    counts[f] = counts.get(f, 0) - 1
                current_blocks = new_blocks
                current_uncovered = new_uncovered
                current_energy = new_energy
                
                if current_uncovered == 0 and len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = list(current_blocks)
                    
        elif move_type < 0.7:
            # Add a block
            # Strategy: If there are uncovered sets, pick one and build a block around it.
            # Otherwise, pick a random block.
            
            if current_uncovered > 0:
                # Find an uncovered set
                # To avoid O(N) scan every time, we could maintain a list, but for now:
                # We can just pick a random point from 0..v-1 and hope? No.
                # Let's pick a random candidate from all_tsubsets and check if uncovered.
                # This is inefficient if uncovered is small.
                # Better: maintain a list of uncovered sets?
                # For simplicity and speed in this scope, let's just pick a random 3-subset 
                # and if it's covered, try again or just add a random block.
                
                # Actually, let's just add a random block. It might cover some uncovered sets.
                # Or, pick a random point p, and a random 2-subset containing p?
                
                # Let's try: pick a random 3-subset s. If it is uncovered, build a block containing s.
                # If covered, just add a random block.
                
                # To make it efficient, let's just add a random block.
                # The SA will explore.
                
                pass # Logic below handles both
            
            # Generate random block
            new_block_set = set(rng.sample(range(v), k))
            new_block = tuple(sorted(new_block_set))
            
            # Check if duplicate
            if new_block in current_blocks:
                continue
                
            # Calculate delta uncovered
            delta_uncovered = 0
            for c in itertools.combinations(new_block, t):
                f = frozenset(c)
                if counts.get(f, 0) == 0:
                    delta_uncovered -= 1 # Reduces uncovered count
            
            new_blocks = current_blocks + [new_block]
            new_uncovered = current_uncovered + delta_uncovered
            new_energy = get_energy(len(new_blocks), new_uncovered)
            
            delta_e = new_energy - old_energy
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                # Accept
                for c in itertools.combinations(new_block, t):
                    f = frozenset(c)
                    counts[f] = counts.get(f, 0) + 1
                current_blocks = new_blocks
                current_uncovered = new_uncovered
                current_energy = new_energy
                
                if current_uncovered == 0 and len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = list(current_blocks)
                    
        else:
            # Swap: Remove one, Add one
            if not current_blocks:
                continue
            idx = rng.randrange(len(current_blocks))
            old_block = current_blocks[idx]
            
            # Generate new block
            new_block_set = set(rng.sample(range(v), k))
            new_block = tuple(sorted(new_block_set))
            
            if new_block == old_block:
                continue
            
            # Calculate deltas
            # Remove old_block
            delta_uncovered_remove = 0
            for c in itertools.combinations(old_block, t):
                f = frozenset(c)
                c_count = counts.get(f, 0)
                if c_count == 1:
                    delta_uncovered_remove += 1
            
            # Add new_block (simulated on top of removed state? No, independent)
            # We need to be careful with counts.
            # Let's calculate the change in counts for the combined operation.
            
            # Simulate counts change
            # Temporarily update counts in a dict? No, just calculate delta.
            
            # For each s in old_block: count decreases by 1
            # For each s in new_block: count increases by 1
            
            # Net change in uncovered:
            # s is in old_block and not in new_block:
            #   if count[s] == 1: becomes 0 -> +1 uncovered
            # s is in new_block and not in old_block:
            #   if count[s] == 0: becomes 1 -> -1 uncovered
            # s is in both:
            #   count unchanged -> no change
            
            # s in old_block only:
            delta_u = 0
            for c in itertools.combinations(old_block, t):
                f = frozenset(c)
                # Check if f is in new_block
                # Efficient check:
                # We can just iterate all t-subsets of old and new?
                # Or simpler:
                # 1. Calculate uncovered added by removing old
                # 2. Calculate uncovered removed by adding new
                # But they interact if a set is in both.
                
                # Let's just do it step by step with a temporary count update?
                # Too slow.
                
                # Let's use the property:
                # new_uncovered = old_uncovered 
                #   + (# sets in old_block with count 1 and NOT in new_block)
                #   - (# sets in new_block with count 0 and NOT in old_block)
                #   + (# sets in both with count 1? No, count becomes 1-1+1=1, so no change)
                #   Wait.
                # If s in both:
                #   old count c.
                #   remove: c-1.
                #   add: c.
                #   Net change 0.
                
                # So:
                # Increase in uncovered: sets in old_block \ new_block with count 1.
                # Decrease in uncovered: sets in new_block \ old_block with count 0.
                
                # We need to check membership in the other block.
                
                pass # Logic below

            # Simplified Swap Logic:
            # 1. Remove old_block. Update counts. Record which sets became uncovered.
            # 2. Add new_block. Update counts. Record which sets became covered.
            # 3. Check if total uncovered changed.
            
            # To do this efficiently without modifying global state until accepted:
            # We can just compute the delta.
            
            # Let's try a different swap strategy:
            # Pick a block to remove.
            # Pick a set of points to add that covers some critical sets.
            
            # For now, let's just do Random Swap with full simulation if accepted?
            # No, we need delta for acceptance.
            
            # Let's implement the delta calculation properly.
            
            # Sets in old_block
            old_sets = [frozenset(c) for c in itertools.combinations(old_block, t)]
            # Sets in new_block
            new_sets = [frozenset(c) for c in itertools.combinations(new_block, t)]
            
            # We need to know for each set in old_sets if it is in new_block.
            # And for each set in new_sets if it is in old_block.
            
            # Convert new_block to a set of points for fast membership?
            # new_block_points = set(new_block)
            # A 3-subset s is in new_block if all 3 points are in new_block_points.
            
            new_block_points = set(new_block)
            old_block_points = set(old_block)
            
            delta_u = 0
            
            # 1. Sets lost by removing old_block
            for s in old_sets:
                # Is s in new_block?
                if s.issubset(new_block_points):
                    # It is in both, so count doesn't change effectively (c-1+1=c)
                    continue
                # It is only in old_block
                c_count = counts.get(s, 0)
                if c_count == 1:
                    delta_u += 1 # Becomes uncovered
            
            # 2. Sets gained by adding new_block
            for s in new_sets:
                # Is s in old_block?
                if s.issubset(old_block_points):
                    # It is in both, so count doesn't change effectively
                    continue
                # It is only in new_block
                c_count = counts.get(s, 0)
                if c_count == 0:
                    delta_u -= 1 # Becomes covered
            
            new_uncovered = current_uncovered + delta_u
            new_blocks_len = len(current_blocks) # Length unchanged in swap
            new_energy = get_energy(new_blocks_len, new_uncovered)
            
            delta_e = new_energy - old_energy
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                # Accept
                # Update counts
                for s in old_sets:
                    counts[s] = counts.get(s, 0) - 1
                for s in new_sets:
                    counts[s] = counts.get(s, 0) + 1
                
                current_blocks[idx] = new_block
                current_uncovered = new_uncovered
                current_energy = new_energy
                
                if current_uncovered == 0 and len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = list(current_blocks)

        T *= cooling

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
