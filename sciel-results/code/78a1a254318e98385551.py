import itertools
import time
import random
from collections import defaultdict

def _greedy_build(v, k, t, seed, time_budget=None):
    """
    Greedy construction: repeatedly pick the block that covers the most uncovered t-subsets.
    To speed up, we pick a random uncovered t-subset and find the best block containing it.
    """
    rng = random.Random(seed)
    all_points = list(range(v))
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subs)
    blocks = []
    blocks_set = set() # Set of frozensets for fast lookup
    
    if not uncovered:
        return []
    
    # Precompute combinations of points for speed?
    # We will generate candidates on the fly.
    
    while uncovered:
        # Pick a random uncovered t-subset
        # To improve, we could pick the one with the fewest "options" but all are symmetric initially.
        # As we add blocks, some triples become "harder" if they are covered by fewer blocks?
        # No, we just pick random for diversity.
        start_sub = rng.choice(list(uncovered))
        
        # Generate all blocks containing start_sub
        # start_sub has t elements. We need k-t more from the remaining v-t points.
        remaining_points = [p for p in all_points if p not in start_sub]
        needed = k - t
        
        best_block = None
        best_gain = -1
        best_candidates = []
        
        # Iterate through combinations of remaining points
        # For v=24, k=7, t=3: remaining=21, needed=4. C(21,4) = 5985.
        for combo in itertools.combinations(remaining_points, needed):
            block_tuple = tuple(sorted(list(start_sub) + list(combo)))
            block_frozen = frozenset(block_tuple)
            
            # Skip if we already have this block (shouldn't happen in greedy, but safe)
            if block_frozen in blocks_set:
                continue
                
            # Calculate gain: how many uncovered t-subsets are covered by this block
            gain = 0
            # We only need to check t-subsets of the block that are in 'uncovered'
            # Since start_sub is in the block and is uncovered, gain is at least 1.
            for tri in itertools.combinations(block_tuple, t):
                if tri in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_candidates = [block_tuple]
            elif gain == best_gain:
                best_candidates.append(block_tuple)
        
        if best_candidates:
            # Pick a random candidate from the best ones to add diversity
            next_block = rng.choice(best_candidates)
            blocks.append(next_block)
            blocks_set.add(frozenset(next_block))
            
            # Update uncovered set
            for tri in itertools.combinations(next_block, t):
                uncovered.discard(tri)
                
    return blocks

def _local_search(blocks, v, k, t, deadline):
    """
    Local search: remove redundant blocks, and try to replace blocks to enable further removals.
    """
    if not blocks:
        return blocks
    
    # Convert to list of tuples for consistency
    blocks = [tuple(b) for b in blocks]
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(blocks)
        
        # 1. Deletion Phase
        # Compute coverage counts
        cover_count = defaultdict(int)
        for b in blocks:
            for tri in itertools.combinations(b, t):
                cover_count[tri] += 1
        
        to_remove = []
        for i, b in enumerate(blocks):
            is_redundant = True
            for tri in itertools.combinations(b, t):
                if cover_count[tri] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                to_remove.append(i)
        
        if to_remove:
            # Remove one at a time to be safe
            idx = to_remove[0]
            blocks.pop(idx)
            changed = True
            continue # Restart loop to recompute counts
        
        # 2. Replacement Phase
        # Try to replace a block with another that covers its essential triples
        # This is a heuristic to "move" coverage to allow other blocks to become redundant.
        
        # Recompute cover_count (it's still valid if no deletion happened)
        # But we need to know which triples are essential for each block.
        
        for i, b in enumerate(blocks):
            # Find essential triples for block b
            essential_tris = []
            for tri in itertools.combinations(b, t):
                if cover_count[tri] == 1:
                    essential_tris.append(tri)
            
            if not essential_tris:
                # Should have been removed in deletion phase
                continue
            
            # If there are too many essential triples, finding a replacement is hard
            # We limit to cases where essential set is small (e.g., <= 2)
            if len(essential_tris) > 2:
                continue
                
            # Try to find a replacement block
            # The replacement must cover all essential_tris
            # Union of essential_tris
            essential_points = set()
            for tri in essential_tris:
                essential_points.update(tri)
            
            if len(essential_points) > k:
                continue
                
            # Generate candidates that contain essential_points
            remaining_points = [p for p in range(v) if p not in essential_points]
            needed = k - len(essential_points)
            
            replacement_found = False
            # Limit search to avoid long loops
            # C(remaining, needed) can be large.
            # For v=24, k=7, essential_points=3 (1 triple), needed=4. C(21,4)=5985.
            # For essential_points=4 (2 triples sharing 1 point? or disjoint?), needed=3.
            
            for combo in itertools.combinations(remaining_points, needed):
                new_block_tuple = tuple(sorted(list(essential_points) + list(combo)))
                new_block_frozen = frozenset(new_block_tuple)
                
                # Check if this block is already in the set (and not the one we are replacing)
                if new_block_frozen == frozenset(b):
                    continue
                if new_block_frozen in [frozenset(x) for x in blocks if x != b]:
                    continue
                    
                # Check if this new block covers all essential_tris
                # It does by construction, but let's verify if t > 1
                # Actually, if essential_points are in the block, and essential_tris are subsets of essential_points?
                # No, essential_tris are specific triples.
                # If essential_tris = [(0,1,2)], essential_points={0,1,2}.
                # New block contains 0,1,2. So it covers (0,1,2).
                # If essential_tris = [(0,1,2), (0,1,3)], essential_points={0,1,2,3}.
                # New block contains 0,1,2,3. It covers (0,1,2) and (0,1,3).
                
                # Accept the replacement
                blocks[i] = new_block_tuple
                changed = True
                replacement_found = True
                break
            
            if replacement_found:
                break # Restart loop to recompute counts
                
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run multiple restarts
    while time.time() < deadline:
        # Check time before starting a new greedy build
        if time.time() > deadline - 5.0:
            break
            
        # Generate a candidate solution
        blocks = _greedy_build(v, k, t, seed)
        
        # Local search to remove redundant blocks
        # Pass deadline to local search to stop if time is up
        blocks = _local_search(blocks, v, k, t, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we found a very good solution, we might want to spend time on local search?
        # For now, just keep restarting.
        # We can stop early if we hit a theoretical lower bound?
        # Lower bound for C(24,7,3) is 69.
        if best_len == 69:
            break
            
        seed += 1
        
    return best_blocks
