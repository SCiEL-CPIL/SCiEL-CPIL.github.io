"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math

def construct(v, k, t):
    """
    Construct a covering C(v, k, t) using a randomized greedy approach with local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets for quick lookup
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast access
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Track which t-subsets are covered
    # Use a bitset-like structure: list of booleans
    covered = [False] * num_t_subsets
    
    def count_uncovered_in_block(block):
        """Count how many t-subsets of the block are currently uncovered."""
        count = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            if not covered[idx]:
                count += 1
        return count
    
    def mark_block_covered(block):
        """Mark all t-subsets of the block as covered."""
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered[idx] = True
    
    def unmark_block_covered(block, blocks):
        """Unmark t-subsets of the block, but only if no other block covers them."""
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            # Check if any other block covers this t-subset
            still_covered = False
            for other_block in blocks:
                if other_block == block:
                    continue
                if set(ts).issubset(set(other_block)):
                    still_covered = True
                    break
            if not still_covered:
                covered[idx] = False
    
    def greedy_cover(uncovered_set, random_seed=0):
        """
        Greedy covering: repeatedly pick a block that covers the most uncovered t-subsets.
        """
        rng = random.Random(random_seed)
        blocks = []
        
        # For efficiency, track uncovered t-subsets
        uncovered = set(uncovered_set)
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Find the best block to add
            best_block = None
            best_gain = 0
            
            # Try to build a block greedily
            # Start with a random uncovered t-subset
            seed_ts = rng.choice(list(uncovered))
            block = list(seed_ts)
            
            # Add points to maximize coverage
            while len(block) < k:
                best_point = None
                best_point_gain = 0
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Compute gain: how many new t-subsets would be covered
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    
                    if gain > best_point_gain:
                        best_point_gain = gain
                        best_point = x
                
                if best_point is None:
                    # Pick a random point
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        best_point = rng.choice(remaining)
                    else:
                        break
                
                block.append(best_point)
            
            block.sort()
            
            # Calculate actual gain
            gain = 0
            for ts in itertools.combinations(block, t):
                if ts in uncovered:
                    gain += 1
            
            if gain > 0:
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
        
        return blocks
    
    # Try multiple random seeds and keep the best result
    best_blocks = None
    best_len = float('inf')
    
    num_attempts = 5
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Reset covered array
        covered = [False] * num_t_subsets
        
        # Get a random permutation of t-subsets for tie-breaking
        rng = random.Random(attempt * 12345 + 678)
        
        # Greedy: repeatedly pick the uncovered t-subset that leads to the best block
        blocks = []
        uncovered = set(all_t_subsets)
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset as seed
            seed_ts = rng.choice(list(uncovered))
            block = list(seed_ts)
            
            # Greedily add points
            while len(block) < k:
                best_point = None
                best_point_gain = 0
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    
                    if gain > best_point_gain:
                        best_point_gain = gain
                        best_point = x
                
                if best_point is None:
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        best_point = rng.choice(remaining)
                    else:
                        break
                
                block.append(best_point)
            
            block.sort()
            
            # Only add if it covers something new
            new_covered = 0
            for ts in itertools.combinations(block, t):
                if ts in uncovered:
                    new_covered += 1
            
            if new_covered > 0:
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
        
        if not uncovered and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
        
        if time.time() > deadline - 20:
            break
    
    # If we still have uncovered, do a final greedy pass
    if best_blocks is None or len(uncovered) > 0:
        # Fallback: simple greedy
        blocks = []
        uncovered = set(all_t_subsets)
        while uncovered:
            seed_ts = min(uncovered)
            block = list(seed_ts)
            while len(block) < k:
                best_point = None
                best_point_gain = 0
                for x in range(v):
                    if x in block:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t-1) 
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_point_gain:
                        best_point_gain = gain
                        best_point = x
                if best_point is None:
                    remaining = [x for x in range(v) if x not in block]
                    best_point = remaining[0] if remaining else None
                    if best_point is None:
                        break
                block.append(best_point)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        best_blocks = blocks
    
    # Local search: try to remove redundant blocks
    if best_blocks and time.time() < deadline - 5:
        for i in range(len(best_blocks) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            # Try removing block i
            candidate = best_blocks[:i] + best_blocks[i+1:]
            # Check if still valid
            all_covered = set()
            for b in candidate:
                for ts in itertools.combinations(b, t):
                    all_covered.add(ts)
            if len(all_covered) == num_t_subsets:
                best_blocks = candidate
    
    return best_blocks
# EVOLVE-BLOCK-END
