import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Pre-compute all t-subsets of v points
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    
    blocks = []
    point_freq = np.zeros(v, dtype=np.int32)
    
    def update_uncovered(block):
        """Remove all t-subsets of 'block' from 'uncovered'."""
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
            
    def gain(x, current_block):
        """Calculate how many new t-subsets adding x would cover."""
        # A subset S is covered if x in S and S\{x} subset current_block
        # Since t=3, S\{x} is a pair from current_block.
        # We iterate over pairs in current_block and check if (pair + x) is in uncovered.
        count = 0
        # Optimization: only consider pairs that are in current_block
        # For t=3, we check all combinations of size t-1 from current_block
        for rest in itertools.combinations(current_block, t - 1):
            # rest is a tuple of t-1 elements
            # We need to check if sorted(rest + (x,)) is in uncovered
            # Constructing the tuple and sorting is costly. 
            # Since current_block is sorted, and x is added, we can insert x in sorted order?
            # Or just use a frozenset? No, uncovered is a set of tuples.
            # Let's just construct the tuple.
            sub = tuple(sorted(rest + (x,)))
            if sub in uncovered:
                count += 1
        return count

    # Greedy Loop
    while uncovered and time.time() < deadline - 5: # Leave 5s for cleanup
        # Start with the lexicographically first uncovered subset to ensure determinism
        # min(uncovered) might be slow if set is large, but v=24, t=3 is small.
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        
        while len(block) < k:
            # Find best point
            best_point = -1
            best_gain = -1
            best_freq = float('inf')
            
            # Candidates are points not in block
            # To speed up, we can check all points and filter, or maintain a list.
            # v=24 is small.
            candidates = [x for x in range(v) if x not in block]
            
            # Vectorized gain calculation idea:
            # For each candidate x, we need count of (rest in C(block, t-1)) s.t. (rest+x) in uncovered.
            # Let's stick to the inner loop for now, it's small.
            # But we can pre-compute the set of "missing pairs" for the current block?
            # No, uncovered changes.
            
            # Optimization: 
            # We want to maximize gain. 
            # Tie-break by lower frequency.
            
            gains = []
            for x in candidates:
                g = gain(x, block)
                if g > 0: # Only consider if it adds value
                    gains.append((g, -point_freq[x], x))
            
            if not gains:
                # If no point adds coverage, we must pick one to complete the block?
                # The problem requires covering ALL t-subsets. 
                # If we are stuck, it means the current block prefix doesn't help?
                # Actually, if uncovered is not empty, there is always some way to cover it.
                # But if we built the block badly, we might have 0 gain for all remaining points.
                # In that case, we just pick an arbitrary point to fill the block, 
                # hoping the block as a whole covers something? 
                # No, if gain is 0 for all, adding any point covers 0 NEW subsets.
                # This implies the subsets containing the current block prefix are already covered?
                # But we started with an uncovered subset. 
                # If block size >= t, it would have covered that initial subset.
                # So if len(block) < t, gain must be > 0 for points completing the t-subset.
                # If len(block) >= t, and we still have uncovered subsets, 
                # it's possible that the current block doesn't help with the *most* urgent ones, 
                # but it must have helped with some.
                # If gains is empty, it means no single point addition covers any NEW uncovered subset.
                # This can happen if the current block is "bad". 
                # We should just pick the point with lowest frequency to finish the block.
                gains = [(0, -point_freq[x], x) for x in candidates]
                
            # Sort by gain desc, then freq asc (so -freq desc)
            gains.sort(key=lambda x: (x[0], x[1]), reverse=True)
            
            best_point = gains[0][2]
            block.append(best_point)
            point_freq[best_point] += 1
            
        block.sort()
        blocks.append(block)
        update_uncovered(block)
        
        # Early exit if we have a very good number of blocks? 
        # Best known is 76. If we get below 72, it's great.
        # We can't know the optimal, but we can stop if time is low.

    # Post-Processing: Remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks.
    # We will iterate and remove blocks that are redundant.
    # We do this in a loop until no changes occur.
    
    # Pre-compute t-subsets for each block for faster checking
    block_subsets = [list(itertools.combinations(b, t)) for b in blocks]
    
    # Build a mapping: subset -> list of block indices containing it
    # Since number of blocks is small (~70-80), and subsets per block is 35,
    # Total entries ~ 2500.
    # Map: subset -> set of indices
    subset_to_blocks = {}
    for i, subs in enumerate(block_subsets):
        for s in subs:
            if s not in subset_to_blocks:
                subset_to_blocks[s] = set()
            subset_to_blocks[s].add(i)
    
    # Check redundancy
    changed = True
    current_blocks = blocks[:]
    current_subsets = block_subsets[:]
    
    # Re-index logic:
    # We'll just mark blocks as removed.
    removed = [False] * len(current_blocks)
    
    while changed:
        changed = False
        for i in range(len(current_blocks)):
            if removed[i]:
                continue
            
            # Check if all subsets of block i are covered by some other block
            redundant = True
            for s in current_subsets[i]:
                # Find blocks containing s, excluding i
                covering_blocks = subset_to_blocks.get(s, set())
                covering_blocks.discard(i)
                if not covering_blocks:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                removed[i] = True
                changed = True
                # Update subset_to_blocks? 
                # For efficiency in next iterations, we should remove i from all sets.
                for s in current_subsets[i]:
                    if s in subset_to_blocks:
                        subset_to_blocks[s].discard(i)
                        if not subset_to_blocks[s]:
                            del subset_to_blocks[s]
                
                # Check time
                if time.time() > deadline - 2:
                    break

    # Construct final list
    final_blocks = [b for i, b in enumerate(current_blocks) if not removed[i]]
    
    return final_blocks
