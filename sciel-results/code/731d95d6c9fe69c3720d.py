"""Improved covering design constructor using randomized greedy and reduction."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t).
    Uses a randomized greedy approach followed by a reduction phase.
    """
    deadline = time.time() + 135.0
    rng = random.Random(42)
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute for each t-subset, the list of candidate blocks (k-subsets) containing it.
    # A block contains a t-subset if we pick k-t points from the remaining v-t points.
    # For v=24, k=7, t=3: pick 4 from 21. ~6000 candidates per t-subset.
    # Total candidates to consider in greedy step is small.
    
    # We don't precompute all blocks for all tsubs as it's memory heavy, 
    # but we can generate them on the fly or cache them.
    # Given v=24 is small, generating on the fly is fine.
    
    def greedy_construct(seed):
        rng_local = random.Random(seed)
        uncovered = set(range(n_tsubs))
        blocks = []
        
        # Map t-subset index to the set of points in it (for quick access)
        # all_tsubs is a list of tuples.
        
        while uncovered:
            if time.time() > deadline - 5:
                break
                
            # Pick a random uncovered t-subset to anchor the new block
            # This helps in diversification
            anchor_idx = rng_local.choice(list(uncovered))
            anchor_tsub = all_tsubs[anchor_idx]
            
            # Generate all k-subsets containing anchor_tsub
            # We need to choose k-t points from the v - len(anchor_tsub) remaining points
            remaining_points = [p for p in range(v) if p not in anchor_tsub]
            candidates = []
            
            # Calculate gain for each candidate block
            best_gain = -1
            best_block = None
            
            # To speed up, we can sample if there are too many, but 5985 is fine.
            for combo in itertools.combinations(remaining_points, k - t):
                block_tuple = tuple(sorted(anchor_tsub + combo))
                
                # Calculate how many uncovered t-subsets this block covers
                gain = 0
                # Iterate through all t-subsets of this block
                for sub in itertools.combinations(block_tuple, t):
                    idx = tsub_to_idx[sub]
                    if idx in uncovered:
                        gain += 1
                
                # Tie-breaking: random or lexicographic
                if gain > best_gain or (gain == best_gain and best_block is not None and rng_local.random() < 0.5):
                    best_gain = gain
                    best_block = block_tuple
            
            if best_block is None:
                break
                
            # Add the block
            blocks.append(list(best_block))
            
            # Update uncovered set
            for sub in itertools.combinations(best_block, t):
                idx = tsub_to_idx[sub]
                uncovered.discard(idx)
                
        return blocks

    def reduce_blocks(blocks):
        """
        Removes redundant blocks from the list.
        """
        if not blocks:
            return blocks
            
        # Count coverage for each t-subset
        # cover_count[i] = number of blocks covering all_tsubs[i]
        cover_count = [0] * n_tsubs
        
        # Initialize counts
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = tsub_to_idx[sub]
                cover_count[idx] += 1
                
        # Iteratively remove blocks that are redundant
        # A block is redundant if all its t-subsets have cover_count > 1
        # Note: We must be careful. If we remove a block, cover_counts decrease.
        # We do this in a loop until no more removals are possible.
        
        changed = True
        while changed:
            changed = False
            # We iterate backwards to handle removals easily
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 2:
                    return blocks
                    
                block = blocks[i]
                is_redundant = True
                for sub in itertools.combinations(block, t):
                    idx = tsub_to_idx[sub]
                    if cover_count[idx] <= 1:
                        is_redundant = False
                        break
                        
                if is_redundant:
                    # Remove block i
                    blocks.pop(i)
                    # Decrement counts
                    for sub in itertools.combinations(block, t):
                        idx = tsub_to_idx[sub]
                        cover_count[idx] -= 1
                    changed = True
                    
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy constructions with different seeds
    # We can also try a "smart" greedy that doesn't pick random anchor,
    # but picks the uncovered subset with the fewest candidate blocks (most constrained)
    # However, for v=24, random anchor is fast and effective.
    
    start_time = time.time()
    attempts = 0
    
    while time.time() < deadline - 10:
        seed = 42 + attempts
        blocks = greedy_construct(seed)
        
        if not blocks:
            continue
            
        # Check validity before reducing (should be valid by construction if loop finished)
        # But if we broke early due to time, it might be invalid.
        # We need to ensure we only return valid covers.
        
        # Quick validity check
        covered = set()
        for block in blocks:
            for sub in itertools.combinations(block, t):
                covered.add(sub)
                
        if len(covered) < n_tsubs:
            # Invalid, try to fix or discard
            # If we broke early, we might have uncovered subsets.
            # We can try to add blocks for remaining uncovered subsets greedily.
            remaining = set(range(n_tsubs)) - set(tsub_to_idx[sub] for sub in covered)
            if remaining:
                # Add simple blocks for remaining
                for idx in remaining:
                    tsub = all_tsubs[idx]
                    # Pick a block containing this tsub
                    # Just pick the first one from candidates
                    remaining_points = [p for p in range(v) if p not in tsub]
                    # Use a deterministic choice for speed
                    combo = tuple(sorted(tsub + remaining_points[:k-t]))
                    blocks.append(list(combo))
        
        # Reduce
        reduced_blocks = reduce_blocks(blocks)
        
        # Final validity check
        covered_final = set()
        for block in reduced_blocks:
            for sub in itertools.combinations(block, t):
                covered_final.add(sub)
                
        if len(covered_final) == n_tsubs:
            if len(reduced_blocks) < best_count:
                best_count = len(reduced_blocks)
                best_blocks = reduced_blocks
                # If we hit the known lower bound or very close, we might stop early?
                # Best known is 76.
                if best_count <= 76:
                    break
                    
        attempts += 1
        # If we have done enough attempts, we can stop
        if attempts > 50:
            break

    if best_blocks is None:
        # Fallback: return a valid cover even if not optimal
        # Use the first successful one or construct a trivial one
        blocks = greedy_construct(0)
        # Ensure validity
        covered = set()
        for block in blocks:
            for sub in itertools.combinations(block, t):
                covered.add(sub)
        remaining = set(range(n_tsubs)) - set(tsub_to_idx[sub] for sub in covered)
        for idx in remaining:
            tsub = all_tsubs[idx]
            remaining_points = [p for p in range(v) if p not in tsub]
            combo = tuple(sorted(tsub + remaining_points[:k-t]))
            blocks.append(list(combo))
        return blocks

    return best_blocks

# EVOLVE-BLOCK-END
