"""Improved probabilistic construction: Greedy initialization with diversity + Local Search for minimization."""
import itertools
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Strategy:
    1. Greedy construction biased towards covering rare/uncovered triples.
    2. Local search (Delete/Replace) to minimize block count.
    """
    # Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Coverage state
    covered = np.zeros(n_tsubs, dtype=np.int32)
    blocks = []
    
    def get_block_tsubs(block):
        """Return list of indices of t-subsets in the block."""
        # block is a sorted tuple
        return [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]

    def update_coverage(block, sign):
        """Update covered array by +/- 1 for all tsubs in block."""
        for idx in get_block_tsubs(block):
            covered[idx] += sign

    def is_covered(idx):
        return covered[idx] > 0

    # Phase 1: Greedy Construction
    # We maintain a list of uncovered triple indices to speed up gain calculation?
    # For v=24, t=3, n_tsubs=2024. It's small enough to just check covered[idx]==0.
    
    random.seed(42)
    
    # To improve on pure greedy, we use a "Randomized Greedy" approach.
    # Pick a random uncovered triple, then try to find a block covering it that also covers many others.
    
    # Initial greedy pass
    while True:
        uncovered_indices = np.where(covered == 0)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a random uncovered triple to "focus" on
        target_idx = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
        target_triple = all_tsubs[target_idx]
        
        best_block = None
        best_score = -1
        
        # Generate candidates that contain the target triple
        # Target triple has 't' points. We need 'k-t' more points.
        target_set = set(target_triple)
        remaining_points = [p for p in range(v) if p not in target_set]
        
        # Sample a few random completions
        for _ in range(20):
            if len(remaining_points) < k - t:
                break
            fillers = random.sample(remaining_points, k - t)
            block = tuple(sorted(list(target_triple) + fillers))
            
            # Calculate score: Gain (uncovered covered) - Penalty (overlap)
            gain = 0
            overlap = 0
            for idx in get_block_tsubs(block):
                if covered[idx] == 0:
                    gain += 1
                else:
                    overlap += 1
            
            # We prioritize gain, but penalize overlap slightly to encourage diversity
            score = gain * 10 - overlap
            
            if score > best_score:
                best_score = score
                best_block = block
        
        # Fallback: if no good block found (shouldn't happen), add a random one
        if best_block is None:
            block = tuple(sorted(random.sample(range(v), k)))
            best_block = block
        
        blocks.append(best_block)
        update_coverage(best_block, 1)
        
        # Safety limit
        if len(blocks) > 500:
            break

    # Phase 2: Local Search Minimization
    # We iterate through blocks and try to remove or replace them.
    
    max_iterations = 100
    for _ in range(max_iterations):
        improved = False
        
        # Iterate in reverse to handle deletion safely
        for i in range(len(blocks) - 1, -1, -1):
            current_block = blocks[i]
            
            # Temporarily remove
            update_coverage(current_block, -1)
            
            # Find triples that are now uncovered (unique to this block)
            unique_triples = []
            for idx in get_block_tsubs(current_block):
                if covered[idx] == 0:
                    unique_triples.append(idx)
            
            if not unique_triples:
                # Block is redundant, remove it
                blocks.pop(i)
                improved = True
                continue
            else:
                # Try to find a replacement block that covers all unique_triples
                # and has minimal overlap with the rest.
                
                # The replacement block must contain all points in the unique triples?
                # No, it must contain each triple.
                # Let U be the set of points in unique_triples.
                # If |U| > k, no single block can cover them all.
                points_in_unique = set()
                for idx in unique_triples:
                    points_in_unique.update(all_tsubs[idx])
                
                if len(points_in_unique) > k:
                    # Cannot replace with a single block that covers all unique triples
                    # Restore and move on
                    update_coverage(current_block, 1)
                    continue
                
                # Try to find a replacement
                best_replacement = None
                best_overlap = float('inf')
                
                # Generate candidates containing points_in_unique
                remaining_points = [p for p in range(v) if p not in points_in_unique]
                needed_fillers = k - len(points_in_unique)
                
                # If we can't fill, restore
                if len(remaining_points) < needed_fillers:
                    update_coverage(current_block, 1)
                    continue
                
                for _ in range(10):
                    fillers = random.sample(remaining_points, needed_fillers)
                    candidate = tuple(sorted(list(points_in_unique) + fillers))
                    
                    # Check if candidate covers all unique_triples
                    # (It should, by construction, since it contains points_in_unique)
                    # But wait, does containing the union of points guarantee covering the triples?
                    # Yes, if triple T is subset of points_in_unique, and points_in_unique is subset of candidate, then T is subset of candidate.
                    
                    # Calculate overlap with current set (excluding current_block which is removed)
                    overlap = 0
                    for idx in get_block_tsubs(candidate):
                        if covered[idx] > 0:
                            overlap += 1
                    
                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_replacement = candidate
                
                if best_replacement is not None and best_replacement != current_block:
                    # Replace
                    blocks[i] = best_replacement
                    update_coverage(best_replacement, 1)
                    improved = True
                else:
                    # Restore original
                    update_coverage(current_block, 1)
        
        if not improved:
            break

    # Final check and repair if any uncovered remain (should be none)
    while True:
        uncovered_indices = np.where(covered == 0)[0]
        if len(uncovered_indices) == 0:
            break
        
        target_idx = uncovered_indices[0]
        target_triple = all_tsubs[target_idx]
        target_set = set(target_triple)
        remaining_points = [p for p in range(v) if p not in target_set]
        
        # Add a block covering this triple
        # Pick random fillers
        if len(remaining_points) >= k - t:
            fillers = random.sample(remaining_points, k - t)
        else:
            fillers = remaining_points[:]
            # If v is small, we might not have enough, but k<=v so this should be fine
            # If k-t > len(remaining), it's impossible, but k<=v ensures enough points
            # Actually if k-t > v-t, i.e., k>v, which is invalid input.
            
        block = tuple(sorted(list(target_triple) + fillers))
        blocks.append(block)
        update_coverage(block, 1)

    return [list(b) for b in blocks]
