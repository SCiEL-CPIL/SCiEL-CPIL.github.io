"""Optimized covering C(24,7,3) using Weighted Greedy and Incremental Local Search."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using:
    1. Weighted Greedy: Prioritizes triples based on 'rarity' of their points.
    2. Incremental Local Search: Uses coverage counts to quickly identify redundant blocks
       and performs swap operations to optimize block efficiency.
    """
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Phase 1: Weighted Greedy Initialization
    blocks = _weighted_greedy(v, k, t, all_triples, triple_to_idx)
    
    # Phase 2: Incremental Local Search Optimization
    blocks = _incremental_local_search(v, k, t, all_triples, triple_to_idx, blocks)
    
    return blocks

def _weighted_greedy(v, k, t, all_triples, triple_to_idx):
    """
    Greedy covering using dynamic weights.
    Weight of a triple is inversely proportional to the frequency of its points in the uncovered set.
    """
    uncovered = set(all_triples)
    blocks = []
    
    # Pre-calculate point frequencies in the full set (static)
    # But we need dynamic frequencies for the uncovered set.
    
    while uncovered:
        # Calculate dynamic point frequencies
        point_freq = [0] * v
        for triple in uncovered:
            for p in triple:
                point_freq[p] += 1
        
        # Calculate weight for each uncovered triple
        # Weight = sum(1 / (freq[p] + 1)) for p in triple
        # We want to pick the block that covers the highest total weight of newly covered triples.
        
        # To speed up, we can just pick a "seed" triple with high weight, then build a block.
        # This is a heuristic approximation of the full greedy set cover.
        
        best_seed_triple = None
        best_seed_weight = -1
        for triple in uncovered:
            w = 0.0
            for p in triple:
                w += 1.0 / (point_freq[p] + 1)
            if w > best_seed_weight:
                best_seed_weight = w
                best_seed_triple = triple
        
        block = list(best_seed_triple)
        
        # Extend block greedily
        while len(block) < k:
            best_point = None
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0.0
                # New triples formed by adding x
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        triple = tuple(sorted([block[i], block[j], x]))
                        if triple in uncovered:
                            w = 0.0
                            for p in triple:
                                w += 1.0 / (point_freq[p] + 1)
                            gain += w
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return blocks

def _incremental_local_search(v, k, t, all_triples, triple_to_idx, blocks, max_passes=10):
    """
    Local search using incremental coverage counts.
    1. Maintain cover_count array.
    2. Remove redundant blocks (those where all covered triples have count > 1).
    3. Try to swap blocks: Replace a block with another that covers its unique triples
       and potentially more.
    """
    blocks = list(blocks)
    num_triples = len(all_triples)
    cover_count = [0] * num_triples
    
    # Initialize coverage counts
    for b in blocks:
        for triple in itertools.combinations(b, t):
            idx = triple_to_idx[triple]
            cover_count[idx] += 1
    
    def update_cover_count(block, delta):
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            cover_count[idx] += delta
    
    # Pass 1: Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            # Check if block i is redundant
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                idx = triple_to_idx[triple]
                if cover_count[idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                update_cover_count(blocks[i], -1)
                blocks.pop(i)
                changed = True
                break
                
    # Pass 2: Swap optimization
    random.seed(42)
    for _ in range(1000): # Fixed number of iterations
        if not blocks:
            break
        idx = random.randint(0, len(blocks) - 1)
        
        # Identify unique triples for block idx
        unique_triples = []
        for triple in itertools.combinations(blocks[idx], t):
            t_idx = triple_to_idx[triple]
            if cover_count[t_idx] == 1:
                unique_triples.append(triple)
                
        if not unique_triples:
            # Redundant, remove it
            update_cover_count(blocks[idx], -1)
            blocks.pop(idx)
            continue
            
        # Check if we can form a new block covering unique_triples
        # The new block must contain all points from unique_triples?
        # No, it must contain each triple.
        # If the union of points in unique_triples is > k, we can't replace with one block.
        candidate_points = set()
        for triple in unique_triples:
            candidate_points.update(triple)
            
        if len(candidate_points) > k:
            continue
            
        # Try to extend candidate_points to a valid block of size k
        # that maximizes additional coverage.
        new_block = list(candidate_points)
        
        # If new_block is not size k, extend it
        while len(new_block) < k:
            best_point = None
            best_gain = -1
            
            for x in range(v):
                if x in new_block:
                    continue
                
                gain = 0
                # Count how many unique_triples are covered by adding x?
                # Actually, we just want to maximize total coverage of *any* triple
                # that is currently under-covered or just to maximize the "value" of the block.
                # Since we must cover unique_triples, and they are already in candidate_points,
                # any extension just adds more coverage.
                # Let's just pick a point that maximizes the number of *new* triples covered.
                
                # To speed up, just pick a random point or one that doesn't conflict?
                # Better: pick point that maximizes overlap with other blocks? No, we want *new* coverage.
                
                # Calculate gain: number of triples formed by (new_block + x) that are in all_triples
                # and have cover_count > 0 (i.e., are already covered? No, we want to cover MORE).
                # Actually, since we are replacing a block, we lose the coverage of blocks[idx].
                # The net gain is (coverage of new_block) - (coverage of blocks[idx] not in unique_triples).
                # This is complex.
                
                # Simpler heuristic: Just fill with random points to ensure we have a valid block.
                # Then check if it's better.
                pass
                
            # For simplicity and speed, let's just pick random points to fill the block
            # and check if it's a valid replacement.
            # A valid replacement must cover all unique_triples.
            # Since candidate_points is a subset of new_block, and unique_triples are subsets of candidate_points,
            # any superset of candidate_points covers unique_triples.
            
            # So we just need to pick (k - len(new_block)) random points.
            available = [x for x in range(v) if x not in new_block]
            if not available:
                break
            new_block.append(random.choice(available))
            
        if len(new_block) != k:
            continue
            
        new_block.sort()
        
        # Check if new_block is better than blocks[idx]
        # Better if it covers at least as many triples, or covers unique_triples and some others.
        # Actually, since we *must* cover unique_triples, and new_block does,
        # we should check if new_block covers *more* other triples than blocks[idx] did.
        
        # Count coverage of new_block
        new_covered = set(itertools.combinations(new_block, t))
        old_covered = set(itertools.combinations(blocks[idx], t))
        
        # Triples in new_block but not in old_block
        added = new_covered - old_covered
        # Triples in old_block but not in new_block (and not in unique_triples)
        # unique_triples are in both? No, unique_triples are in old_block.
        # Are they in new_block? Yes, because candidate_points is in new_block.
        
        # So the only loss is old_covered - new_covered.
        # The gain is added.
        # We want len(added) > len(old_covered - new_covered) ?
        # Or just len(new_covered) > len(old_covered)?
        
        if len(new_covered) > len(old_covered):
            # Swap
            update_cover_count(blocks[idx], -1)
            update_cover_count(new_block, 1)
            blocks[idx] = new_block
            
    return blocks
