"""Optimized covering C(24,7,3) using Information-Theoretic (Entropy Reduction) Greedy."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an Information-Theoretic approach:
    1. Entropy-Reduction Greedy: Prioritize covering 'scarce' or 'isolated' triples
       to maximize the reduction in the entropy of the uncovered triple distribution.
    2. Local Search: Optimize the initial solution by removing redundant blocks
       and replacing blocks with more efficient ones.
    """
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Phase 1: Information-Theoretic Greedy
    blocks = _info_greedy(v, k, t, all_triples, triple_to_idx)
    
    # Phase 2: Local Search Optimization
    blocks = _local_search(v, k, t, all_triples, triple_to_idx, blocks)
    
    return blocks

def _info_greedy(v, k, t, all_triples, triple_to_idx):
    """
    Greedy covering that maximizes coverage gain, with tie-breaking based on
    point scarcity (entropy reduction proxy).
    """
    uncovered = set(all_triples)
    blocks = []
    
    # Point frequency in uncovered set
    point_freq = [0] * v
    for triple in uncovered:
        for p in triple:
            point_freq[p] += 1
    
    while uncovered:
        # Select a "seed" triple. 
        # Heuristic: Pick the triple with the lowest sum of point frequencies (most isolated/rare).
        # This approximates maximizing entropy reduction by addressing the most constrained part first.
        
        min_score = float('inf')
        seed_triple = None
        for triple in uncovered:
            score = 0
            for p in triple:
                score += point_freq[p]
            if score < min_score:
                min_score = score
                seed_triple = triple
        
        block = list(seed_triple)
        
        # Extend the block to size k
        # We want to maximize the number of newly covered triples.
        # Tie-breaker: Prefer points with lower frequency.
        
        while len(block) < k:
            best_point = None
            best_gain = -1
            best_freq = float('inf')
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                # Calculate gain: how many uncovered triples are completed by adding x?
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        triple = tuple(sorted((block[i], block[j], x)))
                        if triple in uncovered:
                            gain += 1
                            
                if gain > best_gain or (gain == best_gain and point_freq[x] < best_freq):
                    best_gain = gain
                    best_freq = point_freq[x]
                    best_point = x
            
            if best_point is None:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered and point_freq
        for triple in itertools.combinations(block, t):
            if triple in uncovered:
                uncovered.discard(triple)
                for p in triple:
                    point_freq[p] -= 1
            
    return blocks

def _local_search(v, k, t, all_triples, triple_to_idx, blocks, max_iters=1000):
    """
    Local search to reduce the number of blocks.
    Uses coverage counts for fast validity checks.
    """
    blocks = [list(b) for b in blocks]
    num_triples = len(all_triples)
    
    # Initialize coverage counts
    cover_count = [0] * num_triples
    
    def update_counts(block, delta):
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            cover_count[idx] += delta
            
    # Count initial coverage
    for b in blocks:
        update_counts(b, 1)
        
    # Pass 1: Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            b = blocks[i]
            redundant = True
            for triple in itertools.combinations(b, t):
                idx = triple_to_idx[triple]
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                update_counts(b, -1)
                blocks.pop(i)
                changed = True
                break
                
    # Pass 2: Replace blocks to increase redundancy elsewhere
    random.seed(42)
    for _ in range(max_iters):
        idx = random.randint(0, len(blocks) - 1)
        b = blocks[idx]
        
        # Find critical triples for this block
        critical = []
        for triple in itertools.combinations(b, t):
            if cover_count[triple_to_idx[triple]] == 1:
                critical.append(triple)
                
        if not critical:
            # This block is fully redundant
            update_counts(b, -1)
            blocks.pop(idx)
            continue
            
        # If critical set is too large, skip
        if len(critical) > 5:
            continue
            
        # Try to find a replacement block B'
        candidate_points = set()
        for triple in critical:
            candidate_points.update(triple)
            
        if len(candidate_points) > k:
            continue
            
        new_block = list(candidate_points)
        
        # Extend new_block
        while len(new_block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in new_block:
                    continue
                gain = 0
                for i in range(len(new_block)):
                    for j in range(i+1, len(new_block)):
                        triple = tuple(sorted((new_block[i], new_block[j], x)))
                        idx_t = triple_to_idx[triple]
                        # Heuristic: Maximize coverage of triples with low cover count
                        if cover_count[idx_t] < 2:
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    
            if best_point is None:
                break
            new_block.append(best_point)
            
        new_block.sort()
        
        if new_block == b:
            continue
            
        if new_block in blocks:
             update_counts(b, -1)
             blocks.pop(idx)
             continue
             
        covered_critical = True
        for triple in critical:
            if not set(triple).issubset(set(new_block)):
                covered_critical = False
                break
        if not covered_critical:
            continue
            
        if len(set(new_block)) != k:
            continue
            
        update_counts(b, -1)
        update_counts(new_block, 1)
        blocks[idx] = new_block
        
    return blocks
