"""Optimized covering C(24,7,3) using Information-Theoretic (Entropy Reduction) Greedy."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an Information-Theoretic approach with random restarts.
    """
    all_triples = set(itertools.combinations(range(v), t))
    
    best_blocks = None
    # Run multiple times with different seeds to escape local minima
    for seed in range(5):
        # Phase 1: Information-Theoretic Greedy
        blocks = _info_greedy(v, k, t, all_triples, seed)
        
        # Phase 2: Efficient Redundancy Removal
        blocks = _remove_redundant(v, k, t, blocks)
        
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            
    return best_blocks

def _info_greedy(v, k, t, all_triples, seed=0):
    """
    Greedy covering that maximizes entropy reduction.
    """
    rng = random.Random(seed)
    uncovered = set(all_triples)
    blocks = []
    
    while uncovered:
        point_freq = [0] * v
        for triple in uncovered:
            for p in triple:
                point_freq[p] += 1
        
        # Score each uncovered triple
        best_triple = None
        best_score = -1
        candidates = []
        for triple in uncovered:
            score = 0.0
            for p in triple:
                score += 1.0 / (point_freq[p] + 1)
            if score > best_score:
                best_score = score
                candidates = [triple]
            elif score == best_score:
                candidates.append(triple)
        
        if not candidates:
            break
            
        best_triple = rng.choice(candidates)
        
        # Build a block around best_triple
        block = list(best_triple)
        
        while len(block) < k:
            best_point = None
            best_gain = -1
            candidates = []
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0.0
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
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            if not candidates:
                break
            best_point = rng.choice(candidates)
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return blocks

def _remove_redundant(v, k, t, blocks):
    """
    Remove redundant blocks efficiently using triple counts.
    """
    blocks = list(blocks)
    if not blocks:
        return blocks
        
    # Count coverage for each triple
    counts = {}
    for b in blocks:
        for triple in itertools.combinations(b, t):
            counts[triple] = counts.get(triple, 0) + 1
    
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            # Check if block i is redundant
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                if counts.get(triple, 0) == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for triple in itertools.combinations(blocks[i], t):
                    counts[triple] -= 1
                blocks.pop(i)
                changed = True
                break
    return blocks
