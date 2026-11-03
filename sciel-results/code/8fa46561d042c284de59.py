"""Optimized covering C(24,7,3) using Information-Theoretic (Entropy Reduction) Greedy."""
import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an Information-Theoretic approach:
    1. Entropy-Reduction Greedy: Prioritize covering 'scarce' or 'isolated' triples
       to maximize the reduction in the entropy of the uncovered triple distribution.
    2. Local Search: Optimize the initial solution by removing redundant blocks
       and replacing blocks with more efficient ones.
    """
    all_triples = set(itertools.combinations(range(v), t))
    
    # Phase 1: Information-Theoretic Greedy
    blocks = _info_greedy(v, k, t, all_triples)
    
    # Phase 2: Optimized Local Search
    blocks = _local_search(v, k, t, all_triples, blocks)
    
    return blocks

def _info_greedy(v, k, t, all_triples):
    """
    Greedy covering that maximizes entropy reduction.
    We use a dynamic weighting scheme based on the 'connectivity' of uncovered triples.
    Triples with low connectivity to other uncovered triples are 'isolated' and
    prioritized because they are harder to cover in bulk.
    """
    uncovered = set(all_triples)
    blocks = []
    
    # Maintain pair counts for uncovered triples to estimate isolation
    # pair_counts[p1, p2] = number of uncovered triples containing p1 and p2
    pair_counts = defaultdict(int)
    for triple in uncovered:
        for i in range(t):
            for j in range(i+1, t):
                pair = (triple[i], triple[j])
                pair_counts[pair] += 1
                
    while uncovered:
        # Find the "most isolated" uncovered triple.
        # Isolation score = sum of 1 / (pair_count(pair) + 1) for pairs in triple.
        # High score = low connectivity = isolated.
        
        best_triple = None
        best_score = -1
        
        for triple in uncovered:
            score = 0.0
            for i in range(t):
                for j in range(i+1, t):
                    pair = (triple[i], triple[j])
                    # If pair count is 1, it's very isolated
                    score += 1.0 / (pair_counts[pair] + 1)
            
            if score > best_score:
                best_score = score
                best_triple = triple
        
        # Build block around best_triple
        block = list(best_triple)
        
        # Extend block to size k
        # Greedy extension: pick point x that maximizes coverage of uncovered triples
        # weighted by their isolation score
        
        while len(block) < k:
            best_point = None
            best_gain = -1
            
            # Pre-compute pairs in current block for gain calculation
            block_pairs = set()
            for i in range(len(block)):
                for j in range(i+1, len(block)):
                    block_pairs.add((block[i], block[j]))
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0.0
                # New triples formed: {a, b, x} where a,b in block
                for p1, p2 in block_pairs:
                    triple = tuple(sorted([p1, p2, x]))
                    if triple in uncovered:
                        # Weight by isolation of this triple
                        # Approximate isolation using current pair_counts
                        w = 0.0
                        # Pairs in this new triple: (p1,p2), (p1,x), (p2,x)
                        w += 1.0 / (pair_counts.get((p1, p2), 0) + 1)
                        w += 1.0 / (pair_counts.get((p1, x), 0) + 1)
                        w += 1.0 / (pair_counts.get((p2, x), 0) + 1)
                        gain += w
                        
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None or best_gain == 0:
                # If no gain, pick a random point to fill the block
                remaining_points = [x for x in range(v) if x not in block]
                if remaining_points:
                    best_point = random.choice(remaining_points)
                else:
                    break
                    
            block.append(best_point)
            
        block.sort()
        if len(block) == k:
            blocks.append(block)
            
            # Update uncovered and pair_counts
            for triple in itertools.combinations(block, t):
                if triple in uncovered:
                    uncovered.discard(triple)
                    # Decrement pair counts
                    for i in range(t):
                        for j in range(i+1, t):
                            pair = (triple[i], triple[j])
                            pair_counts[pair] -= 1
                            if pair_counts[pair] == 0:
                                del pair_counts[pair]
                
    return blocks

def _local_search(v, k, t, all_triples, blocks, max_iters=200):
    """
    Optimized local search with 2-opt (add 1, remove 2).
    """
    blocks = list(blocks)
    
    # Initialize coverage counts
    cover_counts = defaultdict(int)
    for b in blocks:
        for triple in itertools.combinations(b, t):
            cover_counts[triple] += 1
            
    # Pass 1: Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                if cover_counts[triple] <= 1:
                    redundant = False
                    break
            
            if redundant:
                for triple in itertools.combinations(blocks[i], t):
                    cover_counts[triple] -= 1
                blocks.pop(i)
                changed = True
                break
                
    # Pass 2: 2-Opt (Add 1, Remove 2)
    # We try to add a random block, then see if we can remove 2 blocks.
    # If we can, the net change is -1 block.
    
    improved = True
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        
        # Generate a random candidate block
        candidate = random.sample(range(v), k)
        candidate.sort()
        
        # Check if candidate is already in blocks
        if candidate in blocks:
            continue
            
        # Add candidate
        for triple in itertools.combinations(candidate, t):
            cover_counts[triple] += 1
            
        # Try to remove 2 blocks
        # Find blocks that are now redundant
        to_remove = []
        
        for i in range(len(blocks)):
            # Check if blocks[i] is redundant given current cover_counts
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                if cover_counts[triple] <= 1:
                    redundant = False
                    break
            
            if redundant:
                to_remove.append(i)
                if len(to_remove) == 2:
                    break
                    
        if len(to_remove) == 2:
            # Remove these 2 blocks
            for idx in sorted(to_remove, reverse=True):
                b = blocks[idx]
                for triple in itertools.combinations(b, t):
                    cover_counts[triple] -= 1
                blocks.pop(idx)
            
            # We added 1, removed 2. Net -1.
            improved = True
            
        else:
            # Revert candidate
            for triple in itertools.combinations(candidate, t):
                cover_counts[triple] -= 1
            
    # Final redundancy pass
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                if cover_counts[triple] <= 1:
                    redundant = False
                    break
            
            if redundant:
                for triple in itertools.combinations(blocks[i], t):
                    cover_counts[triple] -= 1
                blocks.pop(i)
                changed = True
                break
                
    return blocks
