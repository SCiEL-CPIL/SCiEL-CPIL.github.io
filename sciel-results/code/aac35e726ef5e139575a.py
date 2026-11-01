import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    # Parameters for C(24, 7, 3)
    # v=24, k=7, t=3
    
    if v != 24 or k != 7 or t != 3:
        # Fallback for other parameters, though task specifies 24,7,3
        # Simple greedy fallback
        return _simple_greedy(v, k, t)
        
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Precompute all triples and pair-triple mappings
    all_triples = list(itertools.combinations(range(v), t))
    triple_set = set(all_triples)
    
    # Precompute for each pair, which triples contain it
    pair_to_triples = {}
    for triple in all_triples:
        for pair in itertools.combinations(triple, 2):
            if pair not in pair_to_triples:
                pair_to_triples[pair] = []
            pair_to_triples[pair].append(triple)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy instances with local search
    while time.time() < deadline:
        blocks = _greedy_fast(v, k, t, triple_set, pair_to_triples, random_seed=random.randint(0, 10**6))
        
        # Apply reduction
        blocks = _reduce_blocks(blocks, v, t)
        
        # Try local search to improve
        blocks = _local_search(blocks, v, t, triple_set, pair_to_triples, time_budget=5.0)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
            # If we hit the known best (76), continue trying to beat it
            if best_count <= 76:
                pass

    if best_blocks is None:
        # Should not happen, but safe fallback
        best_blocks = _simple_greedy(v, k, t)
        
    # Final reduction on the best result just in case
    best_blocks = _reduce_blocks(best_blocks, v, t)
    
    # Ensure blocks are lists of ints, distinct, etc.
    return [list(b) for b in best_blocks]

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        # Pick first uncovered
        target = min(uncovered)
        # Find best block containing target
        best_block = None
        best_gain = -1
        # Generate all k-subsets containing target
        # target is a tuple of t elements
        remaining_points = [p for p in range(v) if p not in target]
        need = k - t
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(target + combo))
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        blocks.append(best_block)
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
    return blocks

def _greedy_fast(v, k, t, triple_set, pair_to_triples, random_seed):
    rng = random.Random(random_seed)
    uncovered = set(triple_set)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered triple to start
        unc_list = list(uncovered)
        target = unc_list[rng.randrange(len(unc_list))]
        
        # Find best block containing target
        best_block = None
        best_gain = -1
        
        target_set = set(target)
        remaining_points = [p for p in range(v) if p not in target_set]
        need = k - t
        
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(target + combo))
            
            # Calculate gain using pair_to_triples for speed
            gain = 0
            block_triples = list(itertools.combinations(block, t))
            for sub in block_triples:
                if sub in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
            elif gain == best_gain:
                # Tie-breaking: random pick among best
                if rng.random() < 0.5:
                    best_block = block
                    
        blocks.append(best_block)
        
        # Update uncovered
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
            
    return blocks

def _local_search(blocks, v, t, triple_set, pair_to_triples, time_budget=5.0):
    """
    Try to replace blocks with better ones to reduce the total count.
    A block can be removed if all its triples are covered by other blocks.
    We try to find blocks that are "almost redundant" and can be replaced.
    """
    if not blocks:
        return blocks
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Build coverage counts
    triple_coverage = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            triple_coverage[sub] += 1
    
    # Try to remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            block = blocks[i]
            redundant = True
            for sub in itertools.combinations(block, t):
                if triple_coverage[sub] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove this block
                for sub in itertools.combinations(block, t):
                    triple_coverage[sub] -= 1
                blocks.pop(i)
                changed = True
                break
    
    return blocks

def _reduce_blocks(blocks, v, t):
    """
    Remove redundant blocks.
    A block is redundant if every t-subset in it is covered by at least one other block.
    """
    if not blocks:
        return []
        
    # Build coverage counts
    triple_coverage = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            triple_coverage[sub] += 1
    
    # Iteratively remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            block = blocks[i]
            redundant = True
            for sub in itertools.combinations(block, t):
                if triple_coverage[sub] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove this block
                for sub in itertools.combinations(block, t):
                    triple_coverage[sub] -= 1
                blocks.pop(i)
                changed = True
                break
    
    return blocks
