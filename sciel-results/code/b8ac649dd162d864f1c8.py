import itertools
import time
import random
from collections import defaultdict

def _precompute_triples(v, t):
    """Precompute all t-subsets and their indices."""
    triples = list(itertools.combinations(range(v), t))
    triple_index = {tri: i for i, tri in enumerate(triples)}
    return triples, triple_index

def _greedy_efficiency(v, k, t, seed, deadline):
    """
    Greedy construction.
    """
    rng = random.Random(seed)
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    if not uncovered:
        return []

    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered triple to seed the block
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        block_set = set(block)
        
        # Track triples covered by this specific block so far
        block_covered = set()
        for tri in itertools.combinations(block, t):
            block_covered.add(tri)
            
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Calculate gain: how many *uncovered* triples are covered by adding x
                gain = 0
                if t == 3:
                    # Pairs in block
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            tri = tuple(sorted((block[i], block[j], x)))
                            if tri in uncovered and tri not in block_covered:
                                gain += 1
                else:
                    # General case
                    for comb in itertools.combinations(block, t-1):
                        tri = tuple(sorted(list(comb) + [x]))
                        if tri in uncovered and tri not in block_covered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            if not candidates:
                break
                
            # Pick randomly from top candidates to add diversity
            next_point = rng.choice(candidates)
            block.append(next_point)
            block_set.add(next_point)
            
            # Update block_covered
            for comb in itertools.combinations(block, t):
                block_covered.add(tuple(comb))
        
        if len(block) < k:
            continue
            
        block.sort()
        blocks.append(block)
        
        # Update global uncovered set
        for tri in itertools.combinations(block, t):
            uncovered.discard(tri)
            
    return blocks

def _local_search_remove(blocks, v, k, t, deadline):
    """
    Local search: try to remove redundant blocks.
    """
    if not blocks:
        return blocks
    
    current_blocks = [list(b) for b in blocks]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Build coverage map: triple -> count of blocks covering it
        # Using a dict of sets for flexibility
        triple_blocks = defaultdict(set)
        for i, block in enumerate(current_blocks):
            for tri in itertools.combinations(block, t):
                triple_blocks[tuple(tri)].add(i)
        
        # Try to remove a block
        removed = False
        for i in range(len(current_blocks)):
            # Check if block i is essential
            essential = False
            for tri in itertools.combinations(current_blocks[i], t):
                if len(triple_blocks[tuple(tri)]) == 1:
                    essential = True
                    break
            
            if not essential:
                # Remove block i
                for tri in itertools.combinations(current_blocks[i], t):
                    triple_blocks[tuple(tri)].discard(i)
                    if not triple_blocks[tuple(tri)]:
                        del triple_blocks[tuple(tri)]
                
                current_blocks.pop(i)
                improved = True
                removed = True
                break # Restart loop with new state
        
        if not removed:
            break
            
    return current_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run multiple restarts
    while time.time() < deadline:
        # Phase 1: Greedy Construction
        blocks = _greedy_efficiency(v, k, t, seed, deadline)
        
        if blocks:
            # Phase 2: Local Search (Remove redundant blocks)
            blocks = _local_search_remove(blocks, v, k, t, deadline)
            
        if blocks and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len == 69:
            break
            
        seed += 1
        
    return best_blocks
