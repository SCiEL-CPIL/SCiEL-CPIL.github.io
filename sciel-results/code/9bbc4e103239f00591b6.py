import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute
    points = range(v)
    triples = list(itertools.combinations(points, t))
    triple_map = {tr: i for i, tr in enumerate(triples)}
    num_triples = len(triples)
    
    blocks = list(itertools.combinations(points, k))
    block_points = [list(b) for b in blocks]
    num_blocks = len(blocks)
    
    block_triples = []
    for b in blocks:
        trs = [triple_map[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(trs)
        
    # Inverted index: triple -> list of blocks
    triple_to_blocks = [[] for _ in range(num_triples)]
    for b_idx, trs in enumerate(block_triples):
        for tr in trs:
            triple_to_blocks[tr].append(b_idx)
            
    # Randomized Greedy
    uncovered = [True] * num_triples
    num_uncovered = num_triples
    blocks_in_use = []
    
    # To pick random uncovered, we can just pick a random index and check, or maintain a list.
    # Maintaining a list of uncovered indices is better.
    uncovered_indices = list(range(num_triples))
    
    while uncovered_indices and time.time() < deadline:
        # Pick random uncovered triple
        idx = random.randint(0, len(uncovered_indices) - 1)
        tr_idx = uncovered_indices[idx]
        
        # Pick random block covering it
        candidates = triple_to_blocks[tr_idx]
        if not candidates:
            break # Error
        b_idx = random.choice(candidates)
        
        # Add block
        blocks_in_use.append(b_idx)
        
        # Update uncovered
        # We need to remove all triples covered by b_idx from uncovered_indices
        # This is O(num_triples) if we scan the list.
        # Better: mark as covered, and clean up later?
        # Or use a set.
        
        # Let's switch to a set for uncovered for O(1) removal.
        # But we started with a list. Let's restart with set.
        pass

    # Restart with Set for Uncovered
    uncovered_set = set(range(num_triples))
    blocks_in_use = []
    
    while uncovered_set and time.time() < deadline:
        # Pick a random uncovered triple
        # list(uncovered_set) is slow.
        # We can pick a random index from 0 to num_triples-1 and check if in set?
        # Or just pick the first one?
        # Let's pick the one with fewest candidate blocks to be "smart".
        # But that requires scanning all uncovered.
        
        # Let's just pick a random one by sampling.
        tr_idx = random.choice(list(uncovered_set)) # Slow if set is large initially
        
        # Optimization: Keep a list of uncovered, remove from it.
        # We'll stick to the list method but optimize removal.
        
        # Actually, for v=24, 2024 triples.
        # list(uncovered_set) is 2024 items. Random choice is fast.
        # 76 blocks * 2024 ops = 150k ops. Very fast.
        
        candidates = triple_to_blocks[tr_idx]
        b_idx = random.choice(candidates)
        
        blocks_in_use.append(b_idx)
        
        for tr in block_triples[b_idx]:
            if tr in uncovered_set:
                uncovered_set.remove(tr)
                
    # Local Search: Remove redundant blocks
    counts = np.zeros(num_triples, dtype=np.int32)
    for b_idx in blocks_in_use:
        for tr in block_triples[b_idx]:
            counts[tr] += 1
            
    # Repeat removal until no more can be removed
    while time.time() < deadline:
        removed = False
        # Shuffle to randomize order
        random.shuffle(blocks_in_use)
        for i in range(len(blocks_in_use) - 1, -1, -1):
            b_idx = blocks_in_use[i]
            # Check if removable
            removable = True
            for tr in block_triples[b_idx]:
                if counts[tr] == 1:
                    removable = False
                    break
            if removable:
                blocks_in_use.pop(i)
                for tr in block_triples[b_idx]:
                    counts[tr] -= 1
                removed = True
                break # Restart scan
        if not removed:
            break # No more removable blocks
            
    # Try to improve with swaps?
    # If we have time, try to swap points in blocks to make them more redundant.
    # This is complex. We'll stop here.
    
    result = [list(block_points[b_idx]) for b_idx in blocks_in_use]
    for b in result:
        b.sort()
        
    return result
