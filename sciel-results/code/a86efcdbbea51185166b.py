import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        blocks = _greedy_construct(v, k, t, all_t_subsets, random_seed=attempts)
        
        # Local Search: Remove redundant blocks
        blocks = _remove_redundant(blocks, v, k, t)
        
        # Local Search: Try to swap points to enable more removals?
        # Maybe just try to reduce block size? No, k is fixed.
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # If we hit a very good number, we might stop early?
        # Best known is 335. If we get close, keep trying.
        
        # Check time
        if time.time() > deadline:
            break
            
    return best_blocks if best_blocks else []

def _greedy_construct(v, k, t, all_t_subsets, random_seed=0):
    random.seed(random_seed)
    uncovered = set(all_t_subsets)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered subset to start
        seed_subset = random.choice(list(uncovered))
        block = list(seed_subset)
        
        while len(block) < k:
            # Find point that maximizes coverage of uncovered subsets
            best_point = -1
            best_gain = -1
            
            # Precompute combinations of block of size t-1 to speed up gain calc?
            # Gain of adding x: count of S in uncovered such that S \subseteq block U {x}
            # S has size t. block has size b.
            # S is subset of block U {x} iff S \ {x} is subset of block (if x in S) or S is subset of block (if x not in S, but S is uncovered so not subset of block).
            # So S must contain x.
            # S = {x} U R, where R is a subset of block of size t-1.
            
            # We can iterate over all R in combinations(block, t-1)
            # and check if (x,) + R is in uncovered.
            
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
                
            # Optimize gain calculation
            # For each x, gain = sum(1 for R in combos(block, t-1) if tuple(sorted((x,)+R)) in uncovered)
            
            # To speed up, we can precompute the set of "potential" subsets?
            # Or just compute.
            # t=5, block size grows 5->7.
            # When block size is 5, t-1=4. C(5,4)=5.
            # When block size is 6, t-1=4. C(6,4)=15.
            # When block size is 7, t-1=4. C(7,4)=35.
            # 19 candidates * 35 checks = 665 checks. Fast.
            
            for x in candidates:
                gain = 0
                # R must be subset of block, size t-1
                for R in itertools.combinations(block, t-1):
                    # Create tuple of S = {x} U R
                    # Sort to match key format
                    S = tuple(sorted((x,) + R))
                    if S in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    
            if best_point == -1:
                # No point increases coverage? Should not happen if uncovered is not empty
                # But if we are forced to add a point that doesn't help, pick random
                best_point = random.choice(candidates)
                
            block.append(best_point)
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(tuple(sub))
            
    return blocks

def _remove_redundant(blocks, v, k, t):
    if not blocks:
        return blocks
        
    # Count occurrences of each t-subset
    counter = {}
    for block in blocks:
        for sub in itertools.combinations(block, t):
            key = tuple(sub)
            counter[key] = counter.get(key, 0) + 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for block in blocks:
            redundant = True
            for sub in itertools.combinations(block, t):
                key = tuple(sub)
                if counter.get(key, 0) < 2:
                    redundant = False
                    break
            if redundant:
                # Remove block
                for sub in itertools.combinations(block, t):
                    key = tuple(sub)
                    counter[key] -= 1
                changed = True
            else:
                new_blocks.append(block)
        blocks = new_blocks
        
    return blocks
