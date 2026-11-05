import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    # Leave a safety margin to avoid timeout
    deadline = start_time + 145.0
    
    # Precompute all t-subsets
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper to run one instance of Greedy + Reduction
    def run_instance():
        rng = random.Random()
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block
            # This adds diversity compared to always picking the first
            seed_subset = tuple(rng.choice(list(uncovered)))
            block = list(seed_subset)
            
            while len(block) < k:
                # Find best point to add
                best_gain = -1
                best_points = []
                for x in range(v):
                    if x in block:
                        continue
                    # Calculate gain: how many new t-subsets are covered
                    gain = 0
                    # New t-subsets must include x and t-1 elements from block
                    for combo in itertools.combinations(block, t-1):
                        s = tuple(sorted(combo + (x,)))
                        if s in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if best_gain == 0:
                    # No gain, just pick a random point to fill
                    available = [x for x in range(v) if x not in block]
                    if available:
                        block.append(rng.choice(available))
                    else:
                        break
                else:
                    block.append(rng.choice(best_points))
            
            block.sort()
            # Ensure distinctness (though greedy usually produces distinct blocks)
            # If duplicate, skip? No, just add. The reduction will handle redundancy.
            blocks.append(block)
            
            # Update uncovered
            for s in itertools.combinations(block, t):
                uncovered.discard(tuple(s))
        
        # Reduction: Remove redundant blocks
        # Count coverage
        cover_count = {s: 0 for s in all_t_subsets}
        for b in blocks:
            for s in itertools.combinations(b, t):
                cover_count[tuple(s)] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                b = blocks[i]
                is_redundant = True
                for s in itertools.combinations(b, t):
                    if cover_count[tuple(s)] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove block i
                    for s in itertools.combinations(b, t):
                        cover_count[tuple(s)] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart loop to check other blocks against new counts
        
        return blocks

    # Run multiple instances
    while time.time() < deadline:
        blocks = run_instance()
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
    if best_blocks is None:
        # Fallback: return a valid covering even if it's not optimal
        # Generate one greedy solution
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            seed_subset = tuple(next(iter(uncovered)))
            block = list(seed_subset)
            while len(block) < k:
                best_gain = -1
                best_x = None
                for x in range(v):
                    if x in block: continue
                    gain = 0
                    for combo in itertools.combinations(block, t-1):
                        if tuple(sorted(combo + (x,))) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for s in itertools.combinations(block, t):
                uncovered.discard(tuple(s))
        return blocks
        
    return best_blocks
