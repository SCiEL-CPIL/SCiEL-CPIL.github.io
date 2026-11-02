import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit

    if v != 19 or k != 8 or t != 5:
        # Fallback for other parameters: simple greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            target = min(uncovered)
            best_block = None
            best_gain = -1
            remaining = [x for x in range(v) if x not in target]
            need = k - t
            for comb in itertools.combinations(remaining, need):
                block = tuple(sorted(target + comb))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            if best_block is None:
                break
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
            if time.time() > deadline:
                break
        return blocks

    # Specific optimization for v=19, k=8, t=5
    # Strategy: Density-driven block generation
    # 1. Keep track of uncovered 5-subsets.
    # 2. Pick a random uncovered 5-subset.
    # 3. Generate all 8-blocks containing it (364 candidates).
    # 4. Pick the one covering the most uncovered 5-subsets.
    # 5. Add to solution, update uncovered set.
    # 6. Post-process: remove redundant blocks.
    
    points = list(range(v))
    all_t_subsets = set(itertools.combinations(points, t))
    uncovered = set(all_t_subsets)
    
    def run_greedy(seed):
        random.seed(seed)
        blocks = []
        current_uncovered = set(uncovered)
        
        while current_uncovered:
            if time.time() > deadline - 5.0:
                return blocks
            
            # Pick a random uncovered subset to focus on
            # Using random choice avoids bias towards lexicographically first
            target = random.choice(list(current_uncovered))
            
            # Generate candidates: 8-blocks containing target
            # Target has 5 elements. We need 3 more from the remaining 14.
            remaining_points = [x for x in points if x not in target]
            best_block = None
            best_gain = 0
            
            # Iterate through all combinations of 3 points from remaining 14
            # C(14, 3) = 364. This is fast.
            for extra in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target + extra))
                # Calculate gain: how many uncovered 5-subsets does this block cover?
                gain = 0
                for sub in itertools.combinations(block, t):
                    if sub in current_uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                break
                
            blocks.append(list(best_block))
            
            # Update uncovered set
            for sub in itertools.combinations(best_block, t):
                if sub in current_uncovered:
                    current_uncovered.remove(sub)
        
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Map block to its t-subsets for quick checking
        # We need to check if a block is redundant: i.e., all its t-subsets are covered by other blocks.
        # We can do this by counting coverage.
        
        # To speed this up, we'll use a set of all t-subsets covered by the current set of blocks.
        # But we need to know if removing a block leaves any subset uncovered.
        # So we need coverage counts.
        
        # Efficient approach:
        # 1. Count coverage of each t-subset.
        # 2. If a block has any t-subset with count 1, it's not redundant.
        # 3. If a block has all t-subsets with count > 1, it's redundant. Remove it and decrement counts.
        
        cover_count = {}
        block_ts_list = []
        
        for b in blocks:
            ts_list = list(itertools.combinations(b, t))
            block_ts_list.append(ts_list)
            for ts in ts_list:
                cover_count[ts] = cover_count.get(ts, 0) + 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    return blocks
                
                # Check if block i is redundant
                redundant = True
                for ts in block_ts_list[i]:
                    if cover_count[ts] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for ts in block_ts_list[i]:
                        cover_count[ts] -= 1
                    blocks.pop(i)
                    block_ts_list.pop(i)
                    changed = True
                    break
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    # Each restart is fast (no precomputation), so we can run many.
    while time.time() < deadline - 10.0:
        sol = run_greedy(int(time.time() * 1000) % 100000)
        if not sol:
            continue
            
        sol = remove_redundant(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_blocks = [list(b) for b in sol]
            
        # Early stop if we get a very good solution (heuristic lower bound is 264, best known 335)
        # If we find something close to 335, we can stop.
        if best_len <= 335:
            break

    if best_blocks is None:
        # Fallback
        best_blocks = run_greedy(0)
        if not best_blocks:
            return []
        best_blocks = remove_redundant(best_blocks)

    return best_blocks
