import itertools
import random
import time
import math

def construct(v, k, t):
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N = len(all_blocks)
    
    all_targets = list(itertools.combinations(points, t))
    M = len(all_targets)
    
    target_index = {target: i for i, target in enumerate(all_targets)}
    
    # Precompute subsets for each block
    block_subsets = []
    for block in all_blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(target_index[sub])
        block_subsets.append(subs)
        
    # Precompute target_to_blocks for initialization
    target_to_blocks = [[] for _ in range(M)]
    for i, subs in enumerate(block_subsets):
        for j in subs:
            target_to_blocks[j].append(i)
            
    random.seed(42)
    
    # State variables
    cov = [0] * M
    x = [0] * N  # 1 if selected
    num_blocks = 0
    uncovered_count = M
    
    # Helper: Add block i
    def add_block(i):
        nonlocal num_blocks, uncovered_count
        if x[i]:
            return 0, 0 # No change
        x[i] = 1
        num_blocks += 1
        newly_covered = 0
        for s in block_subsets[i]:
            if cov[s] == 0:
                newly_covered += 1
                uncovered_count -= 1
            cov[s] += 1
        return 1, newly_covered
        
    # Helper: Remove block i
    def remove_block(i):
        nonlocal num_blocks, uncovered_count
        if not x[i]:
            return 0, 0 # No change
        x[i] = 0
        num_blocks -= 1
        newly_uncovered = 0
        for s in block_subsets[i]:
            cov[s] -= 1
            if cov[s] == 0:
                newly_uncovered += 1
                uncovered_count += 1
        return 1, newly_uncovered

    # 1. Initialization: Greedy to get a valid starting point
    # This ensures we start with uncovered_count = 0
    uncovered = list(range(M))
    while uncovered:
        # Pick random uncovered target
        t_idx = random.choice(uncovered)
        candidates = target_to_blocks[t_idx]
        
        best_block = -1
        max_new = -1
        
        for b in candidates:
            if x[b]:
                continue
            # Count how many subsets in b are currently uncovered
            new_cov_count = 0
            for s in block_subsets[b]:
                if cov[s] == 0:
                    new_cov_count += 1
            if new_cov_count > max_new:
                max_new = new_cov_count
                best_block = b
                
        if best_block != -1:
            add_block(best_block)
            # Update uncovered list efficiently? 
            # For init, M is small enough to rebuild or filter.
            # Filter uncovered list
            uncovered = [i for i in uncovered if cov[i] == 0]
            
    # Remove redundant blocks to tighten the initial solution
    removed = True
    while removed:
        removed = False
        for i in range(N):
            if x[i]:
                redundant = True
                for s in block_subsets[i]:
                    if cov[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(i)
                    removed = True

    current_best = num_blocks
    best_x = x.copy()
    best_cov = cov.copy()
    
    start_time = time.time()
    time_limit = 110.0
    
    # 2. Simulated Annealing
    # Objective: Minimize num_blocks + lambda * uncovered_count
    # We want to find a solution with uncovered_count == 0 and minimal num_blocks.
    # If uncovered_count > 0, it's invalid. We penalize heavily.
    lambda_penalty = 100.0 # Large enough to prioritize validity, but allow exploration
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.99995 # Adjusted for time limit
    
    # We will track the best *valid* solution found.
    # We also track the best *overall* state if it gets close to valid?
    # No, the contract requires valid output. So we only update best if uncovered_count == 0.
    
    while time.time() - start_time < time_limit:
        # Choose a random block to toggle
        i = random.randint(0, N - 1)
        
        if x[i] == 0:
            # Try adding
            # Calculate delta E without modifying state first? 
            # Or modify, calculate, and revert if rejected.
            # Modify is faster to calculate exact delta if we just look at cov.
            
            # Count newly covered
            newly_covered = 0
            for s in block_subsets[i]:
                if cov[s] == 0:
                    newly_covered += 1
            
            # Delta E = (1 + lambda * (uncovered_count - newly_covered)) - (0 + lambda * uncovered_count)
            # Delta E = 1 - lambda * newly_covered
            delta_E = 1.0 - lambda_penalty * newly_covered
            
            if delta_E < 0 or (newly_covered > 0 and random.random() < math.exp(-delta_E / T)):
                # Accept
                add_block(i)
                if uncovered_count == 0:
                    if num_blocks < current_best:
                        current_best = num_blocks
                        best_x = x.copy()
                        best_cov = cov.copy()
            # If rejected, do nothing (state unchanged)
            
        else:
            # Try removing
            # Count newly uncovered (subsets with cov[s] == 1)
            newly_uncovered = 0
            for s in block_subsets[i]:
                if cov[s] == 1:
                    newly_uncovered += 1
            
            # Delta E = (-1 + lambda * (uncovered_count + newly_uncovered)) - (0 + lambda * uncovered_count)
            # Delta E = -1 + lambda * newly_uncovered
            delta_E = -1.0 + lambda_penalty * newly_uncovered
            
            if delta_E < 0 or (newly_uncovered == 0 and random.random() < math.exp(-delta_E / T)):
                # Note: If newly_uncovered > 0, delta_E is large positive, unlikely to accept unless T is high.
                # If newly_uncovered == 0, delta_E = -1, always accept.
                remove_block(i)
                if uncovered_count == 0:
                    if num_blocks < current_best:
                        current_best = num_blocks
                        best_x = x.copy()
                        best_cov = cov.copy()
            
        T = max(T * cooling_rate, T_min)
        
        # Optional: If we are stuck in an invalid state for too long, 
        # we might want to nudge towards validity. 
        # The penalty lambda handles this.
        
    # Final check: Ensure best_x corresponds to a valid solution
    # It does by construction (only updated when uncovered_count == 0)
    
    selected = [i for i in range(N) if best_x[i] == 1]
    
    blocks = []
    for i in selected:
        blocks.append(list(all_blocks[i]))
        
    # Ensure distinct and sorted
    unique_blocks = []
    seen = set()
    for b in blocks:
        b.sort()
        t_b = tuple(b)
        if t_b not in seen:
            seen.add(t_b)
            unique_blocks.append(b)
            
    return unique_blocks
