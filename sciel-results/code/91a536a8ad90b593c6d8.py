import itertools
import random
import time
import math

def construct(v, k, t):
    # Pre-computation
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N = len(all_blocks)
    
    all_targets = list(itertools.combinations(points, t))
    M = len(all_targets)
    
    # Map target tuple to index
    target_index = {target: i for i, target in enumerate(all_targets)}
    
    # For each block, store the indices of the t-subsets it covers
    # block_subsets[i] is a list of M-indices
    block_subsets = []
    for block in all_blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(target_index[sub])
        block_subsets.append(subs)
        
    # Greedy Initialization
    uncovered = [1] * M  # 1 if uncovered, 0 if covered
    num_uncovered = M
    selected = [0] * N
    blocks_list = []
    
    # We need to pick blocks greedily.
    # To speed up greedy, we can just iterate through all blocks and pick the one covering most uncovered.
    # But N is 75k. Doing this M times (11k) is 11k * 75k * 56 ops ~ 4.6e9 ops. Too slow in Python.
    # Better Greedy:
    # Maintain a list of uncovered targets.
    # For each uncovered target, find a block that covers it and maximizes coverage of other uncovered targets.
    # This is also potentially slow if done naively.
    
    # Alternative: Use the simple greedy from the seed as initialization.
    # The seed greedy is:
    # uncovered_set = set(all_targets)
    # while uncovered_set:
    #    pick min uncovered target
    #    build block around it greedily
    
    # Let's implement a faster greedy initialization.
    # We can use the fact that we just need *a* valid solution.
    # Randomized Greedy:
    # While there are uncovered targets:
    #   Pick a random uncovered target
    #   Pick a random block that covers it
    #   Add block to solution
    #   Update coverage
    
    # This is O(N * M) worst case but usually fast.
    # Let's try to be smarter.
    # For each block, we know which targets it covers.
    # We can just iterate through blocks randomly and add them if they cover at least one new target.
    # This is a "randomized greedy" or "randomized set cover".
    
    random.seed(42)
    
    # State for SA
    # x: list of 0/1, length N
    # cov: list of ints, length M
    # cost: float
    
    # Initialize with a valid covering using randomized greedy
    cov = [0] * M
    x = [0] * N
    num_uncovered = M
    
    # To make initialization fast, we can pre-compute for each target, the list of blocks that cover it.
    # target_to_blocks[j] = list of block indices i such that j is in block_subsets[i]
    target_to_blocks = [[] for _ in range(M)]
    for i, subs in enumerate(block_subsets):
        for j in subs:
            target_to_blocks[j].append(i)
            
    # Randomized Greedy Init
    while num_uncovered > 0:
        # Pick a random uncovered target
        # We need a list of uncovered targets to pick from.
        # Maintaining this list is expensive.
        # Instead, pick a random target index j. If cov[j] == 0, use it.
        # If not, retry.
        # Since num_uncovered decreases, this becomes efficient.
        while True:
            j = random.randint(0, M - 1)
            if cov[j] == 0:
                break
        
        # Pick a random block from those covering j
        candidates = target_to_blocks[j]
        i = random.choice(candidates)
        
        if x[i] == 0:
            x[i] = 1
            # Update coverage
            for sub_idx in block_subsets[i]:
                if cov[sub_idx] == 0:
                    num_uncovered -= 1
                cov[sub_idx] += 1
                
    # Now we have a valid solution.
    # Calculate initial cost
    P = 1000  # Penalty for uncovered
    current_blocks = sum(x)
    current_uncovered = num_uncovered
    current_cost = current_blocks + P * current_uncovered
    
    # Simulated Annealing
    # We want to minimize cost.
    # Moves:
    # 1. Remove a random selected block
    # 2. Add a random unselected block
    # 3. Swap: remove one selected, add one unselected
    
    # To make moves efficient, we need to quickly find a selected and an unselected block.
    # We can maintain a list of selected indices and unselected indices?
    # Or just scan? Scanning N=75k is fast.
    
    start_time = time.time()
    time_limit = 110.0  # Leave 10s for repair/finalization
    
    T = 500.0  # Initial temperature
    alpha = 0.999  # Cooling rate
    min_T = 0.1
    
    best_cost = current_cost
    best_x = x.copy()
    best_cov = cov.copy()
    
    # Helper to calculate delta cost for a move
    # We will update cov in place and revert if rejected
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a block
            # Find a random selected block
            # Instead of scanning, we can keep a list of selected blocks?
            # For now, scan.
            # Optimization: If we have many selected, scanning is ok.
            # But we can just pick a random index i. If x[i]==1, do it. Else retry.
            # Since x is sparse (maybe 300-400 blocks out of 75k), retrying is bad.
            # Let's maintain a list of selected block indices.
            
            # Actually, let's just scan. 75k checks is < 1ms in Python? Maybe 5ms.
            # 120s / 5ms = 24k iterations. That's small.
            # We need faster moves.
            
            # Let's maintain `selected_indices` list.
            # But updating it on add/remove is O(N) if we use list removal.
            # We can use a set for selected indices.
            
            # Let's re-structure to use sets or just accept the scan cost.
            # Better: Pre-compute a random permutation of block indices? No, x changes.
            
            # Let's use a list of selected blocks for the "remove" move.
            # And a list of unselected for "add" move?
            # Maintaining sorted lists is hard.
            
            # Alternative: Just pick random i.
            # If x[i] == 1: Remove.
            # If x[i] == 0: Add.
            # This naturally balances add/remove based on density.
            # Since density is low, we will mostly try to Add.
            # We want to Remove more often to minimize size.
            # So, bias the move selection.
            
            # Let's try:
            # If x[i] == 1: Remove (prob 0.5) or Swap (prob 0.5)
            # If x[i] == 0: Add (prob 0.5) or Swap (prob 0.5 - need to pick another j)
            
            # This is getting complicated. Let's stick to simple Add/Remove/Swap with explicit index selection.
            
            # To find a selected index fast:
            # We can store the selected blocks in a list `sel_list`.
            # When adding, append. When removing, swap with last and pop.
            # This makes O(1) selection.
            
            # Let's implement this.
            pass # Placeholder for logic restructuring
        
        # Let's restart the loop logic with the `sel_list` optimization in mind.
        
        # For simplicity and robustness within the time limit, I will implement the SA with the `sel_list` optimization.
        
        break # Break out to implement properly below

    # The logic above was just drafting. Here is the actual implementation structure.
    
    # Re-initialize state for the actual run
    cov = [0] * M
    x = [0] * N
    num_uncovered = M
    sel_list = []
    
    # Randomized Greedy Init again (using the logic above)
    while num_uncovered > 0:
        while True:
            j = random.randint(0, M - 1)
            if cov[j] == 0:
                break
        candidates = target_to_blocks[j]
        i = random.choice(candidates)
        if x[i] == 0:
            x[i] = 1
            sel_list.append(i)
            for sub_idx in block_subsets[i]:
                if cov[sub_idx] == 0:
                    num_uncovered -= 1
                cov[sub_idx] += 1
                
    current_blocks = len(sel_list)
    current_uncovered = num_uncovered
    current_cost = current_blocks + P * current_uncovered
    
    best_cost = current_cost
    best_sel_list = sel_list.copy()
    best_cov = cov.copy()
    best_x = x.copy()
    
    start_time = time.time()
    time_limit = 110.0
    
    T = 1000.0
    alpha = 0.9995
    min_T = 1.0
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Choose move
        r = random.random()
        
        if r < 0.5 and sel_list:
            # Remove a block
            idx_in_list = random.randint(0, len(sel_list) - 1)
            i = sel_list[idx_in_list]
            
            # Calculate Delta
            # Remove i
            x[i] = 0
            delta_blocks = -1
            delta_uncovered = 0
            for sub_idx in block_subsets[i]:
                cov[sub_idx] -= 1
                if cov[sub_idx] == 0:
                    delta_uncovered += 1
            
            delta_cost = delta_blocks + P * delta_uncovered
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                current_cost += delta_cost
                current_blocks += delta_blocks
                current_uncovered += delta_uncovered
                # Update sel_list: swap with last, pop
                sel_list[idx_in_list] = sel_list[-1]
                sel_list.pop()
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sel_list = sel_list.copy()
                    best_cov = cov.copy()
                    best_x = x.copy()
            else:
                # Reject: Revert
                x[i] = 1
                for sub_idx in block_subsets[i]:
                    cov[sub_idx] += 1
                # sel_list is not modified on reject? 
                # Wait, I didn't modify sel_list yet.
                # In the accept case, I modified sel_list.
                # In the reject case, I revert x and cov. sel_list is untouched. Correct.
                
        elif r < 0.8:
            # Add a block
            # Pick random i not in sel_list
            # Since sel_list is small, we can just pick random i and check.
            # Or pick from all N.
            i = random.randint(0, N - 1)
            if x[i] == 1:
                continue # Retry
            
            # Calculate Delta
            x[i] = 1
            delta_blocks = 1
            delta_uncovered = 0
            for sub_idx in block_subsets[i]:
                if cov[sub_idx] == 0:
                    delta_uncovered -= 1
                cov[sub_idx] += 1
            
            delta_cost = delta_blocks + P * delta_uncovered
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                current_cost += delta_cost
                current_blocks += delta_blocks
                current_uncovered += delta_uncovered
                sel_list.append(i)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sel_list = sel_list.copy()
                    best_cov = cov.copy()
                    best_x = x.copy()
            else:
                # Reject
                x[i] = 0
                for sub_idx in block_subsets[i]:
                    cov[sub_idx] -= 1
                    
        else:
            # Swap
            # Pick i from sel_list, j not in sel_list
            if not sel_list:
                continue
            idx_in_list = random.randint(0, len(sel_list) - 1)
            i = sel_list[idx_in_list]
            
            j = random.randint(0, N - 1)
            if x[j] == 1:
                continue
            
            # Calculate Delta for removing i and adding j
            # Remove i
            x[i] = 0
            delta_blocks = 0 # Net 0
            delta_uncovered = 0
            for sub_idx in block_subsets[i]:
                cov[sub_idx] -= 1
                if cov[sub_idx] == 0:
                    delta_uncovered += 1
            
            # Add j
            x[j] = 1
            for sub_idx in block_subsets[j]:
                if cov[sub_idx] == 0:
                    delta_uncovered -= 1
                cov[sub_idx] += 1
            
            delta_cost = delta_blocks + P * delta_uncovered
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                current_cost += delta_cost
                current_uncovered += delta_uncovered
                # Update sel_list
                sel_list[idx_in_list] = j
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sel_list = sel_list.copy()
                    best_cov = cov.copy()
                    best_x = x.copy()
            else:
                # Reject: Revert
                x[i] = 1
                x[j] = 0
                for sub_idx in block_subsets[i]:
                    cov[sub_idx] += 1
                for sub_idx in block_subsets[j]:
                    cov[sub_idx] -= 1
        
        T *= alpha
        
    # Use the best solution found
    x = best_x
    cov = best_cov
    
    # Check if valid
    num_uncovered_final = sum(1 for c in cov if c == 0)
    
    if num_uncovered_final > 0:
        # Repair
        # Use greedy to cover remaining
        uncovered_targets = [j for j in range(M) if cov[j] == 0]
        
        while uncovered_targets:
            # Pick random uncovered target
            j = random.choice(uncovered_targets)
            # Pick random block covering it
            candidates = target_to_blocks[j]
            i = random.choice(candidates)
            
            if x[i] == 0:
                x[i] = 1
                # Update cov
                newly_covered = []
                for sub_idx in block_subsets[i]:
                    if cov[sub_idx] == 0:
                        newly_covered.append(sub_idx)
                    cov[sub_idx] += 1
                # Remove newly covered from uncovered_targets
                # This is O(M) if we do it naively.
                # We can just rebuild the list.
                uncovered_targets = [k for k in uncovered_targets if cov[k] == 0]
                
    # Construct output
    blocks = []
    for i in range(N):
        if x[i] == 1:
            blocks.append(list(all_blocks[i]))
            
    # Ensure blocks are distinct and sorted
    # all_blocks are distinct combinations.
    # Sort each block
    for b in blocks:
        b.sort()
        
    # Remove duplicates just in case (shouldn't happen)
    unique_blocks = []
    seen = set()
    for b in blocks:
        t_b = tuple(b)
        if t_b not in seen:
            seen.add(t_b)
            unique_blocks.append(b)
            
    return unique_blocks
