"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math

    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    reciprocals = [1.0 / d for d in denominators]

    # Precompute suffix sums of reciprocals for pruning
    # rem_sum[i] = sum of reciprocals[i] + ... + reciprocals[n-1]
    rem_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        rem_sum[i] = rem_sum[i + 1] + reciprocals[i]

    best_deficit = float('inf')
    best_indices = []
    selection = [False] * n

    def update_best(current_sum):
        nonlocal best_deficit, best_indices
        if current_sum < 1.0:
            deficit = 1.0 - current_sum
            if deficit < best_deficit:
                best_deficit = deficit
                best_indices = [i for i in range(n) if selection[i]]

    # 1. Greedy initial solutions to set a tight upper bound on deficit
    # Strategy A: Greedy from largest denominators (smallest fractions)
    # This allows fine-tuning with small steps.
    current_sum = 0.0
    for i in range(n - 1, -1, -1):
        if current_sum + reciprocals[i] < 1.0:
            current_sum += reciprocals[i]
            selection[i] = True
    if current_sum > 0:
        update_best(current_sum)
    # Reset selection
    selection = [False] * n

    # Strategy B: Greedy from smallest denominators (largest fractions)
    current_sum = 0.0
    for i in range(n):
        if current_sum + reciprocals[i] < 1.0:
            current_sum += reciprocals[i]
            selection[i] = True
    if current_sum > 0:
        update_best(current_sum)
    # Reset selection
    selection = [False] * n

    # 2. Branch and Bound Depth-First Search
    # We iterate indices 0 to n-1 (denominators 2 to 48)
    # We decide to include or exclude each reciprocal.
    # Pruning:
    # If current_sum + rem_sum[idx] < 1 - best_deficit, 
    # then even if we include ALL remaining items, the sum will be 
    # less than 1 - best_deficit, so the deficit will be > best_deficit.
    # Thus, we can prune this branch.
    
    def dfs(idx, current_sum):
        nonlocal best_deficit, best_indices
        
        if time.time() > deadline:
            return

        # Pruning condition
        # Max possible sum from this state is current_sum + rem_sum[idx]
        # If 1 - (current_sum + rem_sum[idx]) > best_deficit, 
        # then min possible deficit in this branch is > best_deficit.
        if 1.0 - (current_sum + rem_sum[idx]) > best_deficit:
            return

        if idx == n:
            update_best(current_sum)
            return

        # Option 1: Include item at idx
        # Only if it doesn't exceed 1
        next_sum = current_sum + reciprocals[idx]
        if next_sum < 1.0:
            selection[idx] = True
            dfs(idx + 1, next_sum)
            selection[idx] = False

        # Option 2: Exclude item at idx
        dfs(idx + 1, current_sum)

    dfs(0, 0.0)

    # Fallback if no solution found (should not happen)
    if not best_indices:
        best_indices = [0]

    return best_indices
# EVOLVE-BLOCK-END
