"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 145.0  # 5s safety margin before 150s

# Strategy:
# The current greedy approach (picking the best item based on a static score) often gets stuck
# in local optima where a combination of lower-density items yields higher total value.
# This program implements a heuristic local search that:
# 1. Builds an initial solution using a greedy density heuristic.
# 2. Performs a time-bounded local search to improve the solution.
#    - Tries single swaps: replace one selected item with one unselected item.
#    - Tries 2-for-1 swaps: replace two selected items with one unselected item.
# 3. Returns the best item to add next based on the improved solution's structure.
#    However, the interface requires picking the *next* item to add to the *current* state.
#    The evaluator builds the solution incrementally.
#    
#    Wait, the interface is: select_next_item(remaining_capacity, remaining_weights, remaining_values) -> index
#    This implies the evaluator maintains the state of "selected" items implicitly or explicitly?
#    No, the evaluator calls this function repeatedly.
#    "picks the next item to add... while building 0/1 knapsack solutions"
#    
#    If the evaluator just appends the returned item to a list, I don't have access to the
#    previously selected items inside this function unless I maintain state.
#    BUT, the function signature only takes remaining_capacity, remaining_weights, remaining_values.
#    It does NOT take the list of already selected items.
#    
#    This is a crucial constraint. I cannot perform local search on the *whole* solution
#    because I don't know what has already been picked.
#    
#    However, I can maintain a global state if the instances are distinct and I can track
#    which instance I'm on? No, the instances are generated, and I don't know the ID.
#    Also, the state (remaining_capacity, weights, values) changes.
#    
#    Actually, looking at the "inspiration 2" which used lookahead and scored 0.9934,
#    and the current program which uses a static score and scored 0.9968,
#    the improvement to 1.001 came from "2-for-1 swaps".
#    
#    How can I do swaps if I don't know the selected items?
#    I can't.
#    
#    Let's re-read the task carefully.
#    "select_next_item(remaining_capacity, remaining_weights, remaining_values)"
#    "remaining (feasible) arrays"
#    
#    If the evaluator filters out infeasible items, then remaining_weights/values are the candidates.
#    I have no memory of past choices.
#    
#    Wait, is it possible that the "remaining_capacity" and "remaining_weights/values"
#    are the *only* state I have? Yes.
#    
#    So, I cannot do global local search.
#    
#    However, I can do a **local lookahead** that simulates the next few steps.
#    Inspiration 2 did this and got 0.9934 (worse than current 0.9968).
#    The 1.001 score came from a program that *did* something with swaps.
#    How?
#    
#    Maybe the "remaining" arrays are not just the unselected items, but the *entire* set
#    of items, and I have to filter by feasibility myself?
#    "index into the remaining (feasible) arrays"
#    If the arrays passed in are the *original* arrays, then I can filter.
#    But if they are the *remaining* arrays, they are already filtered.
#    
#    Let's assume the standard interpretation: The evaluator maintains a list of indices
#    already picked. It passes the *original* weights and values (or the remaining ones?)
#    and the remaining capacity.
#    
#    If it passes the *original* arrays, I can simulate the future.
#    If it passes the *remaining* arrays, I can still simulate the future based on those.
#    
#    Let's try a different strategy: **Simulated Annealing-like greedy with randomization?**
#    No, deterministic is better for scoring stability.
#    
#    Let's look at the "2-for-1 swaps" hint again.
#    "Adding 2-for-1 swaps allows the local search to escape local optima... Measured: 1 -> 1.001."
#    This implies that a program *successfully* implemented local search.
#    How could it have done so without state?
#    
#    Possibility A: The function is called once per instance? No, "picks the next item... while building".
#    Possibility B: The "remaining" arrays actually contain ALL items, and I am expected to
#    know which ones are already picked? No, that's impossible without state.
#    Possibility C: The evaluator passes the *history* or the *current selection*?
#    The signature is fixed: `select_next_item(remaining_capacity, remaining_weights, remaining_values)`.
#    
#    Maybe the "remaining_weights" and "remaining_values" are the items *not yet picked*,
#    but I can infer the picked items from the `remaining_capacity`?
#    No, that's underdetermined.
#    
#    Let's reconsider the "2-for-1" hint.
#    Perhaps the previous program that achieved 1.001 *did* have access to state?
#    Or maybe it used a heuristic that *mimics* the effect of swaps?
#    
#    Alternative Strategy: **Dynamic Programming approximation?**
#    With 100 items and capacity 12.5, DP is feasible if capacity is small.
#    Capacity is 12.5. Weights are likely small (e.g., 0.1 to 1.0).
#    If we discretize capacity to, say, 0.01, the DP table size is 1250 x 100 = 125,000.
#    This is very fast.
#    
#    **New Strategy: Exact DP (or near-exact) for the remaining subproblem.**
#    Since the capacity is small (12.5), we can solve the remaining knapsack problem
#    exactly (or very close to it) using Dynamic Programming.
#    This is a fundamentally different approach from greedy heuristics.
#    It guarantees the optimal choice for the *remaining* capacity, assuming the
#    items passed in are the available ones.
#    
#    If the items passed in are the *remaining* (unselected) items, then solving the
#    knapsack problem for these items with the given capacity gives the optimal set.
#    The "next item" to pick should be the one that is part of the optimal solution
#    and has the highest "priority" or is the first in some ordering?
#    Actually, if I solve the DP, I get the *optimal total value* for the remaining capacity.
#    But I need to return the *index* of the next item to add.
#    
#    If I add item `i`, the evaluator will call me again with `remaining_capacity - w[i]`
#    and the remaining items (excluding `i`).
#    If I always pick the item that is part of the optimal solution for the current state,
#    I will build the optimal solution step-by-step.
#    
#    How to identify which item to pick first from the optimal set?
#    I can perform a "reconstruction" step.
#    1. Compute DP table for current state.
#    2. Check which items are included in the optimal solution.
#    3. Among the included items, pick one. Which one?
#       - The one with the highest value?
#       - The one with the highest density?
#       - Any one of them will lead to an optimal final solution, provided the subsequent
#         choices are also optimal.
#       - If I always pick the item that is part of the optimal solution, and then solve
#         the subproblem optimally again, I will converge to the global optimum.
#    
#    This strategy is **Exact DP with Greedy Reconstruction**.
#    
#    Complexity:
#    - 100 items.
#    - Capacity 12.5.
#    - Discretization: Let's use integer weights scaled by 100 (or 1000).
#    - Max capacity = 1250 (if scale 100).
#    - DP table size: 100 * 1250 = 125,000 entries.
#    - Each entry is O(1) update.
#    - Total ops: ~125,000 per call.
#    - Number of calls: Up to 100 (one per item added).
#    - Total ops per instance: 100 * 125,000 = 12,500,000.
#    - In Python, 12.5M simple ops might take ~1-5 seconds.
#    - 8 instances: ~10-40 seconds.
#    - Limit is 150s. This is feasible.
#    
#    We must be careful with floating point.
#    Weights and values are floats.
#    We can round weights to the nearest integer (scale by 100 or 1000).
#    Values can stay as floats in the DP table, or be rounded if we want integer DP.
#    Since we just need the *index* of the item, we can store the value as float.
#    
#    Let's implement DP.
#    
#    Steps:
#    1. Convert weights to integers (scale by 1000 to be safe, or check min weight).
#    2. Create DP table `dp[j]` = max value achievable with capacity `j`.
#    3. Also track `take[i][j]` = True if item `i` is taken in optimal solution for capacity `j`.
#       Or, simpler: just compute the optimal value, then trace back to see which items are taken.
#       But tracing back is O(N*C).
#       Actually, we just need to find *one* item that is part of the optimal solution.
#       
#    Alternative:
#    For each item `i` in the remaining set:
#       Check if there exists an optimal solution that includes item `i`.
#       This is equivalent to checking if:
#       `value[i] + dp_optimal(remaining_capacity - weight[i], remaining_items_excluding_i)`
#       equals `dp_optimal(remaining_capacity, remaining_items)`.
#       
#    This requires solving the subproblem for each item, which is O(N^2 * C).
#    100^2 * 1250 = 12,500,000 * 100 = 1.25B ops. Too slow in Python.
#    
#    Better approach:
#    1. Compute the optimal value `V_opt` for the current state.
#    2. Iterate through items. For each item `i` that fits:
#       Compute `V_sub` = optimal value for remaining capacity `C - w[i]` using items `j != i`.
#       This is still expensive if done naively.
#       
#    Let's go back to the standard DP reconstruction.
#    1. Compute `dp[i][j]` = max value using first `i` items with capacity `j`.
#    2. `dp[i][j] = max(dp[i-1][j], value[i-1] + dp[i-1][j-w[i-1]])`
#    3. After filling the table, the optimal value is `dp[N][C]`.
#    4. To find which items are in the optimal solution, we can backtrack from `dp[N][C]`.
#       However, we don't need *all* of them, just *one* to pick next.
#       If we pick an item that is in the optimal set, the remaining problem will have
#       an optimal value that is consistent with the global optimum.
#       
#    How to pick which one?
#    Any item in the optimal set is fine.
#    We can find *one* item by backtracking:
#    Start at `i=N, j=C`.
#    While `i > 0`:
#       If `dp[i][j] != dp[i-1][j]` and `dp[i][j] == value[i-1] + dp[i-1][j-w[i-1]]`:
#          Then item `i-1` is in the optimal solution.
#          Return `i-1`.
#       Else:
#          `i = i-1`
#    
#    This finds the *last* item in the optimal solution (in the order of the DP table).
#    Is this a good choice?
#    It might be a heavy item or a light item.
#    Since the order of items in the array is arbitrary, this might be biased.
#    However, it guarantees optimality.
#    
#    Let's refine:
#    We want to return the index in the *original* `remaining_weights` array.
#    The DP table uses 0-indexed items from 0 to N-1.
#    
#    Implementation details:
#    - Scale weights: `w_int = round(w * 1000)`.
#    - Capacity: `C_int = round(C * 1000)`.
#    - If `w_int > C_int`, skip.
#    - DP table size: `(N+1) x (C_int+1)`.
#    - N=100, C_int=12500 (if scale 1000).
#    - Table size: 101 * 12501 ~ 1.26M floats.
#    - Memory: 1.26M * 8 bytes ~ 10MB. Fine.
#    - Time: 1.26M operations per call.
#    - 100 calls per instance: 126M operations.
#    - Python loop: 126M simple ops might take ~10-20 seconds per instance.
#    - 8 instances: 80-160 seconds.
#    - This is close to the limit. We need to optimize.
#    
#    Optimization:
#    - Use NumPy for DP?
#      DP recurrence is sequential in `i`.
#      `dp[i]` depends on `dp[i-1]`.
#      We can update a 1D array `dp[j]` in place?
#      No, 0/1 knapsack requires iterating `j` from `C` down to `w` to avoid using the same item twice.
#      `dp[j] = max(dp[j], dp[j-w] + value)`
#      This is O(N*C) and uses O(C) memory.
#      But we lose the ability to backtrack easily because we overwrite `dp`.
#      
#      To backtrack, we need to know *which* items were taken.
#      We can store a `take` matrix `take[i][j]` (boolean).
#      Or, we can store the full `dp` table `(N+1) x (C_int+1)`.
#      
#      Let's try to speed up the DP loop.
#      Python loops are slow.
#      Can we vectorize?
#      The recurrence `dp[i][j] = max(dp[i-1][j], dp[i-1][j-w] + v)`
#      can be vectorized for a fixed `i` if we have `dp[i-1]`.
#      `dp_i = dp_prev.copy()`
#      `mask = np.arange(C_int+1) >= w`
#      `candidates = dp_prev[mask - w] + v`
#      `dp_i[mask] = np.maximum(dp_i[mask], candidates)`
#      
#      This is much faster!
#      N=100 iterations.
#      Each iteration involves NumPy ops on arrays of size 12500.
#      100 * 12500 = 1.25M elements processed.
#      NumPy handles this in milliseconds.
#      
#      So, we can compute the full DP table `dp_table` of shape `(N+1, C_int+1)`.
#      Then backtrack to find one item.
#      
#      Backtracking:
#      `i = N`
#      `j = C_int`
#      for `i` in `range(N, 0, -1)`:
#          if `dp_table[i][j] != dp_table[i-1][j]`:
#              # Item i-1 was taken
#              return `i-1`
#          # Else, item i-1 was not taken, j remains same
#      If no item found? (Should not happen if V_opt > 0)
#      
#      Wait, what if `dp_table[i][j] == dp_table[i-1][j]` but item i-1 *could* have been taken?
#      The condition `dp_table[i][j] != dp_table[i-1][j]` ensures that taking item i-1 *improved* the value.
#      If it didn't improve, it's not strictly necessary for the optimal value.
#      However, if there are multiple optimal solutions, we might miss an item that is part of *one* optimal solution
#      but not the one that strictly improved the value at each step.
#      
#      Actually, if `dp_table[i][j] == dp_table[i-1][j]`, it means the optimal value using first `i` items
#      is the same as using first `i-1` items. So item `i-1` is not needed.
#      We can safely skip it.
#      
#      This will find *some* item in *some* optimal solution.
#      
#      Let's implement this.
#      
#      Edge cases:
#      - If no item fits, return -1.
#      - If time runs out, fallback to greedy.

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Global deadline check
    if time.time() > _GLOBAL_DEADLINE:
        # Fallback: pick best density item that fits
        if len(remaining_weights) == 0:
            return -1
        weights = np.asarray(remaining_weights, dtype=float)
        values = np.asarray(remaining_values, dtype=float)
        feasible_mask = weights <= remaining_capacity
        if not np.any(feasible_mask):
            return -1
        feasible_indices = np.where(feasible_mask)[0]
        f_weights = weights[feasible_mask]
        f_values = values[feasible_mask]
        densities = f_values / np.maximum(f_weights, 1e-9)
        best_idx_in_f = int(np.argmax(densities))
        return int(feasible_indices[best_idx_in_f])
    
    if len(remaining_weights) == 0:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    
    # Filter feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
        
    # Discretize capacity
    scale = 1000
    C_int = int(round(remaining_capacity * scale))
    if C_int <= 0:
        return -1
        
    f_weights_int = np.round(f_weights * scale).astype(int)
    
    # DP Table
    # dp[i][j] = max value using first i items with capacity j
    dp_table = np.zeros((f_n + 1, C_int + 1), dtype=float)
    
    for i in range(1, f_n + 1):
        w = f_weights_int[i-1]
        v = f_values[i-1]
        prev_row = dp_table[i-1]
        curr_row = dp_table[i]
        
        if w <= C_int:
            candidates = prev_row[:C_int - w + 1] + v
            curr_row[:] = prev_row
            curr_row[w:] = np.maximum(curr_row[w:], candidates)
        else:
            curr_row[:] = prev_row

    # Find all items that are part of SOME optimal solution
    # An item i (0-indexed in f_ arrays, so i+1 in dp_table rows) is in some optimal solution
    # if: value[i] + dp_optimal(first i items, C_int - w[i]) == dp_optimal(first i+1 items, C_int)
    # OR equivalently: dp_table[i+1][C_int] == dp_table[i][C_int] is NOT the right test
    # Correct test: item i is in some optimal solution iff
    #   dp_table[i+1][C_int] == f_values[i] + dp_table[i][C_int - f_weights_int[i]]
    # (assuming C_int >= f_weights_int[i])
    
    opt_val = dp_table[f_n][C_int]
    
    # Collect all items that are in some optimal solution
    optimal_candidates = []
    for i in range(f_n):
        w_i = f_weights_int[i]
        if w_i <= C_int:
            # Check if item i can be part of an optimal solution
            # Using items 0..i-1 (first i items) with capacity C_int - w_i
            # dp_table[i][C_int - w_i] is the optimal value using first i items with capacity C_int - w_i
            val_if_take = f_values[i] + dp_table[i][C_int - w_i]
            if abs(val_if_take - opt_val) < 1e-9:
                optimal_candidates.append(i)
    
    if len(optimal_candidates) == 0:
        # Fallback to greedy
        densities = f_values / np.maximum(f_weights, 1e-9)
        best_idx_in_f = int(np.argmax(densities))
        return int(feasible_indices[best_idx_in_f])
    
    if len(optimal_candidates) == 1:
        return int(feasible_indices[optimal_candidates[0]])
    
    # Multiple optimal candidates: pick the one that maximizes flexibility
    # Flexibility: prefer lighter items (leave more capacity) or higher value density
    # Use a combined score: value + alpha * (remaining_capacity_after)
    # Actually, since all lead to same total value, prefer the one with smallest weight
    # to leave maximum flexibility for subsequent calls
    cand_weights = f_weights_int[optimal_candidates]
    min_w_idx = int(np.argmin(cand_weights))
    best_cand = optimal_candidates[min_w_idx]
    
    return int(feasible_indices[best_cand])

# EVOLVE-BLOCK-END
