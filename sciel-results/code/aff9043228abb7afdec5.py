"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy:
# The task is to pick the next item for a 0/1 knapsack problem incrementally.
# The function `select_next_item` is called with the current remaining capacity and the list
# of remaining (unselected) items. It must return the index of the item to add.
# 
# Since the evaluator builds the solution incrementally and calls this function repeatedly,
# a greedy heuristic that makes locally optimal choices is a strong baseline.
# However, simple density (value/weight) greedy is often suboptimal.
#
# A better approach is to use a **Lookahead** heuristic.
# We can simulate the rest of the process for each candidate item to estimate the total value.
# 
# **Key Insight:** 
# 1. Filter feasible items.
# 2. For each feasible item `i`:
#    a. Tentatively pick `i`.
#    b. Simulate the completion of the knapsack with the remaining items and reduced capacity.
#    c. The simulation should be a strong heuristic (e.g., greedy by density, or a quick local search).
# 3. Pick the item `i` that maximizes the projected total value.
#
# To improve upon simple greedy simulation:
# - We can use a slightly more sophisticated simulator. 
# - Or, we can optimize the greedy simulation itself.
# - A common improvement is to consider "swaps" in the greedy solution, but that's expensive.
# - Another is to use a "density" greedy but with a tie-breaking rule or a secondary metric.
# - Let's try a simulator that does a quick local search (exchange) on the greedy solution.
#   Actually, for N=100, a full DP might be too slow if called 100 times per step.
#   DP is O(N*C). C is ~12.5. If weights are floats, DP is hard.
#   Wait, the problem says "100-item instances with capacity 12.5".
#   If weights are continuous, we can't use integer DP.
#   So we must rely on heuristics.
#
# Let's stick to a robust lookahead with a greedy simulator, but optimize the simulator.
# The current simulator sorts every time. We can pre-sort once per call to select_next_item.
# Then the simulation just iterates.
#
# Algorithm:
# 1. Pre-sort all items by density descending.
# 2. For each feasible candidate `i` (from original list):
#    a. new_cap = capacity - w_i
#    b. Simulate:
#       - Iterate through pre-sorted list.
#       - Skip `i` and any item already "used" (none, since we start fresh).
#       - If item fits, add it.
#    c. Total = v_i + sim_value.
# 3. Return best `i`.
#
# This is O(N^2 log N) per call? No, sorting is O(N log N) once.
# Simulation is O(N) per candidate.
# Total O(N^2 + N log N).
# N=100, so 10,000 ops per call.
# 100 calls per instance. 1,000,000 ops per instance.
# 8 instances. 8,000,000 ops. Very fast.
#
# Can we do better than greedy simulation?
# Maybe a "beam search" of width 2 or 3 in the simulation?
# Or just accept the greedy lookahead.
#
# Let's try to refine the greedy simulation.
# Sometimes, picking the highest density item is bad if it blocks many medium density items.
# But the lookahead already accounts for this by simulating the rest.
# The only error is that the *simulator* uses greedy, which is suboptimal.
# 
# Hypothesis: A lookahead with a *better* simulator will perform better.
# How to make the simulator better quickly?
# 
# Idea: In the simulation, after filling greedily, try a few swaps?
# Or, use a "density" greedy but if there's a tie, pick the one with higher value?
# 
# Let's look at the current score: 1.0015.
# The seed is likely simple greedy.
# The current program is already a lookahead greedy.
# 
# To improve, we can try a **2-step lookahead** in the selection?
# i.e., for each candidate `i`, and for each candidate `j` in the simulation?
# No, that's O(N^3). 100^3 = 1,000,000 per call. 100 calls = 100M. 8 instances = 800M.
# Might be too slow in Python.
# 
# Alternative: **Dynamic Programming with scaling**?
# If weights are floats, we can scale them to integers.
# If capacity is 12.5, and weights are around 0.1, scaling by 10 gives C=125.
# DP table size 100 * 125 = 12,500.
# DP is O(N*C).
# If we run DP for each candidate `i`, that's 100 * 12,500 = 1.25M ops per call.
# 100 calls = 125M ops.
# This is feasible in Python if optimized (using lists/arrays).
# 
# Let's try an exact DP solver for the sub-problem!
# Since the capacity is small (12.5), scaling to integers is a good idea.
# 
# Steps:
# 1. Determine a scaling factor. Let's look at the min weight.
#    Or just use a fixed high precision, e.g., 1000.
#    If C=12.5, scaled C=12500.
#    DP table: 100 * 12500 = 1.25M entries.
#    This is large for memory if we do it naively, but we can use 1D array.
#    Time: O(N * C_scaled).
#    100 * 12500 = 1.25M operations.
#    For each candidate `i` (100 candidates), we need to run DP on N-1 items.
#    Total: 100 * 1.25M = 125M operations per call.
#    100 calls per instance = 12.5B operations.
#    This is too slow in Python.
# 
# So exact DP for each candidate is too slow.
# 
# Back to heuristics.
# 
# What if we use the **exact DP** only once to get a "reference" solution, and then compare?
# No, the state changes.
# 
# Let's refine the **Greedy Lookahead**.
# 
# Improvement 1: **Pre-sorting**.
# Improvement 2: **Tie-breaking**.
# Improvement 3: **Better Simulator**.
# 
# Let's try a simulator that does a **Local Search** on the greedy result.
# 1. Get greedy set S.
# 2. Try to swap any item in S with any item not in S to improve value.
#    - This is O(N^2) per simulation.
#    - 100 candidates * 100^2 swaps = 1M ops per call.
#    - 100 calls = 100M ops.
#    - This is borderline but might be okay.
# 
# Let's implement a **Greedy + Local Search** simulator.
# 
# Actually, let's look at the "Hungarian" hint in the directive.
# "Formulate the selection as a maximum-weight bipartite matching... Hungarian algorithm".
# This is a very different approach.
# But the directive says "KEEP this design".
# So I should stick to the lookahead/search family.
# 
# Let's try to optimize the current lookahead.
# The current code re-sorts in `_simulate_greedy_value`.
# I will pre-sort the items by density once in `select_next_item`.
# Then the simulation just iterates.
# 
# Also, I will add a **deadline guard**.
# 
# Revised Algorithm:
# 1. Pre-sort indices by density (value/weight) descending.
# 2. For each feasible item `i`:
#    a. Simulate greedy fill using the pre-sorted list, skipping `i`.
#    b. Calculate total value.
# 3. Return best `i`.
# 
# This is significantly faster than the current implementation which sorts N times.
# 
# Can we do better?
# Maybe a **beam search** in the simulation?
# Keep top K partial solutions.
# 
# Let's stick to the optimized greedy lookahead first. It's a solid improvement in speed, allowing for more complex logic if needed, or just finishing faster.
# 
# Wait, the score is 1.0015. The seed is likely 1.0.
# So the current lookahead is already an improvement.
# 
# Let's try to add a **2-step lookahead** but only for the top few candidates.
# 1. Evaluate all candidates with 1-step greedy lookahead.
# 2. Pick top 5 candidates.
# 3. For these 5, do a 2-step lookahead (or a local search on the simulation).
# 
# This balances cost and quality.
# 
# Let's implement:
# 1. Pre-sort by density.
# 2. Quick pass: 1-step greedy lookahead for all feasible items.
# 3. Identify top K (e.g., 5) candidates.
# 4. Detailed pass: For top K, run a **Local Search** on the greedy simulation result.
#    - Start with greedy set.
#    - Try to improve by swapping.
# 
# This should be fast and potentially better.
# 
# Implementation details:
# - `simulate_greedy_fast(cap, weights, values, sorted_indices, exclude_idx)`:
#   - Iterates `sorted_indices`.
#   - If `idx == exclude_idx`, skip.
#   - If `weights[idx] <= cap`, add.
#   - Returns (value, selected_indices).
# 
# - `local_search_improve(cap, weights, values, sorted_indices, initial_selected, initial_value)`:
#   - `selected` is a set of indices.
#   - `unselected` is the complement.
#   - Loop:
#     - Try to swap any `in` in `selected` with `out` in `unselected`.
#     - Condition: `w_out - w_in <= cap - (sum_w_selected - w_in)`?
#       Actually, new_cap = cap - (sum_w_selected - w_in + w_out).
#       We need `w_out <= (cap - sum_w_selected) + w_in`.
#     - If swap improves value (`v_out > v_in`), do it.
#     - Repeat until no improvement.
# 
# This local search is O(N^2) per iteration, and might take a few iterations.
# 
# Let's code this up.

def _simulate_greedy_fast(capacity, weights, values, sorted_indices, exclude_idx):
    """
    Simulates greedy fill. Returns (total_value, list_of_selected_indices).
    """
    total_value = 0.0
    current_cap = capacity
    selected = []
    
    for idx in sorted_indices:
        if idx == exclude_idx:
            continue
        w = weights[idx]
        if w <= current_cap:
            total_value += values[idx]
            current_cap -= w
            selected.append(idx)
            if current_cap <= 0:
                break
    
    return total_value, selected

def _local_search_improve(capacity, weights, values, sorted_indices, selected_indices, total_value):
    """
    Attempts to improve the solution by swapping items.
    Returns (improved_value, improved_selected).
    """
    # Convert to set for fast lookup
    selected_set = set(selected_indices)
    unselected_set = set(sorted_indices) - selected_set
    
    # Calculate current weight sum
    current_weight = sum(weights[i] for i in selected_set)
    
    improved = True
    while improved:
        improved = False
        # Try all swaps
        best_swap_gain = 0
        best_swap_in = -1
        best_swap_out = -1
        
        for in_idx in selected_set:
            for out_idx in unselected_set:
                # Check if swap is feasible
                # New weight = current_weight - w_in + w_out
                # Must be <= capacity
                if weights[out_idx] - weights[in_idx] <= capacity - current_weight:
                    gain = values[out_idx] - values[in_idx]
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_in = in_idx
                        best_swap_out = out_idx
        
        if best_swap_gain > 0:
            # Perform swap
            selected_set.remove(best_swap_in)
            selected_set.add(best_swap_out)
            unselected_set.remove(best_swap_out)
            unselected_set.add(best_swap_in)
            current_weight += weights[best_swap_out] - weights[best_swap_in]
            total_value += best_swap_gain
            improved = True
    
    return total_value, list(selected_set)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if len(remaining_weights) == 0:
        return -1
    
    start_time = time.time()
    deadline = start_time + 10.0  # 10s safety margin per call? No, total budget is 120s for 8 instances * 100 calls = 800 calls.
    # 120s / 800 calls = 0.15s per call.
    # Let's use 0.1s per call.
    deadline = start_time + 0.1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    
    # Filter feasible items
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Pre-sort all items by density descending
    # We need to handle zero weight
    densities = np.zeros(n)
    non_zero = weights > 0
    densities[non_zero] = values[non_zero] / weights[non_zero]
    # Items with zero weight get infinite density (or very high)
    densities[~non_zero] = float('inf')
    
    sorted_indices = np.argsort(-densities)
    
    best_index = -1
    best_projected_value = -1.0
    
    # Phase 1: Quick Greedy Lookahead for all feasible items
    candidate_scores = {}
    
    for i_idx in feasible_indices:
        if time.time() > deadline:
            # Fallback to simple greedy pick (highest density feasible)
            return int(sorted_indices[0]) if sorted_indices[0] in feasible_indices else int(feasible_indices[0])
        
        w_i = weights[i_idx]
        v_i = values[i_idx]
        new_cap = remaining_capacity - w_i
        
        sim_val, _ = _simulate_greedy_fast(new_cap, weights, values, sorted_indices, int(i_idx))
        
        projected_val = v_i + sim_val
        candidate_scores[int(i_idx)] = projected_val
        
        if projected_val > best_projected_value:
            best_projected_value = projected_val
            best_index = int(i_idx)
    
    # Phase 2: Local Search for top K candidates
    # Sort candidates by score
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 5
    top_k = sorted_candidates[:5]
    
    for i_idx, score in top_k:
        if time.time() > deadline:
            break
        
        w_i = weights[i_idx]
        v_i = values[i_idx]
        new_cap = remaining_capacity - w_i
        
        sim_val, selected = _simulate_greedy_fast(new_cap, weights, values, sorted_indices, int(i_idx))
        
        # Run local search on the simulated result
        improved_val, _ = _local_search_improve(new_cap, weights, values, sorted_indices, selected, sim_val)
        
        final_projected_val = v_i + improved_val
        
        if final_projected_val > best_projected_value:
            best_projected_value = final_projected_val
            best_index = int(i_idx)
            
    return best_index

# EVOLVE-BLOCK-END
