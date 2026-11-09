import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to feasible items only
    feasible = []
    for i in range(n):
        w = remaining_weights[i]
        v = remaining_values[i]
        if w > 0 and w <= remaining_capacity:
            feasible.append((v, w, i))
        elif w == 0 and v > 0:
            return i  # Free value, take it
    
    if not feasible:
        return -1
    
    # Use DP with scaled weights to find optimal subset
    # Scale by 100 to handle one decimal place
    SCALE = 100
    cap_scaled = int(round(remaining_capacity * SCALE))
    
    # DP: dp[w] = max value achievable with weight <= w
    # Use 1D DP
    dp = [0.0] * (cap_scaled + 1)
    
    # Track which items are used for reconstruction
    # For each weight level, track the last item added
    # Actually, let's just find the best single item to add first
    # by solving the full subproblem and seeing which item is in the optimal solution
    
    # Better: solve the knapsack DP and reconstruct to find which items are optimal
    # Then return the first item (by some ordering) from the optimal set
    
    # Since we need to pick ONE item, let's find the item that appears in the optimal solution
    # with the highest "priority" (e.g., highest density or first in optimal set)
    
    # DP to find optimal value
    for v, w, idx in feasible:
        w_scaled = int(round(w * SCALE))
        if w_scaled > cap_scaled:
            continue
        for c in range(cap_scaled, w_scaled - 1, -1):
            new_val = dp[c - w_scaled] + v
            if new_val > dp[c]:
                dp[c] = new_val
    
    # Reconstruct to find which items are in optimal solution
    # This is tricky with 1D DP. Let's use a different approach.
    
    # Alternative: For each feasible item, compute the best value if we MUST include it
    # Then pick the item with the highest (v + best_remaining_value_after_including_it)
    # But that's O(n^2 * cap) which might be too slow.
    
    # Simpler: Just use density greedy with a small improvement
    # Sort by density descending
    feasible.sort(key=lambda x: x[0] / x[1], reverse=True)
    
    # Greedy fill to get a baseline
    greedy_val = 0
    greedy_items = []
    rem_cap = remaining_capacity
    for v, w, idx in feasible:
        if w <= rem_cap:
            greedy_val += v
            greedy_items.append(idx)
            rem_cap -= w
    
    # Try each item as the first pick, then greedy fill the rest
    best_first = feasible[0][2]
    best_total = greedy_val
    
    # Limit the number of candidates to try for speed
    candidates = feasible[:min(20, len(feasible))]
    
    for v_first, w_first, idx_first in candidates:
        if w_first > remaining_capacity:
            continue
        rem_cap_after = remaining_capacity - w_first
        # Greedy fill with remaining items
        total = v_first
        for v, w, idx in feasible:
            if idx == idx_first:
                continue
            if w <= rem_cap_after:
                total += v
                rem_cap_after -= w
        if total > best_total:
            best_total = total
            best_first = idx_first
    
    return best_first
