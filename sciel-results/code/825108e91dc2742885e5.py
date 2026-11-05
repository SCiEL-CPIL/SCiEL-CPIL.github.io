import numpy as np
import time

# Deadline guard: total wall-clock budget for the entire evaluation.
# The evaluator kills after 150s. We want to be safe, so we use 145s as our hard limit.
# Since the function is called 50 times per instance and there are 8 instances,
# total calls = 400.
# We track global time to ensure we don't exceed the budget.
_START_TIME = time.time()
_DEADLINE = 145.0  # seconds

def _held_karp_next(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Finds the optimal next node for the remaining set using Held-Karp DP.
    This is only called for small n (<= 12 or 13) to keep runtime low.
    Returns the first node in the optimal sequence.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Map unvisited nodes to local indices 0..n-1
    full_mask = (1 << n) - 1
    
    INF = float('inf')
    # dp[mask][i] = min cost to start at current_node, visit nodes in mask, end at unv[i]
    dp = np.full((1 << n, n), INF, dtype=float)
    parent = np.full((1 << n, n), -1, dtype=int)
    
    # Base cases: start at current_node, go to unv[i]
    # We can vectorize this a bit if we want, but loop is fine for n<=13
    for i in range(n):
        dp[1 << i, i] = distance_matrix[current_node, unv[i]]
        
    # Fill DP table
    for mask in range(1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            current_cost = dp[mask, i]
            if current_cost == INF:
                continue
            
            # Try extending to j not in mask
            # To speed up, we can precompute bits not in mask, but for n<=13 it's fine
            for j in range(n):
                if mask & (1 << j):
                    continue
                
                new_mask = mask | (1 << j)
                step_cost = distance_matrix[unv[i], unv[j]]
                new_cost = current_cost + step_cost
                
                if new_cost < dp[new_mask, j]:
                    dp[new_mask, j] = new_cost
                    parent[new_mask, j] = i

    # Find the best ending node i in full_mask such that dp[full_mask][i] + dist(unv[i], dest) is min
    best_cost = INF
    best_last_i = -1
    for i in range(n):
        total_cost = dp[full_mask, i] + distance_matrix[unv[i], destination_node]
        if total_cost < best_cost:
            best_cost = total_cost
            best_last_i = i
            
    if best_last_i == -1:
        return int(unv[0])
        
    # Backtrack to find the first node
    current_mask = full_mask
    current_i = best_last_i
    first_i = -1
    
    while current_mask != 0:
        path_node = current_i
        if current_mask & (current_mask - 1) == 0:
            first_i = current_i
            break
        
        prev_i = parent[current_mask, current_i]
        if prev_i == -1:
            break
        current_mask = current_mask ^ (1 << current_i)
        current_i = prev_i
        
    if first_i == -1:
        first_i = 0
        
    return int(unv[first_i])


def _select_heuristic(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Improved heuristic for larger n.
    Uses a combination of:
    1. Nearest neighbor distance
    2. 2-step lookahead (distance to next closest unvisited)
    3. Regret (difference between best and second-best 2-step costs)
    4. Destination bias (distance to final destination)
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])

    dm = distance_matrix
    d_cur = dm[current_node, unv]
    
    # Compute 2-step lookahead: cost to go from unv[i] to its nearest neighbor in unv
    # sub is the distance matrix among unvisited nodes
    sub = dm[np.ix_(unv, unv)]
    np.fill_diagonal(sub, np.inf)
    d_onward = sub.min(axis=1)
    
    two_step = d_cur + d_onward
    
    # Regret: how much worse is the second-best option compared to the best?
    # We want to prioritize nodes where the difference between best and second-best is large
    # This prevents "locking in" a bad choice early.
    # Sort two_step values
    sorted_two_step = np.sort(two_step)
    best_cost = sorted_two_step[0]
    second_best_cost = sorted_two_step[1] if n > 1 else best_cost
    regret = second_best_cost - best_cost
    
    # Actually, regret per node is more complex. 
    # Let's use a simpler regret: for each node i, compute the cost if we pick it, 
    # and compare to the cost if we pick the best alternative.
    # But that's O(n^2). Let's stick to a simpler proxy.
    
    # Alternative regret: difference between min(two_step) and two_step[i]
    # Nodes with high regret are "risky" to delay, so we might want to pick them sooner?
    # Or nodes with low regret are safe to delay?
    # Standard regret heuristic: pick the node with the highest regret (difference between best and second best options involving that node).
    # Here, let's define regret_i = (second_best_two_step - best_two_step) is global, not per node.
    # Per-node regret: if we pick node i, what is the cost? two_step[i].
    # The "loss" of not picking the best node is (two_step[i] - min(two_step)).
    # So high loss = high regret. We want to minimize regret, so pick the node with smallest loss?
    # That's just nearest neighbor in two_step space.
    
    # Let's use a different approach:
    # Score = w1 * d_cur_norm + w2 * two_step_norm + w3 * d_dest_norm + w4 * regret_norm
    # where regret_norm is high for nodes that are "critical" (i.e., if we don't pick them now, the cost increases a lot).
    
    d_dest = dm[unv, destination_node]
    
    # Normalize each component
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn > 1e-12:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)
    
    d_cur_norm = normalize(d_cur)
    two_step_norm = normalize(two_step)
    d_dest_norm = normalize(d_dest)
    
    # Regret proxy: for each node i, compute how much the two_step cost would increase 
    # if we didn't pick the best node. 
    # Actually, let's compute the "opportunity cost" of picking node i:
    # opp_cost[i] = two_step[i] - min(two_step)
    # High opp_cost means picking this node is much worse than the best option.
    # We want to avoid high opp_cost, so we minimize opp_cost.
    # This is equivalent to minimizing two_step.
    
    # Let's try a different regret: 
    # For each node i, look at its two neighbors in the unvisited set (by distance from i).
    # If both neighbors are far, it's risky.
    # This is getting complex. Let's stick to a blended score.
    
    # Weights: 
    # d_cur: 0.3
    # two_step: 0.4
    # d_dest: 0.2
    # regret: 0.1 (using opp_cost as proxy)
    
    opp_cost = two_step - two_step.min()
    opp_cost_norm = normalize(opp_cost)
    
    score = (0.30 * d_cur_norm + 0.40 * two_step_norm + 0.20 * d_dest_norm + 0.10 * opp_cost_norm)
    
    idx = int(np.argmin(score))
    return int(unv[idx])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 0.5:
        # Return a safe fallback: nearest neighbor
        if n == 0:
            return int(destination_node)
        if n == 1:
            return int(unv[0])
        d_cur = distance_matrix[current_node, unv]
        idx = int(np.argmin(d_cur))
        return int(unv[idx])
    
    # Use exact DP for small remaining sets
    # n <= 12 is safe for speed. 13 might be borderline. Let's use 12.
    if n <= 12:
        return _held_karp_next(current_node, destination_node, unv, distance_matrix)
    else:
        return _select_heuristic(current_node, destination_node, unv, distance_matrix)
