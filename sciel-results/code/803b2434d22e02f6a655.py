"""Insertion-based heuristic for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a global insertion heuristic.
    We simulate inserting each feasible unvisited node into the current route 
    (conceptually) and pick the one that yields the smallest increase in total 
    route length, or the depot if no feasible insertion is found or time is up.
    
    Since we are building the route sequentially, "insertion" here is approximated 
    by evaluating the cost of appending the node vs. the "opportunity cost" of 
    not visiting it now. However, a more powerful differentiator from the current 
    savings+NN blend is to look at the *angle* or *direction* consistency, or 
    simply a pure global minimization of the marginal cost:
    
    Marginal Cost = dist(current, node) + dist(node, depot) - dist(current, depot)
    
    Wait, the current program uses Savings which is effectively minimizing this 
    marginal cost (since minimizing marginal cost is equivalent to maximizing savings 
    if dist(current, depot) is constant).
    
    Let's try a DIFFERENT strategy: **Cheapest Insertion into a Virtual Route**.
    The current program treats the route as a sequence being built. 
    A different framing is to view the problem as packing nodes into the remaining 
    capacity and distance budget.
    
    Actually, the most distinct simple strategy that often outperforms greedy savings 
    in CVRP is **Clarke-Wright Savings** but applied differently, or **Nearest Neighbor 
    with Backtracking**.
    
    Let's try **Angle-Based Heuristic** (Sectoring).
    Strategy:
    1. Calculate the angle of each unvisited node relative to the depot.
    2. Prefer nodes that are in the same "sector" as the current node relative to the depot.
    3. This reduces zig-zagging significantly more than a simple distance blend.
    
    Implementation:
    - Compute vector from Depot to Current: v_cur
    - Compute vector from Depot to Candidate: v_cand
    - Compute dot product to find cosine of angle.
    - Prefer candidates with high cosine (small angle) relative to current direction.
    - Combine with distance penalty.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # Convert to numpy array
    unvisited = np.array(unvisited_nodes)
    
    # 1. Filter by capacity
    feasible = unvisited[demands[unvisited] <= rest_capacity]
    
    if len(feasible) == 0:
        return 0

    # Get coordinates implicitly from distance matrix? No, we only have distance matrix.
    # We cannot compute angles without coordinates. 
    # The task says "distance_matrix" is provided. It does NOT provide coordinates.
    # Therefore, Angle-Based Heuristic is NOT possible unless we assume Euclidean and 
    # try to reconstruct coordinates (too slow/complex) or assume the distance matrix 
    # has structure. We must stick to distance-based metrics.
    
    # Alternative Strategy: **Lookahead Nearest Neighbor with Capacity Pressure**
    # The current program blends Savings and NN.
    # Let's try a **Pure Savings** approach but with a **Tie-Breaker** based on 
    # "Density" (Demand / Distance).
    # Or better: **Global Best-Insertion Approximation**.
    
    # Let's rethink. The current score is 1.08. 
    # Inspiration 2 (1.0749) used "look-ahead penalty for nodes that would leave very little remaining capacity".
    # This suggests that ending a route too early (due to low remaining capacity for future large nodes) is bad.
    
    # New Strategy: **Capacity-Aware Savings**.
    # Standard Savings: S(i,j) = d(i,depot) + d(depot,j) - d(i,j)
    # Modified Savings: Penalize nodes that are "hard to place" later.
    # If a node has high demand, it is "hard to place". If we don't pick it now, 
    # we might be stuck with it later when capacity is fragmented.
    # So, boost the score of high-demand nodes if they fit now.
    
    # Score = Savings + lambda * Demand
    # This encourages packing high-demand nodes into the current route if possible.
    
    unvisited = np.array(unvisited_nodes)
    feasible = unvisited[demands[unvisited] <= rest_capacity]
    
    if len(feasible) == 0:
        return 0

    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_cand = distance_matrix[depot, feasible]
    d_cur_cand = distance_matrix[current_node, feasible]
    
    # Standard Savings
    savings = d_cur_dep + d_dep_cand - d_cur_cand
    
    # Normalize Savings
    s_min = np.min(savings)
    s_max = np.max(savings)
    if s_max == s_min:
        norm_savings = np.zeros_like(savings, dtype=float)
    else:
        norm_savings = (savings - s_min) / (s_max - s_min)
        
    # Demand Score: Prefer higher demand to consolidate
    # Normalize Demand
    dem_cand = demands[feasible]
    d_min = np.min(dem_cand)
    d_max = np.max(dem_cand)
    if d_max == d_min:
        norm_demand = np.zeros_like(dem_cand, dtype=float)
    else:
        norm_demand = (dem_cand - d_min) / (d_max - d_min)
        
    # Distance Penalty: Prefer closer nodes (to keep route compact)
    # This is similar to the current program's NN component, but let's weight it differently.
    # The current program uses alpha=0.7 for savings. 
    # Let's try a different balance: Heavily favor Savings, but use Demand as a tie-breaker/boost.
    # Or, use a "Compactness" metric: dist(current, cand) / (dist(current, depot) + dist(depot, cand))
    # This ratio is low if the node is "on the way".
    
    # Let's try **Ratio Heuristic**:
    # Cost Ratio = d(cur, cand) / (d(cur, depot) + d(depot, cand))
    # Low ratio means high savings.
    # We also want to minimize d(cur, cand) absolute.
    
    # Let's try a **Greedy Set Covering** approximation:
    # Pick the node that minimizes d(cur, cand) + d(cand, depot) - d(cur, depot) 
    # BUT weighted by 1/demand? No, that's the opposite.
    
    # Let's try **Randomized Greedy with Restart** (Simulated Annealing lite)?
    # Too complex for the function signature.
    
    # Let's try **Pure Savings with Demand Boost**.
    # Hypothesis: The current blend (0.7 Savings, 0.3 Dist) might be too conservative on distance,
    # leading to shorter routes but more routes? No, lower total length is the goal.
    # If we save more distance, total length decreases.
    # Why did 1.08 (Savings+Dist) beat 1.12 (Pure Savings in finding)? 
    # Wait, the finding said "Greedy savings... outperforming nearest-neighbor... Measured: 1 -> 1.12".
    # But the current program (1.08) is Savings+Dist.
    # This implies that the specific blend in the current program is *better* than pure savings 
    # in this specific instance set, OR the 1.12 was a different variant.
    # Actually, 1.12 is worse than 1.08 (lower is better). So Pure Savings was worse?
    # Or maybe the 1.12 was a different program.
    # "Measured: 1 -> 1.12" usually means improvement from baseline 1.0 to 1.12.
    # If baseline is 1.0, and 1.12 is the score, that's an improvement.
    # Current score 1.08 is better than 1.12.
    # So the current hybrid is better than the pure savings variant that scored 1.12.
    
    # Let's try **Adaptive Weighting**.
    # If rest_capacity is high, we can afford to take farther nodes (high savings).
    # If rest_capacity is low, we should stick to close nodes (low distance) to ensure 
    # we can return to depot or fit small nodes.
    
    # Capacity Fraction
    # We don't know total capacity, but we know rest_capacity.
    # Let's assume a typical capacity of 50.
    # cap_ratio = rest_capacity / 50.0
    
    # If cap_ratio > 0.5, favor Savings (0.8).
    # If cap_ratio < 0.5, favor Distance (0.6 for dist, 0.4 for savings).
    
    # Let's implement this adaptive strategy.
    
    # Estimate capacity. The problem says "capacity 50".
    CAPACITY = 50.0
    cap_ratio = rest_capacity / CAPACITY
    
    # Calculate components
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_cand = distance_matrix[depot, feasible]
    d_cur_cand = distance_matrix[current_node, feasible]
    
    savings = d_cur_dep + d_dep_cand - d_cur_cand
    
    # Normalize Savings
    s_min = np.min(savings)
    s_max = np.max(savings)
    if s_max == s_min:
        norm_savings = np.zeros_like(savings, dtype=float)
    else:
        norm_savings = (savings - s_min) / (s_max - s_min)
        
    # Normalize Distance (inverted)
    dist_to_cur = d_cur_cand
    dist_min = np.min(dist_to_cur)
    dist_max = np.max(dist_to_cur)
    if dist_max == dist_min:
        norm_dist = np.zeros_like(dist_to_cur, dtype=float)
    else:
        norm_dist = (dist_max - dist_to_cur) / (dist_max - dist_min)
        
    # Adaptive Alpha
    # When capacity is high, we have room to optimize for savings (global structure).
    # When capacity is low, we must be careful not to get stuck; prefer close nodes.
    if cap_ratio > 0.6:
        alpha = 0.8  # More savings
    elif cap_ratio > 0.3:
        alpha = 0.6  # Balanced
    else:
        alpha = 0.4  # More distance (safety)
        
    combined_score = alpha * norm_savings + (1 - alpha) * norm_dist
    
    best_idx = np.argmax(combined_score)
    
    return int(feasible[best_idx])

# EVOLVE-BLOCK-END
