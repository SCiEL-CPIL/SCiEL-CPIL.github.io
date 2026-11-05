"""Rewritten program using Model-and-Optimize approach for CVRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 148.0  # Safety margin before 150s limit

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def _evaluate_candidate(cand_node, current_node, depot, unvisited_set, rest_cap, demands, dist_mat):
    """
    Evaluates a candidate node by looking two steps ahead.
    Returns the estimated total cost if we pick cand_node now.
    Cost = d(current, cand) + d(cand, best_next) + d(best_next, best_next2_or_depot)
    Uses savings-based scoring for consistency with the main heuristic.
    """
    d_cur_cand = dist_mat[current_node, cand_node]
    
    # Update state for the next step
    new_rest_cap = rest_cap - demands[cand_node]
    new_unvisited = unvisited_set - {cand_node}
    
    if not new_unvisited:
        # Must go to depot
        return d_cur_cand + dist_mat[cand_node, depot]
    
    # Find feasible next nodes from cand_node
    feasible_next = [n for n in new_unvisited if demands[n] <= new_rest_cap]
    
    if not feasible_next:
        return d_cur_cand + dist_mat[cand_node, depot]
    
    # For each feasible next node, estimate the 2-step cost:
    # step1_cost = d(cand, next)
    # step2_cost = d(next, best_next2) or d(next, depot)
    # We use savings-based scoring to pick best_next2 for each next candidate.
    
    d_cand_dep = dist_mat[cand_node, depot]
    W_lookahead = 0.5  # Same weight as main heuristic for consistency
    
    best_total_cost = d_cur_cand + d_cand_dep  # Default: go to depot immediately
    
    for n in feasible_next:
        d_cand_n = dist_mat[cand_node, n]
        
        # Estimate cost if we go to depot after n
        cost_go_depot = d_cand_n + dist_mat[n, depot]
        
        # Check if there's a feasible next-next node from n
        rest_after_n = new_rest_cap - demands[n]
        if rest_after_n <= 0:
            # Can't visit another node, must go to depot
            total_cost = d_cur_cand + cost_go_depot
            if total_cost < best_total_cost:
                best_total_cost = total_cost
            continue
        
        # Find best next-next node from n using savings
        d_n_dep = dist_mat[n, depot]
        feasible_next2 = [m for m in new_unvisited if m != n and demands[m] <= rest_after_n]
        
        if not feasible_next2:
            total_cost = d_cur_cand + cost_go_depot
            if total_cost < best_total_cost:
                best_total_cost = total_cost
            continue
        
        # Find best next2 using savings score
        best_next2_cost = d_n_dep  # Default: go to depot
        best_next2_node = None
        
        for m in feasible_next2:
            d_n_m = dist_mat[n, m]
            d_dep_m = dist_mat[depot, m]
            savings_m = d_n_dep + d_dep_m - d_n_m
            score_m = savings_m - W_lookahead * d_n_m
            
            # Cost if we go to m then depot
            cost_via_m = d_n_m + dist_mat[m, depot]
            
            # Compare: going directly to depot vs going via m
            # We want to minimize total cost, so pick the option with lower cost
            # But also consider that m might enable further savings
            # For simplicity, estimate: if score_m > 0 (savings positive), going via m is good
            # Use: cost_via_m if it's better than direct depot, weighted by savings attractiveness
            
            # Simple approach: estimate total cost as d_cand_n + min(d_n_dep, d_n_m + d_m_dep)
            # But also factor in that high-savings nodes are more attractive
            # Let's just compute the minimum of direct depot vs best m
            if cost_via_m < best_next2_cost:
                best_next2_cost = cost_via_m
                best_next2_node = m
        
        # Total cost for this path: current->cand->n->(best_next2 or depot)
        total_cost = d_cur_cand + d_cand_n + best_next2_cost
        if total_cost < best_total_cost:
            best_total_cost = total_cost
    
    return best_total_cost


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # If time is running out, return depot to close route safely
    if not _time_ok():
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Convert to numpy arrays for efficiency
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)
    
    feas_nodes = unvisited_arr[feas_mask]
    
    # Calculate distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Calculate Savings
    # Savings = d[current, depot] + d[depot, c] - d[current, c]
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # Model-and-Optimize Parameter:
    # Score = Savings - W * Distance
    # W is tuned to balance savings vs immediate distance.
    # Lower W favors savings more, creating tighter clusters.
    W = 0.3
    scores = savings - W * d_cur_feas
    
    # Select top candidates for lookahead
    # If few nodes, check all. If many, check top 3-5.
    # For 50 customers, checking all is O(N^2) which is 2500 ops, very fast.
    # But let's limit to top 5 for safety and speed.
    
    n_nodes = len(feas_nodes)
    if n_nodes <= 7:
        top_indices = np.arange(n_nodes)
    else:
        # Get indices of top 7 scores for better exploration
        top_indices = np.argsort(scores)[::-1][:7]
    
    best_candidate_node = int(feas_nodes[0]) # Fallback
    best_candidate_cost = np.inf
    
    # Pre-compute unvisited set for lookahead logic
    # Using a set for O(1) removal
    unvisited_set = set(unvisited_nodes)
    
    for idx in top_indices:
        cand_node = int(feas_nodes[idx])
        
        # Evaluate this candidate with lookahead
        # _evaluate_candidate returns estimated total cost (lower is better)
        est_cost = _evaluate_candidate(cand_node, current_node, depot, unvisited_set, rest_capacity, demands, distance_matrix)
        
        if est_cost < best_candidate_cost:
            best_candidate_cost = est_cost
            best_candidate_node = cand_node
            
        # Early exit if we're getting close to deadline
        if not _time_ok():
            return int(depot)

    return int(best_candidate_node)
# EVOLVE-BLOCK-END
