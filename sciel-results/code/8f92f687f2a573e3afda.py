"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _check_deadline(safety_margin=5.0):
    """Return True if we should stop and return depot."""
    elapsed = time.time() - _get_start_time()
    return elapsed > (120.0 - safety_margin)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot.
    
    Structural change: Use a 2-opt lookahead heuristic combined with 
    capacity-aware nearest neighbor. The key insight is that instead of 
    just minimizing immediate distance, we estimate the impact on the 
    total route length by considering how inserting this node affects 
    the overall path structure.
    
    Strategy:
    1. Filter feasible nodes
    2. For each candidate, compute a score based on:
       - Immediate distance from current
       - Estimated "regret" of not visiting this node now (how much 
         longer the route would be if we skipped it)
       - Capacity efficiency: prefer nodes that use capacity well
    3. Use a look-ahead: for the top few candidates, simulate the next 
       1-2 steps to see which leads to a better overall path
    """
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Check deadline
    if _check_deadline():
        return int(depot)
    
    # Check if we can even fit any more nodes
    min_demand = min(demands[n] for n in unvisited_nodes)
    if min_demand > rest_capacity:
        return int(depot)
    
    # Filter to feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if len(feasible) == 0:
        return int(depot)
    
    # If very few nodes left, use a more careful approach
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited <= 5:
        # For small sets, do a more thorough evaluation
        # Try all permutations of the remaining nodes (up to 5! = 120)
        # and pick the one that minimizes total route length
        best_total = float('inf')
        best_first = feasible[0]
        
        # For n <= 5, we can enumerate
        if n_unvisited <= 5:
            from itertools import permutations
            # We need to find the best sequence starting from current
            # that ends at depot
            # This is TSP on a small set
            
            # Convert to list for permutations
            node_list = list(feasible)
            n_nodes = len(node_list)
            
            if n_nodes <= 8:
                # Enumerate all permutations
                best_perm = None
                best_cost = float('inf')
                
                for perm in permutations(node_list):
                    # Compute total cost: current -> perm[0] -> perm[1] -> ... -> depot
                    cost = distance_matrix[current_node, perm[0]]
                    for i in range(len(perm) - 1):
                        cost += distance_matrix[perm[i], perm[i+1]]
                    cost += distance_matrix[perm[-1], depot]
                    
                    # Check capacity feasibility for this permutation
                    cap = rest_capacity
                    feasible_perm = True
                    for node in perm:
                        if demands[node] > cap:
                            feasible_perm = False
                            break
                        cap -= demands[node]
                    
                    if feasible_perm and cost < best_cost:
                        best_cost = cost
                        best_perm = perm
                
                if best_perm is not None:
                    return int(best_perm[0])
        
        # Fallback: use a heuristic
        best_node = None
        best_score = float('inf')
        
        for n in feasible:
            d_to_node = distance_matrix[current_node, n]
            d_node_to_depot = distance_matrix[n, depot]
            
            # For small sets, heavily weight the return-to-depot distance
            score = d_to_node + 0.9 * d_node_to_depot
            
            # Small penalty for high demand
            score += 0.02 * demands[n]
            
            if score < best_score:
                best_score = score
                best_node = n
        
        return int(best_node)
    
    # For larger sets, use a smarter greedy with lookahead
    # Score each candidate based on:
    # 1. Immediate distance
    # 2. "Regret" - how much worse would it be to not visit this node now
    # 3. Capacity efficiency
    
    # Precompute distances from current to all feasible nodes
    scores = {}
    
    for n in feasible:
        d_to_node = distance_matrix[current_node, n]
        d_node_to_depot = distance_matrix[n, depot]
        demand = demands[n]
        
        # Base score: immediate distance
        score = d_to_node
        
        # Add a component for the return leg, weighted by how many nodes are left
        # When many nodes are left, the return leg matters less
        # When few nodes are left, it matters more
        weight = 0.1 + 0.3 * (1.0 - n_unvisited / 50.0)
        score += weight * d_node_to_depot
        
        # Capacity efficiency: prefer nodes that don't waste capacity
        # If taking this node leaves very little capacity, penalize
        remaining_after = rest_capacity - demand
        if remaining_after < 5:
            score += 0.5  # Penalize leaving very little capacity
        
        # Demand-based adjustment: prefer moderate demands
        # Very high demands might be hard to fit later
        if demand > 7:
            score += 0.1 * demand
        
        scores[n] = score
    
    # Get top candidates for lookahead
    sorted_candidates = sorted(feasible, key=lambda x: scores[x])
    top_k = min(5, len(sorted_candidates))
    top_candidates = sorted_candidates[:top_k]
    
    # Lookahead: for each top candidate, estimate the cost of the next step
    best_node = top_candidates[0]
    best_lookahead_score = float('inf')
    
    for cand in top_candidates:
        # Simulate visiting cand
        new_current = cand
        new_rest_capacity = rest_capacity - demands[cand]
        
        # Find the best next node after cand
        feasible_next = [n for n in unvisited_nodes 
                        if n != cand and demands[n] <= new_rest_capacity]
        
        if len(feasible_next) == 0:
            # Must return to depot
            lookahead_cost = distance_matrix[current_node, cand] + distance_matrix[cand, depot]
        else:
            # Find the best next node
            best_next_score = float('inf')
            for n2 in feasible_next:
                d1 = distance_matrix[current_node, cand]
                d2 = distance_matrix[cand, n2]
                d3 = distance_matrix[n2, depot]
                
                # Weight the return leg based on remaining nodes
                weight2 = 0.1 + 0.3 * (1.0 - (n_unvisited - 1) / 50.0)
                lookahead_cost = d1 + d2 + weight2 * d3
                
                if lookahead_cost < best_next_score:
                    best_next_score = lookahead_cost
            
            lookahead_cost = best_next_score
        
        # Combine immediate score and lookahead
        combined_score = scores[cand] + 0.3 * lookahead_cost
        
        if combined_score < best_lookahead_score:
            best_lookahead_score = combined_score
            best_node = cand
    
    return int(best_node)
# EVOLVE-BLOCK-END
