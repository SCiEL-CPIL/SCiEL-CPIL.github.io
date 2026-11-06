"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin from 150s

# Cache for regret/lookahead computations (limited size)
_score_cache = {}
_CACHE_MAX = 200000

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _START_TIME, _TIME_LIMIT, _score_cache, _CACHE_MAX
    
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    # Convert to numpy array for fast operations
    unvisited = np.array(unvisited_nodes, dtype=np.int64)
    
    # 1. Check capacity constraints
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited])
    
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    n = len(feasible_nodes)
    
    # Cache key: (current, rest_capacity, tuple(feasible_nodes))
    # Use a lighter key: (current, rest_capacity, frozenset(feasible_nodes))
    # But frozenset of numpy array is expensive. Use tuple if n small, else hash of sorted.
    # For speed, we might skip caching if n is large or if time is tight.
    # Let's try a simple cache key.
    if n <= 10:
        cache_key = (current_node, rest_capacity, tuple(feasible_nodes))
    else:
        # For large n, caching might not be as effective or key generation expensive.
        # But let's try a hash of the array.
        # To be safe and fast, maybe don't cache for large n in this simple implementation.
        cache_key = None
    
    if cache_key is not None and cache_key in _score_cache:
        next_node, _ = _score_cache[cache_key]
        return int(next_node)
    
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_from_depot = distance_matrix[depot, feasible_nodes]
    
    # Base savings score
    savings_scores = dist_from_depot - dist_from_current
    
    # Regret-based scoring:
    # For each feasible node i, calculate the regret of not picking it.
    # Regret = (best alternative cost) - (cost of picking i) ?
    # Or as suggested: difference between distance to nearest feasible alternative and second-nearest.
    # Let's interpret "regret" in the context of routing:
    # If we don't pick node i now, we might have to pick it later.
    # The "cost" of delaying node i is related to how well it connects to the current state vs other options.
    
    # A common regret heuristic in VRP/TS is:
    # For each candidate i, look at the best 2nd move if we don't pick i.
    # Here, since we are building a route sequentially, "regret" is tricky.
    # Let's stick to the directive's specific suggestion:
    # "scored by the difference between the distance to its nearest feasible alternative and the distance to its second-nearest feasible alternative"
    # This sounds like it's evaluating the "uniqueness" of a node's position.
    
    # Let's compute a regret score for each feasible node i:
    # 1. Identify the "cost" of connecting i to the route.
    #    Cost_i = dist(current, i) - dist(depot, i)  (Negative of savings)
    # 2. For each i, find the best alternative j != i.
    #    Alternative cost = Cost_j
    # 3. Regret_i = Best_Cost_Alternative - Cost_i
    #    If Cost_i is very low (good savings), and the next best is much higher (worse savings), Regret is high.
    #    This prioritizes nodes that are significantly better than the next best option.
    
    if n > 1:
        # costs is n x n matrix? No, we need cost for each i relative to alternatives.
        # Actually, the "cost" is just the scalar savings_score for each node.
        # Let's use the savings_scores array.
        # For each i, we want the max savings among j != i.
        # savings_scores[i] is the score for i.
        # best_alt[i] = max(savings_scores[j] for j != i)
        
        # To do this efficiently:
        # Find global max and second max of savings_scores.
        # If savings_scores[i] is the global max, best_alt[i] is the second max.
        # Else, best_alt[i] is the global max.
        
        sorted_indices = np.argsort(savings_scores)[::-1]
        max_val = savings_scores[sorted_indices[0]]
        second_max_val = savings_scores[sorted_indices[1]] if n > 1 else -np.inf
        
        # For each i:
        # if savings_scores[i] == max_val (and it's the first one), best_alt = second_max_val
        # else best_alt = max_val
        
        # Handle ties carefully.
        # If there are multiple nodes with max_val, then for any of them, best_alt is still max_val (from another node).
        # So regret would be 0 for all of them?
        
        # Let's compute regret:
        regret_scores = np.zeros(n)
        
        # We can vectorize this slightly.
        # Create a mask for nodes that are the unique global maximum.
        # If unique max, best_alt for that node is second_max.
        # For all others, best_alt is max.
        
        # Simpler loop for clarity and small n (n is usually < 50, but feasible set can be up to 50).
        # 50 is small enough for a Python loop? Maybe, but numpy is better.
        
        # Let's use numpy.
        # max_indices = np.where(savings_scores == max_val)[0]
        # If len(max_indices) == 1:
        #    regret[max_indices[0]] = max_val - second_max_val
        #    regret[others] = max_val - savings_scores[others]
        # Else:
        #    regret[max_indices] = 0  # (max_val - max_val)
        #    regret[others] = max_val - savings_scores[others]
        
        max_mask = (savings_scores == max_val)
        num_max = np.sum(max_mask)
        
        if num_max == 1:
            # The unique maximum has best_alt = second_max
            regret_scores[max_mask] = max_val - second_max_val
            regret_scores[~max_mask] = max_val - savings_scores[~max_mask]
        else:
            # Multiple maximums, best_alt for max nodes is max_val (from another max node)
            regret_scores[max_mask] = 0.0
            regret_scores[~max_mask] = max_val - savings_scores[~max_mask]
        
        # We want to maximize regret?
        # High regret means "I'm much better than the next best, so pick me".
        
        # Combine with savings?
        # The directive says "prioritizing nodes that are uniquely well-positioned".
        # So we should weight regret heavily.
        
        # Let's try a combined score:
        # final_score = alpha * savings_scores + (1-alpha) * regret_scores
        # Or just regret?
        # "Use a regret-based selection that scores each feasible customer by the difference..."
        # This implies the score IS the regret.
        
        # Let's try pure regret first, or a blend.
        # Given savings worked well (1.082), and regret is a refinement, let's blend.
        # If regret is high, it means we have a clear winner.
        
        # Normalize regret to same scale as savings?
        # Savings can be negative or positive. Regret is non-negative.
        
        # Let's try: final_score = savings_scores + 0.5 * regret_scores
        # Or maybe just regret_scores if it's a strong signal.
        
        # Let's look at the "2-step lookahead" inspiration.
        # Lookahead: if we pick i, what is the best next move j?
        # Cost(i) = dist(current, i) + min_j (dist(i, j) - dist(depot, j)) ?
        # This is expensive (n^2). n=50 is 2500 ops, which is fine.
        
        # Let's implement a lightweight lookahead for the top K candidates.
        # K = 5 or 10.
        
        top_k = min(5, n)
        top_indices = sorted_indices[:top_k]
        
        # For each top candidate i, estimate the cost of the next step.
        # We need to know which nodes are still feasible after picking i.
        # rest_capacity' = rest_capacity - demands[i]
        
        lookahead_scores = np.zeros(n)
        
        for idx in top_indices:
            node_i = feasible_nodes[idx]
            new_rest = rest_capacity - demands[node_i]
            
            # Find feasible nodes after picking i
            # We can reuse feasible_nodes, but need to exclude i and check capacity.
            # Since n is small, let's just iterate.
            
            best_next_score = -np.inf
            for jdx in range(n):
                if jdx == idx:
                    continue
                node_j = feasible_nodes[jdx]
                if demands[node_j] <= new_rest:
                    # Score for j given current=i
                    # Savings for j from i: dist(depot, j) - dist(i, j)
                    # We want to maximize this.
                    s_j = distance_matrix[depot, node_j] - distance_matrix[node_i, node_j]
                    if s_j > best_next_score:
                        best_next_score = s_j
            
            # If no next node, best_next_score is -inf (or 0 if we close route)
            # If we close route, cost is dist(i, depot).
            # But the score we are maximizing is "savings".
            # Let's define the lookahead cost as:
            # Immediate cost: dist(current, i)
            # Future cost: min_j (dist(i, j) + ...)
            
            # Let's stick to the savings formulation.
            # Total savings for path current->i->j:
            # (dist(depot, i) - dist(current, i)) + (dist(depot, j) - dist(i, j))
            # We want to maximize this.
            
            immediate_saving = dist_from_depot[idx] - dist_from_current[idx]
            
            if best_next_score == -np.inf:
                # No next node, just return to depot?
                # The "saving" of closing the route isn't a savings in the same sense.
                # Let's just use the immediate saving.
                total_lookahead_score = immediate_saving
            else:
                total_lookahead_score = immediate_saving + best_next_score
            
            lookahead_scores[idx] = total_lookahead_score
        
        # For nodes not in top_k, use their immediate savings (or 0)
        # Let's set them to their immediate savings so they can still be picked if regret is high?
        # Or set them to -inf so we only pick from top_k?
        # The directive says "refine the savings choice", implying we still use savings.
        # Let's use lookahead for top_k, and immediate savings for others.
        
        # Combine regret and lookahead?
        # This is getting complex.
        # Let's try a simpler combination:
        # Score = Savings + Regret + Lookahead_Bonus
        
        # Let's just use the Regret score as the primary driver for the "uniqueness" aspect,
        # and add a small lookahead bonus for the top candidates.
        
        # Actually, the directive specifically asks for "Regret-based selection".
        # Let's try a weighted sum:
        # final_score = savings_scores + regret_scores
        
        # Let's test this.
        
        final_scores = savings_scores + regret_scores
        
        # Add lookahead bonus for top candidates
        # Normalize lookahead scores?
        # Let's just add it directly.
        final_scores[top_indices] += lookahead_scores[top_indices] * 0.5 # Weight for lookahead
        
        best_idx = np.argmax(final_scores)
        
    else:
        # Only one feasible node
        best_idx = 0
    
    next_node = feasible_nodes[best_idx]
    
    if cache_key is not None:
        if len(_score_cache) > _CACHE_MAX:
            _score_cache.clear()
        _score_cache[cache_key] = (next_node, final_scores if n > 1 else savings_scores)
    
    return int(next_node)

# EVOLVE-BLOCK-END
