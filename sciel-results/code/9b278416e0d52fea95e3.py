"""Rewritten program: TSP constructive heuristic with adaptive 3-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using an adaptive 3-step lookahead.
    
    The cost for candidate c is approximated as:
    d(current, c) + min_{b != c} ( d(c, b) + min_{a != b, c} ( d(b, a) + gamma * d(a, dest) ) )
    
    Gamma is adaptive: it starts at 0 and increases as the number of unvisited nodes decreases,
    forcing the heuristic to consider the distance to the final destination more strongly 
    when only a few cities remain.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # Pick the one closer to current, but also consider the return to dest?
        # With 2 nodes left, we go current -> c1 -> c2 -> dest (or c2 -> c1 -> dest).
        # Cost 1: d(cur, c1) + d(c1, c2) + d(c2, dest)
        # Cost 2: d(cur, c2) + d(c2, c1) + d(c1, dest)
        # Since d(c1, c2) == d(c2, c1), we just compare:
        # d(cur, c1) + d(c2, dest) vs d(cur, c2) + d(c1, dest)
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c1, destination_node]
        return c1 if cost1 < cost2 else c2

    # Check deadline early
    if _is_deadline_passed():
        # Fallback to pure nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])

    # Adaptive gamma
    # Start with 0, increase as we get closer to the end
    # For 50 cities, maybe start increasing after 30 cities are visited (20 left)
    # gamma = max(0, 1.0 - n_unvisited / 25.0)
    gamma = max(0.0, 1.0 - (n_unvisited / 25.0))
    
    # Precompute distances from all unvisited to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Submatrix of distances among unvisited nodes
    # shape (n_unvisited, n_unvisited)
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    best_total_cost = np.inf
    best_candidate_idx = 0
    
    # Iterate over each candidate c
    for i in range(n_unvisited):
        # c is unvisited_nodes[i]
        
        # We need min over b != i of:
        # d(c, b) + min over a != b, i of ( d(b, a) + gamma * d(a, dest) )
        
        # Let's create a modified submatrix for the "inner min"
        # For a fixed b, we want min_a ( d(b, a) + gamma * d(a, dest) ) subject to a != b and a != i.
        
        # Construct matrix M where M[b, a] = d(b, a) + gamma * d(a, dest)
        # Note: d(b, a) is submatrix[b, a]. d(a, dest) is dists_to_dest[a].
        # So M = submatrix + gamma * dists_to_dest[np.newaxis, :]
        
        # We exclude candidate i from being 'a' or 'b' in the subsequent steps?
        # Yes, 'b' is chosen from U \ {c}, so b != i.
        # 'a' is chosen from U \ {c, b}, so a != i and a != b.
        
        # To do this efficiently:
        # 1. Create M = submatrix + gamma * dists_to_dest[np.newaxis, :]
        # 2. Set diagonal of M to inf (so a != b)
        # 3. Set column i of M to inf (so a != i, since 'a' is the column index in M[b,a])
        #    Wait, M[b, a] uses row b and col a.
        #    We are iterating over b (which is not i).
        #    For each b, we look at row M[b, :].
        #    We need to exclude a=i. So set M[:, i] = inf.
        #    We need to exclude a=b. So set M[b, b] = inf (diagonal).
        
        # Let's prepare M once outside the loop if possible?
        # M depends on gamma, which is constant for this call.
        # M = submatrix + gamma * dists_to_dest[np.newaxis, :]
        # We need to exclude i from 'a' and 'b'.
        # 'b' is the outer loop variable in the min. We handle b != i by skipping i in the outer min?
        # No, the outer min is over b. We can just compute the vector of costs for all b, then ignore b=i.
        
        # Let's compute the inner min for all b at once.
        # InnerMin[b] = min_a ( M[b, a] ) with constraints a != b and a != i.
        
        # Temporarily modify M? Or use masking?
        # Since n is small (50), we can just do this inside the loop for clarity and safety, 
        # or precompute a version of M that has the diagonal as inf.
        
        # Let's precompute M_base = submatrix + gamma * dists_to_dest[np.newaxis, :]
        # And set diagonal to inf.
        
        # But we also need to exclude a=i.
        # So for each i, we need to set column i to inf in a copy?
        # That's O(n^2) per i, so O(n^3) total. 50^3 = 125,000. Very fast.
        
        # Optimization: 
        # Let's precompute M_base with diagonal inf.
        M_base = submatrix + gamma * dists_to_dest[np.newaxis, :]
        np.fill_diagonal(M_base, np.inf)
        
        # For candidate i:
        # We need to consider b in 0..n-1, b != i.
        # For each such b, we need min_a ( M_base[b, a] ) where a != i.
        # This is equivalent to min( min_{a != i} M_base[b, a] ).
        # We can compute this by taking the min of M_base[b, :] but ignoring index i.
        
        # To do this vectorized for all b:
        # Create a copy of M_base, set column i to inf, then take min along axis 1.
        # This gives a vector inner_mins of shape (n,).
        # inner_mins[b] = min_{a != i, a != b} ( d(b, a) + gamma * d(a, dest) ).
        
        M_for_i = M_base.copy()
        M_for_i[:, i] = np.inf  # Exclude a = i
        
        # Now min over a is just min along axis 1
        inner_mins = np.min(M_for_i, axis=1)
        # inner_mins[b] is the cost of the "next-next" step from b, given we came from i.
        # Note: inner_mins[i] will be inf because M_for_i[i, :] has diagonal inf and col i inf?
        # Actually, row i of M_for_i has M_base[i, j] for j!=i. But we are not using b=i.
        
        # Now we need to compute the cost for each b:
        # cost_b = d(c, b) + inner_mins[b]
        # d(c, b) is submatrix[i, b].
        
        step1_costs = submatrix[i, :] + inner_mins
        
        # We must exclude b = i.
        # Set step1_costs[i] = inf.
        step1_costs[i] = np.inf
        
        # The best continuation cost from c is min(step1_costs).
        best_cont_cost = np.min(step1_costs)
        
        # Total cost for candidate i
        total_cost = dists_from_current[i] + best_cont_cost
        
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_candidate_idx = i
            
    return int(unvisited_nodes[best_candidate_idx])

# EVOLVE-BLOCK-END
