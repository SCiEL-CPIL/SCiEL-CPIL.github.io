# EVOLVE-BLOCK-START
import numpy as np
import time

# We use a module-level cache to store MST-based residual estimates for subsets of unvisited nodes.
# However, since the function is called sequentially to build a tour, the set of unvisited nodes shrinks by 1 each time.
# We can exploit this. But the interface doesn't allow passing state.
# So we must compute efficiently inside the function.

# Key insight: The problem is to build a TSP tour. The evaluator uses "mean closed-tour length".
# The function `select_next_node` is called N-1 times to pick the next node.
# The final step (returning to destination_node) is likely handled by the evaluator or is implicit in the score calculation.
# Actually, the signature includes `destination_node`. In standard TSP, the tour is a cycle.
# Usually, you start at node 0, visit all others, and return to 0.
# Here, `destination_node` is passed. It might be the start node (to which we must return) or a specific target.
# Given "closed-tour length", it's a cycle.
# If `destination_node` is the start node, the last call to `select_next_node` will have `len(unvisited_nodes) == 1`.
# The return value is the last node. The evaluator then adds distance(last_node, destination_node) to close the loop.

# Strategy:
# 1. Compute distances from `current_node` to all `unvisited_nodes`.
# 2. Estimate the "cost to finish" from each candidate `u` in `unvisited_nodes`.
#    If we pick `u`, the remaining problem is a TSP on `unvisited_nodes \ {u}` starting from `u` and ending at `destination_node`.
#    A good lower bound or heuristic estimate for the remaining path is the sum of the Minimum Spanning Tree (MST) of the remaining nodes,
#    plus the cost to connect the start (`u`) and end (`destination_node`) to this tree.
#    Specifically, for a path from `u` to `dest` visiting `S'`, the cost is roughly MST(S' union {u, dest}) - (some adjustment) or simply MST(S') + min_dist(u, S') + min_dist(dest, S')?
#    Actually, a better estimate for the remaining tour cost given we are at `u` and must visit `S'` and return to `dest` (if dest is start) or go to `dest`:
#    Let's assume `destination_node` is the node we must return to (the start).
#    Remaining set `S' = unvisited_nodes \ {u}`.
#    We need a path from `u` to `dest` visiting all nodes in `S'`.
#    Approximation: MST of `S' union {u, dest}`. This is a lower bound on the path length.
#    So, `total_cost_estimate(u) = dist(current_node, u) + MST(S' union {u, dest})`.
#    We want to minimize this.

# Complexity:
# N=50.
# Loop N times.
# Inside each loop, we have ~N candidates.
# For each candidate, we compute MST of ~N nodes.
# MST via Prim's on N=50 nodes is O(N^2) ~ 2500 ops.
# 50 candidates * 2500 ops * 50 steps = 6.25 million ops.
# This is very fast in Python/NumPy.

# Optimization:
# We can precompute the MST of the *entire* unvisited set + current + dest? No, it changes.
# We can use a fast MST implementation.

# Let's implement a helper to compute MST cost of a set of nodes given a distance matrix.
# Since N is small, we can just use a simple O(N^2) Prim's algorithm.

def _mst_cost(nodes, dist_matrix):
    """
    Compute the cost of the Minimum Spanning Tree of the subgraph induced by `nodes`.
    nodes: array of node indices.
    dist_matrix: full NxN distance matrix.
    Returns: MST cost (float).
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Map original indices to 0..n-1 for local Prim
    # We need the subgraph distances.
    # Since N is small, we can just run Prim on the original indices using the full matrix.
    
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    total_cost = 0.0
    
    for _ in range(n):
        # Find the vertex with the minimum distance not in MST
        # We can do this in O(n)
        min_val = np.inf
        min_idx = -1
        for i in range(n):
            if not in_mst[i] and min_dist[i] < min_val:
                min_val = min_dist[i]
                min_idx = i
        
        if min_idx == -1:
            break # Disconnected? Should not happen in Euclidean.
            
        in_mst[min_idx] = True
        total_cost += min_val
        
        # Update distances
        # Neighbor in original graph: nodes[min_idx]
        orig_idx = nodes[min_idx]
        for j in range(n):
            if not in_mst[j]:
                orig_j = nodes[j]
                d = dist_matrix[orig_idx, orig_j]
                if d < min_dist[j]:
                    min_dist[j] = d
                    
    return total_cost

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    # Edge case: only one node left
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If no nodes left (should not happen based on signature)
    if n_unvisited == 0:
        return -1 # Fallback, though invalid
    
    # Get distances from current_node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    best_score = np.inf
    best_node = int(unvisited_nodes[0])
    
    # Iterate over each candidate next node
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # Immediate cost
        immediate_cost = dists_from_current[i]
        
        # Remaining nodes after picking candidate
        # We need to visit all other unvisited nodes, starting from `candidate` and ending at `destination_node`.
        # The set of nodes to include in the MST estimate is:
        # 1. All other unvisited nodes (unvisited_nodes excluding candidate)
        # 2. The candidate itself (start of the remaining path)
        # 3. The destination_node (end of the remaining path)
        
        # Construct the node list for MST
        # Note: candidate is already in unvisited_nodes, so we just take all unvisited_nodes.
        # The MST will connect all unvisited nodes, plus we need to ensure destination_node is included if it's not already in unvisited_nodes.
        # Wait, destination_node is the start node. It is NOT in unvisited_nodes (since we've left it).
        # So the set for MST is: unvisited_nodes (which includes candidate) + destination_node.
        
        # Actually, if we pick `candidate`, the remaining path is:
        # candidate -> (visit all other unvisited nodes) -> destination_node.
        # The MST of {candidate, other_unvisited, destination_node} is a lower bound on the length of this path.
        
        # Let's form the array of nodes for the MST.
        # unvisited_nodes contains candidate and all others.
        # We need to add destination_node to this set.
        
        # Optimization: If destination_node is already in unvisited_nodes? No, it's the start, so it's visited.
        # So we must add it.
        
        # However, creating a new array every time is overhead.
        # We can just pass the list of indices to the MST function.
        
        # The set of nodes is: unvisited_nodes + [destination_node]
        # But wait, if we pick candidate, the "start" of the remaining path is candidate.
        # The MST connects all these nodes.
        # Is MST({unvisited_nodes} U {destination_node}) a good estimate?
        # Yes, it's the minimum cost to connect all these nodes. A path is a connected subgraph, so MST <= Path.
        
        # To make it a better estimate of the *path* specifically (which has 2 endpoints), 
        # a common heuristic is MST + (sum of 2 smallest incident edges to endpoints?) or just MST.
        # Let's stick with MST of the set {unvisited_nodes, destination_node}.
        
        # But we must be careful: the MST is computed on the subgraph induced by these nodes.
        
        # Let's optimize the MST calculation.
        # Since the set of nodes changes by only 1 element (removing the candidate from the "start" perspective? No, the set of nodes to visit is fixed: all unvisited. 
        # The difference is that `candidate` is the START of the remaining path, and `destination_node` is the END.
        # The set of nodes involved in the remaining path is always `unvisited_nodes` U `{destination_node}`.
        # The MST of this fixed set is the SAME for all candidates!
        
        # Wait.
        # If I pick candidate A, the remaining path starts at A.
        # If I pick candidate B, the remaining path starts at B.
        # The set of nodes to visit is the same: `unvisited_nodes`.
        # The destination is the same: `destination_node`.
        # So the set of nodes for the MST is `unvisited_nodes` U `{destination_node}` in BOTH cases.
        # Therefore, the MST cost is IDENTICAL for all candidates.
        
        # If the MST cost is constant, it doesn't help distinguish between candidates!
        # We need an estimate that depends on `candidate`.
        
        # The cost of the remaining path is not just the MST.
        # A better estimate:
        # Cost(path from candidate to dest visiting S) >= MST(S U {candidate, dest}) - (something)?
        # Actually, the MST of S U {dest} is constant.
        # But the path MUST start at `candidate`.
        # So we can estimate the cost as:
        # min_cost_to_connect_candidate_to_MST + MST(S U {dest})?
        # No, that's double counting or inaccurate.
        
        # Let's use a different heuristic:
        # Estimate = dist(current, candidate) + min_dist(candidate, S \ {candidate}) + ... ?
        
        # Alternative: 1-Opt or 2-Opt lookahead?
        # Or: dist(current, candidate) + dist(candidate, nearest_unvisited) + ...
        
        # Let's go back to the "Regret-based" insight from the prompt.
        # "Regret-based selection prioritizes nodes that become expensive to insert later".
        
        # Let's try a simpler, effective heuristic:
        # Score(u) = dist(current, u) + lambda * (min_dist(u, unvisited \ {u}) + min_dist(u, destination_node))
        # This encourages picking nodes that are close to the current node AND well-connected to the rest (low min_dist to others).
        
        # Let's refine:
        # For each candidate u:
        # 1. dist_current = dist(current, u)
        # 2. dist_to_rest = min(dist(u, v) for v in unvisited if v != u)
        # 3. dist_to_dest = dist(u, destination_node)
        
        # Score = dist_current + alpha * dist_to_rest + beta * dist_to_dest
        
        # This is O(N^2) per step. Total O(N^3) = 125,000 ops. Very fast.
        
        # Let's implement this.
        
        # Precompute min_dist to rest for all unvisited nodes at once?
        # dist_matrix[unvisited_nodes, unvisited_nodes] gives an N_unvisited x N_unvisited matrix.
        # We can find the min of each row, excluding the diagonal.
        
    # Precomputation for this step
    # Submatrix of distances between unvisited nodes
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Find min distance to any other unvisited node for each node
    # Set diagonal to inf to exclude self
    np.fill_diagonal(sub_dist, np.inf)
    min_dist_to_rest = np.min(sub_dist, axis=1)
    
    # Distances to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Weights
    # Let's try alpha=1.0, beta=0.5 initially.
    # We can tune these, but let's start with a balanced approach.
    # The immediate cost is the most important.
    
    scores = dists_from_current + 1.0 * min_dist_to_rest + 0.5 * dist_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
