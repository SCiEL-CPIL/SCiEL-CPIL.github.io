import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 145  # Safety margin for 150s limit

def _mst_cost(nodes, dist_matrix):
    """
    Compute the approximate MST cost for a set of nodes using Prim's algorithm.
    This is O(n^2) which is fast for n <= 50.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Start with the first node
    in_mst = [False] * n
    in_mst[0] = True
    min_edge = [dist_matrix[nodes[0], nodes[i]] for i in range(n)]
    
    total_cost = 0.0
    for _ in range(n - 1):
        # Find the minimum edge connecting an MST node to a non-MST node
        min_val = float('inf')
        min_idx = -1
        for i in range(n):
            if not in_mst[i] and min_edge[i] < min_val:
                min_val = min_edge[i]
                min_idx = i
        
        if min_idx == -1:
            break
        
        total_cost += min_val
        in_mst[min_idx] = True
        
        # Update min_edge for the newly added node
        for j in range(n):
            if not in_mst[j]:
                d = dist_matrix[nodes[min_idx], nodes[j]]
                if d < min_edge[j]:
                    min_edge[j] = d
                    
    return total_cost

def _estimate_remaining_cost(current, unvisited, destination, dist_matrix):
    """
    Estimate the cost to complete the tour from 'current' visiting all 'unvisited' and returning to 'destination'.
    Uses a combination of nearest-neighbor path cost and MST-based structural cost.
    """
    if len(unvisited) == 0:
        return dist_matrix[current, destination]
    
    # 1. Nearest Neighbor path cost
    remaining = list(unvisited)
    nn_cost = 0.0
    curr = current
    
    while remaining:
        dists = dist_matrix[curr, remaining]
        min_idx = np.argmin(dists)
        next_node = remaining[min_idx]
        nn_cost += dists[min_idx]
        curr = next_node
        remaining.pop(min_idx)
        
    nn_cost += dist_matrix[curr, destination]
    
    # 2. MST cost for the unvisited nodes
    # The MST is a lower bound on the TSP tour length for that subset.
    # We scale it slightly or combine it to capture structural scattering.
    # A common heuristic is that TSP length is roughly 1.25-1.5 * MST for random points.
    # But here we just use it as a structural penalty.
    # If the MST is high, the points are scattered, and a local choice might be bad.
    
    # To keep it fast and simple, let's just return the NN cost for now, 
    # but the directive suggests using MST. 
    # Let's try a weighted combination. 
    # However, calculating MST for every candidate (50 times) might be expensive if not careful.
    # n=50, MST is O(50^2) = 2500 ops. 50 * 2500 = 125k ops. This is very fast.
    
    mst_c = _mst_cost(unvisited, dist_matrix)
    
    # Combine: The NN cost is a specific path, MST is a lower bound.
    # If we choose a node that keeps the remaining set "compact" (low MST), that's good.
    # But the MST of the *remaining* set after removing a node is what matters.
    # So _estimate_remaining_cost should probably be used in the scoring of candidates,
    # where we compute the MST of the set *excluding* the candidate.
    
    # For this function which is called with a specific 'unvisited' set, 
    # let's just return the NN cost as the primary estimate, 
    # as the structural penalty is better applied in the candidate selection loop.
    return nn_cost + 0.1 * mst_c

def _held_karp_next(current, unvisited, destination, dist_matrix):
    """
    Exact solution for the next node when the number of unvisited nodes is small.
    Uses Held-Karp DP to find the shortest path from 'current' visiting all 'unvisited' and ending at 'destination'.
    Returns the first node in that optimal path.
    """
    n = len(unvisited)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited[0])
    
    # dp[mask][i] = min cost to start at current, visit set 'mask', and end at node 'i' (index in unvisited)
    size = 1 << n
    INF = float('inf')
    dp = [[INF] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]
    
    # Base case: visit only node i
    for i in range(n):
        dp[1 << i][i] = dist_matrix[current, unvisited[i]]
        
    # Fill DP
    for mask in range(size):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == INF:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist_matrix[unvisited[i], unvisited[j]]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    parent[new_mask][j] = i
                    
    # Find the best full path ending at destination
    full_mask = size - 1
    best_end = -1
    best_cost = INF
    for i in range(n):
        total = dp[full_mask][i] + dist_matrix[unvisited[i], destination]
        if total < best_cost:
            best_cost = total
            best_end = i
            
    # Backtrace to find the first node
    cur_mask = full_mask
    cur_node = best_end
    while (cur_mask & (cur_mask - 1)) != 0: # while mask has more than 1 bit
        prev_node = parent[cur_mask][cur_node]
        cur_mask = cur_mask & ~(1 << cur_node)
        cur_node = prev_node
        
    return int(unvisited[cur_node])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Time check
    if time.time() > _DEADLINE - 1.0:
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    # Use exact Held-Karp for small subproblems
    if n_unvisited <= 12:
        return _held_karp_next(current_node, unvisited_nodes, destination_node, distance_matrix)
    
    # For larger n_unvisited, use a hybrid approach with MST-based global structural scoring
    # 1. Compute distances
    dist_current_to_v = distance_matrix[current_node, unvisited_nodes]
    dist_v_to_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    dist_v_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 2. Fast local score: 2-step lookahead
    M_temp = dist_v_to_unvisited + dist_v_to_dest[None, :]
    np.fill_diagonal(M_temp, np.inf)
    best_cost_2step = np.min(M_temp, axis=1)
    quick_score = dist_current_to_v + best_cost_2step
    
    # 3. Select top K candidates for global structural evaluation
    # K should be small enough to allow MST calculation on the remaining sets
    # MST is O(n^2). If n=50, and we check 10 candidates, each remaining set has 49 nodes.
    # 10 * (49^2) ~ 24k ops. Very fast.
    K = min(10, n_unvisited)
    sorted_indices = np.argsort(quick_score)
    candidate_indices = sorted_indices[:K]
    
    best_candidate_node = -1
    best_total_score = float('inf')
    
    for idx in candidate_indices:
        node = unvisited_nodes[idx]
        
        # Cost to go from current to this node
        cost_to_node = dist_current_to_v[idx]
        
        # Remaining nodes after visiting this node
        # Create a list of remaining nodes excluding this one
        # We need to be careful with numpy array slicing to avoid copies if possible, 
        # but for n=50, copies are fine.
        
        # Build the remaining set
        remaining_mask = np.ones(n_unvisited, dtype=bool)
        remaining_mask[idx] = False
        remaining_nodes = unvisited_nodes[remaining_mask]
        
        if len(remaining_nodes) == 0:
            total_cost = cost_to_node + distance_matrix[node, destination_node]
        else:
            # Estimate remaining cost using NN + MST structural penalty
            # 1. NN cost from node through remaining_nodes to destination
            nn_cost = 0.0
            curr = node
            rem_list = list(remaining_nodes)
            while rem_list:
                dists = distance_matrix[curr, rem_list]
                min_idx = np.argmin(dists)
                next_node = rem_list[min_idx]
                nn_cost += dists[min_idx]
                curr = next_node
                rem_list.pop(min_idx)
            nn_cost += distance_matrix[curr, destination_node]
            
            # 2. MST cost of remaining nodes
            # This captures if the remaining set is scattered
            mst_cost = _mst_cost(remaining_nodes, distance_matrix)
            
            # Combine: We want to minimize the total path cost.
            # The NN cost is a specific realization. The MST is a lower bound.
            # A good score might be: nn_cost + alpha * mst_cost?
            # Or just nn_cost, but the directive says "estimates the remaining cost via a fast MST approximation".
            # Perhaps the idea is that NN is myopic, so we add a penalty proportional to the MST of the remainder?
            # If the remainder is scattered (high MST), the NN path might be bad?
            # Actually, for random uniform points, TSP ~ 1.25 * MST.
            # So a better estimate of the remaining TSP tour length is max(nn_cost, 1.25 * mst_cost)?
            # Or a weighted sum?
            # Let's try: estimated_remaining = nn_cost + 0.1 * mst_cost
            # This penalizes choices that leave a scattered set.
            
            estimated_remaining = nn_cost + 0.05 * mst_cost
            
            total_cost = cost_to_node + estimated_remaining
        
        if total_cost < best_total_score:
            best_total_score = total_cost
            best_candidate_node = node
    
    if best_candidate_node != -1:
        return int(best_candidate_node)
    else:
        # Fallback
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
