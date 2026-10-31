# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 125.0  # Leave 25s margin for 150s limit

def _is_timeout():
    return (time.time() - _START_TIME) > _TIMEOUT_LIMIT

def _mst_weight(nodes, distance_matrix):
    """
    Calculate the weight of the Minimum Spanning Tree for the given set of nodes.
    Uses Prim's algorithm.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    dist_to_mst = [float('inf')] * n
    dist_to_mst[0] = 0.0
    visited = [False] * n
    
    total_weight = 0.0
    
    for _ in range(n):
        min_val = float('inf')
        u = -1
        for i in range(n):
            if not visited[i] and dist_to_mst[i] < min_val:
                min_val = dist_to_mst[i]
                u = i
        
        if u == -1:
            break
            
        visited[u] = True
        total_weight += min_val
        
        u_global = nodes[u]
        for v in range(n):
            if not visited[v]:
                v_global = nodes[v]
                d = distance_matrix[u_global, v_global]
                if d < dist_to_mst[v]:
                    dist_to_mst[v] = d
                    
    return total_weight

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour.
    Combines:
    1. MST-based structural cost (from inspirations)
    2. Fiedler vector / Spectral clustering heuristic (from directive)
    3. Exact brute-force tail optimization for small n (from lessons)
    """
    if _is_timeout():
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Exact brute-force for small remaining sets (tail optimization)
    # This ensures the end of the tour is optimal given the start of the tail.
    if n_unvisited <= 5:
        # We need to find the order of visiting unvisited_nodes that minimizes:
        # dist(current, first) + ... + dist(last, destination_node)
        # Since n is small, we can use Held-Karp or simple permutation search?
        # 5! = 120 permutations. Very fast.
        # But we can do better with DP or just simple recursion if n is small.
        # Let's use a simple recursive search with memoization or just itertools.permutations for n<=5.
        
        import itertools
        
        best_cost = float('inf')
        best_first = int(unvisited_nodes[0])
        
        # To avoid overhead of itertools for every call, we can implement a small DP.
        # But n=5 is tiny.
        # State: (mask, last_node) -> min cost to reach last_node from current visiting nodes in mask.
        # We want to end at destination_node, so the path is current -> ... -> last -> destination.
        # Cost = dp[mask][last] + dist(last, destination)
        
        N = n_unvisited
        # Map unvisited_nodes to 0..N-1
        # node_list = unvisited_nodes
        # dp[i][j] where i is bitmask of visited nodes in unvisited set, j is the last node index in unvisited set.
        
        # Initialize dp
        # dp[1 << i][i] = dist(current, unvisited_nodes[i])
        
        # Since N <= 5, max mask is 2^5 = 32.
        # dp is a 32x5 matrix.
        
        dp = np.full((1 << N, N), np.inf)
        
        for i in range(N):
            dp[1 << i][i] = distance_matrix[current_node, unvisited_nodes[i]]
        
        for mask in range(1, 1 << N):
            for last in range(N):
                if not (mask & (1 << last)):
                    continue
                if dp[mask][last] == np.inf:
                    continue
                
                for next_node in range(N):
                    if mask & (1 << next_node):
                        continue
                    
                    new_mask = mask | (1 << next_node)
                    new_cost = dp[mask][last] + distance_matrix[unvisited_nodes[last], unvisited_nodes[next_node]]
                    
                    if new_cost < dp[new_mask][next_node]:
                        dp[new_mask][next_node] = new_cost
        
        # Find the best completion
        full_mask = (1 << N) - 1
        for last in range(N):
            total_cost = dp[full_mask][last] + distance_matrix[unvisited_nodes[last], destination_node]
            if total_cost < best_cost:
                best_cost = total_cost
                # We need to recover the first node.
                # We can store the parent pointers or just re-run to find the first node.
                # Actually, we just need the FIRST node to return.
                # We can reconstruct the path.
                # Let's store parent pointers in a separate array? 
                # Or simpler: since we only return the FIRST node, we can just check which first node leads to the optimal path.
                # Wait, the DP above doesn't track the start.
                # Let's modify the DP to track the start, or just brute force the first node?
                # No, the DP finds the optimal path for a given set of visited nodes ending at `last`.
                # The "first" node is part of the path.
                # We can store `first_node` in the DP state? No, that increases state.
                # Alternative: Run the DP for each possible first node?
                # If we fix the first node `f`, then we start at `f`.
                # dp_f[mask][last] ...
                # Since N<=5, we can just iterate over all permutations if we want to be safe, but DP is better.
                
                # Let's just re-run the logic to find the best first node by checking the optimal path's start.
                # Actually, we can just store the parent pointer `prev[mask][last]` and backtrack to find the start.
                
                # Let's implement a separate helper or just inline the backtracking for the best candidate.
                # To keep it simple and fast for N<=5, let's just use itertools.permutations.
                # 5! = 120. 120 * 5 ops = 600 ops. Very fast.
                pass
        
        # Let's use itertools.permutations for simplicity and correctness for N<=5.
        # It's fast enough.
        for perm in itertools.permutations(unvisited_nodes):
            cost = distance_matrix[current_node, perm[0]]
            for k in range(N - 1):
                cost += distance_matrix[perm[k], perm[k+1]]
            cost += distance_matrix[perm[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_first = perm[0]
                
        return int(best_first)

    # For n_unvisited > 5, use heuristic
    
    # Pre-compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Pre-compute distance matrix between unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Compute Fiedler vector heuristic
    # 1. Build Laplacian L = D - A where A is the adjacency matrix (inverse of distance? or just distance?)
    # Usually spectral clustering uses similarity. Let's use similarity S = 1 / (d + epsilon) or exp(-d^2/sigma^2).
    # Let's use S = 1 / (d + 1e-6) to avoid division by zero.
    # Actually, for TSP, we want to connect clusters.
    # The Fiedler vector of the graph Laplacian helps identify the "cut" or bridge.
    # Nodes with large |Fiedler component| are often on the boundary or in separate clusters.
    # We want to pick a node that bridges the clusters.
    # A common heuristic is to pick the node with the largest absolute value in the Fiedler vector?
    # Or the one that minimizes the cut?
    # Let's compute the Fiedler vector of the unvisited subgraph.
    
    # Similarity matrix
    # To avoid numerical issues with 1/d, let's use a Gaussian kernel: S_ij = exp(-d_ij^2 / sigma^2)
    # sigma can be estimated from the mean distance.
    mean_d = np.mean(d_unvisited)
    if mean_d < 1e-6:
        mean_d = 1.0
    sigma = mean_d
    
    S = np.exp(-(d_unvisited ** 2) / (sigma ** 2))
    np.fill_diagonal(S, 0) # No self-loops
    
    # Degree matrix
    D = np.sum(S, axis=1)
    L = np.diag(D) - S
    
    # Compute eigenvalues and eigenvectors
    # We only need the 2nd smallest eigenvalue's eigenvector (Fiedler vector).
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # eigvecs are columns. The 2nd smallest eigenvalue is index 1.
        fiedler = eigvecs[:, 1]
    except np.linalg.LinAlgError:
        fiedler = np.zeros(n_unvisited)
    
    # Heuristic: Nodes with high absolute Fiedler component are "central" to the split?
    # Actually, in spectral clustering, the Fiedler vector partitions the graph.
    # We want to pick a node that connects the two sides.
    # A node with a value close to 0 in the Fiedler vector is on the boundary?
    # Or a node with a large value is in one cluster?
    # Let's try to minimize the "cut" cost.
    # Alternative: Use the Fiedler vector to estimate the "distance" to the rest of the cluster.
    
    # Let's combine with MST.
    # Score(u) = dist(current, u) + alpha * MST(U \ {u}) + beta * |fiedler[u]|
    
    # Why |fiedler[u]|?
    # If a node is strongly in one cluster (high |fiedler|), it might be "safe" to visit now?
    # Or if it's on the boundary (low |fiedler|), it might be a good bridge?
    # Let's try minimizing the score, so we prefer nodes with LOW |fiedler|?
    # Or maybe we want to visit nodes that are "well-connected" to the rest.
    
    # Let's look at the inspiration 1: it used minimax (max dist to other).
    # Let's stick to the MST term as the primary structural cost, as it is a lower bound.
    # The Fiedler term is a secondary "connectivity" heuristic.
    
    # Let's calculate MST for each candidate.
    best_score = float('inf')
    best_node = int(unvisited_nodes[0])
    
    alpha = 1.0
    beta = 0.1
    
    for i, u in enumerate(unvisited_nodes):
        immediate_cost = dist_from_current[i]
        
        # MST of remaining nodes
        remaining_mask = np.ones(n_unvisited, dtype=bool)
        remaining_mask[i] = False
        remaining_nodes = unvisited_nodes[remaining_mask]
        
        if len(remaining_nodes) == 0:
            mst_w = 0.0
        else:
            mst_w = _mst_weight(remaining_nodes, distance_matrix)
            
        # Fiedler term
        # We want to penalize nodes that are "isolated" or "bridges" that might break connectivity?
        # Actually, if we pick a node, we remove it from the graph.
        # We want the remaining graph to stay connected.
        # Nodes with high degree (in the similarity graph) are good to keep?
        # Let's use the degree from the similarity matrix as a proxy for connectivity.
        # Or just use the Fiedler value.
        # Let's try: Score = immediate + alpha * MST + beta * (1 - degree[i])
        # Higher degree -> lower score.
        
        degree = D[i]
        
        # Normalize degree?
        # Let's just use the raw degree.
        
        total_score = immediate_cost + alpha * mst_w - beta * degree
        
        if total_score < best_score:
            best_score = total_score
            best_node = int(u)
            
    return best_node

# EVOLVE-BLOCK-END
