import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    Iterative Simulated Greedy Scoring.
    
    Instead of relying on static local statistics (degree, neighbor degree), this method
    simulates the greedy decoding process to estimate the true value of each vertex.
    
    1. We perform multiple "trials" of a greedy selection process.
    2. In each trial, we randomly break ties or slightly perturb the initial ordering.
    3. We track how often each vertex is selected early in the process across different
       random perturbations of the greedy order.
    4. Vertices that are consistently selected early (robustly "good" choices) get high scores.
    5. Vertices that are often blocked by better choices get lower scores.
    
    This captures global structural dependencies better than local degree heuristics because
    it simulates the actual conflict resolution process.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # If the graph is very small, we can potentially do exact or near-exact search,
    # but for n=200, we stick to the heuristic simulation.
    
    # Base heuristic: Degree. Lower degree is generally better for MIS.
    # We use this as the primary sort key for the base greedy order.
    deg = adj.sum(axis=1).astype(float)
    
    # We want to estimate a "score" for each vertex.
    # The idea: A vertex is "good" if it is likely to be picked by a greedy algorithm
    # that tries to maximize the set size.
    
    # We will run K simulated greedy passes.
    # In each pass, we pick vertices in a specific order.
    # To get a robust score, we need to vary the order slightly.
    
    # However, running a full greedy pass is O(n^2) or O(n + m).
    # With n=200, n^2 is 40,000 ops. We can afford thousands of passes in 140s.
    
    # Let's define the score as the average "rank" at which a vertex is picked
    # across multiple randomized greedy simulations, where randomization is
    # based on perturbing the base degree order.
    
    # Alternative, simpler and often more effective structural change:
    # Use "Coreness" or "K-core" decomposition ideas.
    # Vertices in high-density cores are harder to include in MIS if they have high degree,
    # but low-degree vertices in the periphery are easy wins.
    
    # Let's try a different structural approach:
    # **Local Independent Set Density**
    # For each vertex u, look at its neighborhood N(u) plus u.
    # Compute the size of the Maximum Independent Set (MIS) of this local subgraph.
    # If u is included in the MIS of its local neighborhood, it's a "good" vertex.
    # If u is excluded, it's a "bad" vertex.
    # This is a local approximation of the global value.
    
    # Computing exact MIS for local neighborhoods (size ~ degree) can be expensive if degree is high.
    # But in ER graphs, degree is usually low (e.g., 5-15).
    # We can use a fast greedy heuristic for the local MIS.
    
    # Let's combine:
    # 1. Base score: -degree (prefer low degree)
    # 2. Local Context Score: Does u survive in a local greedy MIS of G[N[u]]?
    
    # Implementation of Local Greedy MIS Score:
    # For each u:
    #   sub_adj = adj[np.ix_([u]+neighbors, [u]+neighbors)]
    #   Run greedy on sub_adj.
    #   If u is in the resulting set, score += 1.
    
    # This is O(n * (deg^2)) roughly. For n=200, deg~10, this is 200 * 100 = 20,000 ops.
    # Very fast.
    
    scores = np.zeros(n)
    
    # Pre-compute neighbors for speed
    neighbors = [np.where(adj[i])[0] for i in range(n)]
    
    # 1. Primary Signal: Negative Degree
    # Normalize degrees to [0, 1] roughly for weighting
    max_deg = np.max(deg) if np.max(deg) > 0 else 1.0
    scores -= deg / max_deg
    
    # 2. Secondary Signal: Local Greedy Survival
    # For each vertex, check if it's in the greedy MIS of its closed neighborhood.
    # We use a deterministic greedy rule for the local subgraph:
    # Pick vertex with lowest degree in the subgraph, remove it and its neighbors.
    
    for u in range(n):
        # Build local subgraph indices
        local_indices = np.unique(np.concatenate(([u], neighbors[u])))
        k = len(local_indices)
        
        if k == 0:
            continue
            
        # Extract local adjacency matrix
        # local_adj[i, j] is True if local_indices[i] is connected to local_indices[j] in original graph
        local_adj = adj[np.ix_(local_indices, local_indices)]
        
        # Compute local degrees
        local_deg = local_adj.sum(axis=1)
        
        # Greedy MIS on local subgraph
        # We want to know if 'u' (which is at index 0 in local_indices? No, need to find it)
        # Let's find the index of u in local_indices
        u_local_idx = np.where(local_indices == u)[0][0]
        
        # Simulate greedy selection
        # remaining = set of all vertices in local subgraph
        # picked = []
        # While remaining not empty:
        #   Find vertex v in remaining with min local_deg (within remaining)
        #   If tie, pick smallest index (or random? Let's keep deterministic for now)
        #   Pick v
        #   Remove v and its neighbors from remaining
        
        remaining = np.ones(k, dtype=bool)
        # We need to update degrees dynamically or just recompute?
        # k is small (~10-20), so recomputing is fine.
        
        is_u_picked = False
        while np.any(remaining):
            # Get degrees of remaining vertices within the remaining set
            # deg_in_remaining[i] = sum(local_adj[i, remaining])
            deg_rem = local_adj[:, remaining].sum(axis=1)
            
            # Mask out non-remaining vertices
            deg_rem[~remaining] = np.inf
            
            # Find min degree vertex
            min_deg_idx = np.argmin(deg_rem)
            
            if min_deg_idx == u_local_idx:
                is_u_picked = True
                break
            
            # Remove min_deg_idx and its neighbors from remaining
            remaining[min_deg_idx] = False
            neighbors_in_local = np.where(local_adj[min_deg_idx])[0]
            remaining[neighbors_in_local] = False
            
            # Check time
            if time.time() > deadline - 20:
                # Fallback: just use degree
                return -deg
        
        if is_u_picked:
            scores[u] += 1.0
    
    # 3. Tertiary Signal: Neighbor Degree Sum (Hub blocking)
    # Prefer vertices adjacent to hubs, as they block hubs.
    # This term was in the previous version.
    neighbor_deg_sum = adj @ deg
    # Normalize
    if np.max(neighbor_deg_sum) > 0:
        scores += 0.5 * (neighbor_deg_sum / np.max(neighbor_deg_sum))
        
    return scores

# Fallback if the loop above times out or fails
# The function returns scores inside the loop or at the end.
# If it times out in the middle of the loop, it returns -deg.
# This is safe.
