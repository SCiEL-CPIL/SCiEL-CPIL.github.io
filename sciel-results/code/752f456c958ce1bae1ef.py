"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    Uses a combination of local structural features and iterative local search
    to directly optimize the greedy decoder's output.
    """
    # Ensure adjacency is a numpy array
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    # Precompute neighbor lists for fast vectorized deactivation
    neighbors_list = []
    for i in range(n):
        neighbors_list.append(np.where(adj[i])[0])
    
    # Compute degree
    degree = np.array([len(neighbors_list[i]) for i in range(n)], dtype=float)
    
    # Compute neighbor-degree-sum (opportunity cost signal)
    # For each vertex i, sum of degrees of its neighbors
    neighbor_degree_sum = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            neighbor_degree_sum[i] = np.sum(degree[neighbors_list[i]])
    
    # Compute Conflict Radius: number of edges among neighbors
    # This measures local clustering density.
    conflict_radius = np.zeros(n)
    try:
        if time.time() < deadline:
            A_sq = adj.astype(float) @ adj.T
            # Element-wise product of adj and A_sq, then sum over neighbors
            # adj[v, u] is 1 if u is neighbor of v
            # A_sq[v, u] is common neighbors
            # Sum gives 2 * conflict_radius
            conflict_radius = 0.5 * (adj.astype(float) * A_sq).sum(axis=1)
        else:
            conflict_radius = np.zeros_like(degree)
    except Exception:
        conflict_radius = np.zeros_like(degree)

    # Normalize degree to [0, 1]
    dmin = degree.min()
    dmax = degree.max()
    if dmax > dmin:
        degree_norm = (degree - dmin) / (dmax - dmin)
    else:
        degree_norm = np.zeros_like(degree)

    # Normalize conflict radius to [0, 1]
    cr_min = conflict_radius.min()
    cr_max = conflict_radius.max()
    if cr_max > cr_min:
        cr_norm = (conflict_radius - cr_min) / (cr_max - cr_min)
    else:
        cr_norm = np.zeros_like(conflict_radius)

    # Normalize neighbor degree sum to [0, 1]
    nds_min = neighbor_degree_sum.min()
    nds_max = neighbor_degree_sum.max()
    if nds_max > nds_min:
        nds_norm = (neighbor_degree_sum - nds_min) / (nds_max - nds_min)
    else:
        nds_norm = np.zeros_like(neighbor_degree_sum)

    # Composite score:
    # Directive: prioritize lowest conflict radius.
    # Also low degree is generally beneficial for MIS.
    # Penalize being connected to high-degree nodes (hubs).
    # We negate because higher score is selected first.
    w1 = 1.0
    w2 = 0.5
    w3 = 0.3

    score = -w1 * degree_norm - w2 * cr_norm - w3 * nds_norm

    # Fast Simulation Function
    def simulate(s):
        """
        Simulates the greedy decoder and returns the size of the independent set.
        """
        order = np.argsort(-s) # Descending score
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                # Deactivate neighbors
                if len(neighbors_list[v]) > 0:
                    active[neighbors_list[v]] = False
        return count

    # Local Search
    best_size = simulate(score)
    
    # Time budget: use remaining time for search
    max_time = min(10.0, deadline - time.time())
    if max_time <= 0:
        return score
        
    rng = np.random.default_rng(42)
    
    # We will perform random swaps.
    # To improve convergence, we can bias swaps towards vertices that are close in rank.
    # However, random swaps over the whole set often work well for N=200.
    
    while time.time() - start_time < start_time + max_time:
        # Pick two random distinct vertices
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Swap scores
        score[i], score[j] = score[j], score[i]
        
        # Evaluate
        new_size = simulate(score)
        
        if new_size > best_size:
            best_size = new_size
        else:
            # Revert
            score[i], score[j] = score[j], score[i]
        
    return score
# EVOLVE-BLOCK-END
