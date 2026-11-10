# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Graph-theoretic centrality-based heuristic for ant colony orienteering.
    
    Structural change: instead of hand-crafted prize/distance ratios, we model
    the heuristic as a linear combination of graph-theoretic centrality features:
      1. Prize-weighted eigenvector centrality (global structural attractor)
      2. Degree centrality on the feasible-move graph
      3. Betweenness centrality (hub nodes that connect clusters)
      4. Origin-proximity centrality (closeness to start)
    
    Each feature is computed on the full graph, then combined into a per-edge
    heuristic H[i,j] that reflects how attractive destination j is from
    source i, weighted by global structural importance of j.
    
    The key insight: nodes that are structural hubs (high betweenness, high
    eigenvector centrality) should be visited even if locally slightly farther,
    because they provide access to many other high-value nodes.
    """
    start_time = time.time()
    deadline = start_time + 300.0

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Safety check for finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(distance_matrix)):
        return np.ones((n, n), dtype=float) * 1e-9
    
    eps = 1e-9
    
    # --- Feature 1: Prize-weighted eigenvector centrality ---
    # Build a weighted adjacency where edge weight = prize[j] / (d[i,j] + d[j,0])
    # This captures "how valuable is j as a destination from anywhere"
    cost = distance_matrix + distance_matrix[:, 0][None, :]  # d[i,j] + d[j,0]
    cost_safe = np.maximum(cost, eps)
    
    # Weighted adjacency: A[i,j] = prize[j] / cost[i,j]
    # (prize of destination, cost of reaching it)
    A = prizes[None, :] / cost_safe
    np.fill_diagonal(A, 0.0)
    
    # Compute eigenvector centrality via power iteration
    # x_{k+1} = A^T x_k / ||A^T x_k||
    x = np.ones(n) / n
    for _ in range(50):
        x_new = A.T @ x
        norm = np.linalg.norm(x_new)
        if norm < eps:
            break
        x_new /= norm
        if np.max(np.abs(x_new - x)) < 1e-10:
            x = x_new
            break
        x = x_new
    eig_centrality = np.maximum(x, 0.0)
    
    # --- Feature 2: Degree centrality on feasible-move graph ---
    # A node j has high degree centrality if many nodes i can feasibly reach j
    # (i.e., d[i,j] + d[j,0] <= max_len)
    feasible = (cost <= max_len).astype(float)
    np.fill_diagonal(feasible, 0.0)
    degree_cent = feasible.sum(axis=0) / max(n - 1, 1)
    
    # --- Feature 3: Betweenness centrality (approximate via shortest paths) ---
    # Use the distance matrix to compute a simple betweenness measure:
    # For each pair (a, b), node j is "between" if d[a,j] + d[j,b] ≈ d[a,b]
    # We approximate: betweenness[j] = sum over (a,b) of indicator that j lies
    # on a shortest path from a to b, weighted by prize[j]
    
    # Efficient approximation: for each j, count how many pairs (a,b) have
    # d[a,j] + d[j,b] close to d[a,b]
    # This is O(n^3) which is fine for n=50
    betweenness = np.zeros(n)
    for j in range(n):
        if j == 0:
            continue
        # d[a,j] + d[j,b] for all a,b
        dj = distance_matrix[:, j]  # d[a,j]
        djT = distance_matrix[j, :]  # d[j,b]
        # d[a,b]
        dab = distance_matrix  # d[a,b]
        # Check if j is approximately on shortest path: |d[a,j]+d[j,b] - d[a,b]| < tol
        diff = np.abs(dj[:, None] + djT[None, :] - dab)
        # Count pairs where j is on a shortest path (within tolerance)
        on_path = (diff < 1e-6).astype(float)
        # Exclude trivial cases where a==j or b==j
        on_path[j, :] = 0
        on_path[:, j] = 0
        # Weight by prize of intermediate nodes? No, just count
        betweenness[j] = on_path.sum()
    
    # Normalize betweenness
    max_btw = np.max(betweenness)
    if max_btw > eps:
        betweenness /= max_btw
    else:
        betweenness = np.zeros(n)
    
    # --- Feature 4: Origin-proximity centrality ---
    # Nodes close to origin are more accessible and reduce return cost
    dist_to_origin = distance_matrix[:, 0]
    max_dist = np.max(dist_to_origin)
    if max_dist > eps:
        origin_cent = 1.0 - dist_to_origin / max_dist
    else:
        origin_cent = np.ones(n)
    origin_cent[0] = 0.0  # origin itself
    
    # --- Combine features into a node-attractiveness score ---
    # Each feature is a vector of length n (per-node importance)
    # We normalize each to [0, 1]
    def normalize_01(v):
        vmin, vmax = np.min(v), np.max(v)
        if vmax - vmin < eps:
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)
    
    eig_n = normalize_01(eig_centrality)
    deg_n = normalize_01(degree_cent)
    btw_n = normalize_01(betweenness)
    orig_n = normalize_01(origin_cent)
    
    # Prize itself as a feature
    prize_n = normalize_01(prizes)
    
    # Linear combination with learned-like weights
    # Weights chosen to balance local prize with global structure:
    # - Prize is primary (0.40)
    # - Eigenvector centrality captures "influence" in the prize graph (0.25)
    # - Betweenness captures hub-ness (0.15)
    # - Degree centrality captures accessibility (0.10)
    # - Origin proximity captures feasibility (0.10)
    w_prize = 0.40
    w_eig = 0.25
    w_btw = 0.15
    w_deg = 0.10
    w_orig = 0.10
    
    node_score = (w_prize * prize_n + w_eig * eig_n + w_btw * btw_n +
                  w_deg * deg_n + w_orig * orig_n)
    
    # --- Build edge heuristic H[i,j] ---
    # H[i,j] should reflect:
    #   (a) How attractive node j is globally (node_score[j])
    #   (b) How easy it is to go from i to j and back (cost penalty)
    #   (c) Whether the move is feasible
    #
    # Use a multiplicative form: H[i,j] = node_score[j] * exp(-beta * cost[i,j])
    # This gives a smooth, positive, finite matrix where:
    #   - High node_score[j] makes j attractive from all i
    #   - Low cost[i,j] makes the move attractive
    #   - The exponential decay is tunable via beta
    
    # Choose beta based on max_len to scale appropriately
    # We want exp(-beta * max_len) to be small (~0.01-0.1)
    # beta = -ln(0.1) / max_len ≈ 2.3 / max_len
    beta = 2.3 / max(max_len, eps)
    
    # H[i,j] = node_score[j] * exp(-beta * cost[i,j])
    # But we also want to strongly penalize infeasible moves
    H = node_score[None, :] * np.exp(-beta * cost_safe)
    
    # For infeasible moves (cost > max_len), set to very small value
    infeasible = (cost > max_len)
    H[infeasible] = eps
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, eps)
    
    # Check time
    if time.time() > deadline - 1.0:
        return H
    
    # Final safety: ensure shape is correct
    if H.shape != (n, n):
        return np.ones((n, n), dtype=float) * 1e-9
    
    return H
# EVOLVE-BLOCK-END
