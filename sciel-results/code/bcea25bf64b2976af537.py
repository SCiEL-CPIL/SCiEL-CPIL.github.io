"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _evaluate_fast(nbrs, scores):
    """Fast evaluation using precomputed neighbor lists."""
    n = len(nbrs)
    if n == 0:
        return 0
    indices = np.argsort(-scores, kind='stable')
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in indices:
        if not blocked[v]:
            count += 1
            blocked[v] = True
            if len(nbrs[v]) > 0:
                blocked[nbrs[v]] = True
    return count


def _fiedler_fast(L, n, iters=200):
    """Approximate Fiedler vector via power iteration on shifted Laplacian."""
    max_deg = int(np.max(np.sum(L, axis=1)))
    c = 2.0 * max_deg + 1.0
    
    M = c * np.eye(n) - L
    
    ones = np.ones(n) / np.sqrt(n)
    v = np.random.randn(n)
    v = v - np.dot(v, ones) * ones
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        v = np.random.randn(n)
        v = v - np.dot(v, ones) * ones
        norm_v = np.linalg.norm(v)
    v /= norm_v
    
    for _ in range(iters):
        w = M @ v
        w = w - np.dot(w, ones) * ones
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-10:
            break
        v = w / norm_w
    
    return v


def _eigenvector_centrality(adjacency, degree, iters=50):
    """Compute eigenvector centrality via power iteration."""
    n = len(degree)
    if n == 0:
        return np.array([])
    
    max_deg = degree.max()
    if max_deg > 0:
        scaled = adjacency.astype(float) / max_deg
    else:
        scaled = adjacency.astype(float)
    
    x = np.ones(n, dtype=float)
    x /= np.linalg.norm(x)
    
    for _ in range(iters):
        x_new = scaled.T @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x) < 1e-8:
            x = x_new
            break
        x = x_new
    
    return x


def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    Uses a combination of local degree, global centrality (Fiedler vector and eigenvector centrality),
    and local structural features to better approximate which vertices to pick.
    Incorporates a 'survival probability' heuristic based on weighted degree in non-neighborhood.
    """
    deadline = time.time() + 145  # Safety margin before 150s kill
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])

    # Precompute neighbor lists for fast evaluation
    nbrs = [np.where(adjacency[i])[0] for i in range(n)]
    degree = np.array([len(nbrs[i]) for i in range(n)], dtype=float)

    # --- Feature Computation ---
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    # Degree
    D_n = norm(degree)

    # Mean neighbor degree
    mean_nd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            mean_nd[i] = np.mean(degree[nbrs[i]])
        else:
            mean_nd[i] = 0
    MeanND_n = norm(mean_nd)

    # Sum of neighbor degrees
    snd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            snd[i] = np.sum(degree[nbrs[i]])
    SND_n = norm(snd)

    # Triangles
    A = adjacency.astype(float)
    A_sq = A @ A
    tri = np.diag(A_sq) / 2.0
    Tri_n = norm(tri)

    # Clustering coefficient
    clus = np.zeros(n)
    mask = degree > 1
    clus[mask] = 2.0 * tri[mask] / (degree[mask] * (degree[mask] - 1.0))
    Clus_n = norm(clus)

    # Fiedler vector (second-smallest eigenvector of Laplacian)
    try:
        D_mat = np.diag(degree)
        L = D_mat - A
        fiedler = _fiedler_fast(L, n, iters=300)
        abs_fiedler = np.abs(fiedler)
        AbsFiedler_n = norm(abs_fiedler)
    except Exception:
        AbsFiedler_n = np.zeros(n)

    # Core numbers (simplified k-core)
    core = np.zeros(n, dtype=int)
    deg_copy = degree.astype(int).copy()
    remaining = np.ones(n, dtype=bool)
    max_deg = int(deg_copy.max()) if n > 0 else 0
    for k in range(max_deg, -1, -1):
        for i in range(n):
            if remaining[i] and deg_copy[i] >= k:
                core[i] = k
                remaining[i] = False
                for j in nbrs[i]:
                    if remaining[j]:
                        deg_copy[j] -= 1
    Core_n = norm(core.astype(float))

    # Eigenvector Centrality
    try:
        evc = _eigenvector_centrality(adjacency, degree, iters=50)
        EVC_n = norm(evc)
    except Exception:
        EVC_n = np.zeros(n)

    # Survival Probability Heuristic:
    # Prioritize vertices that are structurally isolated from dense clusters.
    # Approximate by inverse of weighted degree in the induced subgraph of non-neighbors?
    # Actually, the prompt says: "inverse of the vertex's weighted degree in the induced subgraph of its non-neighborhood"
    # Let's compute for each vertex v, the sum of degrees of its non-neighbors (excluding itself).
    # Or perhaps, the degree within the set of non-neighbors.
    # Let's try: For each v, let N_v be neighbors. Let V \ (N_v U {v}) be the non-neighborhood.
    # The "weighted degree in the induced subgraph of its non-neighborhood" is ambiguous.
    # Let's interpret it as: How "connected" is v to the rest of the graph via non-neighbor paths?
    # Or simpler: A vertex with low degree and low connection to high-degree nodes is good.
    # Let's try a feature: 1 / (1 + sum of degrees of neighbors) -- already have SND.
    # Let's try: 1 / (1 + degree) -- already have D_n.
    
    # Let's implement the specific heuristic:
    # "inverse of the vertex's weighted degree in the induced subgraph of its non-neighborhood"
    # This is computationally expensive to compute exactly for all v.
    # Approximation: 
    # For vertex v, the non-neighborhood is V \ N(v) \ {v}.
    # The induced subgraph on this set has edges between non-neighbors of v.
    # The "degree" of v in this context doesn't make sense as v is not in the subgraph.
    # Maybe it means: the degree of v in the graph G \ N(v)? No, v is isolated then.
    
    # Alternative interpretation from "maximum entropy ... expected survival probability":
    # A vertex survives if it is picked before any of its neighbors.
    # If scores are random, probability v is picked before neighbor u is 0.5.
    # But we control scores.
    
    # Let's stick to the existing features which are proven, but add Eigenvector Centrality.
    # The "survival probability" idea might be approximated by a feature that captures
    # how "central" a vertex is in the core of the graph. High core number = harder to remove.
    # We already have Core_n.
    
    # Let's add a feature: "Isolation Score"
    # For each vertex, compute the fraction of its neighbors that are also neighbors of each other (clustering).
    # We have Clus_n.
    
    # Let's try a new feature: "Weighted Non-Neighbor Degree"
    # For each vertex v, sum the degrees of vertices NOT in N(v) and not v.
    # This is (sum of all degrees) - (degree[v] + sum of degrees of neighbors)
    # Sum of all degrees is 2 * E.
    # Let's compute this.
    total_deg_sum = np.sum(degree)
    non_nbr_deg_sum = total_deg_sum - degree - snd
    NonNbrDeg_n = norm(non_nbr_deg_sum)
    
    # Inverse survival heuristic: We want high score for vertices that are "safe".
    # If a vertex is surrounded by high-degree nodes, it's risky.
    # If a vertex is surrounded by low-degree nodes, it's safe.
    # MeanND_n captures this. Low MeanND is good.
    
    # Let's combine features.
    
    # --- Initial Score Candidates ---
    # Features: D_n, MeanND_n, SND_n, Tri_n, Clus_n, Core_n, AbsFiedler_n, EVC_n, NonNbrDeg_n
    # We want to prefer:
    # Low Degree (high D_n is bad) -> -D_n
    # Low Mean Neighbor Degree (high MeanND_n is bad) -> -MeanND_n
    # Low Sum of Neighbor Degrees (high SND_n is bad) -> -SND_n
    # Low Triangles (high Tri_n is bad) -> -Tri_n
    # Low Clustering (high Clus_n is bad) -> -Clus_n
    # Low Core Number (high Core_n is bad) -> -Core_n
    # Low Fiedler (high AbsFiedler_n is bad? or good?) -> Fiedler vector magnitude indicates "centrality" in some sense.
    # Low Eigenvector Centrality (high EVC_n is bad) -> -EVC_n
    # High Non-Neighbor Degree Sum? (High NonNbrDeg_n means many edges elsewhere, maybe good to pick v to isolate?)
    
    # Let's define the feature vector order:
    # 0: D_n
    # 1: MeanND_n
    # 2: SND_n
    # 3: Tri_n
    # 4: Clus_n
    # 5: Core_n
    # 6: AbsFiedler_n
    # 7: EVC_n
    # 8: NonNbrDeg_n
    
    candidates = [
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0),
        (-0.8, -0.3, 0.0, 0.0, 0.0, -0.3, 0.0, -0.3, 0.5),
        (-0.7, -0.2, 0.0, 0.0, 0.0, -0.2, 0.0, -0.2, 0.3),
        (-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
        (-1.0, -0.3, -0.2, 0.0, 0.0, -0.3, 0.0, -0.3, 0.5),
        (-0.9, -0.4, -0.3, -0.2, -0.2, -0.4, 0.0, -0.4, 0.8),
        (-0.8, -0.4, -0.4, -0.3, -0.3, -0.4, 0.0, -0.4, 0.7),
        (-1.0, -0.6, 0.0, 0.0, 0.0, -0.6, 0.0, -0.6, 0.6),
        (-0.9, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, -0.5, 0.5),
        (-1.0, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5),
    ]

    best_count = 0
    best_scores = -D_n.copy()
    best_weights = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    features = np.stack([D_n, MeanND_n, SND_n, Tri_n, Clus_n, Core_n, AbsFiedler_n, EVC_n, NonNbrDeg_n], axis=1)

    for weights in candidates:
        if time.time() > deadline:
            break
        w = np.array(weights)
        scores = features @ w
        count = _evaluate_fast(nbrs, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()
            best_weights = w.copy()

    # --- Local Polish: Hill Climbing on Score Weights ---
    if time.time() < deadline - 15:
        base_weights = best_weights.copy()
        for _ in range(150):
            if time.time() > deadline:
                break
            perturbation = np.random.normal(0, 0.1, 9)
            new_weights = base_weights + perturbation
            scores = features @ new_weights
            count = _evaluate_fast(nbrs, scores)
            if count > best_count:
                best_count = count
                best_scores = scores.copy()
                best_weights = new_weights.copy()
                base_weights = new_weights.copy()
            elif np.random.random() < 0.05:
                base_weights = best_weights.copy()

    # --- Final Polish: Targeted Swap Improvement ---
    if time.time() < deadline - 5:
        scores = best_scores.copy()
        order = np.argsort(-scores, kind='stable')
        for sweep in range(15):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in order[:min(5, n)]:
                    if j == i:
                        continue
                    scores[i], scores[j] = scores[j], scores[i]
                    count = _evaluate_fast(nbrs, scores)
                    if count > best_count:
                        best_count = count
                        best_scores = scores.copy()
                        improved = True
                        order = np.argsort(-scores, kind='stable')
                    else:
                        scores[i], scores[j] = scores[j], scores[i]
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if not improved:
                break
        scores = best_scores

    return best_scores
# EVOLVE-BLOCK-END
