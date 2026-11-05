"""
Optimized Greedy MIS via Two-Phase Search:
1. Global Heuristic Weight Optimization (Low-dimensional space)
2. Local Vertex Score Polish (High-dimensional space)

This approach leverages the insight that optimizing weights for structural features
provides a robust global structure, which is then refined by local swaps on vertex scores.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin below 150s limit

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5
    
    # Precompute neighbor lists for faster greedy evaluation
    neighbors = [np.where(A_bool[i])[0] for i in range(N)]

    # --- Fast Greedy Evaluator ---
    def evaluate(scores):
        """
        Simulates the greedy decoder:
        1. Sort vertices by score descending.
        2. Iterate through sorted list.
        3. If vertex is not blocked, pick it.
        4. Mark it and its neighbors as blocked.
        Returns the count of picked vertices.
        """
        order = np.argsort(-scores, kind='stable')
        
        blocked = np.zeros(N, dtype=bool)
        count = 0
        
        for v in order:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                if neighbors[v].size > 0:
                    blocked[neighbors[v]] = True
        
        return count

    # --- Feature Extraction ---
    
    # 1. Degree
    D = A.sum(axis=1)
    
    # 2. Clustering Coefficient
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0
    D_safe = np.where(D > 0, D, 1.0)
    Clus = np.zeros(N)
    mask = D >= 2
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # 3. Mean Neighbor Degree
    SND = A @ D
    MeanND = SND / D_safe

    # 4. Spectral Centrality (Principal Eigenvector)
    x = np.ones(N) / N
    for _ in range(30):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x) < 1e-9:
            break
        x = x_new
    Spectral = x

    # 5. Fiedler Vector
    L = np.diag(D) - A
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        L_sparse = csr_matrix(L)
        k_eig = min(3, N - 1)
        eigvals, eigvecs = eigsh(L_sparse, k=k_eig, which='SM')
        idx_sort = np.argsort(eigvals)
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]
        if len(eigvals) >= 2:
            Fiedler = np.abs(eigvecs[:, 1])
        else:
            Fiedler = Spectral
    except Exception:
        Fiedler = Spectral

    # Normalize features to [0, 1]
    def norm(x):
        min_val = np.min(x)
        max_val = np.max(x)
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    Clus_n = norm(Clus)
    MeanND_n = norm(MeanND)
    Spectral_n = norm(Spectral)
    Fiedler_n = norm(Fiedler)
    
    # "Openness": Vertices with many neighbors that are NOT connected to each other.
    OpenPairs = np.maximum(0, 0.5 * D * (D - 1) - Tri)
    Open_n = norm(OpenPairs)

    # Store features in a matrix for vectorized weight combination
    # Features: [D_n, Clus_n, MeanND_n, Spectral_n, Fiedler_n, Open_n]
    # Note: We want to minimize degree/clustering/centrality, so we will use negative weights or invert logic in score calc.
    # Let's define score = -w_d*D_n - w_c*Clus_n - w_s*Spectral_n + w_f*Fiedler_n + w_o*Open_n ...
    # Actually, let's just compute the score vector directly from weights.
    
    def compute_scores(weights):
        # weights: [w_d, w_c, w_mn, w_sp, w_f, w_o]
        # Higher score -> picked first.
        # We want low degree, low clustering, low spectral centrality.
        # We want high fiedler (periphery), high openness.
        w_d, w_c, w_mn, w_sp, w_f, w_o = weights
        scores = -w_d * D_n - w_c * Clus_n - w_mn * MeanND_n - w_sp * Spectral_n + w_f * Fiedler_n + w_o * Open_n
        return scores

    rng = np.random.default_rng(42)
    
    # --- Phase 1: Global Weight Optimization ---
    # Search space: 6 weights, each in [-1, 1] roughly.
    # Use random restarts + local search on weights.
    
    best_global_size = -1
    best_global_weights = None
    best_global_scores = None

    # Initial candidates
    init_weights = [
        [1.0, 0.5, 0.5, 0.3, 0.5, 0.5],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.8, 0.4, 0.4, 0.2, 0.6, 0.4],
    ]
    
    for w in init_weights:
        s = compute_scores(w)
        size = evaluate(s)
        if size > best_global_size:
            best_global_size = size
            best_global_weights = np.array(w)
            best_global_scores = s

    # Local search on weights
    weight_dim = 6
    max_weight_iters = 5000
    weight_trial_count = 0
    
    while weight_trial_count < max_weight_iters:
        if time.time() > DEADLINE - 30.0: # Leave time for local polish
            break
            
        # Perturb weights
        new_w = best_global_weights.copy()
        noise = rng.normal(0, 0.1, weight_dim)
        new_w += noise
        # Clip to reasonable range
        new_w = np.clip(new_w, -2.0, 2.0)
        
        s = compute_scores(new_w)
        size = evaluate(s)
        
        if size > best_global_size:
            best_global_size = size
            best_global_weights = new_w
            best_global_scores = s
            weight_trial_count = 0
        else:
            weight_trial_count += 1
            
        # Occasionally restart with random weights
        if weight_trial_count % 500 == 0:
            rand_w = rng.uniform(-1.0, 1.0, weight_dim)
            s = compute_scores(rand_w)
            size = evaluate(s)
            if size > best_global_size:
                best_global_size = size
                best_global_weights = rand_w
                best_global_scores = s
                weight_trial_count = 0

    # --- Phase 2: Local Vertex Score Polish ---
    # Start from the best global scores and refine the vertex ordering.
    
    best_size = best_global_size
    best_scores = best_global_scores.copy()
    
    if best_scores is None:
        # Fallback if global search failed
        best_scores = -D_n
        best_size = evaluate(best_scores)

    # Local search on 200D scores
    max_local_iters = 20000
    local_trial_count = 0
    
    while local_trial_count < max_local_iters:
        if time.time() > DEADLINE - 5.0:
            break
            
        new_scores = best_scores.copy()
        r = rng.random()
        
        if r < 0.6:
            # Swap two random vertices
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i != j:
                new_scores[i], new_scores[j] = new_scores[j], new_scores[i]
            else:
                continue
        elif r < 0.8:
            # Perturb a few vertices
            k = rng.integers(1, 5)
            idxs = rng.choice(N, k, replace=False)
            noise = rng.normal(0, 0.05, k)
            new_scores[idxs] += noise
        else:
            # Small blend with global best
            alpha = rng.uniform(0.8, 0.95)
            new_scores = alpha * best_scores + (1 - alpha) * best_global_scores

        size = evaluate(new_scores)
        
        if size > best_size:
            best_size = size
            best_scores = new_scores
            local_trial_count = 0
        else:
            local_trial_count += 1
            
        # If stuck, try a more aggressive perturbation
        if local_trial_count > 1000:
            # Restart local search from a slightly perturbed global best
            noise = rng.normal(0, 0.1, N)
            candidate = best_global_scores + noise
            size_c = evaluate(candidate)
            if size_c > best_size:
                best_size = size_c
                best_scores = candidate
            local_trial_count = 0

    return best_scores
# EVOLVE-BLOCK-END
