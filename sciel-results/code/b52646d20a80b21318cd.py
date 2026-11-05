"""
Optimized Greedy MIS via Fiedler Vector Scoring:
Uses the Fiedler vector (second-smallest eigenvector of Laplacian) to identify
vertices minimally entangled in the graph's connectivity structure.
Vertices with large |Fiedler| values are peripheral and naturally separated
from dense cores, making them good candidates for early selection.
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

    # --- Compute Fiedler Vector ---
    D = A.sum(axis=1)
    
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        
        L = np.diag(D) - A
        L_sparse = csr_matrix(L)
        k_eig = min(3, N - 1)
        eigvals, eigvecs = eigsh(L_sparse, k=k_eig, which='SM')
        idx_sort = np.argsort(eigvals)
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]
        
        if len(eigvals) >= 2:
            fiedler = eigvecs[:, 1]
        else:
            # Fallback: use principal eigenvector
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
            fiedler = x
    except Exception:
        # Fallback: use principal eigenvector
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
        fiedler = x

    # Normalize Fiedler to [0, 1]
    def norm(x):
        min_val = np.min(x)
        max_val = np.max(x)
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)
    
    fiedler_norm = norm(np.abs(fiedler))
    
    # Also compute degree for potential blending
    D_safe = np.where(D > 0, D, 1.0)
    D_n = norm(D)
    
    # --- Phase 1: Try different scoring strategies ---
    rng = np.random.default_rng(42)
    
    best_size = -1
    best_scores = None
    
    # Strategy 1: Pure Fiedler (high |Fiedler| = peripheral = good)
    s1 = fiedler_norm
    size1 = evaluate(s1)
    if size1 > best_size:
        best_size = size1
        best_scores = s1.copy()
    
    # Strategy 2: Fiedler + inverse degree (prefer low degree AND peripheral)
    s2 = 0.7 * fiedler_norm + 0.3 * (1 - D_n)
    size2 = evaluate(s2)
    if size2 > best_size:
        best_size = size2
        best_scores = s2.copy()
    
    # Strategy 3: Fiedler - degree (penalize high degree more)
    s3 = fiedler_norm - 0.5 * D_n
    size3 = evaluate(s3)
    if size3 > best_size:
        best_size = size3
        best_scores = s3.copy()
    
    # Strategy 4: Rank-based Fiedler
    fiedler_rank = np.argsort(np.argsort(fiedler_norm)) / N
    s4 = fiedler_rank
    size4 = evaluate(s4)
    if size4 > best_size:
        best_size = size4
        best_scores = s4.copy()
    
    # Strategy 5: Fiedler with slight random perturbation for tie-breaking
    s5 = fiedler_norm + rng.normal(0, 0.01, N)
    size5 = evaluate(s5)
    if size5 > best_size:
        best_size = size5
        best_scores = s5.copy()
    
    # --- Phase 2: Local Search to Refine ---
    if best_scores is None:
        best_scores = fiedler_norm
        best_size = evaluate(best_scores)
    
    max_local_iters = 30000
    local_trial_count = 0
    
    while local_trial_count < max_local_iters:
        if time.time() > DEADLINE - 5.0:
            break
            
        new_scores = best_scores.copy()
        r = rng.random()
        
        if r < 0.5:
            # Swap two random vertices
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i != j:
                new_scores[i], new_scores[j] = new_scores[j], new_scores[i]
            else:
                continue
        elif r < 0.7:
            # Perturb a few vertices
            k = rng.integers(1, 6)
            idxs = rng.choice(N, k, replace=False)
            noise = rng.normal(0, 0.05, k)
            new_scores[idxs] += noise
        elif r < 0.85:
            # Blend with Fiedler base
            alpha = rng.uniform(0.8, 0.95)
            new_scores = alpha * best_scores + (1 - alpha) * fiedler_norm
        else:
            # More aggressive: swap multiple pairs
            for _ in range(rng.integers(2, 5)):
                i = rng.integers(0, N)
                j = rng.integers(0, N)
                if i != j:
                    new_scores[i], new_scores[j] = new_scores[j], new_scores[i]

        size = evaluate(new_scores)
        
        if size > best_size:
            best_size = size
            best_scores = new_scores
            local_trial_count = 0
        else:
            local_trial_count += 1
            
        # If stuck, try restarting from Fiedler with noise
        if local_trial_count > 2000:
            noise = rng.normal(0, 0.1, N)
            candidate = fiedler_norm + noise
            size_c = evaluate(candidate)
            if size_c > best_size:
                best_size = size_c
                best_scores = candidate
            local_trial_count = 0

    return best_scores
# EVOLVE-BLOCK-END
