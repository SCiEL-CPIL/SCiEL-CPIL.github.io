"""
Combined Greedy MIS: LP-Guided Local Search.

Strategy:
1. Use Linear Programming (LP) relaxation (inspiration 2) to obtain a high-quality
   continuous solution vector y* that approximates the optimal independent set indicator.
2. Use structural features (Degree, Clustering, Core, Spectral) from the current program
   to construct a robust baseline score.
3. Initialize the local search (from current program) with a hybrid score that blends
   the LP solution and the structural features.
4. Perform intensive local search on the score vector to maximize the greedy decoder's
   independent set size.

The LP solution provides a strong "direction" for the scores, reducing the search space
and allowing the local search to converge faster and potentially to a better local optimum.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

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

    # Precompute neighbor lists
    neighbors = [np.where(A_bool[i])[0] for i in range(N)]
    degrees = np.array([len(neighbors[i]) for i in range(N)], dtype=float)

    # --- Feature Extraction ---

    # 1. Degree
    D = degrees

    # 2. Clustering Coefficient
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0
    D_safe = np.where(D > 0, D, 1.0)
    Clus = np.zeros(N)
    mask = D >= 2
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # 3. Core Number (k-core decomposition)
    core = np.zeros(N, dtype=int)
    deg = D.copy().astype(int)
    active = np.ones(N, dtype=bool)
    max_deg = int(D.max())
    if max_deg > 0:
        for k in range(1, max_deg + 1):
            changed = True
            while changed:
                changed = False
                for i in range(N):
                    if active[i] and deg[i] < k:
                        active[i] = False
                        core[i] = k - 1
                        changed = True
                        for nb in neighbors[i]:
                            if active[nb]:
                                deg[nb] -= 1
        for i in range(N):
            if active[i]:
                core[i] = max_deg

    # 4. Spectral (Fiedler / Principal Eigenvector)
    # Use power iteration for the largest eigenvalue (Perron).
    # Low centrality is good for MIS.
    x = np.ones(N) / N
    for _ in range(20):
        x_new = A @ x
        norm_x = np.linalg.norm(x_new)
        if norm_x < 1e-12: break
        x_new /= norm_x
        if np.linalg.norm(x_new - x) < 1e-9: break
        x = x_new
    Spec_n = x

    # Normalize features to [0, 1]
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = np.min(x)
        max_val = np.max(x)
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    Clus_n = norm(Clus)
    Core_n = norm(core)
    Spec_n = norm(Spec_n)

    # --- Fast Greedy Evaluator ---
    def evaluate(scores):
        order = np.argsort(-scores, kind='stable')
        blocked = np.zeros(N, dtype=np.uint8)
        count = 0
        for v in order:
            if blocked[v] == 0:
                count += 1
                blocked[v] = 1
                if neighbors[v].size > 0:
                    blocked[neighbors[v]] = 1
        return count

    # --- Phase 1: LP Relaxation (Inspiration 2) ---
    lp_scores = None
    try:
        if time.time() < DEADLINE - 20:
            from scipy.optimize import linprog
            from scipy.sparse import coo_matrix

            i, j = np.triu_indices(N, k=1)
            mask = A_bool[i, j]
            i = i[mask]
            j = j[mask]
            m = len(i)

            if m == 0:
                lp_scores = np.ones(N)
            else:
                row_indices = np.concatenate([i, j])
                col_indices = np.concatenate([i, j])
                data = np.ones(len(row_indices)) * -1.0

                A_ub = coo_matrix((data, (row_indices, col_indices)), shape=(m, N)).tocsr()
                b_ub = -np.ones(m)
                c = np.ones(N)
                bounds = [(0, 1)] * N

                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    y_opt = res.x
                    # We want to maximize the independent set size.
                    # The LP minimizes sum(y) s.t. y_i + y_j >= 1.
                    # So y_i = 1 means "likely in MIS".
                    # We want high scores for vertices likely in MIS.
                    # So score = y_opt is a good direction.
                    lp_scores = y_opt
    except Exception:
        lp_scores = None

    # --- Phase 2: Hybrid Initialization ---
    # Combine LP scores with structural features.
    # If LP failed, use structural features only.
    if lp_scores is not None:
        # Normalize LP scores to [0, 1]
        lp_n = norm(lp_scores)
        # Blend: 0.7 * LP + 0.3 * Structural
        # Structural score: -D_n - Clus_n - Core_n - Spec_n (lower is better for features, so negate for score)
        struct_score = -(D_n + Clus_n + Core_n + Spec_n)
        struct_score_n = norm(struct_score)
        best_scores = 0.7 * lp_n + 0.3 * struct_score_n
    else:
        # Fallback to structural only
        best_scores = -(D_n + Clus_n + Core_n + Spec_n)

    # --- Phase 3: Local Search on Vertex Scores ---
    current_size = evaluate(best_scores)
    
    rng = np.random.default_rng(42)
    
    # Polish: Swap vertices to improve greedy outcome
    # Strategy: 
    # 1. Random swaps
    # 2. Targeted swaps: Swap a picked vertex with a blocked neighbor
    
    iterations = 0
    max_iterations = 50000
    
    while time.time() < DEADLINE - 2 and iterations < max_iterations:
        iterations += 1
        
        # Random swap of two vertices
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        if i == j:
            continue
            
        # Try swapping
        old_si = best_scores[i]
        old_sj = best_scores[j]
        best_scores[i] = old_sj
        best_scores[j] = old_si
        
        new_size = evaluate(best_scores)
        
        if new_size > current_size:
            current_size = new_size
        else:
            # Revert
            best_scores[i] = old_si
            best_scores[j] = old_sj
            
        # Occasionally try a targeted swap: 
        # Swap a vertex with one of its neighbors if it's currently blocking it
        if rng.random() < 0.2:
             # Get current greedy order and blocked status
             order = np.argsort(-best_scores, kind='stable')
             blocked = np.zeros(N, dtype=np.uint8)
             picked = []
             for v in order:
                 if blocked[v] == 0:
                     picked.append(v)
                     blocked[v] = 1
                     if neighbors[v].size > 0:
                         blocked[neighbors[v]] = 1
             
             # Pick a random picked vertex
             if picked:
                 v = picked[rng.integers(0, len(picked))]
                 # Pick a random neighbor of v
                 if neighbors[v].size > 0:
                     u = neighbors[v][rng.integers(0, neighbors[v].size)]
                     # Swap v and u
                     old_sv = best_scores[v]
                     old_su = best_scores[u]
                     best_scores[v] = old_su
                     best_scores[u] = old_sv
                     
                     new_size = evaluate(best_scores)
                     if new_size > current_size:
                         current_size = new_size
                     else:
                         best_scores[v] = old_sv
                         best_scores[u] = old_su

    return best_scores
# EVOLVE-BLOCK-END
