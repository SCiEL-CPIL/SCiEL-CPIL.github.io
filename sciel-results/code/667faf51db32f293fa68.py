"""
Optimized Greedy MIS via Local Search on Vertex Scores.

This program generates priority scores for a greedy decoder.
It uses a two-phase approach:
1. Generate diverse initial score vectors based on graph structural features (Degree, Clustering, Spectral).
2. Perform local search (swaps and perturbations) on the score vectors to maximize the resulting independent set size, 
   using the greedy decoder as the objective function.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 125.0  # Safety margin below 150s limit, allowing ~10s for cleanup

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5
    
    # Precompute neighbor lists for faster greedy evaluation
    # Using lists of indices is faster than boolean slicing in a loop for N=200
    neighbors = [np.where(A_bool[i])[0] for i in range(N)]
    # Also store neighbor sets for O(1) membership checks if needed
    neighbor_sets = [set(n.tolist()) for n in neighbors]
    
    # For faster evaluation, precompute neighbor masks as Python integers (bitmasks)
    # This allows very fast "is vertex v blocked?" checks
    neighbor_masks = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        neighbor_masks[i] = mask

    # --- Fast Greedy Evaluator with Early Termination ---
    def evaluate(scores):
        """
        Simulates the greedy decoder:
        1. Sort vertices by score descending.
        2. Iterate through sorted list.
        3. If vertex is not blocked (not neighbor of any picked vertex), pick it.
        4. Mark it and its neighbors as blocked.
        Returns the count of picked vertices.
        """
        order = np.argsort(-scores, kind='stable')
        
        blocked_mask = 0  # Bitmask of blocked vertices
        count = 0
        
        # Upper bound: max possible picks is N - count_of_blocked_so_far
        # We can stop early if remaining vertices can't beat current best
        for v in order:
            if not (blocked_mask >> v) & 1:
                count += 1
                blocked_mask |= (1 << v) | neighbor_masks[v]
        
        return count

    # --- Feature Extraction for Initial Candidates ---
    
    # 1. Degree
    D = A.sum(axis=1)
    
    # 2. Clustering Coefficient (approximated or exact)
    # For N=200, exact triangle count is feasible
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0
    D_safe = np.where(D > 0, D, 1.0)
    # Standard clustering coefficient: 2*Triangles / (deg*(deg-1))
    # Avoid division by zero
    Clus = np.zeros(N)
    mask = D >= 2
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # 3. Mean Neighbor Degree
    SND = A @ D
    MeanND = SND / D_safe

    # 4. Spectral Centrality (Principal Eigenvector)
    # Vertices with high spectral centrality are central. We want low centrality for independent set.
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

    # 5. Fiedler Vector (second-smallest eigenvector of Laplacian)
    # Identifies vertices minimally entangled in graph connectivity
    # High absolute Fiedler values indicate vertices on periphery/sparse regions
    L = np.diag(D) - A
    try:
        # Use scipy for efficient eigenvalue computation
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        L_sparse = csr_matrix(L)
        # Get 3 smallest eigenvalues to ensure we get the Fiedler vector
        # (eigsh requires k < N for 'SM' target)
        k_eig = min(3, N - 1)
        eigvals, eigvecs = eigsh(L_sparse, k=k_eig, which='SM')
        # Sort by eigenvalue
        idx_sort = np.argsort(eigvals)
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]
        # Fiedler vector is the second smallest (index 1)
        if len(eigvals) >= 2:
            Fiedler = np.abs(eigvecs[:, 1])
        else:
            # Fallback to principal eigenvector if only one mode found
            Fiedler = Spectral
    except Exception:
        # Fallback: use spectral as proxy
        Fiedler = Spectral

    # Normalize features to [0, 1] for consistent weighting
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
    # This suggests the vertex is in a sparse region.
    # OpenPairs = C(D, 2) - Tri
    OpenPairs = np.maximum(0, 0.5 * D * (D - 1) - Tri)
    Open_n = norm(OpenPairs)

    # --- Phase 1: Initialize with Strong Heuristics ---
    
    candidates = []
    
    # H1: Min Degree (Classic)
    candidates.append(-D_n)
    
    # H2: Min Degree, High Openness
    candidates.append(-D_n + 0.5 * Open_n)
    
    # H3: Min Degree, Low Clustering
    candidates.append(-D_n - 0.5 * Clus_n)
    
    # H4: Min Degree, Low Mean Neighbor Degree (prefer peripheral)
    candidates.append(-D_n - 0.5 * MeanND_n)
    
    # H5: Low Spectral Centrality
    candidates.append(-Spectral_n)
    
    # H6: Mix of Degree and Spectral
    candidates.append(-0.7 * D_n - 0.3 * Spectral_n)
    
    # H7: High Fiedler (peripheral/sparse regions)
    candidates.append(Fiedler_n)
    
    # H8: Min Degree, High Fiedler
    candidates.append(-0.5 * D_n + 0.5 * Fiedler_n)
    
    # H9: Min Degree, High Fiedler, Low Clustering
    candidates.append(-0.4 * D_n + 0.4 * Fiedler_n - 0.2 * Clus_n)
    
    # H10: Random initialization
    rng = np.random.default_rng(42)
    candidates.append(rng.normal(0, 1, N))

    best_size = -1
    best_scores = None

    for c in candidates:
        size = evaluate(c)
        if size > best_size:
            best_size = size
            best_scores = c.copy()
            
    # --- Phase 2: Local Search ---
    # Strategy:
    # 1. Try swapping scores of two random vertices.
    # 2. Try adding small Gaussian noise.
    # 3. Occasionally revert to a base heuristic + noise to escape local optima.
    # 4. Targeted "unblock" moves: boost scores of vertices that are frequently blocked.

    # We can do many iterations in 120s. N=200.
    # Each eval is ~200 argsorts + 200 checks. Very fast.
    
    trial_count = 0
    max_trials = 120000 # Increased budget with faster evaluation
    
    # Track which vertices are often blocked to guide targeted moves
    block_counts = np.zeros(N, dtype=int)
    
    while trial_count < max_trials:
        if time.time() > DEADLINE:
            break
            
        # Decide move type
        r = rng.random()
        new_scores = best_scores.copy()
        
        if r < 0.35:
            # Swap move: Pick two random indices and swap their scores.
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i != j:
                new_scores[i], new_scores[j] = new_scores[j], new_scores[i]
            else:
                continue # No change
        elif r < 0.55:
            # 2-opt neighborhood: Swap two vertices that are close in ranking
            # This is a more structured move than random swap
            order = np.argsort(-best_scores, kind='stable')
            # Pick a random position in the ranking
            pos = rng.integers(0, N - 1)
            # Swap vertices at pos and pos+1 (adjacent in ranking)
            i = order[pos]
            j = order[pos + 1]
            new_scores[i], new_scores[j] = new_scores[j], new_scores[i]
        elif r < 0.70:
            # Adjacent swap: Pick a vertex and swap with a nearby-ranked vertex
            # This makes smaller, more controlled changes
            i = rng.integers(0, N)
            # Find rank of i
            rank_i = np.where(np.argsort(-best_scores, kind='stable') == i)[0][0]
            # Pick a vertex within +/- 3 ranks (tighter neighborhood)
            delta_rank = rng.integers(-3, 4)
            rank_j = rank_i + delta_rank
            if 0 <= rank_j < N:
                order = np.argsort(-best_scores, kind='stable')
                j = order[rank_j]
                new_scores[i], new_scores[j] = new_scores[j], new_scores[i]
            else:
                continue
        elif r < 0.85:
            # Perturbation move: Add small noise to a few random vertices
            k = rng.integers(1, 5) # Perturb 1 to 4 vertices (smaller perturbations)
            idxs = rng.choice(N, k, replace=False)
            noise = rng.normal(0, 0.03, k)
            new_scores[idxs] += noise
        elif r < 0.92:
            # Targeted unblock: boost scores of frequently blocked vertices
            # Find vertices with high block counts
            top_blocked = np.argsort(-block_counts)[:10]
            if len(top_blocked) > 0:
                # Pick one of the most blocked vertices and boost it
                v = top_blocked[rng.integers(0, min(10, len(top_blocked)))]
                # Boost its score significantly
                new_scores[v] += rng.uniform(0.1, 0.5)
                # Optionally, also slightly reduce scores of its neighbors
                if rng.random() < 0.5:
                    for nb in neighbors[v]:
                        new_scores[nb] -= rng.uniform(0.01, 0.05)
            else:
                continue
        else:
            # Restart/Escape: Blend with a random base heuristic including Fiedler
            w_d = rng.uniform(-1, 0)
            w_c = rng.uniform(-0.5, 0.5)
            w_s = rng.uniform(-0.5, 0.5)
            w_f = rng.uniform(0, 0.5)
            base = w_d * D_n + w_c * Clus_n + w_s * Spectral_n + w_f * Fiedler_n
            # Blend with current best to stay in the neighborhood
            alpha = rng.uniform(0.6, 0.9)
            new_scores = alpha * best_scores + (1 - alpha) * base
            
        size = evaluate(new_scores)
        
        if size > best_size:
            best_size = size
            best_scores = new_scores
            # Reset trial counter for success-based patience
            trial_count = 0
        else:
            trial_count += 1
        
        # Update block counts periodically to guide targeted moves
        if trial_count % 100 == 0:
            # Simulate the greedy process to count how often each vertex is blocked
            order = np.argsort(-best_scores, kind='stable')
            blocked_mask = 0
            for v in order:
                if not (blocked_mask >> v) & 1:
                    blocked_mask |= (1 << v) | neighbor_masks[v]
                else:
                    block_counts[v] += 1
            # Decay block counts to focus on recent behavior
            block_counts *= 0.8
            
        # Early exit if stuck for too many trials
        if trial_count > 3000:
            # Diversify: jump to a new random neighborhood
            base_heuristics = [
                -D_n,
                -0.5 * D_n + 0.5 * Fiedler_n,
                Fiedler_n,
                -0.7 * D_n - 0.3 * Spectral_n,
                -0.3 * D_n + 0.3 * Fiedler_n - 0.2 * Clus_n
            ]
            base = base_heuristics[rng.integers(0, len(base_heuristics))]
            alpha = rng.uniform(0.4, 0.8)
            best_scores = alpha * best_scores + (1 - alpha) * base
            best_size = evaluate(best_scores)
            trial_count = 0

    # --- Phase 3: Final Refinement (if time permits) ---
    if time.time() < DEADLINE - 5.0 and best_scores is not None:
        # Systematic polish: try all adjacent-in-ranking swaps
        improved = True
        while improved and time.time() < DEADLINE - 2.0:
            improved = False
            order = np.argsort(-best_scores, kind='stable')
            for pos in range(N - 1):
                if time.time() > DEADLINE - 2.0:
                    break
                
                i = order[pos]
                j = order[pos + 1]
                
                # Store old values
                si = best_scores[i]
                sj = best_scores[j]
                
                # Swap
                best_scores[i] = sj
                best_scores[j] = si
                
                size = evaluate(best_scores)
                
                if size > best_size:
                    best_size = size
                    improved = True
                    # Keep swap and restart scan from beginning
                    order = np.argsort(-best_scores, kind='stable')
                    pos = -1  # Will be incremented to 0
                else:
                    # Revert swap
                    best_scores[i] = si
                    best_scores[j] = sj
        
        # Additional targeted improvements: try boosting specific vertices
        if time.time() < DEADLINE - 1.0:
            # Try boosting each vertex individually
            for v in range(N):
                if time.time() > DEADLINE - 1.0:
                    break
                
                original_score = best_scores[v]
                # Try increasing score
                best_scores[v] = original_score + 0.1
                size = evaluate(best_scores)
                if size > best_size:
                    best_size = size
                else:
                    best_scores[v] = original_score
                
                # Try decreasing score
                best_scores[v] = original_score - 0.1
                size = evaluate(best_scores)
                if size > best_size:
                    best_size = size
                else:
                    best_scores[v] = original_score

    if best_scores is None:
        return -D.astype(float)

    return best_scores
# EVOLVE-BLOCK-END
