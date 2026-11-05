import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin
    
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
    
    # Ensure symmetric and no self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5
    
    # Precompute neighbor lists for fast greedy decoding
    nbrs = [np.where(A_bool[i])[0] for i in range(N)]
    
    def greedy_decode(scores):
        """
        Fast greedy decoder.
        Returns the size of the independent set.
        """
        order = np.argsort(-scores, kind='stable')
        blocked = np.zeros(N, dtype=bool)
        count = 0
        for v in order:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                if len(nbrs[v]) > 0:
                    blocked[nbrs[v]] = True
        return count

    # --- Feature Computation ---
    
    # 1. Degree
    D = A.sum(axis=1)
    
    # 2. Average Neighbor Degree (Sum of Neighbor Degrees / Degree)
    # High avg neighbor degree means the vertex is attached to hubs.
    # Picking such a vertex (if it has low degree) blocks the hub, which is efficient.
    SND = A @ D
    with np.errstate(divide='ignore', invalid='ignore'):
        MeanND = np.where(D > 0, SND / D, 0.0)
        
    # 3. Principal Eigenvector (Perron Vector)
    # This captures global structure. In bipartite-like graphs, the sign
    # separates the two partitions. Even in general graphs, it highlights
    # the "core" vs "periphery".
    eig_vec = np.zeros(N)
    try:
        if time.time() < DEADLINE - 10.0:
            # For N=200, eigh is very fast (~ms)
            eigvals, eigvecs = np.linalg.eigh(A)
            # The largest eigenvalue corresponds to the last column in ascending order
            eig_vec = eigvecs[:, -1]
            # Normalize to unit norm or max abs 1 for stable weighting
            max_abs = np.max(np.abs(eig_vec))
            if max_abs > 0:
                eig_vec = eig_vec / max_abs
    except Exception:
        eig_vec = np.zeros(N)
        
    # 4. Degree Centrality (normalized)
    D_n = D / (D.max() + 1e-9)
    MeanND_n = MeanND / (MeanND.max() + 1e-9)
    
    # --- Initial Score Search ---
    
    # We will search over weights for the composite score:
    # Score = w_d * (-D_n) + w_m * MeanND_n + w_e * eig_vec
    
    best_size = 0
    best_scores = np.zeros(N)
    
    # Grid search over weights
    # w_d: weight for -Degree (always positive, we want low degree)
    # w_m: weight for MeanNeighborDegree (positive, we want to block hubs)
    # w_e: weight for Eigenvector (can be positive or negative depending on sign convention, 
    #       but we normalized it. We'll try both signs implicitly by range)
    
    w_d_candidates = [0.5, 1.0, 1.5, 2.0]
    w_m_candidates = [0.0, 0.5, 1.0, 1.5]
    w_e_candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for wd in w_d_candidates:
        for wm in w_m_candidates:
            for we in w_e_candidates:
                if time.time() > DEADLINE - 20.0:
                    break
                
                scores = wd * (-D_n) + wm * MeanND_n + we * eig_vec
                
                # Add a small random noise to break ties and explore
                # Actually, let's just evaluate this deterministic score first
                sz = greedy_decode(scores)
                
                if sz > best_size:
                    best_size = sz
                    best_scores = scores.copy()
    
    # --- Local Refinement via Pair Swaps ---
    # The greedy result is sensitive to the exact order.
    # We can improve by swapping the priority of pairs of vertices.
    # If swapping v_i and v_j increases the independent set size, we keep it.
    
    current_scores = best_scores.copy()
    current_size = best_size
    rng = np.random.default_rng(42)
    
    # We will perform a series of randomized local search steps.
    # Each step picks a random pair and tries to swap them.
    # To be efficient, we can restrict swaps to vertices that are "close" in rank
    # or just random. Random is safer for coverage.
    
    max_swap_attempts = 50000
    no_improve_count = 0
    max_no_improve = 500
    
    for _ in range(max_swap_attempts):
        if time.time() > DEADLINE - 5.0:
            break
        
        if no_improve_count > max_no_improve:
            # Restart with a small perturbation to escape local optima
            current_scores += rng.normal(0, 0.01, size=N)
            current_size = greedy_decode(current_scores)
            if current_size > best_size:
                best_size = current_size
                best_scores = current_scores.copy()
            no_improve_count = 0
            continue
            
        # Pick two distinct random vertices
        i, j = rng.choice(N, size=2, replace=False)
        
        # Swap scores
        current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        
        new_size = greedy_decode(current_scores)
        
        if new_size > current_size:
            current_size = new_size
            no_improve_count = 0
            if new_size > best_size:
                best_size = new_size
                best_scores = current_scores.copy()
        else:
            # Revert swap
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            no_improve_count += 1
            
    return best_scores
