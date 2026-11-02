import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

def construct():
    # Number of unique distances
    M = N * (N - 1) // 2
    
    def unpack_D(params):
        D_mat = np.zeros((N, N))
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                D_mat[i, j] = params[idx]
                D_mat[j, i] = params[idx]
                idx += 1
        return D_mat

    def edm_to_gram(D_mat):
        J = np.eye(N) - np.ones((N, N)) / N
        G = -0.5 * J @ D_mat @ J
        return G

    def gram_to_coords(G):
        # Project to PSD and rank D
        # Symmetrize
        G = (G + G.T) / 2
        evals, evecs = np.linalg.eigh(G)
        # Keep top D eigenvalues, clip negatives to 0
        evals_clipped = np.clip(evals, 0, None)
        # We want rank D. So we keep the top D.
        # If we just clip negatives, rank might be > D.
        # To enforce rank D, we zero out eigenvalues beyond D.
        # But which ones? The largest D.
        # Sort eigenvalues in descending order
        sort_idx = np.argsort(evals_clipped)[::-1]
        evals_sorted = evals_clipped[sort_idx]
        evecs_sorted = evecs[:, sort_idx]
        
        # Keep top D
        top_evals = evals_sorted[:D]
        top_evecs = evecs_sorted[:, :D]
        
        # Reconstruct G_rank_D
        G_rank = top_evecs @ np.diag(top_evals) @ top_evecs.T
        # Coordinates
        X = top_evecs * np.sqrt(top_evals)
        return X

    def objective(params):
        D_mat = unpack_D(params)
        
        # Ensure D is non-negative? 
        # The ratio is on distances. If D has negatives, it's invalid.
        # We can penalize negative distances.
        neg_pen = np.sum(np.maximum(0, -D_mat[~np.eye(N, dtype=bool)]))
        
        G = edm_to_gram(D_mat)
        
        # Eigenvalues for constraints
        evals = np.linalg.eigvalsh(G)
        
        # PSD Penalty: sum of negative eigenvalues
        psd_pen = np.sum(np.maximum(0, -evals))
        
        # Rank Penalty: sum of eigenvalues beyond rank D
        # Sort ascending to get smallest at end? No, we want to penalize the "extra" rank.
        # If we want rank <= D, we should penalize the eigenvalues that are NOT in the top D.
        # So sort descending, take indices D to N-1.
        sort_idx = np.argsort(evals)[::-1]
        evals_sorted = evals[sort_idx]
        rank_pen = np.sum(evals_sorted[D:])
        
        # Compute ratio from D_mat (or G)
        # Using D_mat is direct.
        dists_sq = D_mat[~np.eye(N, dtype=bool)]
        min_d = np.min(dists_sq)
        max_d = np.max(dists_sq)
        
        if min_d < 1e-12:
            return 1e10 + neg_pen + psd_pen + rank_pen
        
        ratio = max_d / min_d
        
        # Penalty weights
        # We want the constraints to be strong initially.
        # Dynamic penalty or fixed? Fixed is simpler.
        alpha = 100.0
        total = ratio + alpha * (neg_pen + psd_pen + rank_pen)
        
        return total

    # Initialization
    # Start with a Fibonacci sphere configuration to have a good initial EDM
    golden_angle = np.pi * (3 - np.sqrt(5))
    X_init = np.zeros((N, D))
    for i in range(N):
        z = 1 - 2 * i / (N - 1)
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        X_init[i, 0] = r * np.cos(theta)
        X_init[i, 1] = r * np.sin(theta)
        X_init[i, 2] = z
        
    # Compute initial EDM
    dists_init = np.sum((X_init[:, np.newaxis, :] - X_init[np.newaxis, :, :])**2, axis=2)
    # Extract upper triangle
    idx = np.triu_indices(N, 1)
    D_init = dists_init[idx]
    
    # Optimize
    res = minimize(objective, D_init, method='Powell', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    # Recover coordinates from the optimized EDM
    D_opt = unpack_D(res.x)
    G_opt = edm_to_gram(D_opt)
    X_opt = gram_to_coords(G_opt)
    
    # Ensure distinct points (add small noise if needed, though optimization should handle it)
    # Check if points are distinct
    # If not, fallback
    if np.any(np.isclose(X_opt[:, np.newaxis, :], X_opt[np.newaxis, :, :], atol=1e-6)):
         # Fallback to init
         return X_init
         
    return X_opt
