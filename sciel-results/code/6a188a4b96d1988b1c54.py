import numpy as np
import time

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 150s kill

    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=bool)
    A_float = A.astype(float)

    # 1. Degree
    deg = A.sum(axis=1).astype(float)
    has_edge = deg > 0

    # 2. Mean Neighbor Degree
    sum_neighbor_deg = A_float @ deg
    mean_neighbor_deg = np.zeros(n, dtype=float)
    mean_neighbor_deg[has_edge] = sum_neighbor_deg[has_edge] / deg[has_edge]

    # 3. Sum of Inverse Neighbor Degrees (lower = neighbors are hubs)
    inv_deg = 1.0 / (deg + 1.0)
    sum_inv_neighbor_deg = A_float @ inv_deg

    # 4. Exact Triangle Count per vertex
    A2 = A_float @ A_float
    tri_counts = 0.5 * np.sum(A_float * A2, axis=1)

    # 5. Clustering Coefficient (local density)
    max_tri = deg * (deg - 1) / 2.0
    max_tri[max_tri == 0] = 1.0
    clustering = tri_counts / max_tri
    clustering[~has_edge] = 0.0

    def zscore(x):
        x = x.astype(float)
        mean = x.mean()
        std = x.std()
        if std < 1e-9:
            return np.zeros_like(x)
        return (x - mean) / std

    z_deg = zscore(deg)
    z_mean_nbr = zscore(mean_neighbor_deg)
    z_sum_inv = zscore(sum_inv_neighbor_deg)
    z_tri = zscore(tri_counts)
    z_clust = zscore(clustering)

    # Base structural score
    # -deg: low degree good
    # +mean_nbr: high neighbor degree good (block hubs)
    # -sum_inv: low sum of inverse (neighbors are high-degree) good
    # -tri: low triangle (sparse, less local redundancy) good
    # -clust: low clustering (less dense local structure) good
    w_deg = 1.0
    w_mean_nbr = 0.5
    w_sum_inv = 0.5
    w_tri = 0.2
    w_clust = 0.3

    scores = (-w_deg * z_deg
              + w_mean_nbr * z_mean_nbr
              - w_sum_inv * z_sum_inv
              - w_tri * z_tri
              - w_clust * z_clust)

    BIG = 1e9
    scores[~has_edge] = BIG

    if time.time() > deadline:
        return scores

    # Spectral Refinement
    # Vertices with high magnitude in low-frequency non-trivial eigenvectors
    # are at the boundaries of communities/clusters, which are "safer" to pick
    try:
        if time.time() < deadline - 10.0:
            D = np.diag(deg)
            L = D - A_float
            
            eigvals, eigvecs = np.linalg.eigh(L)
            
            # Take the first 3 non-trivial eigenvectors (indices 1, 2, 3)
            # Skip index 0 (constant)
            spectral_scores = np.zeros(n)
            num_vecs = min(3, n - 1)
            for k in range(1, num_vecs + 1):
                spectral_scores += np.abs(eigvecs[:, k])
            
            s_max = spectral_scores.max()
            if s_max > 1e-12:
                spectral_scores = spectral_scores / s_max
            else:
                spectral_scores = np.zeros(n)
                
            # Combine scores
            # Normalize base scores to [0, 1] for weighting
            mask_non_iso = has_edge
            if mask_non_iso.any():
                s_min = scores[mask_non_iso].min()
                s_max_base = scores[mask_non_iso].max()
                s_range = max(s_max_base - s_min, 1e-12)
                s_norm = (scores - s_min) / s_range
                s_norm[~mask_non_iso] = 1.0
            else:
                s_norm = np.zeros(n)

            w_base = 1.0
            w_spec = 0.5
            
            scores = w_base * s_norm + w_spec * spectral_scores
            
            scores[~has_edge] = BIG
    except Exception:
        pass

    if time.time() > deadline:
        return scores

    # Fixed-point refinement: s[i] = 1 / (1 + sum_{j in nbr(i)} s[j])
    # This approximates a "soft" independent set where neighbors compete.
    # Vertices with high scores have low-scoring neighbors (less competition).
    try:
        if time.time() < deadline - 5.0:
            # Initialize s with normalized scores (higher is better)
            s = scores.copy()
            s_min = s[has_edge].min() if has_edge.any() else 0
            s_max = s[has_edge].max() if has_edge.any() else 1
            s_range = max(s_max - s_min, 1e-12)
            s = (s - s_min) / s_range
            s[~has_edge] = BIG
            
            # We want to amplify the difference between "safe" and "unsafe" vertices.
            # The iteration s = 1/(1+sum_nbr_s) makes s high when neighbors have low s.
            # This is a bit counter-intuitive if we want high score = safe.
            # Let's invert: let u = 1/s (or similar) or just use the previous logic.
            
            # Actually, let's use the logic from Inspiration 1 directly but blend it.
            # s[i] = 1 / (1 + sum_{j in nbr(i)} s[j])
            # If i has high degree neighbors with high s, i's s drops.
            # This effectively propagates "unavailability".
            
            s_fp = 1.0 / (deg + 1.0).copy()
            s_fp[~has_edge] = BIG
            
            for it in range(50):
                if time.time() > deadline - 2.0:
                    break
                alpha = 0.5
                nbr_sum = A_float @ s_fp
                new_s_fp = 1.0 / (1.0 + nbr_sum)
                new_s_fp[~has_edge] = BIG
                damped_s_fp = (1 - alpha) * s_fp + alpha * new_s_fp
                damped_s_fp[~has_edge] = BIG
                
                if np.allclose(damped_s_fp, s_fp, rtol=1e-5, atol=1e-7):
                    s_fp = damped_s_fp
                    break
                s_fp = damped_s_fp
            
            # Blend fixed-point score with structural score
            # Normalize s_fp
            s_fp_min = s_fp[has_edge].min() if has_edge.any() else 0
            s_fp_max = s_fp[has_edge].max() if has_edge.any() else 1
            s_fp_range = max(s_fp_max - s_fp_min, 1e-12)
            s_fp_norm = (s_fp - s_fp_min) / s_fp_range
            s_fp_norm[~has_edge] = 1.0
            
            scores = 0.7 * s_norm + 0.3 * s_fp_norm
            scores[~has_edge] = BIG
    except Exception:
        pass

    if time.time() > deadline:
        return scores

    # Final Tie-breaker: Lower degree is slightly preferred
    if deg.max() > 0:
        deg_rank = -deg / (deg.max() + 1)
    else:
        deg_rank = 0.0
    scores = scores + 1e-6 * deg_rank

    if time.time() > deadline:
        return scores

    return scores
