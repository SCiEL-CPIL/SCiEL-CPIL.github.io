import numpy as np
import time

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s / 150s kill

    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=bool)
    A_float = A.astype(float)

    deg = A.sum(axis=1).astype(float)
    has_edge = deg > 0

    # Isolated vertices: always safe to pick -> top priority
    BIG = 1e9

    # Initialize with inverse degree
    s = 1.0 / (deg + 1.0).copy()
    s[~has_edge] = BIG

    # Fixed-point iteration with damping for smoother convergence
    # s[i] = 1 / (1 + sum_{j in nbr(i)} s[j])
    # This captures "safety" — vertices whose neighbors have low total score are safer
    alpha = 0.5  # damping factor
    for it in range(300):
        if time.time() > deadline - 10.0:
            break
        nbr_sum = A_float @ s
        new_s = 1.0 / (1.0 + nbr_sum)
        new_s[~has_edge] = BIG
        # Damped update for stability
        damped_s = (1 - alpha) * s + alpha * new_s
        damped_s[~has_edge] = BIG
        if np.allclose(damped_s, s, rtol=1e-7, atol=1e-9):
            s = damped_s
            break
        s = damped_s

    # Spectral refinement: use Fiedler vector magnitude as a secondary signal
    # Vertices at the periphery (high |Fiedler component|) are less central
    # and thus safer to include in an independent set
    try:
        if time.time() < deadline - 5.0:
            D = np.diag(deg)
            L = D - A_float
            
            if n <= 500:
                eigvals, eigvecs = np.linalg.eigh(L)
                positive_mask = eigvals > 1e-10
                if positive_mask.any():
                    idx = np.where(positive_mask)[0][0]
                    fiedler = eigvecs[:, idx]
                else:
                    fiedler = np.zeros(n)
                
                fiedler_score = np.abs(fiedler)
                
                fmax = fiedler_score.max()
                if fmax > 1e-12:
                    fiedler_score = fiedler_score / fmax
                else:
                    fiedler_score = np.zeros(n)
                
                # Adaptive spectral weight: stronger when base scores are close
                s_spread = s.max() - s.min()
                if s_spread > 1e-12:
                    spectral_weight = min(0.3, 0.1 + 0.2 * (1.0 / (1.0 + s_spread)))
                else:
                    spectral_weight = 0.3
                
                scores = s + spectral_weight * fiedler_score
            else:
                scores = s.copy()
        else:
            scores = s.copy()
    except Exception:
        scores = s.copy()

    # Local neighborhood refinement: adjust scores based on neighbor score distribution
    # Vertices with neighbors that have very high scores are more "contested"
    try:
        if time.time() < deadline - 2.0:
            # Compute neighbor score statistics
            nbr_max = np.zeros(n)
            nbr_mean = np.zeros(n)
            for i in range(n):
                if has_edge[i]:
                    nbrs = np.where(A[i])[0]
                    if len(nbrs) > 0:
                        nbr_max[i] = scores[nbrs].max()
                        nbr_mean[i] = scores[nbrs].mean()
            
            # Penalty for being adjacent to high-scoring vertices
            # Normalize the penalty
            max_nbr_max = nbr_max.max()
            if max_nbr_max > 1e-12:
                contested_penalty = nbr_max / max_nbr_max
                # Subtract a small penalty for high contestation
                scores = scores - 0.05 * contested_penalty
                scores[~has_edge] = BIG
            else:
                scores[~has_edge] = BIG
        else:
            scores[~has_edge] = BIG
    except Exception:
        scores[~has_edge] = BIG

    # Isolated vertices must remain at top
    scores[~has_edge] = BIG

    # Small tie-breaker: lower degree slightly preferred
    deg_rank = -deg / (deg.max() + 1) if deg.max() > 0 else 0.0
    scores = scores + 1e-6 * deg_rank

    if time.time() > deadline:
        return scores

    return scores
