import numpy as np
import time

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s / 150s kill

    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=bool)

    deg = A.sum(axis=1).astype(float)
    has_edge = deg > 0

    # Isolated vertices: always safe to pick -> top priority
    BIG = 1e9

    # Initialize with inverse degree
    s = 1.0 / (deg + 1.0).copy()
    s[~has_edge] = BIG

    # Fixed-point iteration: s[i] = 1 / (1 + sum_{j in nbr(i)} s[j])
    # This captures "safety" — vertices whose neighbors have low total score are safer
    for it in range(200):
        if time.time() > deadline:
            break
        nbr_sum = A.astype(float) @ s
        new_s = 1.0 / (1.0 + nbr_sum)
        new_s[~has_edge] = BIG
        if np.allclose(new_s, s, rtol=1e-6, atol=1e-8):
            s = new_s
            break
        s = new_s

    # Spectral refinement: use Fiedler vector magnitude as a secondary signal
    # Vertices at the periphery (high |Fiedler component|) are less central
    # and thus safer to include in an independent set
    try:
        if time.time() < deadline - 5.0:
            # Build Laplacian
            A_float = A.astype(float)
            D = np.diag(deg)
            L = D - A_float
            
            # For disconnected graphs, the Fiedler vector corresponds to the
            # second smallest eigenvalue. We compute it via eigendecomposition
            # of the Laplacian. For n=200, this is fast.
            if n <= 500:
                eigvals, eigvecs = np.linalg.eigh(L)
                # Sort by eigenvalue (eigh returns in ascending order)
                # Fiedler vector is the eigenvector for the second smallest eigenvalue
                # If the graph is connected, eigvals[1] is the Fiedler value
                # For disconnected graphs, there may be multiple zero eigenvalues
                # We take the eigenvector corresponding to the smallest positive eigenvalue
                positive_mask = eigvals > 1e-10
                if positive_mask.any():
                    idx = np.where(positive_mask)[0][0]
                    fiedler = eigvecs[:, idx]
                else:
                    fiedler = np.zeros(n)
                
                # Use magnitude of Fiedler vector as a score component
                # Higher magnitude = more peripheral = safer to pick
                fiedler_score = np.abs(fiedler)
                
                # Normalize fiedler_score to [0, 1]
                fmax = fiedler_score.max()
                if fmax > 1e-12:
                    fiedler_score = fiedler_score / fmax
                else:
                    fiedler_score = np.zeros(n)
                
                # Combine: base score (safety from neighbor competition) 
                # plus a small spectral component (peripherality)
                # Weight the spectral component lightly so it refines, not dominates
                spectral_weight = 0.1
                scores = s + spectral_weight * fiedler_score
            else:
                scores = s.copy()
        else:
            scores = s.copy()
    except Exception:
        scores = s.copy()

    # Isolated vertices must remain at top
    scores[~has_edge] = BIG

    # Small tie-breaker: lower degree slightly preferred
    deg_rank = -deg / (deg.max() + 1) if deg.max() > 0 else 0.0
    scores = scores + 1e-6 * deg_rank

    if time.time() > deadline:
        return scores

    return scores
