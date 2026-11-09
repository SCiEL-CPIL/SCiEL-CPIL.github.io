import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 240.0  # Safety margin before 300s kill

    n = distance_matrix.shape[0]
    dm = distance_matrix.astype(float)
    d = demands.astype(float)
    coords = np.asarray(coordinates, dtype=float)

    # --- 1. Build weighted graph Laplacian ---
    # Weight w[i,j] = 1 / (distance + eps)  (closer = stronger edge)
    eps = 1e-9
    W = 1.0 / (dm + eps)
    np.fill_diagonal(W, 0.0)

    # Degree matrix
    degree = W.sum(axis=1)
    D = np.diag(degree)

    # Laplacian L = D - W
    L = D - W

    # --- 2. Compute Fiedler vector (2nd smallest eigenvector) ---
    # Use scipy for efficient eigendecomposition
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        L_sp = csr_matrix(L)
        # Find the 3 smallest eigenvalues/vectors (index 0 is trivial, 1 is Fiedler)
        k = min(3, n - 1)
        eigenvalues, eigenvectors = eigsh(L_sp, k=k, which='SM')
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        fiedler = eigenvectors[:, 1]  # 2nd smallest
    except Exception:
        # Fallback: use numpy eig for small matrices
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            fiedler = eigenvectors[:, 1]
        except Exception:
            fiedler = np.random.randn(n)
            fiedler -= fiedler.mean()

    # --- 3. Compute edge attractiveness based on spectral coherence ---
    # For edge (i, j), compute how "coherent" it is in the spectral embedding.
    # Edges within the same cluster (similar Fiedler values) get high attractiveness.
    # Edges crossing cluster boundaries (different Fiedler values) get low attractiveness.

    # Spectral distance: |fiedler[i] - fiedler[j]|
    fiedler_diff = np.abs(fiedler[:, None] - fiedler[None, :])

    # Normalize fiedler_diff to [0, 1]
    max_fd = fiedler_diff.max()
    if max_fd > 0:
        fiedler_diff_norm = fiedler_diff / max_fd
    else:
        fiedler_diff_norm = np.zeros_like(fiedler_diff)

    # Attractiveness: high when fiedler_diff is small (same cluster)
    # Use exponential decay: exp(-alpha * fiedler_diff_norm)
    # Also incorporate distance: closer edges are more attractive
    base = 1.0 / (dm + eps)

    # Combine: spectral coherence * distance proximity
    alpha = 2.0  # Controls how strongly we penalize cross-cluster edges
    spectral_factor = np.exp(-alpha * fiedler_diff_norm)

    # Heuristic: product of base distance attractiveness and spectral coherence
    heuristic = base * spectral_factor

    # --- 4. Additional refinement: use higher-order spectral info ---
    # If we have more eigenvectors, use them to build a richer embedding
    try:
        if eigenvectors.shape[1] >= 3:
            # Use 2nd and 3rd eigenvectors for 2D embedding
            embed_2d = eigenvectors[:, 1:3]
            # Compute pairwise distances in embedding space
            diff_2d = embed_2d[:, None, :] - embed_2d[None, :, :]
            embed_dist = np.sqrt((diff_2d ** 2).sum(axis=2))
            max_ed = embed_dist.max()
            if max_ed > 0:
                embed_dist_norm = embed_dist / max_ed
            else:
                embed_dist_norm = np.zeros_like(embed_dist)
            spectral_factor_2d = np.exp(-alpha * embed_dist_norm)
            # Blend with 1D spectral factor
            spectral_factor = 0.5 * spectral_factor + 0.5 * spectral_factor_2d
            heuristic = base * spectral_factor
    except Exception:
        pass

    # --- 5. Greedy evaluation to tune alpha ---
    def _greedy_tour_cost(h, dm, demands, cap):
        current = 0
        remaining_cap = cap
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        total_dist = 0.0

        while True:
            eligible = [j for j in range(1, n) if not visited[j] and demands[j] <= remaining_cap]
            if not eligible:
                total_dist += dm[current, 0]
                current = 0
                remaining_cap = cap
                if visited[1:].all():
                    break
                continue

            # Pick best eligible by heuristic
            best_j = max(eligible, key=lambda j: h[current, j])
            total_dist += dm[current, best_j]
            remaining_cap -= demands[best_j]
            visited[best_j] = True
            current = best_j

            if visited[1:].all():
                total_dist += dm[current, 0]
                break

        return total_dist

    # Try several alpha values
    best_alpha = 2.0
    best_cost = float('inf')

    alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    for a in alphas:
        if time.time() > deadline - 15:
            break

        spectral_factor_a = np.exp(-a * fiedler_diff_norm)
        h_test = base * spectral_factor_a
        h_test = np.where(h_test > 0, h_test, 1e-9)

        cost = _greedy_tour_cost(h_test, dm, d, capacity)

        if cost < best_cost:
            best_cost = cost
            best_alpha = a

    # Final heuristic with best alpha
    spectral_factor_final = np.exp(-best_alpha * fiedler_diff_norm)
    heuristic = base * spectral_factor_final

    # Ensure positivity and finiteness
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)

    # Shape check
    if heuristic.shape != dm.shape:
        heuristic = 1.0 / (dm + eps)

    return heuristic
