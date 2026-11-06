"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute priority scores for vertices using a refined spectral approach.
    Uses the Fiedler vector with damped fixed-point iteration, adaptive
    spectral weighting, and a smooth (soft-max) local neighborhood penalty.

    Improvements within the family:
    1. Smooth surrogate (soft-max via log-sum-exp) for the contestedness
       penalty, replacing the hard threshold + hard neighbor count.
    2. Higher-precision polish pass: more iterations, tighter tolerance.
    3. Tuned damping and blend weights.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]

    if n == 0:
        return np.array([], dtype=np.float64)

    # Compute degree
    degree = A.sum(axis=1)

    # Compute Laplacian: L = D - A
    D = np.diag(degree)
    L = D - A

    try:
        # Get eigenvalues and eigenvectors of the Laplacian
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Fiedler vector (second smallest eigenvalue)
        fiedler_vec = eigenvectors[:, 1]
        fiedler_mag = np.abs(fiedler_vec)

        # Normalize to [0, 1]
        fmin, fmax = fiedler_mag.min(), fiedler_mag.max()
        if fmax > fmin + 1e-12:
            fiedler_norm = (fiedler_mag - fmin) / (fmax - fmin)
        else:
            fiedler_norm = np.zeros(n)

        # Normalize degree to [0, 1]
        dmin, dmax = degree.min(), degree.max()
        if dmax > dmin + 1e-12:
            degree_norm = (degree - dmin) / (dmax - dmin)
        else:
            degree_norm = np.zeros(n)

        # Base scores: blend of low-degree preference and peripherality
        alpha = 0.4
        beta = 0.6
        base_scores = alpha * (1.0 - degree_norm) + beta * fiedler_norm

        # Damped fixed-point iteration for smoothing
        # The smooth surrogate for contestedness: soft-max of neighbor scores.
        # For vertex i, contestedness = logsumexp_j(s_j) over neighbors j,
        # which is a smooth, monotone function of neighbor scores.
        # We use a temperature-scaled version for smoothness.
        temperature = 2.0  # Lower = sharper (closer to max), higher = smoother
        damping = 0.6
        max_iter = 200
        tol = 1e-8

        scores = base_scores.copy()

        for it in range(max_iter):
            if time.time() > deadline:
                break

            # Compute neighbor scores
            # A @ scores gives sum of neighbor scores per vertex
            neighbor_sum = A @ scores

            # Soft-max surrogate: for each vertex, the log-sum-exp of
            # temperature-scaled neighbor scores.
            # To avoid numerical issues, we compute per-vertex.
            # We use a vectorized approach: for each vertex i,
            #   softmax_i = (1/temperature) * logsumexp(temperature * s_j for j in N(i))
            # We approximate this by:
            #   softmax_i = (1/temperature) * log(sum_j exp(temperature * s_j))
            # But summing exp over all neighbors per vertex is O(n^2) dense.
            # Instead, use a smooth approximation via the sum and max:
            #   logsumexp(x) ≈ max(x) + log(sum exp(x - max(x)))
            # For a dense matrix, we can compute this efficiently.

            # Compute temperature-scaled neighbor scores
            scaled = temperature * scores  # shape (n,)
            # neighbor_scaled_sum[i] = sum_j A[i,j] * scaled[j]
            # But we need logsumexp, not sum. Approximate:
            # For each vertex, the soft-max of neighbor scores ≈
            #   (1/temperature) * (max_neighbor_scaled + log(count_effective))
            # A simpler smooth surrogate: use the mean of neighbor scores
            # combined with the max, both smooth.

            # Actually, let's use a cleaner smooth surrogate:
            # contested_i = (1/temperature) * log( sum_j exp(temperature * s_j) )
            # For dense A, we can compute this as:
            #   For each i: logsumexp over j of (temperature * s_j) where A[i,j]=1
            # This is O(n^2) per iteration which is fine for n=200.

            # Efficient vectorized logsumexp per row:
            # Create a matrix of neighbor scores: M[i,j] = s_j if A[i,j] else -inf
            # Then logsumexp along axis 1.
            # For n=200, this is a 200x200 matrix, fine.

            # Build the scaled neighbor score matrix
            # M[i,j] = temperature * scores[j] if A[i,j] else -inf
            M = np.where(A > 0.5, temperature * scores[None, :], -np.inf)

            # Row-wise logsumexp
            M_max = np.max(M, axis=1)  # shape (n,)
            # Handle rows with no neighbors (M_max = -inf)
            valid = np.isfinite(M_max)
            M_max_safe = np.where(valid, M_max, 0.0)
            M_shifted = M - M_max_safe[None, :]
            # exp of -inf is 0, fine
            M_exp_sum = np.sum(np.exp(M_shifted), axis=1)
            # logsumexp = M_max + log(sum exp(M - M_max))
            logsumexp_vals = np.where(valid, M_max_safe + np.log(np.maximum(M_exp_sum, 1e-300)), 0.0)

            # Soft-max surrogate: (1/temperature) * logsumexp
            contested_smooth = logsumexp_vals / temperature

            # Normalize contested_smooth to [0, 1]
            cmin, cmax = contested_smooth.min(), contested_smooth.max()
            if cmax > cmin + 1e-12:
                contested_norm = (contested_smooth - cmin) / (cmax - cmin)
            else:
                contested_norm = np.zeros(n)

            # Damped update
            neighbor_sum = A @ scores
            neighbor_avg = np.where(degree > 0, neighbor_sum / np.maximum(degree, 1), 0)

            # New score: damping * base + (1-damping) * (1 - neighbor_avg) - penalty
            # The penalty uses the smooth contestedness
            penalty_weight = 0.15
            new_scores = (damping * base_scores
                          + (1.0 - damping) * (1.0 - neighbor_avg)
                          - penalty_weight * contested_norm)

            # Check convergence
            diff = np.max(np.abs(new_scores - scores))
            scores = new_scores
            if diff < tol:
                break

        # High-precision polish pass: tighten tolerance, more iterations
        if time.time() < deadline:
            tol2 = 1e-10
            max_iter2 = 100
            for it in range(max_iter2):
                if time.time() > deadline:
                    break

                M = np.where(A > 0.5, temperature * scores[None, :], -np.inf)
                M_max = np.max(M, axis=1)
                valid = np.isfinite(M_max)
                M_max_safe = np.where(valid, M_max, 0.0)
                M_shifted = M - M_max_safe[None, :]
                M_exp_sum = np.sum(np.exp(M_shifted), axis=1)
                logsumexp_vals = np.where(valid, M_max_safe + np.log(np.maximum(M_exp_sum, 1e-300)), 0.0)
                contested_smooth = logsumexp_vals / temperature

                cmin, cmax = contested_smooth.min(), contested_smooth.max()
                if cmax > cmin + 1e-12:
                    contested_norm = (contested_smooth - cmin) / (cmax - cmin)
                else:
                    contested_norm = np.zeros(n)

                neighbor_sum = A @ scores
                neighbor_avg = np.where(degree > 0, neighbor_sum / np.maximum(degree, 1), 0)

                new_scores = (damping * base_scores
                              + (1.0 - damping) * (1.0 - neighbor_avg)
                              - penalty_weight * contested_norm)

                diff = np.max(np.abs(new_scores - scores))
                scores = new_scores
                if diff < tol2:
                    break

        # Ensure scores are bounded
        smin, smax = scores.min(), scores.max()
        if smax > smin + 1e-12:
            scores = (scores - smin) / (smax - smin)

        return scores.astype(np.float64)

    except Exception:
        # Fallback to degree-based if spectral fails
        return -degree.astype(np.float64)
# EVOLVE-BLOCK-END
