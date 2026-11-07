# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Derivative-free optimizer using adaptive coordinate descent with
    dynamic rotation of the search basis via principal directions of
    recent evaluation history. Uses top-2 principal directions to align
    with ridge structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        B = self.budget

        # Initialize at origin (center of box) for f13 sharp ridge
        x = np.zeros(d)
        fx = func(x)
        calls = 1

        # Maintain a history of (direction, improvement) pairs for PCA
        history_dirs = []  # list of unit vectors of successful moves
        history_imps = []  # magnitudes of improvement

        # Initial basis: identity
        basis = np.eye(d)

        # Per-axis step sizes for adaptive line search
        steps = np.full(d, 0.5)

        # PCA update interval
        pca_interval = 15
        # Minimum number of history entries before rotating
        min_hist = 8

        # Track consecutive failures per axis for step adjustment
        fail_counts = np.zeros(d, dtype=int)

        for i in range(B - 1):
            # Determine search directions: current basis vectors
            # For each axis in the rotated frame, try a line search
            for k in range(d):
                if calls >= B:
                    break
                direction = basis[:, k]
                # Adaptive line search: try several step sizes
                best_x = x
                best_f = fx
                found_improvement = False

                # Try positive and negative directions with decreasing steps
                for sign in [1, -1]:
                    for mult in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                        if calls >= B:
                            break
                        trial = x + sign * steps[k] * mult * direction
                        trial = np.clip(trial, -5, 5)
                        ft = func(trial)
                        calls += 1
                        if ft < best_f:
                            best_f = ft
                            best_x = trial
                            found_improvement = True

                if found_improvement and best_f < fx:
                    # Record the move
                    move = best_x - x
                    norm = np.linalg.norm(move)
                    if norm > 1e-12:
                        history_dirs.append(move / norm)
                        history_imps.append(fx - best_f)
                    x = best_x
                    fx = best_f
                    # Increase step size on success
                    steps[k] = min(steps[k] * 1.2, 3.0)
                    fail_counts[k] = 0
                else:
                    # No improvement; shrink step
                    steps[k] = max(steps[k] * 0.7, 0.005)
                    fail_counts[k] += 1

            # Periodically update the basis using PCA of recent history
            if len(history_dirs) >= min_hist and (i + 1) % pca_interval == 0:
                # Take the most recent history entries
                n_recent = min(len(history_dirs), 20)
                recent_dirs = np.array(history_dirs[-n_recent:])
                recent_imps = np.array(history_imps[-n_recent:])
                # Weight by improvement magnitude (normalized)
                weights = recent_imps / (np.sum(recent_imps) + 1e-12)
                # Weighted covariance
                cov = np.zeros((d, d))
                for j in range(len(recent_dirs)):
                    w = weights[j]
                    cov += w * np.outer(recent_dirs[j], recent_dirs[j])
                # Eigendecomposition to get principal directions
                try:
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    # Sort by eigenvalue descending
                    idx = np.argsort(eigvals)[::-1]
                    
                    # Use top 2 principal directions explicitly, fill rest with orthogonal complement
                    top_dirs = eigvecs[:, idx[:2]]
                    
                    # Build orthonormal basis starting with top 2 directions
                    # Use QR to orthonormalize the top 2
                    Q, R = np.linalg.qr(top_dirs)
                    # Ensure right-handed system for the first two
                    if np.linalg.det(Q[:, :2]) < 0:
                        Q[:, 0] = -Q[:, 0]
                    
                    # Complete the basis with remaining eigenvectors
                    if d > 2:
                        remaining = eigvecs[:, idx[2:]]
                        # Orthonormalize against Q
                        for col in remaining.T:
                            for j in range(Q.shape[1]):
                                col -= np.dot(col, Q[:, j]) * Q[:, j]
                        # QR on the combined set
                        combined = np.column_stack([Q, remaining])
                        Q_full, R_full = np.linalg.qr(combined)
                        if np.linalg.det(Q_full) < 0:
                            Q_full[:, 0] = -Q_full[:, 0]
                        basis = Q_full
                    else:
                        basis = Q
                except np.linalg.LinAlgError:
                    pass

            # Keep history bounded
            if len(history_dirs) > 40:
                history_dirs = history_dirs[-40:]
                history_imps = history_imps[-40:]

        return x, fx
# EVOLVE-BLOCK-END
