"""RBF Basis: Fixed-Center Radial Basis Functions for Symbolic Regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural Change: Fixed-Center RBF Basis.
    Uses 10 parameters as weights for 10 Gaussian RBFs with fixed centers.
    Centers are placed at corners, center, and midpoints of the normalized
    input space to ensure good coverage. The model is linear in parameters,
    so BFGS faces a convex landscape.
    """
    # Normalize inputs to [0, 1] range
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 39.0
    t_n = (temp - 1.0) / 11.0
    p_n = (pH - 1.0) / 4.0

    # Stack normalized inputs into a matrix of shape (n, 4)
    X = np.column_stack((b_n, s_n, t_n, p_n))

    # Define 10 fixed centers in the normalized 4D space:
    # 1. Center: (0.5, 0.5, 0.5, 0.5)
    # 2-9. Corners and midpoints
    # 10. Another strategic point
    centers = np.array([
        [0.5, 0.5, 0.5, 0.5],      # 1. Center
        [0.0, 0.0, 0.0, 0.0],      # 2. Min corner
        [1.0, 1.0, 1.0, 1.0],      # 3. Max corner
        [0.0, 0.0, 1.0, 1.0],      # 4. Mixed corner
        [1.0, 1.0, 0.0, 0.0],      # 5. Mixed corner
        [0.0, 1.0, 0.0, 1.0],      # 6. Mixed corner
        [1.0, 0.0, 1.0, 0.0],      # 7. Mixed corner
        [0.5, 0.0, 0.5, 0.0],      # 8. Midpoint edge
        [0.5, 1.0, 0.5, 1.0],      # 9. Midpoint edge
        [0.0, 0.5, 1.0, 0.5],      # 10. Midpoint edge
    ])

    # Compute squared distances from each point to each center
    # X: (n, 4), centers: (10, 4)
    # diff: (n, 10, 4)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff**2, axis=2)  # (n, 10)

    # RBF width parameter (gamma). Smaller gamma = wider bumps.
    # We use a fixed gamma for simplicity.
    gamma = 2.0  # 1/(2*sigma^2), so sigma = 1/sqrt(2*gamma) ≈ 0.5

    # Compute RBF activations: exp(-gamma * sq_dist)
    rbf_values = np.exp(-gamma * sq_dist)  # (n, 10)

    # Linear combination of RBF values
    pred = rbf_values @ params

    return pred

# EVOLVE-BLOCK-END
