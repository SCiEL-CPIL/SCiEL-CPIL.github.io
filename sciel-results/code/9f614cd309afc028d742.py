# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    def score(points):
        """Calculate R = (d_min/d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return ratio ** 2

    def objective(params):
        """Minimize -R"""
        points = params.reshape((n, 2))
        return -score(points)

    # Start with a 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.flatten(), Y.flatten()])
    best_points = points.copy()
    best_score = score(points)

    # Try multiple random initializations and optimize
    np.random.seed(42)
    n_trials = 50
    for trial in range(n_trials):
        if time.time() > deadline - 60:
            break

        # Random initialization near the grid
        if trial == 0:
            init = best_points.copy()
        else:
            init = best_points + np.random.randn(n, 2) * 0.1

        result = minimize(objective, init.flatten(), method='L-BFGS-B', 
                         options={'maxiter': 1000, 'ftol': 1e-10})
        
        if result.success and result.fun < 0:
            trial_score = -result.fun
            if trial_score > best_score:
                best_score = trial_score
                best_points = result.x.reshape((n, 2))

    # Final check
    if time.time() > deadline - 30:
        return best_points

    # One more refinement with tighter tolerance
    result = minimize(objective, best_points.flatten(), method='L-BFGS-B',
                     options={'maxiter': 5000, 'ftol': 1e-12})
    if result.success and -result.fun > best_score:
        best_points = result.x.reshape((n, 2))

    return best_points

# EVOLVE-BLOCK-END
