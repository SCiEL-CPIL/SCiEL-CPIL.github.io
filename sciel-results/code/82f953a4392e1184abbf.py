# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16

    def objective(x):
        points = x.reshape(n, 2)
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return -(d_min / d_max)

    # Generate initial configurations
    initial_points = []

    # 4x4 grid
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    initial_points.append(grid.flatten())

    # Points on a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_points.append(circle.flatten())

    # Hexagonal pattern
    hex_points = []
    hex_points.append([0, 0])
    for k in range(6):
        angle = k * np.pi / 3
        hex_points.append([np.cos(angle), np.sin(angle)])
    for k in range(12):
        angle = k * np.pi / 6
        r = 2
        hex_points.append([r * np.cos(angle), r * np.sin(angle)])
    hex_points = hex_points[:n]
    initial_points.append(np.array(hex_points).flatten())

    # Random starts
    np.random.seed(42)
    for _ in range(30):
        initial_points.append(np.random.randn(n * 2))

    best_points = None
    best_score = -np.inf

    for x0 in initial_points:
        result = minimize(objective, x0, method='SLSQP', options={'maxiter': 3000, 'ftol': 1e-12})

        if result.success:
            points = result.x.reshape(n, 2)
            dists = pdist(points)
            if len(dists) > 0:
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max > 0:
                    score = d_min / d_max
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()

    if best_points is None:
        best_points = initial_points[0].reshape(n, 2)

    return best_points

# EVOLVE-BLOCK-END
