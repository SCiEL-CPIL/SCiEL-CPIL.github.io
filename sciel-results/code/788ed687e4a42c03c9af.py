# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2
    deadline = time.time() + 600 - 30  # 30s safety margin

    # Precompute pair indices for efficient distance calculation
    i_idx, j_idx = np.triu_indices(n, k=1)
    num_pairs = len(i_idx)

    def compute_score(pts):
        """Compute R = (d_min / d_max)^2"""
        if pts.shape != (n, 2):
            return 0.0
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """Objective to minimize: -log(d_min) + log(d_max) (equivalent to minimizing d_max/d_min)"""
        pts = x.reshape(n, 2)
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min < 1e-10 or d_max < 1e-10:
            return 1e10
        return np.log(d_max) - np.log(d_min)

    def objective_grad(x):
        """Gradient of objective: d/dx [log(d_max) - log(d_min)]"""
        pts = x.reshape(n, 2)
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min < 1e-10 or d_max < 1e-10:
            return np.zeros_like(x)
        
        grad = np.zeros_like(x)
        
        # Gradient of log(d_max)
        idx_max = np.argmax(d)
        i, j = i_idx[idx_max], j_idx[idx_max]
        diff = pts[i] - pts[j]
        grad[2*i] += diff / d_max
        grad[2*i+1] += diff[1] / d_max
        grad[2*j] -= diff / d_max
        grad[2*j+1] -= diff[1] / d_max
        
        # Gradient of -log(d_min)
        idx_min = np.argmin(d)
        i, j = i_idx[idx_min], j_idx[idx_min]
        diff = pts[i] - pts[j]
        grad[2*i] -= diff / d_min
        grad[2*i+1] -= diff[1] / d_min
        grad[2*j] += diff / d_min
        grad[2*j+1] += diff[1] / d_min
        
        return grad

    # Generate initial configurations
    def generate_hexagonal():
        """Generate a hexagonal-like pattern for 16 points"""
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # Ring 1: 6 points
        for k in range(6):
            angle = k * np.pi / 3
            pts.append([np.cos(angle), np.sin(angle)])
        # Ring 2: 9 points (approximate hexagonal)
        for k in range(9):
            angle = k * 2 * np.pi / 9 + np.pi/9
            r = 2.0
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        return np.array(pts[:16])

    def generate_random(seed):
        """Generate random points"""
        np.random.seed(seed)
        return np.random.randn(n, 2)

    def generate_grid():
        """Generate a 4x4 grid"""
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((xx.ravel(), yy.ravel()))

    # Initial points: hexagonal, random, grid
    initial_configs = [
        generate_hexagonal().flatten(),
        generate_random(42).flatten(),
        generate_grid().flatten(),
        generate_random(123).flatten(),
        generate_random(999).flatten(),
    ]

    best_pts = None
    best_score = -1.0

    # Evaluate initial configs
    for cfg in initial_configs:
        pts = cfg.reshape(n, 2)
        score = compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    # Run optimization from each initial config
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective,
                cfg,
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline:
                break
            
            pts = result.x.reshape(n, 2)
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue

    # If we have time, do more random restarts
    num_restarts = 0
    while time.time() < deadline - 5:
        seed = 1000 + num_restarts
        cfg = generate_random(seed).flatten()
        try:
            result = minimize(
                objective,
                cfg,
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline:
                break
            
            pts = result.x.reshape(n, 2)
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass
        num_restarts += 1

    if best_pts is None:
        # Fallback to random
        np.random.seed(42)
        best_pts = np.random.randn(n, 2)

    return best_pts

# EVOLVE-BLOCK-END
