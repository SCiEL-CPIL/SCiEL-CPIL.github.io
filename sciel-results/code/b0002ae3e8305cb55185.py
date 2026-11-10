# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    deadline = time.time() + 600.0
    
    def compute_metric(points):
        """Compute (d_min/d_max)^2"""
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0
        return (d_min / d_max) ** 2
    
    def soft_min_max_objective(params, beta):
        """
        Smooth surrogate for -log(d_min/d_max) using log-sum-exp.
        We minimize this to maximize d_min/d_max.
        """
        points = params.reshape(n, d)
        dists_sq = pdist(points, 'sqeuclidean')
        # Use squared distances for numerical stability
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-15:
            return 1e10
        ratio = d_min_sq / d_max_sq
        if ratio < 1e-30:
            return 1e10
        return -np.log(ratio)
    
    def objective_with_beta(params, beta):
        return soft_min_max_objective(params, beta)
    
    # Initial guess: Fibonacci sphere
    def fibonacci_sphere(n_points, radius=1.0):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2
            radius_at_y = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            points.append([x * radius, y * radius, z * radius])
        return np.array(points)
    
    # Try multiple initializations
    best_points = None
    best_score = -1.0
    
    initializations = []
    
    # 1. Fibonacci sphere
    initializations.append(fibonacci_sphere(n))
    
    # 2. Regular tetrahedron + extra points (symmetric start)
    # Start with octahedron (6 points) and add more
    octa = np.array([
        [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
    ], dtype=float)
    # Add 8 more points at face centers of octahedron (which form a cube)
    cube = np.array([
        [1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],
        [-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]
    ], dtype=float) / np.sqrt(3)
    init2 = np.vstack([octa, cube])
    initializations.append(init2)
    
    # 3. Random with fixed seed
    np.random.seed(42)
    init3 = np.random.randn(n, d)
    init3 /= np.linalg.norm(init3, axis=1, keepdims=True)
    initializations.append(init3)
    
    # 4. Another random
    np.random.seed(123)
    init4 = np.random.randn(n, d)
    init4 /= np.linalg.norm(init4, axis=1, keepdims=True)
    initializations.append(init4)
    
    # 5. Icosahedron-like (12 points) + 2 more
    # Regular icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 2 points at poles
    poles = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
    init5 = np.vstack([ico, poles])
    initializations.append(init5)
    
    for init in initializations:
        if time.time() > deadline - 60:
            break
        
        params = init.flatten()
        
        # Adaptive beta annealing
        betas = [50, 80, 120, 200, 300, 500]
        current_params = params.copy()
        
        for beta in betas:
            if time.time() > deadline - 30:
                break
            
            # Use SLSQP optimization
            try:
                result = minimize(
                    lambda p: objective_with_beta(p, beta),
                    current_params,
                    method='SLSQP',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success or result.fun < 1e10:
                    current_params = result.x
            except Exception:
                pass
        
        # Final high-precision polish with large beta
        if time.time() > deadline - 15:
            break
        
        try:
            result = minimize(
                lambda p: objective_with_beta(p, 1000),
                current_params,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.success or result.fun < 1e10:
                current_params = result.x
        except Exception:
            pass
        
        points = current_params.reshape(n, d)
        score = compute_metric(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # If no good result, fallback to random
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    return best_points


# EVOLVE-BLOCK-END
