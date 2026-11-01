import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import itertools

def construct():
    # Octahedron vertices
    octa = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    
    # Cube vertices
    cube = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ], dtype=float)
    
    def get_rotation_matrix(rx, ry, rz):
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def objective(params):
        b, rx, ry, rz = params
        R = get_rotation_matrix(rx, ry, rz)
        cube_rot = (cube * b) @ R.T
        points = np.vstack([octa, cube_rot])
        
        # Check distinctness
        # If any two points are too close, return large penalty
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        max_d = np.max(dists)
        
        if min_d < 1e-6:
            return 1e10
        
        return (max_d / min_d) ** 2

    best_score = 1e10
    best_params = None
    
    # Initial guesses
    guesses = [
        [1.0, 0, 0, 0],
        [1.0, np.pi/4, 0, 0],
        [1.0, 0, np.pi/4, 0],
        [0.8, 0, 0, 0],
        [1.2, 0, 0, 0],
        [0.5, 0, 0, 0],
        [2.0, 0, 0, 0],
    ]
    
    # Add random guesses
    np.random.seed(42)
    for _ in range(10):
        b = np.random.uniform(0.5, 2.0)
        rx = np.random.uniform(-np.pi, np.pi)
        ry = np.random.uniform(-np.pi, np.pi)
        rz = np.random.uniform(-np.pi, np.pi)
        guesses.append([b, rx, ry, rz])
        
    for guess in guesses:
        try:
            res = minimize(objective, guess, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to seed
        return np.array([[i,j,k] for i in range(3) for j in range(3) for k in range(3)][:14], dtype=float)

    b, rx, ry, rz = best_params
    R = get_rotation_matrix(rx, ry, rz)
    cube_rot = (cube * b) @ R.T
    points = np.vstack([octa, cube_rot])
    
    return points
