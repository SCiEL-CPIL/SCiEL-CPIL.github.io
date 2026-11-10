# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
import sys

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _objective(points_flat):
    """Objective function to minimize: -log(d_min/d_max) with smoothing."""
    n = 16
    points = points_flat.reshape(n, 2)
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    
    # Handle edge cases
    if d_max < 1e-10 or d_min < 1e-10:
        return 1e6
    
    ratio = d_min / d_max
    if ratio <= 0:
        return 1e6
    return -np.log(ratio)

def _soft_objective(points_flat, temperature=0.01):
    """Soft objective using log-sum-exp for smoother optimization."""
    n = 16
    points = points_flat.reshape(n, 2)
    d = pdist(points)
    
    # Use soft-min and soft-max via log-sum-exp
    # soft-min = -logsumexp(-d/temperature) * temperature
    # soft-max = logsumexp(d/temperature) * temperature
    inv_d = -d / temperature
    inv_d_max = np.max(inv_d)
    soft_min = -temperature * (inv_d_max + np.log(np.sum(np.exp(inv_d - inv_d_max))))
    
    d_scaled = d / temperature
    d_max_scaled = np.max(d_scaled)
    soft_max = temperature * (d_max_scaled + np.log(np.sum(np.exp(d_scaled - d_max_scaled))))
    
    if soft_max < 1e-10 or soft_min < 1e-10:
        return 1e6
    
    ratio = soft_min / soft_max
    if ratio <= 0:
        return 1e6
    return -np.log(ratio)

def _generate_initial_points(seed=0):
    """Generate diverse initial points."""
    n = 16
    np.random.seed(seed)
    
    strategies = [
        # Random
        lambda: np.random.randn(n, 2),
        # Circle
        lambda: np.array([[np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)] for i in range(n)]),
        # Hexagonal-ish grid
        lambda: _hex_grid(n),
        # Square grid
        lambda: _square_grid(n),
        # Random with scaling
        lambda: np.random.randn(n, 2) * np.random.uniform(0.5, 2.0),
    ]
    
    idx = seed % len(strategies)
    return strategies[idx]()

def _hex_grid(n):
    """Generate hexagonal grid points."""
    # Arrange 16 points in a hexagonal pattern
    # 3-4-3-3-3 or similar arrangement
    points = []
    # Center
    points.append([0, 0])
    # Ring 1: 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([np.cos(angle), np.sin(angle)])
    # Ring 2: 9 more points (but we only need 16 total, so 9 more)
    for i in range(9):
        angle = 2 * np.pi * i / 9
        points.append([2*np.cos(angle), 2*np.sin(angle)])
    
    points = np.array(points[:n])
    return points

def _square_grid(n):
    """Generate square grid points."""
    # 4x4 grid
    xs = np.linspace(-1.5, 1.5, 4)
    ys = np.linspace(-1.5, 1.5, 4)
    X, Y = np.meshgrid(xs, ys)
    points = np.column_stack([X.flatten(), Y.flatten()])
    return points[:n]

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = 600  # seconds
    safety_margin = 30  # seconds
    
    n = 16
    best_points = None
    best_score = 0.0
    
    # Try multiple initializations
    n_starts = 20
    
    for i in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - safety_margin:
            break
        
        # Generate initial points
        if i < 4:
            # Use structured initializations
            if i == 0:
                x0 = _hex_grid(n).flatten()
            elif i == 1:
                x0 = _square_grid(n).flatten()
            elif i == 2:
                angles = 2 * np.pi * np.arange(n) / n
                x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
            else:
                # Random
                np.random.seed(i)
                x0 = np.random.randn(n, 2).flatten()
        else:
            # Random initializations with different seeds
            np.random.seed(100 + i)
            x0 = np.random.randn(n, 2).flatten()
        
        # Optimize using SLSQP
        try:
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < _objective(x0):
                points = result.x.reshape(n, 2)
                score = _compute_score(points)
                
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    
                    # If good score, try to polish with soft objective
                    if score > 10 and (time.time() - start_time) < deadline - 60:
                        try:
                            # Polish with softer objective
                            result2 = minimize(
                                lambda x: _soft_objective(x, temperature=0.001),
                                result.x,
                                method='SLSQP',
                                options={'maxiter': 100, 'ftol': 1e-14}
                            )
                            if result2.fun < result.fun:
                                points2 = result2.x.reshape(n, 2)
                                score2 = _compute_score(points2)
                                if score2 > best_score:
                                    best_score = score2
                                    best_points = points2.copy()
                        except:
                            pass
        except:
            pass
    
    # If no optimization succeeded, return a reasonable default
    if best_points is None:
        best_points = _hex_grid(n)
    
    # Final check
    if best_points.shape != (16, 2):
        best_points = _hex_grid(n)
    
    return best_points

# EVOLVE-BLOCK-END
