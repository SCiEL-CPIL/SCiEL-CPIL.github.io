"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points."""
from scipy.optimize import basinhopping, minimize
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations
import time

N = 13

# EVOLVE-BLOCK-START
_DEADLINE = 150.0
_START = time.time()
_SAFETY = 10.0

# Pre-compute triangle indices
TRI_INDICES = np.array(list(combinations(range(N), 3)))
NUM_TRI = len(TRI_INDICES)

def _min_triangle_area_exact(pts):
    """Minimum area of any triangle formed by 3 of the points. Vectorized."""
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _convex_hull_area(pts):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    return _min_triangle_area_exact(pts) / hull_area

def _neg_score(x):
    """Negative score for minimization."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def _step(x, stepsize):
    """
    Generate a new point by adding random noise.
    We use a stepsize that is large enough to escape local minima.
    """
    return x + stepsize * np.random.normal(size=x.shape)

def _local_polish(x0, max_iter=2000):
    """
    Perform a local search using Nelder-Mead to refine the starting point x0.
    """
    try:
        res = minimize(_neg_score, x0, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
        return res.x
    except Exception:
        return x0

def _generate_initial_points():
    """
    Generate a diverse set of initial points.
    """
    inits = []
    
    # 1. Regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    pts_reg = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    inits.append(pts_reg)
    
    # 2. Regular 12-gon + Center
    angles_12 = 2 * np.pi * np.arange(12) / 12
    pts_12 = np.column_stack([0.5 + 0.4 * np.cos(angles_12), 0.5 + 0.4 * np.sin(angles_12)])
    pts_12_c = np.vstack([pts_12, np.array([0.5, 0.5])])
    inits.append(pts_12_c)
    
    # 3. Jittered Grid
    grid_x = np.linspace(0.1, 0.9, 4)
    grid_y = np.linspace(0.1, 0.9, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    rng = np.random.RandomState(123)
    indices_to_keep = list(range(16))
    removed = rng.choice(16, 3, replace=False)
    indices_to_keep = [i for i in indices_to_keep if i not in removed]
    pts_grid = grid_pts[indices_to_keep]
    inits.append(pts_grid)
    
    # 4. Random points
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts_rand = rng.rand(N, 2) * 0.8 + 0.1
        inits.append(pts_rand)
        
    return inits

def construct():
    global _START, _DEADLINE
    elapsed = time.time() - _START
    remaining = _DEADLINE - _SAFETY - elapsed
    
    if remaining < 10.0:
        # Fallback
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])

    best_x = None
    best_score = -np.inf
    
    inits = _generate_initial_points()
    
    # Evaluate initial points
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_x = init.flatten()

    # Use Basin Hopping
    # Basin Hopping is a global optimization method that uses a local minimizer
    # and a stochastic step to escape local minima.
    
    # We run basin hopping from the best initial point found.
    # We also run it from a few other good initial points if time permits.
    
    # Parameters for Basin Hopping
    # stepsize: how much random noise to add
    # miniter: minimum number of iterations for the local minimizer
    # maxiter: maximum number of basin hopping steps
    
    # We need to be careful with the time budget.
    # Each basin hopping step involves a local minimization (Nelder-Mead).
    # Nelder-Mead can be slow if it doesn't converge quickly.
    # We limit maxiter in the local minimizer.
    
    rng = np.random.RandomState(123)
    
    # Try a few different step sizes and starting points
    candidates = [best_x]
    # Add top 2 other inits
    for init in inits[:2]:
        candidates.append(init.flatten())
    
    for x_start in candidates:
        if time.time() - _START > _DEADLINE - _SAFETY - 5.0:
            break
        
        try:
            # Basin Hopping
            # We use a custom step function
            # The local minimizer is Nelder-Mead
            
            # We run for a limited number of iterations to ensure we finish in time
            # Each 'step' in basin hopping is a local optimization.
            # Let's allow up to 50 basin hopping steps.
            # Each local optimization might take ~0.1s. 50 * 0.1s = 5s.
            
            res = basinhopping(
                _neg_score, 
                x_start, 
                stepsize=0.1, # Initial step size
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}},
                niter=50, # Number of basin hopping steps
                stepgen=_step,
                disp=False
            )
            
            if res.fun < best_score * -1:
                best_score = -res.fun
                best_x = res.x
                
        except Exception as e:
            # If basin hopping fails, continue
            pass

    # Final local polish on the best result found
    if time.time() - _START < _DEADLINE - _SAFETY - 2.0:
        polished_x = _local_polish(best_x, max_iter=2000)
        polished_score = -_neg_score(polished_x)
        
        if polished_score > best_score:
            best_score = polished_score
            best_x = polished_x

    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
