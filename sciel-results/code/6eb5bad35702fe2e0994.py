import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import random

N = 13
DEADLINE = 145.0

def _all_triangles_idx():
    """Precompute indices for all unique triples (i, j, k) with i < j < k."""
    idxs = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                idxs.append((i, j, k))
    return np.array(idxs)

TRI_IDX = _all_triangles_idx()

def _min_triangle_area_vec(points):
    """
    Vectorized computation of the minimum triangle area.
    points: (N, 2) array
    """
    # points[TRI_IDX[:, 0]] has shape (M, 2)
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    # Cross product in 2D: (p2-p1) x (p3-p1)
    # Result shape (M,)
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0 # Fallback to avoid division by zero or errors, though hull should exist for random points

def _score(points):
    """
    Raw score: min triangle area / hull area.
    """
    min_area = _min_triangle_area_vec(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate for the negative score to be minimized.
    We want to maximize min_area / hull_area.
    Equivalent to minimizing - (min_area / hull_area).
    
    Using log-sum-exp trick for min:
    min(x) = -lim_{beta->inf} (1/beta) * log(sum(exp(-beta * x)))
    
    Let f(points) = min_area / hull_area.
    We want to minimize -f(points).
    
    Let A_i be triangle areas.
    min(A) = -lim (1/beta) log(sum(exp(-beta * A_i)))
    
    So f = ( - (1/beta) log(sum(exp(-beta * A_i))) ) / H
    We want to minimize -f = ( (1/beta) log(sum(exp(-beta * A_i))) ) / H
    
    To avoid numerical instability with large beta, we can scale.
    However, for L-BFGS, we just need a smooth function.
    Let's use a slightly softer beta to allow gradients to flow, then increase or use exact min for final check.
    Actually, let's just optimize the negative soft-min area directly, and handle hull area separately or assume it's roughly constant?
    No, hull area changes.
    
    Let's define objective to minimize:
    obj = - ( min_area / hull_area )
    
    Approximation:
    min_area ≈ - (1/beta) * log( sum( exp(-beta * A_i) ) )
    
    So obj ≈ (1/beta) * log( sum( exp(-beta * A_i) ) ) / hull_area
    
    Let's compute this.
    """
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft min
    # To prevent overflow in exp, subtract max? 
    # exp(-beta * A). If A is small, -beta*A is large negative -> exp is small.
    # If A is large, -beta*A is large positive? No, A is positive.
    # Wait. min(A) is the smallest A.
    # log-sum-exp for min:
    # m = min(A)
    # log(sum(exp(-beta * A_i))) = -beta * m + log(sum(exp(-beta * (A_i - m))))
    # So (1/beta) * log(sum(exp(-beta * A_i))) = -m + (1/beta) * log(sum(exp(-beta * (A_i - m))))
    # As beta -> inf, this goes to -m.
    
    # Let's just use the direct formula with a safe beta.
    # If beta is too high, gradients vanish for non-min triangles.
    # If beta is too low, it's an average, not a min.
    # Start with beta=100.
    
    # Shift for numerical stability
    min_a = np.min(areas)
    # We want to approximate min(areas).
    # soft_min = - (1/beta) * log(sum(exp(-beta * (areas - min_a)))) - min_a
    # This is stable.
    
    shifted = areas - min_a
    # exp(-beta * shifted)
    # shifted >= 0.
    # exp(-beta * shifted) <= 1.
    # Sum is between 1 and M.
    # log is safe.
    
    soft_min_val = min_a - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10 # Penalty for degenerate hull
    
    return - (soft_min_val / hull_area)

def _negative_score_exact(points):
    """
    Exact negative score for final verification.
    """
    min_area = _min_triangle_area_vec(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return - (min_area / hull_area)

def _halton_sequence(n, base):
    """Generate first n Halton sequence values."""
    result = []
    f = 1.0
    for i in range(1, n + 1):
        r = 0.0
        f = 1.0
        index = i
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        result.append(r)
    return np.array(result)

def _halton_points():
    """Generate Halton sequence points in unit square."""
    x = _halton_sequence(N, 2)
    y = _halton_sequence(N, 3)
    return np.column_stack((x, y))

def _regular_polygon():
    """Generate points on a regular polygon."""
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Center at 0.5, 0.5, radius 0.5
    x = 0.5 + 0.5 * np.cos(theta)
    y = 0.5 + 0.5 * np.sin(theta)
    return np.column_stack((x, y))

def _jittered_grid():
    """Generate a jittered grid."""
    # 4x4 grid = 16 points. We need 13.
    # Let's take a 4x4 grid and remove 3 corners? Or just use first 13?
    # Better: 3x5 grid = 15, remove 2?
    # Let's just use a 4x4 grid and jitter it, then take 13.
    # Actually, a specific arrangement might be better.
    # Let's create a 4x4 grid in [0,1]x[0,1]
    coords = np.linspace(0, 1, 4)
    grid = np.array([[x, y] for x in coords for y in coords])
    # Remove 3 points to get 13. Let's remove the corners?
    # Or just take the first 13.
    # Let's try a specific pattern:
    # Rows of 4, 4, 4, 1?
    # Or 3, 4, 3, 3?
    # Let's just use the first 13 of the grid and jitter.
    pts = grid[:13].copy()
    # Add small jitter
    np.random.seed(42)
    pts += np.random.uniform(-0.05, 0.05, pts.shape)
    return pts

def _center_ring():
    """
    One point in center, rest on a ring.
    """
    center = np.array([0.5, 0.5])
    theta = np.linspace(0, 2 * np.pi, N - 1, endpoint=False)
    radius = 0.4
    ring_x = 0.5 + radius * np.cos(theta)
    ring_y = 0.5 + radius * np.sin(theta)
    ring = np.column_stack((ring_x, ring_y))
    return np.vstack([center, ring])

def construct():
    start_time = time.time()
    
    # List of initializers
    initializers = [
        _halton_points,
        _regular_polygon,
        _jittered_grid,
        _center_ring,
        lambda: np.random.rand(N, 2) * 0.8 + 0.1,
        lambda: np.random.rand(N, 2) * 0.8 + 0.1,
        lambda: np.random.rand(N, 2) * 0.8 + 0.1,
    ]
    
    best_points = None
    best_score = -np.inf
    
    # We will run L-BFGS-B on each initializer.
    # L-BFGS-B is fast, so we can do many restarts.
    
    for i, init_func in enumerate(initializers):
        if time.time() > DEADLINE - 5.0:
            break
            
        # Generate initial points
        x0 = init_func()
        
        # Ensure points are within a reasonable bound to help optimization
        # L-BFGS-B bounds: [0, 1] for x and y?
        # The problem is scale invariant, but normalizing to hull area makes it well-defined.
        # However, keeping points in [0,1] helps prevent them from flying apart.
        bounds = [(0.0, 1.0)] * (2 * N)
        
        # Flatten points for L-BFGS-B
        x0_flat = x0.flatten()
        
        # Objective function
        def objective(x_flat):
            pts = x_flat.reshape(N, 2)
            return _soft_min_score(pts, beta=100.0)
        
        # Run optimization
        try:
            res = minimize(objective, x0_flat, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e9: # Check if it didn't fail catastrophically
                x_opt = res.x.reshape(N, 2)
                # Calculate exact score
                score = _score(x_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = x_opt.copy()
                    
        except Exception:
            pass
            
    # If we still have time, do a final polish on the best points with higher beta or Nelder-Mead
    if best_points is not None and time.time() < DEADLINE - 10.0:
        # Try to refine with a smoother beta or just use the best found
        # L-BFGS-B with soft-min is usually good enough.
        # Let's try one more round with a different beta or local search if time permits.
        # Given the time limit, just return the best found so far.
        
        # One more optimization pass with a tighter beta on the best point
        x0_flat = best_points.flatten()
        bounds = [(0.0, 1.0)] * (2 * N)
        
        def objective_tight(x_flat):
            pts = x_flat.reshape(N, 2)
            return _soft_min_score(pts, beta=500.0)
            
        try:
            res = minimize(objective_tight, x0_flat, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.success:
                x_opt = res.x.reshape(N, 2)
                score = _score(x_opt)
                if score > best_score:
                    best_score = score
                    best_points = x_opt.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = _halton_points()
        
    return best_points
