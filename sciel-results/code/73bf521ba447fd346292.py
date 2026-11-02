import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _fibonacci_sphere(n):
    """Generate n points on unit sphere via Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_deg(points, m=50000):
    """
    Estimate covering radius in degrees by sampling m points on sphere.
    For each sample, find nearest configuration point, compute angular distance.
    Return max angular distance in degrees.
    """
    # Generate sample points on sphere
    y = _fibonacci_sphere(m)
    # Compute dot products: y @ points.T -> (m, N)
    dots = y @ points.T
    # Nearest point for each y: argmax of dots (since angular dist = arccos(dot))
    nearest_idx = np.argmax(dots, axis=1)
    nearest_dots = dots[np.arange(m), nearest_idx]
    # Clip to [-1, 1] for numerical safety
    nearest_dots = np.clip(nearest_dots, -1.0, 1.0)
    angles_rad = np.arccos(nearest_dots)
    max_angle_rad = np.max(angles_rad)
    return np.degrees(max_angle_rad)

def _smooth_objective(params, points):
    """
    Objective: minimize the maximum angular distance.
    We use a smooth approximation via log-sum-exp on squared angular distances.
    Actually, simpler: we want to push points apart. A good objective is to minimize
    the sum of inverse of min distances, or use a repulsive potential.
    
    Better approach: minimize the maximum distance directly is non-smooth.
    Use a soft-max: for a sample of directions, compute min distance to any point,
    then take a soft max over those.
    
    Alternative: use the fact that covering radius = max over y of min_i arccos(y.p_i)
    = max over y of arccos(max_i (y.p_i))
    
    So we can write: for each sample y, f(y) = arccos(max_i(y.p_i))
    And we want min over P of max_y f(y).
    
    A smooth surrogate: use a log-sum-exp style. But let's try a simpler repulsive potential:
    Objective = sum_{i<j} -log(1 - p_i.p_j)  (pushes points apart)
    This is smooth and encourages uniform distribution.
    """
    P = np.zeros((N, 3))
    for i in range(N):
        P[i] = _param_to_point(params[3*i:3*i+3])
    
    # Compute pairwise dot products
    dots = P @ P.T
    # Diagonal is 1, off-diagonal < 1
    # We want to minimize the maximum angular distance, which is equivalent to
    # maximizing the minimum dot product between points? No, that's for packing.
    # For covering, we want the convex hull to have small circumradii.
    
    # Let's use a repulsive potential: sum of 1/(1-dots) for i<j
    # This blows up when points coincide.
    off_diag_mask = ~np.eye(N, dtype=bool)
    off_diag_dots = dots[off_diag_mask]
    # Avoid log of zero or negative
    off_diag_dots = np.clip(off_diag_dots, -0.999, 0.999)
    # Use -log(1 - dots) as repulsive term
    return np.sum(-np.log(1 - off_diag_dots))

def _param_to_point(p):
    """Convert 3 parameters to a unit vector (just normalize)."""
    norm = np.linalg.norm(p)
    if norm < 1e-15:
        return np.array([1.0, 0.0, 0.0])
    return p / norm

def _point_to_param(pt):
    """Just the point itself (identity for this parameterization)."""
    return pt

def construct():
    t_start = time.time()
    deadline = t_start + 120.0  # 120s limit, leave margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_score = _covering_radius_deg(best_points)
    
    # Try to improve via optimization
    # Parameterize each point as 3 free coordinates (normalize during evaluation)
    x0 = points.flatten()
    
    # Define objective and gradient for scipy minimize
    def objective(x):
        P = np.zeros((N, 3))
        for i in range(N):
            P[i] = _param_to_point(x[3*i:3*i+3])
        
        # Compute covering radius estimate
        # Use a moderate number of samples for speed
        m_sample = 5000
        y = _fibonacci_sphere(m_sample)
        dots = y @ P.T
        nearest_dots = np.max(dots, axis=1)
        nearest_dots = np.clip(nearest_dots, -1.0, 1.0)
        angles = np.arccos(nearest_dots)
        # Use soft max for smoothness: t * log(sum(exp(angles/t)))
        t_soft = 0.1
        max_angle = t_soft * np.log(np.sum(np.exp(angles / t_soft)))
        return max_angle
    
    def gradient(x):
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros_like(x)
        f0 = objective(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = objective(x_plus)
            grad[i] = (f_plus - f0) / eps
        return grad
    
    # Use L-BFGS-B with numerical gradient
    try:
        result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                         options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 2000})
        
        if result.success or result.fun < _covering_radius_deg(best_points):
            new_points = np.zeros((N, 3))
            for i in range(N):
                new_points[i] = _param_to_point(result.x[3*i:3*i+3])
            new_score = _covering_radius_deg(new_points)
            if new_score < best_score:
                best_points = new_points
                best_score = new_score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_points
    
    # Try a second round with more samples and finer optimization
    try:
        x0 = best_points.flatten()
        result2 = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000})
        
        if result2.success:
            new_points = np.zeros((N, 3))
            for i in range(N):
                new_points[i] = _param_to_point(result2.x[3*i:3*i+3])
            new_score = _covering_radius_deg(new_points)
            if new_score < best_score:
                best_points = new_points
                best_score = new_score
    except Exception:
        pass
    
    # Final polish: check if we're still within deadline
    if time.time() < deadline - 10:
        # Try a few random restarts with perturbation
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            # Perturb best points
            perturbed = best_points + 0.1 * np.random.randn(N, 3)
            # Renormalize
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            perturbed = perturbed / norms
            
            x0 = perturbed.flatten()
            try:
                result_r = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                                   options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000})
                new_points = np.zeros((N, 3))
                for i in range(N):
                    new_points[i] = _param_to_point(result_r.x[3*i:3*i+3])
                new_score = _covering_radius_deg(new_points)
                if new_score < best_score:
                    best_points = new_points
                    best_score = new_score
            except Exception:
                pass
    
    # Final validity check
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-10) or np.any(~np.isfinite(best_points)):
        # Fall back to Fibonacci spiral
        return _fibonacci_sphere(N)
    
    return best_points / norms
