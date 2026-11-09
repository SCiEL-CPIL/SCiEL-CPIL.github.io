import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _score(points):
    """Compute the score: largest common radius."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points inside unit circle
    norms = np.hypot(points[:, 0], points[:, 1])
    if np.any(norms > 1.0):
        return 0.0
    # Min distance from any point to boundary
    boundary_min = 1.0 - np.max(norms)
    # Min pairwise distance / 2
    if N > 1:
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
    else:
        min_pair = boundary_min
    return min(min_pair, boundary_min)

def _objective(x):
    """
    Objective function to minimize.
    We want to maximize the minimum distance.
    We approximate min(d_ij) using a soft-min: -log(sum(exp(-k * d_ij))) / k
    This is differentiable and smooth.
    We also penalize boundary violations.
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    # Soft min of distances
    # Use a large k to make it sharp, but not too large to cause overflow
    k = 50.0
    # d_min_soft = -1/k * log(sum(exp(-k * d)))
    # To avoid overflow in exp, subtract max d? No, we want min d.
    # exp(-k * d) is largest for smallest d.
    # Let's use: soft_min = -1/k * log(sum(exp(-k * d)))
    # This is equivalent to min(d) when k -> infinity.
    
    # Numerical stability:
    # max_val = np.max(d)
    # log_sum_exp = max_val + log(sum(exp(d - max_val)))
    # We want -log(sum(exp(-k*d)))
    # Let y = -k*d. We want -1/k * log(sum(exp(y)))
    # max_y = np.max(y) = -k * min(d)
    # sum(exp(y)) = exp(max_y) * sum(exp(y - max_y))
    # log(sum(exp(y))) = max_y + log(sum(exp(y - max_y)))
    # soft_min = -1/k * (max_y + log(sum(exp(y - max_y))))
    
    y = -k * d
    # Mask diagonal (inf)
    y_flat = y.flatten()
    # Remove inf
    y_valid = y_flat[np.isfinite(y_flat)]
    
    if len(y_valid) == 0:
        return 0.0
    
    max_y = np.max(y_valid)
    # If max_y is very negative, exp(y - max_y) is fine.
    # If max_y is very positive, exp(y - max_y) <= 1.
    # Wait, y = -k*d. d > 0. So y < 0.
    # max_y is the least negative (closest to 0), corresponding to min d.
    
    exp_sum = np.sum(np.exp(y_valid - max_y))
    log_sum_exp = max_y + np.log(exp_sum)
    
    d_min_soft = -log_sum_exp / k
    
    # Boundary constraint
    # We want pts to be inside unit circle.
    # Penalty: sum(max(0, |p| - 1)^2)
    norms = np.hypot(pts[:, 0], pts[:, 1])
    boundary_violation = np.maximum(0, norms - 1.0)
    boundary_penalty = np.sum(boundary_violation ** 2)
    
    # We want to maximize d_min_soft.
    # So minimize -d_min_soft + penalty.
    # Scale penalty heavily to ensure validity.
    return -d_min_soft + 100.0 * boundary_penalty

def _gradient(x):
    """
    Numerical gradient or analytical?
    Analytical is better for L-BFGS-B performance.
    d/dx ( -1/k * log(sum(exp(-k * d_ij))) )
    Let S = sum(exp(-k * d_ij))
    f = -1/k * log(S)
    df/dx_i = -1/k * (1/S) * dS/dx_i
    dS/dx_i = sum_j d(exp(-k * d_ij))/dx_i
    d(exp(-k * d_ij))/dx_i = exp(-k * d_ij) * (-k) * d(d_ij)/dx_i
    d(d_ij)/dx_i = (x_i - x_j) / d_ij  (for i != j)
    
    So df/dx_i = -1/k * (1/S) * sum_j [ exp(-k * d_ij) * (-k) * (x_i - x_j) / d_ij ]
               = (1/S) * sum_j [ exp(-k * d_ij) * (x_i - x_j) / d_ij ]
    
    Boundary gradient:
    g_b = 200 * sum_i max(0, |p_i| - 1) * p_i / |p_i|
    """
    pts = x.reshape(-1, 2)
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    k = 50.0
    y = -k * d
    y_flat = y.flatten()
    y_valid = y_flat[np.isfinite(y_flat)]
    
    if len(y_valid) == 0:
        return np.zeros_like(x)
    
    max_y = np.max(y_valid)
    exp_terms = np.exp(y - max_y)
    # Zero out diagonal
    np.fill_diagonal(exp_terms, 0.0)
    
    S = np.sum(exp_terms)
    
    if S == 0:
        return np.zeros_like(x)
    
    grad = np.zeros_like(pts)
    
    # Pairwise contribution
    # For each i, sum over j
    # term_ij = exp(-k * d_ij) * (p_i - p_j) / d_ij
    # We can vectorize this.
    # exp_terms is (N, N).
    # diff is (N, N, 2).
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # d is (N, N)
    # Avoid division by zero on diagonal
    safe_d = np.where(d > 1e-8, d, 1.0)
    
    # term = exp_terms[:,:,np.newaxis] * diff / safe_d[:,:,np.newaxis]
    term = exp_terms[:, :, np.newaxis] * diff / safe_d[:, :, np.newaxis]
    
    # Sum over j (axis 1) to get grad for each i
    grad_pair = np.sum(term, axis=1)
    
    # Scale by 1/S
    grad_pair /= S
    
    # Boundary contribution
    norms = np.hypot(pts[:, 0], pts[:, 1])
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    
    # Penalty: 100 * sum(max(0, norm - 1)^2)
    # Grad: 100 * 2 * max(0, norm - 1) * p / norm
    mask = norms > 1.0
    grad_boundary = np.zeros_like(pts)
    if np.any(mask):
        grad_boundary[mask] = 200.0 * (norms[mask, np.newaxis] - 1.0) * pts[mask] / safe_norms[mask, np.newaxis]
    
    total_grad = grad_pair + grad_boundary
    
    return total_grad.flatten()

def _hexagonal_initial(spacing):
    """Generate hexagonal lattice points inside unit disk."""
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    # Select N closest to origin
    norms = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]

def _random_initial():
    """Generate random initial configuration."""
    points = []
    for _ in range(N):
        while True:
            x = np.random.uniform(-0.9, 0.9)
            y = np.random.uniform(-0.9, 0.9)
            if x * x + y * y <= 0.81:
                points.append([x, y])
                break
    return np.array(points)

def _ring_initial():
    """Generate ring-based initial configuration."""
    points = []
    points.append([0.0, 0.0])
    radii = [0.3, 0.55, 0.75, 0.9]
    counts = [6, 12, 18, 13]
    for r, n in zip(radii, counts):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points[:N])

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Initializations
    initializations = []
    
    # Hexagonal
    for spacing in [0.20, 0.22, 0.24, 0.26, 0.28]:
        pts = _hexagonal_initial(spacing)
        if pts is not None:
            initializations.append(pts)
            
    # Ring
    initializations.append(_ring_initial())
    
    # Random
    np.random.seed(42)
    for _ in range(5):
        initializations.append(_random_initial())
    
    # Optimization loop
    for init_points in initializations:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        x0 = init_points.flatten()
        
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                jac=_gradient,
                options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-10}
            )
        except Exception:
            continue
            
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        optimized_points = res.x.reshape(-1, 2)
        
        # Check validity
        norms = np.hypot(optimized_points[:, 0], optimized_points[:, 1])
        if np.any(norms > 1.0):
            # Clip if slightly outside
            mask = norms > 1.0
            optimized_points[mask] = optimized_points[mask] / norms[mask, np.newaxis] * 0.999
            
        score = _score(optimized_points)
        
        if score > best_score:
            best_score = score
            best_points = optimized_points.copy()
            
        # If we got close to the known best, do a final polish with tighter tolerance
        if best_score > 0.125:
             # One more run with higher precision
             try:
                 res2 = minimize(
                     _objective, 
                     best_points.flatten(), 
                     method='L-BFGS-B', 
                     jac=_gradient,
                     options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-12}
                 )
                 if time.time() - START_TIME > TIME_LIMIT - 2:
                     break
                 polished_points = res2.x.reshape(-1, 2)
                 norms = np.hypot(polished_points[:, 0], polished_points[:, 1])
                 if np.any(norms > 1.0):
                     mask = norms > 1.0
                     polished_points[mask] = polished_points[mask] / norms[mask, np.newaxis] * 0.999
                 score2 = _score(polished_points)
                 if score2 > best_score:
                     best_score = score2
                     best_points = polished_points.copy()
             except Exception:
                 pass

    # Fallback
    if best_points is None:
        best_points = _hexagonal_initial(0.25)
        if best_points is None:
            best_points = np.zeros((N, 2))

    return best_points
