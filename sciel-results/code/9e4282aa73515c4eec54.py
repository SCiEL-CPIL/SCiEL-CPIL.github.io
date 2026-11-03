"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

def _compute_score(points):
    """Compute the maximum radius for given points."""
    n = len(points)
    if n < 2:
        return 0.0
    # Min distance between any pair / 2
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    min_pair_r = min_pair / 2.0
    
    # Min distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    return min(min_pair_r, min_boundary)

def _feasible(points, r):
    """Check if all circles of radius r fit without overlap and inside unit circle."""
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms + r > 1.0 + 1e-12):
        return False
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    if np.any(dists < 2.0 * r - 1e-12):
        return False
    return True

def _constraint_push(points, r, max_iter=200):
    """Iteratively push points to satisfy constraints for radius r."""
    pts = points.copy()
    for _ in range(max_iter):
        # Push from boundary
        norms = np.linalg.norm(pts, axis=1)
        viol_b = norms + r - 1.0
        mask_b = viol_b > 1e-10
        if np.any(mask_b):
            for i in np.where(mask_b)[0]:
                pts[i] *= (1.0 - r) / norms[i]
        
        # Push from overlaps
        changed = False
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < 2.0 * r - 1e-10:
                    if d < 1e-15:
                        # Randomly separate
                        pts[i] += np.array([0.01, 0.01])
                        pts[j] -= np.array([0.01, 0.01])
                    else:
                        diff = pts[j] - pts[i]
                        push = (2.0 * r - d) / d * 0.5
                        pts[i] -= diff * push
                        pts[j] += diff * push
                    changed = True
        if not changed:
            break
    
    # Final boundary clip
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1.0 - r
    if np.any(mask):
        pts[mask] *= (1.0 - r) / norms[mask, np.newaxis]
    
    return pts

def _soft_min_surrogate(points, beta=100.0):
    """
    Compute a smooth surrogate for the min of:
    - pairwise distances / 2
    - boundary distances (1 - norm)
    using log-sum-exp style soft-min.
    """
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    
    # Soft-min of pairwise distances / 2
    pair_vals = dists / 2.0
    # Exclude diagonal (inf)
    flat_vals = pair_vals.flatten()
    flat_vals = flat_vals[~np.isinf(flat_vals)]
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_vals = 1.0 - norms
    
    # Combine all values
    all_vals = np.concatenate([flat_vals, boundary_vals])
    
    # Soft-min via log-sum-exp
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    soft_min = m - np.log(np.sum(exp_vals)) / beta
    
    return soft_min

def _smooth_objective(x, beta=100.0):
    """Smooth objective: negative soft-min radius + boundary penalty."""
    pts = x.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary_violation = np.maximum(0, norms - 1.0)
    penalty = 100.0 * np.sum(boundary_violation)
    radius = _soft_min_surrogate(pts, beta)
    return -radius + penalty

def _smooth_grad(x, beta=100.0):
    """Gradient of the smooth objective (numerical)."""
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _smooth_objective(x, beta)
    for i in range(len(x)):
        x_p = x.copy()
        x_p[i] += eps
        grad[i] = (_smooth_objective(x_p, beta) - f0) / eps
    return grad

def _objective(x, beta=100.0):
    """Objective function for optimization: negative soft-min radius + boundary penalty."""
    return _smooth_objective(x, beta)

def _ring_initialization():
    """
    Generate initial configuration with concentric rings.
    Known optimal packing has: 1 center, then rings of 6, 12, 18, 13 = 50
    """
    points = []
    
    # Center
    points.append([0.0, 0.0])
    
    # Ring 1: 6 circles
    r1 = 0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 12 circles
    r2 = 0.50
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12  # offset
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 18 circles
    r3 = 0.75
    for i in range(18):
        angle = 2 * np.pi * i / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # Ring 4: 13 circles (partial outer ring)
    r4 = 0.90
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi / 13
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(points)

def _hexagonal_initial():
    """Generate initial hexagonal lattice packing."""
    spacing = 0.25
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _polish_smooth(points, start_time, deadline):
    """
    Final high-precision polish using smooth surrogate with increasing beta.
    """
    current = points.copy()
    current_score = _compute_score(current)
    
    # Gradually increase beta to sharpen the approximation
    betas = [30, 60, 120, 240, 480, 960]
    
    for beta in betas:
        if time.time() > deadline - 3:
            break
        
        def obj(x):
            return _smooth_objective(x, beta)
        
        # Use L-BFGS-B with the smooth objective
        try:
            result = minimize(
                obj,
                current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            new_points = result.x.reshape(N, 2)
            
            # Check validity
            if np.all(np.isfinite(new_points)):
                norms = np.linalg.norm(new_points, axis=1)
                if np.max(norms) <= 1.0 + 1e-12:
                    # Clip to unit circle
                    norms = np.linalg.norm(new_points, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        new_points[mask] /= norms[mask, np.newaxis] * 1.0001
                    new_score = _compute_score(new_points)
                    if new_score > current_score:
                        current = new_points
                        current_score = new_score
        except Exception:
            pass
    
    return current, current_score

def _binary_search_polish(points, start_time, deadline):
    """
    Binary search on radius with constraint push for high-precision polish.
    """
    current = points.copy()
    current_score = _compute_score(current)
    
    # Binary search on radius
    lo = current_score
    hi = min(0.5, 1.0 / np.sqrt(N))  # Upper bound
    
    # Try to improve
    for _ in range(30):
        if time.time() > deadline - 2:
            break
        
        mid = (lo + hi) / 2.0
        
        # Try to find a feasible configuration at radius mid
        pts = _constraint_push(current, mid, max_iter=100)
        
        if _feasible(pts, mid):
            lo = mid
            current = pts
            current_score = mid
        else:
            hi = mid
    
    return current, current_score

def _local_search_polish(points, start_time, deadline):
    """
    Local search polish: try small perturbations and accept improvements.
    """
    current = points.copy()
    current_score = _compute_score(current)
    rng = np.random.default_rng(123)
    
    step_size = 0.01
    for _ in range(5000):
        if time.time() > deadline - 2:
            break
        
        # Random perturbation
        idx = rng.integers(0, N)
        delta = rng.normal(0, step_size, size=2)
        new_points = current.copy()
        new_points[idx] += delta
        
        # Keep in circle
        norm = np.linalg.norm(new_points[idx])
        if norm > 0.999:
            new_points[idx] *= 0.999 / norm
        
        new_score = _compute_score(new_points)
        
        if new_score > current_score:
            current = new_points
            current_score = new_score
            step_size = max(step_size * 0.99, 1e-6)
        else:
            step_size = min(step_size * 1.05, 0.1)
    
    return current, current_score

def _parametric_points(radii, angles):
    """
    Build 50 points from ring parameters.
    radii: array of 4 values (ring 1-4 radii, ring 0 is center)
    angles: array of 4 values (ring 1-4 rotation offsets)
    Ring sizes: [1, 6, 12, 18, 13]
    """
    ring_sizes = [1, 6, 12, 18, 13]
    points = []
    
    # Center
    points.append([0.0, 0.0])
    
    for ring_idx in range(1, 5):
        size = ring_sizes[ring_idx]
        r = radii[ring_idx - 1]
        angle_offset = angles[ring_idx - 1]
        for i in range(size):
            angle = 2 * np.pi * i / size + angle_offset
            points.append([r * np.cos(angle), r * np.sin(angle)])
    
    return np.array(points)

def _parametric_objective(params):
    """
    Objective for parametric optimization: negative soft-min radius.
    params: 8 values (4 radii, 4 angles)
    """
    radii = params[:4]
    angles = params[4:8]
    
    points = _parametric_points(radii, angles)
    
    # Check if all points are inside unit circle
    norms = np.linalg.norm(points, axis=1)
    if np.max(norms) > 1.0:
        return 100.0  # Large penalty
    
    radius = _soft_min_surrogate(points, beta=50.0)
    return -radius

def _optimize_parametric(start_time, deadline):
    """
    Optimize ring radii and angles using differential evolution.
    Returns best points and score.
    """
    # Bounds for radii (must be positive and not too large)
    # Ring 1: ~0.2-0.35, Ring 2: ~0.4-0.6, Ring 3: ~0.6-0.8, Ring 4: ~0.8-0.95
    bounds = [
        (0.15, 0.40),  # r1
        (0.35, 0.65),  # r2
        (0.55, 0.85),  # r3
        (0.75, 0.98),  # r4
        (0.0, 2 * np.pi),  # angle1
        (0.0, 2 * np.pi),  # angle2
        (0.0, 2 * np.pi),  # angle3
        (0.0, 2 * np.pi),  # angle4
    ]
    
    # Initial guess
    x0 = np.array([0.25, 0.50, 0.75, 0.90, 0.0, np.pi/12, 0.0, np.pi/13])
    
    # Run differential evolution
    try:
        result_de = differential_evolution(
            _parametric_objective,
            bounds,
            seed=42,
            maxiter=200,
            tol=1e-10,
            polish=False,
            init='sobol'
        )
        
        if time.time() > deadline - 20:
            return None, -np.inf
        
        best_params = result_de.x
        best_points = _parametric_points(best_params[:4], best_params[4:8])
        best_score = _compute_score(best_points)
        
        # Polish the parametric result with full-point optimization
        if time.time() < deadline - 10:
            best_points, best_score = _polish_smooth(best_points, start_time, deadline)
        
        return best_points, best_score
    except Exception:
        return None, -np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Phase 1: Parametric optimization (low-dimensional)
    param_points, param_score = _optimize_parametric(start_time, deadline)
    
    # Phase 2: Traditional initializations
    initializations = [
        ("ring", _ring_initialization()),
        ("hex", _hexagonal_initial()),
    ]
    
    best_points = None
    best_score = -np.inf
    
    # Include parametric result if valid
    if param_points is not None and param_score > best_score:
        best_points = param_points.copy()
        best_score = param_score
    
    for name, init_points in initializations:
        if time.time() > deadline - 60:
            break
        
        score = _compute_score(init_points)
        if score > best_score:
            best_score = score
            best_points = init_points.copy()
    
    # Phase 3: Simulated annealing with adaptive step size and fast scoring
    rng = np.random.default_rng(42)
    current_points = best_points.copy()
    current_score = best_score
    
    # Precompute initial distances for fast updates
    T = 0.003  # Initial temperature
    T_min = 1e-10
    cooling = 0.99985
    max_iter = 500000
    
    # Track global min boundary for fast updates
    norms = np.linalg.norm(current_points, axis=1)
    global_min_boundary = 1.0 - np.max(norms)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 40:
            break
        
        # Random perturbation with adaptive scale
        idx = rng.integers(0, N)
        scale = max(T, 1e-7) * 0.12
        delta = rng.normal(0, scale, size=2)
        new_points = current_points.copy()
        new_points[idx] += delta
        
        # Ensure points stay in unit circle
        norm_idx = np.linalg.norm(new_points[idx])
        if norm_idx > 1.0:
            new_points[idx] = new_points[idx] / norm_idx * 0.9995
        
        # Fast score: only compute distances involving idx
        diffs = new_points - new_points[idx]
        dists = np.linalg.norm(diffs, axis=1)
        dists[idx] = np.inf
        min_pair = np.min(dists)
        min_pair_r = min_pair / 2.0
        
        # Update global min boundary
        new_norms = norms.copy()
        new_norms[idx] = np.linalg.norm(new_points[idx])
        new_min_boundary = 1.0 - np.max(new_norms)
        
        new_score = min(min_pair_r, new_min_boundary)
        
        # Accept or reject
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / max(T, 1e-12)):
            current_points = new_points
            current_score = new_score
            norms = new_norms
            global_min_boundary = new_min_boundary
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= cooling
        if T < T_min:
            break
    
    # Final high-precision polish with smooth surrogate
    if time.time() < deadline - 15:
        best_points, best_score = _polish_smooth(best_points, start_time, deadline)
    
    # Binary search polish for high precision
    if time.time() < deadline - 5:
        best_points, best_score = _binary_search_polish(best_points, start_time, deadline)
    
    # Local search polish
    if time.time() < deadline - 2:
        best_points, best_score = _local_search_polish(best_points, start_time, deadline)
    
    # Final constraint push to ensure validity
    final_r = _compute_score(best_points)
    if final_r > 0:
        best_points = _constraint_push(best_points, final_r, max_iter=50)
        final_r2 = _compute_score(best_points)
        if final_r2 >= final_r - 1e-10:
            best_points = best_points
    
    return best_points
# EVOLVE-BLOCK-END
