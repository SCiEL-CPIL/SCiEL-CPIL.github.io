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

def _objective(x, beta=100.0):
    """Objective function for optimization: negative soft-min radius + boundary penalty."""
    pts = x.reshape(N, 2)
    
    # Hard penalty for points outside unit circle
    norms = np.linalg.norm(pts, axis=1)
    boundary_violation = np.maximum(0, norms - 1.0)
    penalty = 100.0 * np.sum(boundary_violation)
    
    # Negative soft-min radius
    radius = _soft_min_surrogate(pts, beta)
    
    return -radius + penalty

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
            return _objective(x, beta)
        
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

def _polish_exact(points, start_time, deadline):
    """
    Final polish using direct optimization of the exact objective.
    Uses penalty method to handle the min constraint.
    """
    current = points.copy()
    current_score = _compute_score(current)
    
    # Use a sequence of penalty parameters
    penalty_weights = [10, 50, 200, 1000]
    
    for pw in penalty_weights:
        if time.time() > deadline - 2:
            break
        
        def exact_obj(x):
            pts = x.reshape(N, 2)
            # Compute actual radius
            dists = cdist(pts, pts)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            min_pair_r = min_pair / 2.0
            
            norms = np.linalg.norm(pts, axis=1)
            min_boundary = 1.0 - np.max(norms)
            
            r = min(min_pair_r, min_boundary)
            
            # Penalty for violations
            violation = 0.0
            if min_pair_r < r:
                violation += (r - min_pair_r)
            if min_boundary < r:
                violation += (r - min_boundary)
            
            return -r + pw * violation**2
        
        try:
            result = minimize(
                exact_obj,
                current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            new_points = result.x.reshape(N, 2)
            
            if np.all(np.isfinite(new_points)):
                norms = np.linalg.norm(new_points, axis=1)
                if np.max(norms) <= 1.0 + 1e-12:
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
    
    # Final exact polish
    if time.time() < deadline - 3:
        best_points, best_score = _polish_exact(best_points, start_time, deadline)
    
    return best_points
# EVOLVE-BLOCK-END
