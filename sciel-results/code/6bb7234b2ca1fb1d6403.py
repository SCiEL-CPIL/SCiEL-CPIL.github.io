"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
# Safety margin: Evaluator kills at 150s. We aim to finish well before 120s.
# Let's target 115s for computation to be safe.
_DEADLINE = _START_TIME + 115.0


def _score(positions):
    """
    Compute the valid packing radius (min of pairwise dist/2 and dist to boundary).
    Returns 0 if invalid (non-finite, outside container).
    """
    n = len(positions)
    if not np.all(np.isfinite(positions)):
        return 0.0
    
    # Check container constraint
    radii = np.hypot(positions[:, 0], positions[:, 1])
    if np.any(radii > 1.0):
        return 0.0
        
    min_boundary = 1.0 - np.max(radii)
    if min_boundary < 0:
        return 0.0
        
    if n < 2:
        return min_boundary

    # Pairwise distances
    # Using broadcasting for vectorization
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_boundary)


def _soft_min(values, beta):
    """Compute the soft minimum of an array of values."""
    v = np.array(values)
    m = np.min(v)
    # Avoid underflow/overflow issues by shifting
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))


def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n, 2) array of centers
    Returns (objective_value, gradient)
    Vectorized gradient computation for speed.
    """
    centers = params.reshape(n, 2)
    
    # Pairwise distances
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = centers[i_idx, :] - centers[j_idx, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    boundary_dists = 1.0 - norms
    
    # Combine all constraints
    vals = np.concatenate([dists / 2.0, boundary_dists])
    
    obj = -_soft_min(vals, beta)
    
    # Compute gradient
    m = np.min(vals)
    exp_vals = np.exp(-beta * (vals - m))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    # Pairwise part - vectorized
    # v_k = dist_k / 2
    # d(v_k)/d(centers[i]) = 0.5 * (centers[i] - centers[j]) / dist_k
    # d(v_k)/d(centers[j]) = -0.5 * (centers[j] - centers[i]) / dist_k = -0.5 * (centers[i] - centers[j]) / dist_k
    
    pair_weights = weights[:len(i_idx)]
    safe_dists = np.where(dists > 1e-8, dists, 1.0)
    unit_vecs = diff / safe_dists[:, np.newaxis]  # (n_pairs, 2)
    weighted_vecs = pair_weights[:, np.newaxis] * 0.5 * unit_vecs  # (n_pairs, 2)
    
    # Accumulate gradients for pairwise contributions
    # For each pair (i, j): grad[i] += w*0.5*vec, grad[j] -= w*0.5*vec
    np.add.at(grad, i_idx, weighted_vecs)
    np.add.at(grad, j_idx, -weighted_vecs)
    
    # Boundary part - vectorized
    # v_k = 1.0 - norm_k
    # d(v_k)/d(centers[i]) = - centers[i] / norm_k
    
    bound_weights = weights[len(i_idx):]
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    # grad[i] -= w_i * centers[i] / norm_i
    grad -= bound_weights[:, np.newaxis] * centers / safe_norms[:, np.newaxis]
    
    return obj, -grad


def _optimize_lbfgs(centers, beta, maxiter, gtol):
    """
    Optimize using L-BFGS-B.
    """
    try:
        res = minimize(
            _objective_and_gradient, 
            centers.flatten(), 
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'disp': False}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers


def _get_hex_lattice(n, seed_offset=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(42 + seed_offset)
    # Estimate spacing based on N
    s_est = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    
    points = []
    max_range = int(2.0 / s_est) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s_est + (j % 2) * s_est / 2.0
            y = j * s_est * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) < n:
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    # Add small random perturbation to break symmetry
    points = points + rng.normal(0, 0.01, points.shape)
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]


def _get_ring_init(n, seed=42):
    """
    Generate a ring-based initialization that better matches expected optimal geometry.
    Concentric rings with points spaced evenly around each ring.
    """
    rng = np.random.default_rng(seed)
    
    # For 50 circles, a good ring structure:
    # Ring 0: 1 point (center)
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # Ring 4: 13 points
    # Total: 50
    
    rings = [1, 6, 12, 18, 13]
    # Adjust if n != 50
    while sum(rings) > n:
        rings[-1] -= 1
        if rings[-1] <= 0:
            rings.pop()
    while sum(rings) < n:
        rings.append(1)
    
    points = []
    # Estimate optimal ring radii
    # For equal circles of radius r in a unit circle, rings at approximately:
    # r_i ≈ i * 2r (rough estimate)
    # With r ≈ 0.126, spacing ≈ 0.252
    ring_spacing = 0.25
    
    for ring_idx, count in enumerate(rings):
        if count == 0:
            continue
        if ring_idx == 0:
            # Center point
            points.append([0.0, 0.0])
        else:
            radius = ring_idx * ring_spacing
            # Add jitter to avoid perfect symmetry
            angle_offset = rng.uniform(0, 2 * np.pi)
            for k in range(count):
                angle = angle_offset + 2 * np.pi * k / count
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                # Small radial jitter
                r_jitter = rng.normal(0, 0.01)
                x += r_jitter * np.cos(angle)
                y += r_jitter * np.sin(angle)
                points.append([x, y])
    
    points = np.array(points[:n])
    
    # Ensure all points are within the unit circle
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 0.95
    if np.any(mask):
        points[mask] *= (0.95 / norms[mask])[:, np.newaxis]
    
    return points


def _get_random_init(n, seed):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.85)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)


def _polish(centers, start_time):
    """
    Progressively polish the solution with increasing beta.
    """
    current = centers.copy()
    best_score = _score(current)
    
    # Start with lower beta, increase for finer resolution
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0]
    
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        maxiter = 150
        gtol = 1e-8
        
        polished = _optimize_lbfgs(current, beta, maxiter, gtol)
        
        if np.all(np.isfinite(polished)):
            # Project to ensure validity if needed (L-BFGS might go slightly out)
            radii = np.hypot(polished[:, 0], polished[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                polished[mask] *= (0.999 / radii[mask])[:, None]
            
            r = _score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score


def construct():
    """
    Construct positions for N circles in the unit disk.
    Returns an array of shape (N, 2).
    """
    start_time = time.time()
    
    # 1. Generate multiple diverse initializations
    inits = []
    
    # Hex lattice
    inits.append(_get_hex_lattice(N, seed_offset=0))
    
    # Ring-based init (better matches expected optimal geometry)
    inits.append(_get_ring_init(N, seed=42))
    inits.append(_get_ring_init(N, seed=123))
    
    # Random inits
    for seed in [1, 2, 3]:
        inits.append(_get_random_init(N, seed))
    
    # 2. Optimize each initialization
    best_pts = None
    best_score = 0.0
    
    # Allocate time budget for each init
    # We have ~115s. Let's spend most of it on the best few.
    # Strategy: Quick pass on all, then deep pass on best.
    
    # Phase 1: Quick optimization (low beta, few iters)
    for init in inits:
        if time.time() - start_time > _DEADLINE - 30:
            break
            
        # Quick polish
        optimized, score = _polish(init, start_time)
        
        if score > best_score:
            best_score = score
            best_pts = optimized
    
    if best_pts is None:
        # Fallback
        best_pts = inits[0]
        best_score = _score(best_pts)
    
    # Phase 2: Deep optimization on the best candidate
    if time.time() - start_time < _DEADLINE - 5:
        # Use higher betas and more iterations
        current = best_pts.copy()
        
        # Use a tighter schedule with more aggressive beta progression
        betas = [200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
        
        for beta in betas:
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            maxiter = 300
            gtol = 1e-10
            
            optimized = _optimize_lbfgs(current, beta, maxiter, gtol)
            
            if np.all(np.isfinite(optimized)):
                radii = np.hypot(optimized[:, 0], optimized[:, 1])
                mask = radii > 1.0
                if np.any(mask):
                    optimized[mask] *= (0.999 / radii[mask])[:, None]
                
                score = _score(optimized)
                if score > best_score:
                    best_score = score
                    best_pts = optimized
                    current = optimized
    
    # Final validation and correction
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    # Use a slightly conservative radius to avoid floating point issues at the boundary
    max_allowed_radius = 1.0 - best_score - 1e-12
    
    mask = radii > max_allowed_radius
    if np.any(mask):
        best_pts[mask] *= (max_allowed_radius / radii[mask])[:, None]
        
    return best_pts

# EVOLVE-BLOCK-END
