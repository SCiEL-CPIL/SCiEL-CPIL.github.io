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
    # We want to maximize the minimum of (pair_dists/2, boundary_dists)
    # So we minimize -soft_min(pair_dists/2, boundary_dists)
    vals = np.concatenate([dists / 2.0, boundary_dists])
    
    obj = -_soft_min(vals, beta)
    
    # Compute gradient
    # d/dx (-soft_min) = - d/dx (soft_min)
    # soft_min = m - (1/beta) * log(sum(exp(-beta*(v-m))))
    # Let S = sum(exp(-beta*(v-m)))
    # d(soft_min)/dx = - (1/beta) * (1/S) * sum(d(exp(-beta*(v-m)))/dx)
    # exp(-beta*(v-m)) = exp(-beta*v) * exp(beta*m)
    # d(exp(-beta*v))/dx = -beta * exp(-beta*v) * dv/dx
    # So d(soft_min)/dx = - (1/beta) * (1/S) * sum(-beta * exp(-beta*v) * dv/dx)
    #                    = (1/S) * sum(exp(-beta*v) * dv/dx)
    # Note: exp(-beta*v) is proportional to exp(-beta*(v-m)), so we can use weights w_k = exp(-beta*(v_k-m))/S
    # Then d(soft_min)/dx = sum(w_k * dv_k/dx)
    # And d(obj)/dx = - sum(w_k * dv_k/dx)
    
    m = np.min(vals)
    exp_vals = np.exp(-beta * (vals - m))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    # Pairwise part
    # v_k = dist_k / 2
    # d(dist_k)/d(centers[i]) = (centers[i] - centers[j]) / dist_k
    # d(dist_k)/d(centers[j]) = (centers[j] - centers[i]) / dist_k
    # d(v_k)/d(centers[i]) = 0.5 * (centers[i] - centers[j]) / dist_k
    
    pair_weights = weights[:len(i_idx)]
    for k in range(len(i_idx)):
        w = pair_weights[k]
        if w < 1e-12:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[k]
        if d > 1e-8:
            vec = (centers[i] - centers[j]) / d
            # d(v_k)/d(centers[i]) = 0.5 * vec
            # d(v_k)/d(centers[j]) = -0.5 * vec
            grad[i] += w * 0.5 * vec
            grad[j] -= w * 0.5 * vec
    
    # Boundary part
    # v_k = 1.0 - norm_k
    # d(v_k)/d(centers[i]) = - centers[i] / norm_k
    
    bound_weights = weights[len(i_idx):]
    for i in range(n):
        w = bound_weights[i]
        if w < 1e-12:
            continue
        norm = norms[i]
        if norm > 1e-8:
            # d(v_i)/d(centers[i]) = - centers[i] / norm
            grad[i] -= w * centers[i] / norm
    
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
    # Area of circle ~ pi. Area per point ~ pi/N.
    # Hex lattice area per point ~ s^2 * sqrt(3)/2
    # s ~ sqrt(2 * pi / (N * sqrt(3)))
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
        # Fallback or adjust
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
        
        # Increase iterations as beta increases? Or keep fixed?
        # Higher beta means sharper min, might need more iterations to converge.
        maxiter = 100
        gtol = 1e-7
        
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
        
        # Use a tighter schedule
        betas = [200.0, 500.0, 1000.0, 2000.0, 5000.0]
        
        for beta in betas:
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            maxiter = 200
            gtol = 1e-9
            
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
    max_allowed_radius = 1.0 - best_score
    
    mask = radii > max_allowed_radius
    if np.any(mask):
        best_pts[mask] *= (max_allowed_radius / radii[mask])[:, None]
        
    return best_pts

# EVOLVE-BLOCK-END
