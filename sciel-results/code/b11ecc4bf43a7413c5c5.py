"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _ring_initial(radii=None):
    """Generate ring-structured initialization matching known optimal geometry.
    Structure: 1 center + rings of 6, 12, 18, 13 = 50 circles."""
    rings = [1, 6, 12, 18, 13]
    if radii is None:
        radii = [0.0, 0.22, 0.45, 0.68, 0.88]
    
    points = []
    for ring_idx, n in enumerate(rings):
        r = radii[ring_idx]
        if n == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
    
    return np.array(points)


def _compute_score(points):
    """Compute the maximum common radius for given points."""
    if len(points) < 2:
        return 0.0
    # Vectorized pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(len(points), k=1)
    min_pair = np.min(dists[iu]) / 2.0
    norms = np.hypot(points[:, 0], points[:, 1])
    min_boundary = 1.0 - np.max(norms)
    return min(min_pair, min_boundary)


def _soft_min_obj_grad(points_flat, beta):
    """Objective and gradient for L-BFGS-B using soft-min via log-sum-exp."""
    points = points_flat.reshape(N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_dists = 1.0 - norms
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    scaled = -beta * all_dists
    max_val = np.max(scaled)
    exp_scaled = np.exp(scaled - max_val)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp  # softmax weights
    
    # Compute gradients
    grad = np.zeros_like(points_flat)
    
    # Pair distance gradients
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = weights[idx]
        d = pair_dists[idx]
        if d > 1e-15:
            direction = (points[i] - points[j]) / d
            grad[2*i] += w * direction[0]
            grad[2*i+1] += w * direction[1]
            grad[2*j] -= w * direction[0]
            grad[2*j+1] -= w * direction[1]
    
    # Boundary distance gradients
    for idx in range(N):
        w = weights[N + idx]
        norm = norms[idx]
        if norm > 1e-15:
            grad[2*idx] -= w * points[idx, 0] / norm
            grad[2*idx+1] -= w * points[idx, 1] / norm
    
    soft_min_val = -(max_val + np.log(sum_exp)) / beta
    obj = -soft_min_val
    return obj, grad


def _local_repulsion_polish(points, max_iters=300, tol=1e-13):
    """Exact local search: push apart overlapping pairs and pull inward if outside."""
    points = points.copy()
    n = len(points)
    
    for iteration in range(max_iters):
        if time.time() > _deadline:
            break
        improved = False
        
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        norms = np.hypot(points[:, 0], points[:, 1])
        
        # Find current minimum radius
        iu = np.triu_indices(n, k=1)
        min_pair = np.min(dists[iu]) / 2.0
        min_boundary = 1.0 - np.max(norms)
        current_r = min(min_pair, min_boundary)
        
        # For each pair that is too close, push them apart
        step = 1e-4
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            d = dists[i, j]
            if d < 2 * current_r - 1e-10:
                if d > 1e-15:
                    direction = (points[i] - points[j]) / d
                    overlap = (2 * current_r - d) * 0.5
                    move = overlap * 0.15
                    points[i] += direction * move
                    points[j] -= direction * move
                    improved = True
        
        # For each point too close to boundary, pull inward
        for i in range(n):
            norm_i = norms[i]
            if norm_i > 1.0 - current_r + 1e-10:
                if norm_i > 1e-15:
                    direction = points[i] / norm_i
                    overlap = (norm_i - (1.0 - current_r)) * 0.5
                    move = overlap * 0.15
                    points[i] -= direction * move
                    improved = True
        
        # Recompute score
        new_score = _compute_score(points)
        if new_score <= current_r - tol:
            break
        if not improved:
            break
    
    return points


def _single_point_push(points, i, r, max_iters=50):
    """Push point i away from all others and inward if needed."""
    points = points.copy()
    for _ in range(max_iters):
        if time.time() > _deadline:
            break
        moved = False
        # Push away from all other points
        for j in range(len(points)):
            if i == j:
                continue
            d = np.linalg.norm(points[i] - points[j])
            if d < 2*r - 1e-12 and d > 1e-15:
                direction = (points[i] - points[j]) / d
                move = (2*r - d) * 0.5 * 0.3
                points[i] += direction * move
                moved = True
        # Pull inward if too close to boundary
        norm_i = np.linalg.norm(points[i])
        if norm_i > 1.0 - r + 1e-12 and norm_i > 1e-15:
            direction = points[i] / norm_i
            move = (norm_i - (1.0 - r)) * 0.5 * 0.3
            points[i] -= direction * move
            moved = True
        if not moved:
            break
    return points


def construct():
    global _deadline
    t_start = time.time()
    _deadline = t_start + 140.0  # Safety margin before 150s kill
    
    # Start with ring-structured initialization
    best_points = _ring_initial()
    best_score = _compute_score(best_points)
    
    # Try multiple ring configurations
    ring_configs = [
        [0.0, 0.22, 0.45, 0.68, 0.88],
        [0.0, 0.20, 0.42, 0.65, 0.85],
        [0.0, 0.25, 0.50, 0.72, 0.90],
        [0.0, 0.18, 0.38, 0.60, 0.82],
    ]
    for radii in ring_configs:
        pts = _ring_initial(radii)
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # Multi-restart L-BFGS-B optimization with increasing beta
    rng = np.random.default_rng(42)
    
    for restart in range(15):
        if time.time() > _deadline - 20.0:
            break
        
        if restart == 0:
            x0 = best_points.flatten().copy()
        else:
            # Perturb best points with varying noise
            noise_scale = 0.04 * np.exp(-0.2 * restart)
            x0 = best_points.flatten().copy()
            x0 += rng.normal(0, noise_scale, size=2*N)
            # Ensure points stay in unit circle
            pts = x0.reshape(N, 2)
            norms = np.hypot(pts[:, 0], pts[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.999
            x0 = pts.flatten()
        
        # Use L-BFGS-B with analytic gradient
        for beta in [20.0, 40.0, 80.0, 160.0]:
            if time.time() > _deadline - 12.0:
                break
            
            result = minimize(
                _soft_min_obj_grad,
                x0,
                method='L-BFGS-B',
                args=(beta,),
                jac=True,
                options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
            )
            
            if result.success or result.fun < 1e10:
                opt_points = result.x.reshape(N, 2)
                
                # Verify validity
                if np.all(np.isfinite(opt_points)):
                    norms = np.hypot(opt_points[:, 0], opt_points[:, 1])
                    if np.all(norms <= 1.0):
                        score = _compute_score(opt_points)
                        if score > best_score:
                            best_score = score
                            best_points = opt_points.copy()
                            x0 = opt_points.flatten().copy()
    
    # Final polish with high beta
    if time.time() < _deadline - 8.0:
        for beta in [300.0, 600.0, 1200.0]:
            if time.time() > _deadline - 5.0:
                break
            result = minimize(
                _soft_min_obj_grad,
                best_points.flatten(),
                method='L-BFGS-B',
                args=(beta,),
                jac=True,
                options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 20}
            )
            if result.success or result.fun < 1e10:
                opt_points = result.x.reshape(N, 2)
                if np.all(np.isfinite(opt_points)):
                    norms = np.hypot(opt_points[:, 0], opt_points[:, 1])
                    if np.all(norms <= 1.0):
                        score = _compute_score(opt_points)
                        if score > best_score:
                            best_score = score
                            best_points = opt_points.copy()
    
    # Exact local repulsion polish
    if time.time() < _deadline - 3.0:
        polished = _local_repulsion_polish(best_points, max_iters=400, tol=1e-14)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    return best_points
# EVOLVE-BLOCK-END
