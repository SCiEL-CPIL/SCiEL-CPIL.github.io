"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(pts):
    """Compute the packing score (minimum radius)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Container constraint
    dists_center = np.sqrt(np.sum(pts**2, axis=1))
    if np.any(dists_center >= 1.0):
        return 0.0
    r_container = 1.0 - np.max(dists_center)
    if r_container <= 0:
        return 0.0
    # Pairwise distances
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Exclude self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_container, r_pair)

def _negative_score(pts):
    """Negative of score for minimization."""
    return -_score(pts)

def _hex_lattice_init():
    """Generate initial points using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _random_init():
    """Generate random initial points inside the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _ring_init():
    """Generate initial points on concentric rings."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    # Rings
    n_remaining = N - 1
    ring_radius = 0.3
    while n_remaining > 0 and ring_radius < 0.95:
        # How many points fit on this ring?
        circumference = 2 * np.pi * ring_radius
        min_spacing = 0.25  # rough estimate
        n_on_ring = int(circumference / min_spacing)
        n_on_ring = min(n_on_ring, n_remaining)
        if n_on_ring <= 0:
            n_on_ring = 1
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        ring_pts = np.column_stack([ring_radius * np.cos(angles), ring_radius * np.sin(angles)])
        pts.append(ring_pts)
        n_remaining -= n_on_ring
        ring_radius += 0.15
    pts = np.vstack(pts)
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        # Fill with random points
        extra = N - len(pts)
        angles = np.random.uniform(0, 2 * np.pi, extra)
        radii = np.sqrt(np.random.uniform(0.5, 0.95, extra))
        extra_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts = np.vstack([pts, extra_pts])
    return pts

def _soft_min_score(pts, beta=10.0):
    """Soft approximation of the packing score using log-sum-exp."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Container constraint (soft)
    dists_center = np.sqrt(np.sum(pts**2, axis=1))
    container_violation = np.maximum(0, dists_center - 1.0)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Soft min of pairwise distances
    # Use log-sum-exp trick: min(x) = -1/beta * log(sum(exp(-beta*x)))
    # For numerical stability, subtract max
    min_dists = dists[dists < np.inf]
    if len(min_dists) == 0:
        return 0.0
    # Soft min of pairwise distances
    beta_pair = 5.0
    shifted = -beta_pair * min_dists
    shifted -= np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_pair = -1.0 / beta_pair * (log_sum + np.max(shifted) * 0)  # This is getting complicated
    # Let's use a simpler approach: direct soft min
    # soft_min = -1/beta * log(sum(exp(-beta * x)))
    # For stability: let m = min(x), then soft_min = m - 1/beta * log(sum(exp(-beta*(x-m))))
    m = np.min(min_dists)
    exp_terms = np.exp(-beta_pair * (min_dists - m))
    soft_min_pair = m - np.log(np.sum(exp_terms)) / beta_pair
    # Soft min with container
    # r = min(soft_min_pair/2, 1 - max(dists_center))
    # Use soft min of these two
    r1 = soft_min_pair / 2.0
    r2 = 1.0 - np.max(dists_center)
    # Soft min of r1 and r2
    beta_final = 10.0
    m_final = min(r1, r2)
    exp_terms_final = np.exp(-beta_final * np.array([r1, r2]) + beta_final * m_final)
    soft_min_final = m_final - np.log(np.sum(exp_terms_final)) / beta_final
    # Penalty for container violation
    penalty = 100.0 * np.sum(container_violation**2)
    return soft_min_final - penalty

def _negative_soft_score(pts, beta=10.0):
    return -_soft_min_score(pts, beta)

def _gradient_soft_score(pts, beta=10.0):
    """Numerical gradient of the soft score."""
    eps = 1e-6
    grad = np.zeros_like(pts)
    for i in range(len(pts)):
        for j in range(2):
            pts_plus = pts.copy()
            pts_plus[i, j] += eps
            f_plus = _soft_min_score(pts_plus, beta)
            pts_minus = pts.copy()
            pts_minus[i, j] -= eps
            f_minus = _soft_min_score(pts_minus, beta)
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
    return -grad  # Negative because we want to maximize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    # Generate multiple initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(5):
        inits.append(_random_init())
    
    best_pts = inits[0].copy()
    best_score = _score(best_pts)
    
    # Try to improve each initialization
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        # First, try direct optimization with SLSQP
        try:
            res = minimize(
                _negative_score,
                init_pts,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if res.success or res.fun < -best_score:
                candidate = res.x
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate.copy()
        except Exception:
            pass
        
        # Try BFGS with soft objective
        if time.time() > deadline - 15.0:
            break
        try:
            res2 = minimize(
                _negative_soft_score,
                init_pts,
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-8}
            )
            candidate2 = res2.x
            cand_score2 = _score(candidate2)
            if cand_score2 > best_score:
                best_score = cand_score2
                best_pts = candidate2.copy()
        except Exception:
            pass
    
    # Final polish with high precision
    if time.time() < deadline - 5.0 and best_score > 0:
        try:
            res_final = minimize(
                _negative_score,
                best_pts,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if res_final.success:
                candidate_final = res_final.x
                cand_score_final = _score(candidate_final)
                if cand_score_final > best_score:
                    best_score = cand_score_final
                    best_pts = candidate_final.copy()
        except Exception:
            pass
    
    # Ensure all points are strictly inside the unit circle
    dists_center = np.sqrt(np.sum(best_pts**2, axis=1))
    mask = dists_center < 1.0
    if not np.all(mask):
        # Scale points that are outside
        for i in range(len(best_pts)):
            if dists_center[i] >= 1.0:
                scale = 0.9999 / dists_center[i]
                best_pts[i] *= scale
    
    return best_pts
# EVOLVE-BLOCK-END
