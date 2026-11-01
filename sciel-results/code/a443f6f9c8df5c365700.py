"""Combined program: soft-min L-BFGS-B optimization for circle packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    return min(r_cont, r_pair)


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = 0.0
    for spacing in np.linspace(0.15, 0.35, 80):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    if best_pts is not None:
        return best_pts
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _ring_init():
    """Generate ring-based initialization."""
    centers = [[0.0, 0.0]]
    remaining = N - 1
    num_rings = 3
    points_per_ring = [remaining // num_rings] * num_rings
    for i in range(remaining - sum(points_per_ring)):
        points_per_ring[i % num_rings] += 1
    for ring_idx, count in enumerate(points_per_ring):
        if count <= 0:
            continue
        radius = 0.3 + ring_idx * 0.25
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for angle in angles:
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(centers[:N])


def _golden_spiral_init():
    """Generate initialization using golden angle spiral."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    centers = []
    for i in range(N):
        r = np.sqrt((i + 0.5) / N) * 0.9
        theta = i * golden_angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        centers.append([x, y])
    return np.array(centers)


def _soft_min(values, beta):
    """Compute soft minimum using log-sum-exp trick."""
    v = np.asarray(values)
    m = np.min(v)
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))


def _objective_and_gradient(params, beta):
    """
    Objective to minimize: -soft_min(min_pair_dists/2, 1 - norms)
    Returns (objective, gradient) for L-BFGS-B.
    """
    centers = params.reshape(N, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq + 1e-30)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms
    
    # All constraint values
    vals = np.concatenate([pair_dists / 2.0, bound_dists])
    
    # Soft minimum
    m = np.min(vals)
    exp_terms = np.exp(-beta * (vals - m))
    S = np.sum(exp_terms)
    weights = exp_terms / S
    obj = -(m - (1.0 / beta) * np.log(S))
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pair distance gradients
    pair_weights = weights[:N*(N-1)//2]
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w < 1e-12:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-10:
            vec = (centers[i] - centers[j]) / d
            grad[i] += w * vec / 2.0
            grad[j] -= w * vec / 2.0
    
    # Boundary gradients
    bound_weights = weights[N*(N-1)//2:]
    for i in range(N):
        w = bound_weights[i]
        if w < 1e-12:
            continue
        if norms[i] > 1e-10:
            grad[i] -= w * centers[i] / norms[i]
    
    return obj, -grad


def _optimize_from_start(centers, beta, maxiter=80, gtol=1e-8):
    """Run L-BFGS-B optimization from given starting point."""
    try:
        res = minimize(
            _objective_and_gradient,
            centers.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        result = res.x.reshape(N, 2)
        if np.all(np.isfinite(result)):
            return result
    except Exception:
        pass
    return centers


def _progressive_polish(centers, start_time):
    """Progressively increase beta for finer optimization."""
    current = centers.copy()
    best_score = _score(current)
    
    betas = [30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0]
    for beta in betas:
        if time.time() > _DEADLINE - 25:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=60, gtol=1e-8)
        if np.all(np.isfinite(polished)):
            r = _score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score


def _local_search(centers, start_time, max_attempts=8):
    """Local search with random perturbations + re-optimization."""
    current = centers.copy()
    current_score = _score(current)
    rng = np.random.default_rng(12345)
    
    for attempt in range(max_attempts):
        if time.time() > _DEADLINE - 30:
            break
        
        # Random perturbation
        noise_scale = 0.03 / (1 + attempt * 0.5)
        noisy = current + rng.normal(0, noise_scale, current.shape)
        
        # Keep inside container
        norms = np.linalg.norm(noisy, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            noisy[mask] *= 0.95 / norms[mask, np.newaxis]
        
        # Optimize perturbed solution
        optimized = _optimize_from_start(noisy, beta=100.0, maxiter=40, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            r = _score(optimized)
            if r > current_score:
                current_score = r
                current = optimized.copy()
    
    return current, current_score


def _final_high_precision(centers, start_time):
    """Final high-precision polish with very high beta."""
    current = centers.copy()
    best_score = _score(current)
    
    for beta in [5000.0, 10000.0, 20000.0]:
        if time.time() > _DEADLINE - 5:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=100, gtol=1e-10)
        if np.all(np.isfinite(polished)):
            r = _score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score


def construct():
    start_time = time.time()
    
    # Generate multiple initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    inits.append(_golden_spiral_init())
    
    # Add a few random initializations
    rng = np.random.default_rng(42)
    for _ in range(2):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(rng.uniform(0, 0.81, N))
        inits.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Score all initializations
    best_pts = None
    best_score = 0.0
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
    
    if best_pts is None:
        best_pts = inits[0]
        best_score = _score(best_pts)
    
    # Progressive polish from best initialization
    try:
        if time.time() < _DEADLINE - 60:
            polished, pscore = _progressive_polish(best_pts, start_time)
            if pscore > best_score:
                best_pts = polished
                best_score = pscore
    except Exception:
        pass
    
    # Local search to escape local minima
    try:
        if time.time() < _DEADLINE - 35:
            searched, sscore = _local_search(best_pts, start_time)
            if sscore > best_score:
                best_pts = searched
                best_score = sscore
    except Exception:
        pass
    
    # Final high-precision polish
    try:
        if time.time() < _DEADLINE - 8:
            final, fscore = _final_high_precision(best_pts, start_time)
            if fscore > best_score:
                best_pts = final
                best_score = fscore
    except Exception:
        pass
    
    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        return inits[0]
    
    return best_pts
