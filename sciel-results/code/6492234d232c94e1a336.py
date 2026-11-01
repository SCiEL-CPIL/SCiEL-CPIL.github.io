"""Structural rewrite: direct radius optimization with contact-graph initialization."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
_DEADLINE = time.time() + 135.0


def _score(centers):
    """Compute packing score: min(min_pair_dist/2, min_boundary_margin)."""
    if len(centers) != N or not np.all(np.isfinite(centers)):
        return 0.0
    norms = np.linalg.norm(centers, axis=1)
    if np.any(norms > 1.0):
        return 0.0
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_bound = np.min(1.0 - norms)
    return float(min(min_pair, min_bound))


def _contact_graph_init():
    """
    Initialize based on the known optimal packing structure:
    1 center + rings of circles. The known packing has:
    - 1 center circle
    - Ring 1: 6 circles (hexagonal)
    - Ring 2: 12 circles
    - Ring 3: 18 circles  
    - Ring 4: 13 circles (partial, to make 50)
    Total: 1+6+12+18+13 = 50
    """
    centers = []
    r = 0.1258  # known optimal radius
    
    # Center
    centers.append([0.0, 0.0])
    
    # Ring 1: 6 circles at distance ~2r from center
    d1 = 2 * r
    for i in range(6):
        angle = i * np.pi / 3.0
        centers.append([d1 * np.cos(angle), d1 * np.sin(angle)])
    
    # Ring 2: 12 circles at distance ~3.5r from center (hexagonal lattice)
    d2 = 3.5 * r
    for i in range(12):
        angle = i * np.pi / 6.0 + np.pi / 12.0
        centers.append([d2 * np.cos(angle), d2 * np.sin(angle)])
    
    # Ring 3: 18 circles at distance ~5r
    d3 = 5.0 * r
    for i in range(18):
        angle = i * 2 * np.pi / 18.0
        centers.append([d3 * np.cos(angle), d3 * np.sin(angle)])
    
    # Ring 4: 13 circles at distance ~6.5r (partial ring)
    d4 = 6.5 * r
    for i in range(13):
        angle = i * 2 * np.pi / 13.0 + 0.1
        centers.append([d4 * np.cos(angle), d4 * np.sin(angle)])
    
    centers = np.array(centers[:N])
    # Small perturbation to break exact symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    
    # Ensure inside unit circle
    norms = np.linalg.norm(centers, axis=1)
    mask = norms > 0.98
    if np.any(mask):
        centers[mask] *= 0.98 / norms[mask]
    
    return centers


def _hex_init(seed=0):
    """Hexagonal lattice initialization."""
    rng = np.random.default_rng(100 + seed)
    s = 0.25
    points = []
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        while len(points) < N:
            angle = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0, 0.9)
            points = np.vstack([points, [[rad * np.cos(angle), rad * np.sin(angle)]]])
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    pts = points[idx[:N]]
    pts += rng.normal(0, 0.01, pts.shape)
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 0.95
    if np.any(mask):
        pts[mask] *= 0.95 / norms[mask]
    return pts


def _ring_init(seed=0):
    """Ring-based initialization with varying ring sizes."""
    rng = np.random.default_rng(200 + seed)
    centers = [[0.0, 0.0]]
    remaining = N - 1
    rings = [6, 12, 18, 13]
    radii = [0.25, 0.5, 0.75, 0.92]
    for count, rad in zip(rings, radii):
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for a in angles:
            centers.append([rad * np.cos(a), rad * np.sin(a)])
    centers = np.array(centers[:N])
    centers += rng.normal(0, 0.01, centers.shape)
    norms = np.linalg.norm(centers, axis=1)
    mask = norms > 0.95
    if np.any(mask):
        centers[mask] *= 0.95 / norms[mask]
    return centers


def _spiral_init(seed=0):
    """Golden angle spiral initialization."""
    rng = np.random.default_rng(300 + seed)
    golden = np.pi * (3 - np.sqrt(5))
    centers = []
    for i in range(N):
        r = np.sqrt((i + 0.5) / N) * 0.95
        theta = i * golden
        centers.append([r * np.cos(theta), r * np.sin(theta)])
    centers = np.array(centers)
    centers += rng.normal(0, 0.01, centers.shape)
    norms = np.linalg.norm(centers, axis=1)
    mask = norms > 0.95
    if np.any(mask):
        centers[mask] *= 0.95 / norms[mask]
    return centers


def _random_init(seed=0):
    """Random initialization."""
    rng = np.random.default_rng(400 + seed)
    centers = []
    for _ in range(N):
        angle = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, 0.9)
        centers.append([rad * np.cos(angle), rad * np.sin(angle)])
    return np.array(centers)


def _objective(params, penalty):
    """
    Objective: maximize r subject to:
    - |c_i - c_j| >= 2r for all i < j
    - |c_i| + r <= 1 for all i
    
    We minimize: -r + penalty * sum(violations^2)
    """
    n = N
    centers = params[:2*n].reshape(n, 2)
    r = params[2*n]
    
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, 2*r + 1.0)  # avoid self-constraint
    iu = np.triu_indices(n, 1)
    pair_dists = d[iu]
    
    # Violations: pair_dists - 2r should be >= 0
    pair_viol = np.maximum(0, 2*r - pair_dists)
    
    # Boundary: 1 - |c_i| - r should be >= 0
    norms = np.linalg.norm(centers, axis=1)
    bound_viol = np.maximum(0, norms + r - 1.0)
    
    penalty_term = penalty * (np.sum(pair_viol**2) + np.sum(bound_viol**2))
    
    return -r + penalty_term


def _objective_grad(params, penalty):
    """Gradient of the objective."""
    n = N
    centers = params[:2*n].reshape(n, 2)
    r = params[2*n]
    
    d = cdist(centers, centers)
    np.fill_diagonal(d, 2*r + 1.0)
    iu = np.triu_indices(n, 1)
    pair_dists = d[iu]
    
    pair_viol = np.maximum(0, 2*r - pair_dists)
    norms = np.linalg.norm(centers, axis=1)
    bound_viol = np.maximum(0, norms + r - 1.0)
    
    obj = -r + penalty * (np.sum(pair_viol**2) + np.sum(bound_viol**2))
    
    grad = np.zeros_like(params)
    
    # Gradient w.r.t. r
    grad[2*n] = -1.0
    # From pair violations: d/dr of penalty * sum(max(0, 2r - d_ij)^2)
    active_pair = pair_viol > 1e-12
    if np.any(active_pair):
        grad[2*n] += penalty * 2 * np.sum(2 * pair_viol[active_pair])
    # From boundary violations: d/dr of penalty * sum(max(0, |c_i| + r - 1)^2)
    active_bound = bound_viol > 1e-12
    if np.any(active_bound):
        grad[2*n] += penalty * 2 * np.sum(bound_viol[active_bound])
    
    # Gradient w.r.t. centers
    # Pair constraints
    for k in range(len(iu[0])):
        if pair_viol[k] > 1e-12:
            i, j = iu[0][k], iu[1][k]
            diff = centers[i] - centers[j]
            dist = pair_dists[k]
            if dist > 1e-10:
                dir_vec = diff / dist
                # d/dc_i of penalty * (2r - dist)^2 = penalty * 2*(2r-dist) * (-1) * d(dist)/d(c_i)
                # d(dist)/d(c_i) = dir_vec, d(dist)/d(c_j) = -dir_vec
                coeff = penalty * 2 * pair_viol[k]
                grad[2*i:2*i+2] += coeff * dir_vec
                grad[2*j:2*j+2] -= coeff * dir_vec
    
    # Boundary constraints
    for i in range(n):
        if bound_viol[i] > 1e-12:
            if norms[i] > 1e-10:
                dir_vec = centers[i] / norms[i]
                coeff = penalty * 2 * bound_viol[i]
                grad[2*i:2*i+2] += coeff * dir_vec
    
    return obj, grad


def _optimize_radius(centers, max_penalty=1e6, start_penalty=1.0, maxiter=200):
    """
    Optimize the radius by iteratively increasing penalty.
    Returns (centers, r).
    """
    n = N
    r = _score(centers) * 0.9  # start slightly below current score
    params = np.concatenate([centers.flatten(), [r]])
    
    penalty = start_penalty
    best_centers = centers.copy()
    best_r = r
    
    while penalty < max_penalty and time.time() < _DEADLINE - 10:
        try:
            res = minimize(
                _objective_grad,
                params,
                args=(penalty,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'gtol': 1e-8}
            )
            if res.success or res.status in [1, 2, 3, 4]:
                new_params = res.x
                new_centers = new_params[:2*n].reshape(n, 2)
                new_r = new_params[2*n]
                
                # Check validity
                if np.all(np.isfinite(new_centers)):
                    actual_score = _score(new_centers)
                    if actual_score > best_r:
                        best_r = actual_score
                        best_centers = new_centers.copy()
                        params = new_params.copy()
                else:
                    break
        except Exception:
            break
        
        penalty *= 5.0
    
    return best_centers, best_r


def _force_directed(centers, steps=50, learning_rate=0.001):
    """
    Force-directed relaxation: push apart overlapping circles,
    pull toward center if too far out.
    """
    n = N
    current = centers.copy()
    
    for step in range(steps):
        if time.time() > _DEADLINE - 15:
            break
        
        forces = np.zeros_like(current)
        r = _score(current)
        
        # Pairwise repulsion
        d = cdist(current, current)
        np.fill_diagonal(d, np.inf)
        
        for i in range(n):
            for j in range(i+1, n):
                dist = d[i, j]
                if dist < 2*r * 1.05:  # if overlapping or close
                    if dist > 1e-10:
                        dir_vec = (current[i] - current[j]) / dist
                        strength = (2*r - dist) * 10.0
                        forces[i] += dir_vec * strength
                        forces[j] -= dir_vec * strength
        
        # Boundary force
        norms = np.linalg.norm(current, axis=1)
        for i in range(n):
            if norms[i] > 1.0 - r * 1.05:
                if norms[i] > 1e-10:
                    dir_vec = current[i] / norms[i]
                    strength = (norms[i] + r - 1.0) * 10.0
                    forces[i] -= dir_vec * strength
        
        # Update
        current += learning_rate * forces
        # Project back inside
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            current[mask] *= 0.99 / norms[mask]
        
        # Adaptive learning rate
        if step % 10 == 9:
            learning_rate *= 0.8
    
    return current


def _refine_with_lbfgs(centers):
    """Final refinement using L-BFGS-B on the soft-min formulation."""
    n = N
    
    def soft_min_obj(params, beta=1000.0):
        c = params.reshape(n, 2)
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        iu = np.triu_indices(n, 1)
        vals = np.concatenate([d[iu] / 2.0, 1.0 - np.linalg.norm(c, axis=1)])
        m = np.min(vals)
        exp_terms = np.exp(-beta * (vals - m))
        return -(m - (1.0/beta) * np.log(np.sum(exp_terms)))
    
    def soft_min_grad(params, beta=1000.0):
        c = params.reshape(n, 2)
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        iu = np.triu_indices(n, 1)
        pair_dists = d[iu]
        norms = np.linalg.norm(c, axis=1)
        vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
        
        m = np.min(vals)
        exp_vals = np.exp(-beta * (vals - m))
        S = np.sum(exp_vals)
        weights = exp_vals / S
        
        obj = -(m - (1.0/beta) * np.log(S))
        
        grad = np.zeros_like(c)
        pair_w = weights[:n*(n-1)//2]
        for k in range(len(pair_w)):
            if pair_w[k] < 1e-12:
                continue
            i, j = iu[0][k], iu[1][k]
            dist = pair_dists[k]
            if dist > 1e-10:
                vec = (c[i] - c[j]) / dist
                grad[i] += pair_w[k] * vec / 2.0
                grad[j] -= pair_w[k] * vec / 2.0
        
        bound_w = weights[n*(n-1)//2:]
        for i in range(n):
            if bound_w[i] < 1e-12:
                continue
            if norms[i] > 1e-10:
                grad[i] -= bound_w[i] * c[i] / norms[i]
        
        return obj, -grad
    
    try:
        res = minimize(
            soft_min_grad,
            centers.flatten(),
            args=(1000.0,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 300, 'gtol': 1e-10}
        )
        if np.all(np.isfinite(res.x)):
            return res.x.reshape(n, 2)
    except Exception:
        pass
    
    return centers


def construct():
    start = time.time()
    
    # Generate multiple initializations
    inits = [
        _contact_graph_init(),
        _hex_init(0),
        _hex_init(1),
        _ring_init(0),
        _ring_init(1),
        _spiral_init(0),
        _random_init(0),
    ]
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Force-directed relaxation + radius optimization for each init
    for i, init_pts in enumerate(inits):
        if time.time() > _DEADLINE - 50:
            break
        
        # Force-directed relaxation
        relaxed = _force_directed(init_pts, steps=30, learning_rate=0.002)
        
        # Radius optimization
        opt_pts, opt_r = _optimize_radius(relaxed, max_penalty=1e5, start_penalty=1.0, maxiter=100)
        
        score = _score(opt_pts)
        if score > best_score:
            best_score = score
            best_pts = opt_pts.copy()
    
    if best_pts is None:
        best_pts = inits[0]
        best_score = _score(best_pts)
    
    # Phase 2: Further force-directed refinement
    if time.time() < _DEADLINE - 30:
        refined = _force_directed(best_pts, steps=50, learning_rate=0.001)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined
    
    # Phase 3: High-precision L-BFGS refinement
    if time.time() < _DEADLINE - 10:
        final_pts = _refine_with_lbfgs(best_pts)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts
    
    # Safety check
    final_score = _score(best_pts)
    if final_score <= 0 or not np.all(np.isfinite(best_pts)):
        return inits[0]
    
    return best_pts
