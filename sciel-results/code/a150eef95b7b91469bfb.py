"""Heilbronn 11 triangles: Barycentric soft-min optimization with DE exploration."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

DEADLINE = 145  # Safety margin from 150s limit

# Precompute all triplets for N=11 points
# There are C(11, 3) = 165 triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Barycentric coordinates: vertices of the triangle
# V0 = (0, 0), V1 = (1, 0), V2 = (0.5, sqrt(3)/2)
_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, S3H])


def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian coordinates.
    bary has shape (n, 3) with columns (lambda0, lambda1, lambda2) where each row sums to 1.
    """
    # bary @ [V0, V1, V2]^T
    return bary[:, 0:1] * _V0 + bary[:, 1:2] * _V1 + bary[:, 2:3] * _V2


def _cart_to_bary(pts):
    """Convert Cartesian coordinates to barycentric coordinates.
    pts has shape (n, 2).
    Returns barycentric coordinates with shape (n, 3) where each row sums to 1.
    """
    # Solve: p = l0*V0 + l1*V1 + l2*V2, l0+l1+l2=1
    # V0=(0,0), V1=(1,0), V2=(0.5, S3H)
    # x = l1*1 + l2*0.5
    # y = l2*S3H
    # So l2 = y/S3H, l1 = x - 0.5*l2, l0 = 1 - l1 - l2
    l2 = pts[:, 1] / S3H
    l1 = pts[:, 0] - 0.5 * l2
    l0 = 1.0 - l1 - l2
    bary = np.column_stack([l0, l1, l2])
    # Clip to ensure validity (handle floating point errors)
    bary = np.clip(bary, 0, 1)
    # Normalize to sum to 1
    bary /= bary.sum(axis=1, keepdims=True)
    return bary


def _project_to_triangle(pts):
    """
    Project points into the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _min_triangle_area(pts):
    """
    Compute the minimum area of any triangle formed by three points.
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _soft_min_area(pts, beta):
    """
    Soft minimum of triangle areas using log-sum-exp.
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _bary_objective(bary_flat, beta):
    """Objective function in barycentric coordinates.
    bary_flat is a 1D array of length 3*N.
    """
    bary = bary_flat.reshape(N, 3)
    # Normalize to ensure valid barycentric coordinates
    bary = np.clip(bary, 0, 1)
    bary /= bary.sum(axis=1, keepdims=True)
    pts = _bary_to_cart(bary)
    return -_soft_min_area(pts, beta)


def _bary_hard_objective(bary_flat):
    """Hard objective function in barycentric coordinates for final polish."""
    bary = bary_flat.reshape(N, 3)
    bary = np.clip(bary, 0, 1)
    bary /= bary.sum(axis=1, keepdims=True)
    pts = _bary_to_cart(bary)
    return -_min_triangle_area(pts)


def _penalized_hard_min(bary_flat, penalty_coeff=100.0):
    """Penalized objective that discourages points on the boundary."""
    bary = bary_flat.reshape(N, 3)
    bary = np.clip(bary, 0, 1)
    bary /= bary.sum(axis=1, keepdims=True)
    pts = _bary_to_cart(bary)
    area = _min_triangle_area(pts)
    # Penalty for points too close to boundary (barycentric coords near 0)
    min_bary = np.min(bary, axis=1)
    boundary_penalty = penalty_coeff * np.sum(np.minimum(min_bary, 0.05))
    return -(area - boundary_penalty)


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts)
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    def _generate_initial_configs():
        """Generate diverse starting points in barycentric coordinates."""
        configs = []
        
        # 1. Symmetric base: Corners, midpoints, centroid, etc.
        base_sym_cart = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, S3H],
            [0.5, 0.0],
            [0.75, S3H / 2],
            [0.25, S3H / 2],
            [1/3, S3H/3],
            [1/6, S3H/6],
            [5/6, S3H/6],
            [1/3, 2*S3H/3],
            [2/3, 2*S3H/3],
        ])
        base_sym_bary = _cart_to_bary(base_sym_cart)
        configs.append(base_sym_bary.flatten())

        # 2. Random valid barycentric configurations
        for _ in range(8):
            # Generate random barycentric coordinates
            raw = rng.random((N, 3))
            bary = raw / raw.sum(axis=1, keepdims=True)
            configs.append(bary.flatten())

        # 3. Perturbed symmetric base
        for _ in range(5):
            perturbed = base_sym_bary + rng.normal(0, 0.05, size=(N, 3))
            perturbed = np.clip(perturbed, 0, 1)
            perturbed /= perturbed.sum(axis=1, keepdims=True)
            configs.append(perturbed.flatten())

        return configs

    def _simulated_annealing(bary0, T_start=0.1, T_end=0.001, n_iter=5000):
        """Simulated Annealing in barycentric space."""
        current_bary = bary0.copy()
        current_bary = np.clip(current_bary, 0, 1)
        current_bary /= current_bary.sum(axis=1, keepdims=True)
        current_pts = _bary_to_cart(current_bary)
        current_area = _min_triangle_area(current_pts)
        
        best_bary = current_bary.copy()
        best_area = current_area
        
        n_iterations = n_iter
        for i in range(n_iterations):
            if not _elapsed_ok(5):
                break
                
            # Temperature schedule (exponential decay)
            T = T_start * (T_end / T_start) ** (i / n_iterations)
            
            # Generate perturbation in barycentric space
            perturbation = rng.normal(0, 0.02, size=(N, 3))
            new_bary = current_bary + perturbation
            new_bary = np.clip(new_bary, 0, 1)
            new_bary /= new_bary.sum(axis=1, keepdims=True)
            
            new_pts = _bary_to_cart(new_bary)
            new_area = _min_triangle_area(new_pts)
            
            # Acceptance criterion
            if new_area >= current_area:
                current_bary = new_bary
                current_area = new_area
                if new_area > best_area:
                    best_area = new_area
                    best_bary = new_bary.copy()
            else:
                # Accept with probability exp(-(current_area - new_area) / T)
                # Note: we're maximizing, so delta = new_area - current_area < 0
                delta = new_area - current_area
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_bary = new_bary
                    current_area = new_area
        
        return best_bary, best_area

    def _optimize_single_bary(bary0, beta_schedule):
        """Run L-BFGS-B with a schedule of beta values in barycentric space."""
        current_bary = bary0.copy()
        
        for beta in beta_schedule:
            if not _elapsed_ok(10):
                break
                
            try:
                res = minimize(
                    _bary_objective,
                    current_bary.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                current_bary = res.x.reshape(N, 3)
                current_bary = np.clip(current_bary, 0, 1)
                current_bary /= current_bary.sum(axis=1, keepdims=True)
                pts = _bary_to_cart(current_bary)
                _update_best(pts)
            except Exception:
                pass
                
        return current_bary

    # Phase 1: Multi-start optimization with annealing
    configs = _generate_initial_configs()
    
    # Beta schedule: start smooth, get sharp
    beta_schedule = [10, 50, 200, 1000]

    for config in configs:
        if not _elapsed_ok(20):
            break
        _optimize_single_bary(config, beta_schedule)

    # Phase 2: Simulated Annealing on the best candidate
    if best_pts is not None and _elapsed_ok(30):
        best_bary = _cart_to_bary(best_pts)
        # Run SA with multiple restarts
        for sa_iter in range(3):
            if not _elapsed_ok(10):
                break
            sa_bary, sa_area = _simulated_annealing(
                best_bary, 
                T_start=0.05, 
                T_end=0.001, 
                n_iter=3000
            )
            sa_pts = _bary_to_cart(sa_bary)
            _update_best(sa_pts)
            if _elapsed_ok(5):
                # Polish with L-BFGS
                bary_polished = _optimize_single_bary(sa_bary, [500, 1000])
                pts_polished = _bary_to_cart(bary_polished)
                _update_best(pts_polished)

    # Phase 3: Powell Polish on the best candidate (in Cartesian space for robustness)
    if best_pts is not None and _elapsed_ok(15):
        def powell_obj(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_min_triangle_area(p)
        
        try:
            res = minimize(
                powell_obj,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 100, 'maxfev': 5000, 'xrtol': 1e-10, 'fatol': 1e-12}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass

    # Phase 4: Final fine polish with Nelder-Mead on the absolute best
    if best_pts is not None and _elapsed_ok(5):
        def nm_obj_final(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_min_triangle_area(p)
        
        try:
            res = minimize(
                nm_obj_final,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        # Absolute fallback: random points
        pts = rng.random((N, 2))
        for i in range(N):
            u, v = pts[i]
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = S3H * v
            pts[i] = [x, y]
        best_pts = _project_to_triangle(pts)

    return best_pts
