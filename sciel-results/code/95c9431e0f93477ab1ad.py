"""Heilbronn-11: maximize minimum triangle area using barycentric coordinates and soft-min optimization."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 140.0  # safety margin from 150s limit

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)


def _bary_to_cart(bary):
    """Convert barycentric coordinates (n,3) to Cartesian (n,2)."""
    return bary[:, 0:1] * V0 + bary[:, 1:2] * V1 + bary[:, 2:3] * V2


def _cart_to_bary(cart):
    """Convert Cartesian (n,2) to barycentric (n,3)."""
    # barycentric: point = a*V0 + b*V1 + c*V2, a+b+c=1
    # x = b + 0.5*c, y = (sqrt(3)/2)*c
    c = cart[:, 1] / S3H
    b = cart[:, 0] - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def _project_bary(bary):
    """Project barycentric coordinates to valid region (all >= 0, sum=1)."""
    bary = bary.copy()
    # Ensure non-negative
    bary = np.maximum(bary, 0.0)
    # Normalize
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-10)
    bary = bary / s
    return bary


def _min_triangle_area(cart):
    """Compute minimum area of any triangle formed by 3 points."""
    pi = cart[_T_I]
    pj = cart[_T_J]
    pk = cart[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _all_areas(cart):
    """Compute all triangle areas."""
    pi = cart[_T_I]
    pj = cart[_T_J]
    pk = cart[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _soft_min(areas, beta):
    """Soft minimum approximation using log-sum-exp."""
    # -log(sum(exp(-beta * areas))) / beta approximates min(areas) for large beta
    min_val = np.min(areas)
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_val - np.log(np.sum(np.exp(-beta * shifted))) / beta


def _objective(x, beta):
    """Objective function: negative soft-min area (to minimize)."""
    bary = _project_bary(x.reshape(N, 3))
    cart = _bary_to_cart(bary)
    areas = _all_areas(cart)
    return -_soft_min(areas, beta)


def _objective_grad(x, beta):
    """Gradient of the objective (negative soft-min area)."""
    bary = _project_bary(x.reshape(N, 3))
    cart = _bary_to_cart(bary)
    areas = _all_areas(cart)
    
    # Gradient of soft-min
    min_val = np.min(areas)
    shifted = areas - min_val
    weights = np.exp(-beta * shifted)
    weights = weights / np.sum(weights)
    
    # Gradient of each area w.r.t. points
    # area = 0.5 * |cross|, but we use abs so sign matters
    pi = cart[_T_I]
    pj = cart[_T_J]
    pk = cart[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    
    # d(area)/d(point) for each point in the triangle
    # For triangle (i,j,k): area = 0.5*|cross|
    # d(area)/d(pi) = -0.5*(dy2, -dx2) * sign(cross) ... this gets complex
    # Let's use numerical gradient instead for simplicity
    
    return None  # Fall back to numerical gradient


def _nelder_mead_optimize(bary_start, beta, max_iter, time_limit):
    """Run Nelder-Mead optimization with time limit."""
    x0 = bary_start.flatten()
    
    result = minimize(
        _objective,
        x0,
        args=(beta,),
        method='Nelder-Mead',
        options={
            'maxiter': max_iter,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    bary_opt = _project_bary(result.x.reshape(N, 3))
    cart_opt = _bary_to_cart(bary_opt)
    return cart_opt, -result.fun


def _push_apart_worst(cart, step_size=0.01):
    """Push apart the worst (min-area) triangle's vertices."""
    areas = _all_areas(cart)
    min_idx = np.argmin(areas)
    i, j, k = _TRIPLETS[min_idx]
    
    new_cart = cart.copy()
    centroid = (cart[i] + cart[j] + cart[k]) / 3.0
    
    for idx in [i, j, k]:
        direction = cart[idx] - centroid
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
            new_cart[idx] = cart[idx] + step_size * direction
    
    # Project back to triangle
    bary = _cart_to_bary(new_cart)
    bary = _project_bary(bary)
    return _bary_to_cart(bary)


def _random_perturb(cart, rng, sigma=0.02):
    """Random perturbation in Cartesian, then project."""
    new_cart = cart + rng.normal(0, sigma, size=(N, 2))
    bary = _cart_to_bary(new_cart)
    bary = _project_bary(bary)
    return _bary_to_cart(bary)


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_cart = None
    best_area = 0.0
    
    def _elapsed_ok(margin=5.0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(cart):
        nonlocal best_cart, best_area
        area = _min_triangle_area(cart)
        if area > best_area:
            best_area = area
            best_cart = cart.copy()
    
    # Generate diverse initial configurations
    inits = []
    
    # Init 1: Symmetric with corners and midpoints
    init1_cart = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.25, S3H/2],
        [0.75, S3H/2],
        [0.5, S3H/3],
        [0.333, S3H/6],
        [0.667, S3H/6],
        [0.5, 2*S3H/3],
        [0.3, 0.15],
    ])
    inits.append(init1_cart)
    
    # Init 2: Grid-like
    init2_cart = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, S3H/2],
        [0.75, S3H/2],
        [0.5, S3H/2],
        [0.5, S3H/6],
        [0.5, 2*S3H/3],
        [0.5, 0.0],
    ])
    inits.append(init2_cart)
    
    # Init 3: Random spread
    init3_cart = _random_perturb(np.zeros((N, 2)), rng, sigma=0.5)
    inits.append(init3_cart)
    
    # Init 4: Concentric rings
    init4_cart = np.zeros((N, 2))
    init4_cart[0] = [0.5, S3H/3]  # center
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    for i, a in enumerate(angles):
        init4_cart[1+i] = [0.5 + 0.25*np.cos(a), S3H/3 + 0.25*np.sin(a)*S3H/S3]
    angles2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    for i, a in enumerate(angles2):
        init4_cart[5+i] = [0.5 + 0.4*np.cos(a), S3H/3 + 0.4*np.sin(a)*S3H/S3]
    init4_cart = _bary_to_cart(_project_bary(_cart_to_bary(init4_cart)))
    inits.append(init4_cart)
    
    # Evaluate all inits
    for init in inits:
        _update_best(init)
    
    if best_cart is None:
        return np.array([[0.5, S3H/3]] * N)
    
    # Phase A: Nelder-Mead with soft-min from multiple starts
    if _elapsed_ok(30):
        betas = [10, 30, 100]
        for beta in betas:
            if not _elapsed_ok(10):
                break
            
            # Try from current best
            bary_start = _cart_to_bary(best_cart)
            bary_start = _project_bary(bary_start)
            
            cart_opt, area_opt = _nelder_mead_optimize(
                bary_start, beta, max_iter=2000, time_limit=20
            )
            _update_best(cart_opt)
            
            # Also try from a perturbed version
            if _elapsed_ok(10):
                bary_pert = _cart_to_bary(_random_perturb(best_cart, rng, 0.05))
                bary_pert = _project_bary(bary_pert)
                cart_opt2, area_opt2 = _nelder_mead_optimize(
                    bary_pert, beta, max_iter=1000, time_limit=15
                )
                _update_best(cart_opt2)
    
    # Phase B: Iterative push-apart refinement
    if _elapsed_ok(20):
        current = best_cart.copy()
        current_area = _min_triangle_area(current)
        step_size = 0.005
        
        for it in range(2000):
            if not _elapsed_ok(5):
                break
            
            candidate = _push_apart_worst(current, step_size)
            cand_area = _min_triangle_area(candidate)
            
            if cand_area > current_area:
                current = candidate
                current_area = cand_area
                _update_best(current)
            else:
                # Small random nudge to escape
                current = _random_perturb(current, rng, 0.002)
                current_area = _min_triangle_area(current)
            
            if it % 100 == 0:
                step_size *= 0.95
    
    # Phase C: Fine-tuning with very small steps
    if _elapsed_ok(10):
        current = best_cart.copy()
        current_area = _min_triangle_area(current)
        
        for it in range(500):
            if not _elapsed_ok(3):
                break
            
            # Try pushing apart worst triangle
            candidate = _push_apart_worst(current, 0.001)
            cand_area = _min_triangle_area(candidate)
            
            if cand_area > current_area:
                current = candidate
                current_area = cand_area
                _update_best(current)
            else:
                # Tiny random perturbation
                current = _random_perturb(current, rng, 0.0005)
                current_area = _min_triangle_area(current)
    
    if best_cart is None:
        return np.array([[0.5, S3H/3]] * N)
    
    return _bary_to_cart(_project_bary(_cart_to_bary(best_cart)))
