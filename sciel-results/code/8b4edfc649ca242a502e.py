"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: Multi-start Nelder-Mead with Sobol-sequence initialization,
3-4-4 layer structure, and soft-min smooth objective for better convergence.
"""
import numpy as np
import time
from scipy.optimize import minimize, sobol
import itertools
import math

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
N_TRIP = len(_TRIPLETS)

DEADLINE = 146.0

def _bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    u = (S3H - pts[:, 1]) / S3H
    v = 2.0 * pts[:, 0] - 1.0 + u
    return np.column_stack([u, v])

def _project_bary_to_valid(bary):
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack([u, v])

def _all_triangle_areas(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas

def _min_triangle_area(pts):
    return np.min(_all_triangle_areas(pts))

def _soft_min(areas, beta):
    """Log-sum-exp based soft minimum, smoother than hard min."""
    # Use -log-sum-exp(-areas/beta) * beta as a smooth approximation to min
    scaled = -areas / beta
    max_scaled = np.max(scaled)
    if max_scaled < -100:
        return np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -beta * log_sum

def _layered_init(rng, seed_idx):
    """
    Initialize with a 3-4-4 layer structure relative to centroid.
    Layer 0: 3 points near corners
    Layer 1: 4 points in a ring around centroid
    Layer 2: 4 points in an outer ring
    """
    bary = np.zeros((N, 2))
    center = np.array([1/3, 1/3])
    
    # Layer 0: 3 points near the three vertices (barycentric corners)
    corners = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    for i in range(3):
        # Move slightly inward from corner
        pull = 0.15 + rng.random() * 0.1
        bary[i] = corners[i] * (1 - pull) + center * pull
    
    # Layer 1: 4 points in a ring around center
    r1 = 0.18 + rng.random() * 0.08
    angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2]) + rng.random() * 0.3
    for i in range(4):
        u = center[0] + r1 * np.cos(angles1[i])
        v = center[1] + r1 * np.sin(angles1[i])
        bary[3+i] = [u, v]
    
    # Layer 2: 4 points in an outer ring
    r2 = 0.32 + rng.random() * 0.1
    angles2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]) + rng.random() * 0.3
    for i in range(4):
        u = center[0] + r2 * np.cos(angles2[i])
        v = center[1] + r2 * np.sin(angles2[i])
        bary[7+i] = [u, v]
    
    return _project_bary_to_valid(bary)

def _sobol_init(seed_offset=0):
    """Generate a well-distributed initial point using Sobol sequence."""
    # Use Sobol to generate N points in the unit square, then map to triangle
    sampler = sobol.Sobol(d=2, scramble=True, seed=seed_offset)
    sampler.fast_forward(2**8)
    samples = sampler.random(2**8)
    
    # Pick a random subset of N points from the Sobol sequence
    # This gives us low-discrepancy points in the unit square
    idx = np.random.default_rng(seed_offset).choice(2**8, N, replace=False)
    pts_sq = samples[idx]
    
    # Map from unit square to triangle: for each point (s1, s2),
    # use s1 to choose which "half" and s2 for position
    bary = np.zeros((N, 2))
    for i in range(N):
        s1, s2 = pts_sq[i]
        # Transform to barycentric: u = s1 * (1 - s2), v = s2 * (1 - s1)
        # Actually use: u = s1, v = s2*(1-s1) to fill triangle
        u = s1
        v = s2 * (1 - s1)
        bary[i] = [u, v]
    
    return _project_bary_to_valid(bary)

def _nelder_mead_polish(bary0, max_iter=2000, beta=0.003):
    """Polish with Nelder-Mead on soft-min objective."""
    x0 = bary0.flatten()
    
    def nm_objective(x):
        b = x.reshape(N, 2)
        b = _project_bary_to_valid(b)
        cart = _bary_to_cart(b)
        areas = _all_triangle_areas(cart)
        return -_soft_min(areas, beta)
    
    try:
        res = minimize(nm_objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12,
                               'adaptive': True})
        return _project_bary_to_valid(res.x.reshape(N, 2))
    except Exception:
        return bary0

def _nelder_mead_hard(bary0, max_iter=2000):
    """Polish with Nelder-Mead on exact min area."""
    x0 = bary0.flatten()
    
    def nm_objective(x):
        b = x.reshape(N, 2)
        b = _project_bary_to_valid(b)
        cart = _bary_to_cart(b)
        return -_min_triangle_area(cart)
    
    try:
        res = minimize(nm_objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12,
                               'adaptive': True})
        return _project_bary_to_valid(res.x.reshape(N, 2))
    except Exception:
        return bary0

def _gradient_ascent_step(bary, steps=50, lr=0.01):
    """Subgradient ascent on the minimum area."""
    current_bary = bary.copy()
    
    for step in range(steps):
        cart = _bary_to_cart(current_bary)
        areas = _all_triangle_areas(cart)
        min_area = np.min(areas)
        
        tol = min_area * 0.02 + 1e-10
        active_mask = areas <= min_area + tol
        active_indices = np.where(active_mask)[0]
        
        total_grad = np.zeros((N, 2))
        for idx in active_indices:
            i, j, k = _TRIPLETS[idx]
            pi = cart[i]
            pj = cart[j]
            pk = cart[k]
            dx1 = pj[0] - pi[0]
            dy1 = pj[1] - pi[1]
            dx2 = pk[0] - pi[0]
            dy2 = pk[1] - pi[1]
            cross = dx1 * dy2 - dy1 * dx2
            sign = np.sign(cross)
            if abs(cross) < 1e-15:
                sign = 1.0
            coeff = 0.5 * sign
            total_grad[i] += coeff * np.array([dy1 - dy2, dx2 - dx1])
            total_grad[j] += coeff * np.array([dy2, -dx2])
            total_grad[k] += coeff * np.array([-dy1, dx1])
        
        total_grad /= max(1, len(active_indices))
        
        # Convert to barycentric gradient
        grad_u = total_grad[:, 0] * (-0.5) + total_grad[:, 1] * (-S3H)
        grad_v = total_grad[:, 0] * (0.5) + total_grad[:, 1] * (-S3H)
        bary_grad = np.column_stack([grad_u, grad_v])
        
        current_bary = current_bary + lr * bary_grad
        current_bary = _project_bary_to_valid(current_bary)
        
        new_cart = _bary_to_cart(current_bary)
        new_min = _min_triangle_area(new_cart)
        if new_min < min_area:
            lr *= 0.5
        else:
            lr = min(lr * 1.05, 0.03)
    
    return current_bary

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_bary = None
    best_area = 0.0
    best_cart = None

    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_bary, best_area, best_cart
        bary = _project_bary_to_valid(bary)
        cart = _bary_to_cart(bary)
        area = _min_triangle_area(cart)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
            best_cart = cart.copy()
        return area

    # Generate diverse initial configurations
    init_configs = []
    
    # Layered inits with different random seeds
    for i in range(12):
        init_configs.append(_layered_init(rng, i))
    
    # Sobol-based inits
    for i in range(10):
        init_configs.append(_sobol_init(seed_offset=100 + i))
    
    # Symmetric inits
    center = np.array([1/3, 1/3])
    for r1, r2, offset in [(0.15, 0.30, 0.0), (0.20, 0.35, 0.1), (0.25, 0.40, 0.2)]:
        bary = np.zeros((N, 2))
        corners = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        for i in range(3):
            bary[i] = corners[i] * 0.85 + center * 0.15
        for i in range(4):
            angle = i * np.pi / 2 + offset
            bary[3+i] = center + r1 * np.array([np.cos(angle), np.sin(angle)])
        for i in range(4):
            angle = i * np.pi / 2 + np.pi/4 + offset
            bary[7+i] = center + r2 * np.array([np.cos(angle), np.sin(angle)])
        init_configs.append(_project_bary_to_valid(bary))
    
    # Random inits
    for _ in range(10):
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        if np.any(mask):
            scale = 1.0 / (u[mask] + v[mask])
            u[mask] *= scale
            v[mask] *= scale
        init_configs.append(np.column_stack([u, v]))
    
    # Evaluate all initial configs
    for bary0 in init_configs:
        if not _elapsed_ok(40):
            break
        _update_best(bary0)
    
    # Main optimization: multi-start Nelder-Mead
    max_starts = len(init_configs)
    for start_idx in range(max_starts):
        if not _elapsed_ok(20):
            break
        
        current_bary = init_configs[start_idx].copy()
        
        # Phase 1: Quick gradient ascent
        if _elapsed_ok(15):
            current_bary = _gradient_ascent_step(current_bary, steps=40, lr=0.015)
            _update_best(current_bary)
        
        # Phase 2: Nelder-Mead with soft min
        if _elapsed_ok(10):
            current_bary = _nelder_mead_polish(current_bary, max_iter=1500, beta=0.003)
            _update_best(current_bary)
        
        # Phase 3: Nelder-Mead with hard min for final polish
        if _elapsed_ok(8):
            current_bary = _nelder_mead_hard(current_bary, max_iter=1000)
            _update_best(current_bary)
        
        # Phase 4: Perturbation and re-optimize for promising starts
        if _elapsed_ok(12) and start_idx < max_starts // 2:
            for _ in range(2):
                if not _elapsed_ok(8):
                    break
                perturbed = current_bary + rng.normal(0, 0.02, size=(N, 2))
                perturbed = _project_bary_to_valid(perturbed)
                perturbed = _nelder_mead_polish(perturbed, max_iter=800, beta=0.003)
                _update_best(perturbed)
                if _elapsed_ok(5):
                    perturbed = _nelder_mead_hard(perturbed, max_iter=500)
                    _update_best(perturbed)
                current_bary = perturbed
    
    # Final intensive polish on the best found
    if best_bary is not None and _elapsed_ok(10):
        polished = _nelder_mead_polish(best_bary, max_iter=2000, beta=0.002)
        _update_best(polished)
        
        if _elapsed_ok(5):
            polished = _nelder_mead_hard(best_bary, max_iter=1500)
            _update_best(polished)
        
        # Small perturbation + polish
        if _elapsed_ok(4):
            for scale in [0.01, 0.005]:
                if not _elapsed_ok(2):
                    break
                perturbed = best_bary + rng.normal(0, scale, size=(N, 2))
                perturbed = _project_bary_to_valid(perturbed)
                perturbed = _nelder_mead_hard(perturbed, max_iter=800)
                _update_best(perturbed)

    if best_cart is None:
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        if np.any(mask):
            scale = 1.0 / (u[mask] + v[mask])
            u[mask] *= scale
            v[mask] *= scale
        bary = np.column_stack([u, v])
        best_cart = _bary_to_cart(bary)
        
    return best_cart
