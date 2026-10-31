"""Rewritten program for Heilbronn triangles task."""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2

# Precompute all (i, j, k) triples
_TRIPLES = np.array(list(combinations(range(N), 3)))

# Barycentric basis vectors
# Point = b0*(0,0) + b1*(1,0) + b2*(0.5, H)
# So x = b1 + 0.5*b2, y = H*b2
_B1 = np.array([1.0, 0.0])
_B2 = np.array([0.5, H])

def _bary_to_cart(bary):
    """Convert (N, 3) barycentric coords to (N, 2) Cartesian."""
    # bary[:, 0] is b0 = 1 - b1 - b2, so we ignore it
    return bary[:, 1:2] * _B1 + bary[:, 2:3] * _B2

def _cart_to_bary(cart):
    """Convert (N, 2) Cartesian to (N, 3) barycentric."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / H
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _min_area_bary(bary_flat):
    """
    Compute minimum triangle area from barycentric coords (flat array of length 3*N).
    Returns negative of min area (for minimization).
    """
    bary = bary_flat.reshape(N, 3)
    cart = _bary_to_cart(bary)
    # Compute all triangle areas
    pi = cart[_TRIPLES[:, 0]]
    pj = cart[_TRIPLES[:, 1]]
    pk = cart[_TRIPLES[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return -np.min(areas)

def _soft_min_area_bary(bary_flat, beta=20.0):
    """
    Soft minimum of triangle areas using log-sum-exp trick.
    Returns negative soft-min (for minimization).
    """
    bary = bary_flat.reshape(N, 3)
    cart = _bary_to_cart(bary)
    pi = cart[_TRIPLES[:, 0]]
    pj = cart[_TRIPLES[:, 1]]
    pk = cart[_TRIPLES[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Soft min: -logsumexp(-beta * areas) / beta
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a))))
    soft_min = max_a - np.log(np.sum(np.exp(-beta * (areas - max_a)))) / beta
    return -soft_min

def _clamp_bary(bary):
    """Clamp barycentric coords to valid region (all >= 0, sum = 1)."""
    bary = np.maximum(bary, 0.0)
    # Normalize
    s = np.sum(bary, axis=1, keepdims=True)
    s = np.maximum(s, 1e-12)
    bary = bary / s
    return bary

def _objective_de(x):
    """
    Objective for differential evolution.
    x is a flat array of length 2*N (Cartesian coords).
    Returns negative min area (to minimize).
    """
    cart = x.reshape(N, 2)
    bary = _cart_to_bary(cart)
    bary = _clamp_bary(bary)
    cart = _bary_to_cart(bary)
    pi = cart[_TRIPLES[:, 0]]
    pj = cart[_TRIPLES[:, 1]]
    pk = cart[_TRIPLES[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return -np.min(areas)

def _local_refine(cart, time_limit=5.0):
    """
    Local refinement using Powell's method on the hard objective.
    Works in Cartesian space with projection to triangle.
    """
    start_time = time.time()
    best_cart = cart.copy()
    best_score = -_min_area_bary(_cart_to_bary(cart).flatten())
    
    # Powell's method on Cartesian coordinates
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        cart = x.reshape(N, 2)
        bary = _cart_to_bary(cart)
        bary = _clamp_bary(bary)
        cart2 = _bary_to_cart(bary)
        pi = cart2[_TRIPLES[:, 0]]
        pj = cart2[_TRIPLES[:, 1]]
        pk = cart2[_TRIPLES[:, 2]]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return -np.min(areas)
    
    try:
        result = minimize(obj, cart.flatten(), method='Powell',
                         options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-10})
        if result.fun < best_score:
            best_cart = result.x.reshape(N, 2)
            best_score = result.fun
    except Exception:
        pass
    
    return best_cart, -best_score

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Structured: vertices, midpoints, centroid
    sym = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, H],
        [0.5, H/3],
        [0.25, H/6],
        [0.75, H/6],
        [0.5, 2*H/3],
        [0.125, H/12],
        [0.875, H/12],
        [0.375, H/4],
        [0.625, H/4]
    ])
    configs.append(sym)
    
    # 2. Grid-like
    grid = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, H],
        [0.5, H*0.33],
        [0.25, H*0.17],
        [0.75, H*0.17],
        [0.5, H*0.67],
        [0.15, H*0.08],
        [0.85, H*0.08],
        [0.35, H*0.4],
        [0.65, H*0.4]
    ])
    configs.append(grid)
    
    # 3. Random configs
    rng = np.random.default_rng(42)
    for seed in range(5):
        pts = rng.random((N, 2))
        pts[:, 1] *= H
        bary = _cart_to_bary(pts)
        bary = _clamp_bary(bary)
        pts = _bary_to_cart(bary)
        configs.append(pts)
    
    return configs

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_cart = None
    best_score = -1.0
    
    configs = _generate_initial_configs()
    
    # Phase 1: Global search with differential evolution
    # Use a subset of time for DE
    de_time_limit = 60.0
    
    # Bounds for DE: x in [0, 1], y in [0, H]
    bounds = [(0, 1), (0, H)] * N
    
    try:
        result_de = differential_evolution(
            _objective_de, bounds,
            maxiter=50, popsize=15,
            tol=1e-6, seed=42,
            polish=False,
            workers=1
        )
        if time.time() < deadline:
            cart = result_de.x.reshape(N, 2)
            bary = _cart_to_bary(cart)
            bary = _clamp_bary(bary)
            cart = _bary_to_cart(bary)
            score = -_min_area_bary(bary.flatten())
            if score > best_score:
                best_score = score
                best_cart = cart
    except Exception:
        pass
    
    # Phase 2: Local refinement on best config and other good configs
    if time.time() < deadline - 10:
        # Refine the best from DE
        if best_cart is not None:
            remaining = deadline - time.time() - 5
            cart, score = _local_refine(best_cart, time_limit=min(remaining, 30.0))
            if score > best_score:
                best_score = score
                best_cart = cart
    
    # Phase 3: Try local refinement on initial configs if time allows
    if time.time() < deadline - 5:
        for config in configs:
            if time.time() >= deadline - 5:
                break
            remaining = deadline - time.time() - 3
            if remaining < 2:
                break
            try:
                cart, score = _local_refine(config, time_limit=min(remaining, 10.0))
                if score > best_score:
                    best_score = score
                    best_cart = cart
            except Exception:
                continue
    
    # Final: if we still have time, do another round of local refinement on best
    if best_cart is not None and time.time() < deadline - 3:
        remaining = deadline - time.time() - 2
        if remaining > 1:
            try:
                cart, score = _local_refine(best_cart, time_limit=remaining)
                if score > best_score:
                    best_score = score
                    best_cart = cart
            except Exception:
                pass
    
    if best_cart is None:
        best_cart = configs[0]
    
    return best_cart
