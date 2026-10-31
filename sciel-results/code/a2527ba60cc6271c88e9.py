"""
Heilbronn triangles: maximize min triangle area for 11 points in unit equilateral triangle.
Strategy: Barycentric coordinate optimization with force-directed relaxation,
simulated annealing, and gradient refinement.
"""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

_SQRT3 = np.sqrt(3.0)
_H = _SQRT3 / 2.0
_EPS = 1e-9

# Precompute all C(11,3) = 165 index triples
_IDX = np.array(list(combinations(range(N), 3)))
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]

# Barycentric to Cartesian: point = b0*(0,0) + b1*(1,0) + b2*(0.5, H)
# So x = b1 + 0.5*b2, y = H*b2
# Inverse: b2 = y/H, b1 = x - 0.5*b2, b0 = 1 - b1 - b2


def _bary_to_cart(bary):
    """Convert barycentric (n,3) to Cartesian (n,2)."""
    b0, b1, b2 = bary[:, 0], bary[:, 1], bary[:, 2]
    x = b1 + 0.5 * b2
    y = _H * b2
    return np.column_stack([x, y])


def _cart_to_bary(cart):
    """Convert Cartesian (n,2) to barycentric (n,3)."""
    x, y = cart[:, 0], cart[:, 1]
    b2 = y / _H
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _clamp_bary(bary):
    """Clamp barycentric coordinates to valid simplex."""
    bary = np.array(bary, dtype=float).copy()
    bary = np.clip(bary, _EPS, None)
    # Normalize to sum to 1
    s = bary.sum(axis=1, keepdims=True)
    bary /= s
    return bary


def _areas_from_bary(bary):
    """Compute all 165 triangle areas from barycentric coords."""
    cart = _bary_to_cart(bary)
    pi = cart[_I]
    pj = cart[_J]
    pk = cart[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _min_area_bary(bary):
    return float(np.min(_areas_from_bary(bary)))


def _soft_min_bary(bary, beta):
    """Smooth approximation of -min(areas) using log-sum-exp."""
    areas = _areas_from_bary(bary)
    neg_beta = -beta * areas
    m = np.max(neg_beta)
    if m > 500:
        neg_beta = neg_beta - m
        m = 0.0
    s = np.sum(np.exp(neg_beta))
    if s < 1e-300:
        return 1e30
    return (m + np.log(s)) / beta


def _objective_soft(params, beta):
    bary = _clamp_bary(params.reshape(N, 3))
    return _soft_min_bary(bary, beta)


def _objective_hard(params):
    bary = _clamp_bary(params.reshape(N, 3))
    return -_min_area_bary(bary)


def _generate_hexagonal_init():
    """Generate a hexagonal-like arrangement in barycentric coords."""
    # Hexagonal grid points in barycentric
    h = _H
    pts_cart = np.array([
        [0.5, h/6],
        [0.25, h/6],
        [0.75, h/6],
        [0.375, h/3],
        [0.625, h/3],
        [0.5, h/2],
        [0.25, h/2],
        [0.75, h/2],
        [0.375, 2*h/3],
        [0.625, 2*h/3],
        [0.5, 5*h/6],
    ])
    return _clamp_bary(_cart_to_bary(pts_cart))


def _generate_structured_init():
    """Generate structured symmetric arrangement."""
    h = _H
    pts_cart = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.167, h/12],
        [0.833, h/12],
        [0.333, h/4],
        [0.667, h/4],
    ])
    return _clamp_bary(_cart_to_bary(pts_cart))


def _generate_perturbed_init(seed):
    """Generate perturbed hexagonal arrangement."""
    rng = np.random.default_rng(seed)
    base = _generate_hexagonal_init()
    bary = base.copy()
    bary += rng.normal(0, 0.05, bary.shape)
    return _clamp_bary(bary)


def _generate_random_init(seed):
    """Generate random barycentric points."""
    rng = np.random.default_rng(seed)
    bary = rng.dirichlet(np.ones(N))
    return bary


def _force_directed_relax(bary, target_dist, n_iter=200, time_limit=10.0):
    """
    Force-directed relaxation: treat points as nodes, want pairwise distances
    to be at least target_dist. Minimize sum of (max(0, target_dist - d))^2.
    """
    bary = _clamp_bary(np.array(bary, dtype=float)).copy()
    start_time = time.time()
    best_bary = bary.copy()
    best_score = _min_area_bary(bary)
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        cart = _bary_to_cart(bary)
        # Compute pairwise distances
        diff = cart[:, np.newaxis, :] - cart[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Compute forces: repulsive for close pairs
        forces = np.zeros_like(cart)
        for i in range(N):
            for j in range(i + 1, N):
                d = dist[i, j]
                if d < target_dist and d > 1e-12:
                    # Repulsive force proportional to (target - d)
                    f = (target_dist - d) / d
                    diff_vec = cart[i] - cart[j]
                    forces[i] += f * diff_vec
                    forces[j] -= f * diff_vec
        
        # Update positions
        step = 0.01 * (1.0 - it / n_iter) + 0.001
        cart_new = cart + step * forces
        bary_new = _cart_to_bary(cart_new)
        bary_new = _clamp_bary(bary_new)
        
        score = _min_area_bary(bary_new)
        if score > best_score:
            best_score = score
            best_bary = bary_new.copy()
        bary = bary_new
    
    return best_bary, best_score


def _simulated_annealing(start_bary, time_limit=10.0, seed=None):
    """Simulated Annealing in barycentric space."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    bary = _clamp_bary(np.array(start_bary, dtype=float))
    current_score = _min_area_bary(bary)
    
    best_bary = bary.copy()
    best_score = current_score
    
    T = 0.02
    T_min = 1e-9
    alpha = 0.9998
    step_size = 0.1
    
    start_time = time.time()
    n_iter = 0
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
        
        n_iter += 1
        frac = n_iter / max(1, int(time_limit * 100))
        
        # Adaptive number of points to move
        if frac < 0.3:
            n_move = int(rng.integers(2, 4))
        elif frac < 0.7:
            n_move = int(rng.integers(1, 3))
        else:
            n_move = 1
        
        idxs = rng.choice(N, size=n_move, replace=False)
        old_bary = bary[idxs].copy()
        
        for idx in idxs:
            noise = rng.normal(0, step_size, 3)
            bary[idx] += noise
        
        bary = _clamp_bary(bary)
        new_score = _min_area_bary(bary)
        
        delta = new_score - current_score
        if delta > 0 or (T > T_min and rng.random() < np.exp(delta / T)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_bary = bary.copy()
        else:
            bary[idxs] = old_bary
            bary = _clamp_bary(bary)
        
        T *= alpha
        step_size *= 0.9999
        
    return best_bary, best_score


def _gradient_refine(bary, beta_start=50.0, beta_end=5000.0, n_steps=20, time_limit=10.0):
    """Annealed gradient refinement in barycentric space."""
    x0 = _clamp_bary(np.array(bary, dtype=float)).flatten()
    start_time = time.time()
    best_x = x0.copy()
    best_score = _min_area_bary(best_x.reshape(N, 3))
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        frac = step / max(1, n_steps - 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        
        try:
            res = minimize(
                _objective_soft,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 10000},
            )
            cand = _clamp_bary(res.x.reshape(N, 3))
            s = _min_area_bary(cand)
            if s > best_score:
                best_score = s
                best_x = cand.flatten()
                x0 = best_x.copy()
        except Exception:
            break
    
    return _clamp_bary(best_x.reshape(N, 3)), best_score


def _nelder_mead_polish(bary, time_limit=5.0):
    """Nelder-Mead refinement in barycentric space."""
    x0 = _clamp_bary(np.array(bary, dtype=float)).flatten()
    start_time = time.time()
    best_x = x0.copy()
    best_score = _min_area_bary(best_x.reshape(N, 3))
    
    try:
        res = minimize(
            _objective_hard,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'maxfev': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                     'adaptive': True},
        )
        if time.time() - start_time < time_limit:
            cand = _clamp_bary(res.x.reshape(N, 3))
            s = _min_area_bary(cand)
            if s > best_score:
                best_score = s
                best_x = cand
    except Exception:
        pass
    
    return _clamp_bary(best_x.reshape(N, 3)), best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_bary = None
    best_score = -1.0
    
    def _update(bary, score):
        nonlocal best_bary, best_score
        if score > best_score:
            best_score = score
            best_bary = np.array(bary, dtype=float).copy()
    
    # Generate diverse seeds
    seeds = []
    seeds.append(_generate_hexagonal_init())
    seeds.append(_generate_structured_init())
    
    for i in range(8):
        seeds.append(_generate_perturbed_init(1000 + i * 7))
    for i in range(5):
        seeds.append(_generate_random_init(2000 + i * 13))
    
    # Phase 1: Force-directed relaxation on each seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 40:
            break
        remaining = deadline - 40 - time.time()
        per = min(3.0, remaining / max(1, len(seeds) - i))
        if per < 0.5:
            break
        try:
            fd_bary, fd_score = _force_directed_relax(seed, target_dist=0.25, time_limit=per)
            _update(fd_bary, fd_score)
        except Exception:
            continue
    
    if best_bary is None:
        best_bary = seeds[0]
        best_score = _min_area_bary(best_bary)
    
    # Phase 2: SA on best result
    if time.time() < deadline - 30:
        remaining = deadline - 30 - time.time()
        if remaining > 3:
            try:
                sa_bary, sa_score = _simulated_annealing(best_bary, time_limit=min(15.0, remaining), seed=42)
                _update(sa_bary, sa_score)
            except Exception:
                pass
    
    # Phase 3: Gradient refinement
    if time.time() < deadline - 20:
        remaining = deadline - 20 - time.time()
        if remaining > 3:
            try:
                grad_bary, grad_score = _gradient_refine(
                    best_bary, beta_start=30, beta_end=5000,
                    n_steps=25, time_limit=min(15.0, remaining)
                )
                _update(grad_bary, grad_score)
            except Exception:
                pass
    
    # Phase 4: Nelder-Mead polish
    if time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        if remaining > 2:
            try:
                nm_bary, nm_score = _nelder_mead_polish(best_bary, time_limit=min(5.0, remaining))
                _update(nm_bary, nm_score)
            except Exception:
                pass
    
    # Phase 5: Another SA round with different seed
    if time.time() < deadline - 8:
        remaining = deadline - 8 - time.time()
        if remaining > 3:
            try:
                sa2_bary, sa2_score = _simulated_annealing(best_bary, time_limit=min(8.0, remaining), seed=9999)
                _update(sa2_bary, sa2_score)
            except Exception:
                pass
    
    # Phase 6: Final gradient polish with high beta
    if time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 2:
            try:
                final_bary, final_score = _gradient_refine(
                    best_bary, beta_start=500, beta_end=15000,
                    n_steps=15, time_limit=min(8.0, remaining)
                )
                _update(final_bary, final_score)
            except Exception:
                pass
    
    # Phase 7: Final Nelder-Mead
    if time.time() < deadline - 2:
        remaining = deadline - 2 - time.time()
        if remaining > 1:
            try:
                nm2_bary, nm2_score = _nelder_mead_polish(best_bary, time_limit=min(3.0, remaining))
                _update(nm2_bary, nm2_score)
            except Exception:
                pass
    
    if best_bary is None:
        best_bary = seeds[0]
    
    # Convert to Cartesian and return
    cart = _bary_to_cart(best_bary)
    # Final clamp to ensure validity
    cart = np.clip(cart, _EPS, None)
    cart[:, 1] = np.clip(cart[:, 1], _EPS, _H - _EPS)
    x_min = cart[:, 1] / _SQRT3 + _EPS
    x_max = 1.0 - cart[:, 1] / _SQRT3 - _EPS
    cart[:, 0] = np.clip(cart[:, 0], x_min, x_max)
    
    return cart

# EVOLVE-BLOCK-END
