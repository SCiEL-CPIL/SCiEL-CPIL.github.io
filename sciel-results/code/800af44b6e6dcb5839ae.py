"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.
Uses soft-min differentiable surrogate with L-BFGS-B, differential evolution global search,
and adaptive beta annealing for robust convergence."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    x = bary[:, 1] * V1[0] + bary[:, 2] * V2[0]
    y = bary[:, 1] * V1[1] + bary[:, 2] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert Cartesian (N, 2) to barycentric (N, 3)."""
    x = pts[:, 0]
    y = pts[:, 1]
    b2 = 2.0 * y / np.sqrt(3)
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _clamp_bary(bary):
    """Ensure valid barycentric coordinates."""
    bary = np.clip(bary, 0.0, 1.0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    bary = bary / s
    return bary

def _all_triangle_areas(pts):
    """Compute all triangle areas among all triples."""
    idx = _IDX_TRIPLES
    pi = pts[idx[:, 0]]
    pj = pts[idx[:, 1]]
    pk = pts[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples."""
    return _all_triangle_areas(pts).min()

def _softmin_objective(bary_flat, beta=50.0):
    """Differentiable soft-min objective: negative soft-min of triangle areas."""
    bary = bary_flat.reshape(N, 3)
    bary = _clamp_bary(bary)
    pts = _bary_to_cart(bary)
    areas = _all_triangle_areas(pts)
    # Soft min: -log-sum-exp(-beta * areas) / beta
    # This approximates min(areas) as beta -> infinity
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min = -log_sum / beta
    return -soft_min

def _hard_objective(bary_flat):
    """Hard objective: negative min triangle area."""
    bary = bary_flat.reshape(N, 3)
    bary = _clamp_bary(bary)
    pts = _bary_to_cart(bary)
    return -_min_triangle_area(pts)

def _score_bary(bary_flat):
    """Score a barycentric configuration (higher is better)."""
    bary = bary_flat.reshape(N, 3)
    bary = _clamp_bary(bary)
    pts = _bary_to_cart(bary)
    return _min_triangle_area(pts)

def _generate_structured_init(rng):
    """Generate a structured initial configuration with symmetry."""
    pts = []
    center = (V0 + V1 + V2) / 3.0
    pts.append(center)
    
    # Ring 1: 5 points
    for i in range(5):
        angle = 2 * np.pi * i / 5
        r = 0.15 + rng.uniform(-0.02, 0.02)
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        pts.append([x, y])
    
    # Ring 2: 5 points
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi / 5
        r = 0.30 + rng.uniform(-0.02, 0.02)
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        pts.append([x, y])
    
    pts = np.array(pts)
    bary = _cart_to_bary(pts)
    bary = _clamp_bary(bary)
    return bary.flatten()

def _generate_random_init(rng):
    """Generate a random initial configuration."""
    bary = rng.uniform(0.05, 0.95, size=(N, 3))
    bary = _clamp_bary(bary)
    return bary.flatten()

def _l_bfgs_softmin(bary_flat, beta=50.0, maxiter=200):
    """Optimize using L-BFGS-B with soft-min objective."""
    bounds = [(0.0, 1.0)] * (3 * N)
    try:
        res = minimize(
            _softmin_objective, bary_flat, args=(beta,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x
    except Exception:
        return bary_flat

def _l_bfgs_hard(bary_flat, maxiter=100):
    """Polish using L-BFGS-B with hard objective (Nelder-Mead fallback)."""
    bounds = [(0.0, 1.0)] * (3 * N)
    try:
        res = minimize(
            _hard_objective, bary_flat,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return res.x
    except Exception:
        return bary_flat

def _beta_anneal_refine(bary_flat, betas=(20.0, 40.0, 80.0, 160.0, 320.0)):
    """Refine with increasing beta values."""
    current = bary_flat
    for beta in betas:
        current = _l_bfgs_softmin(current, beta=beta, maxiter=150)
    # Final hard polish
    current = _l_bfgs_hard(current, maxiter=500)
    return current

def _differential_evolution_global(rng, maxiter=300, popsize=15):
    """Global search using differential evolution."""
    bounds = [(0.0, 1.0)] * (3 * N)
    
    def de_objective(x):
        return _hard_objective(x)
    
    try:
        result = differential_evolution(
            de_objective, bounds,
            maxiter=maxiter, popsize=popsize,
            seed=int(rng.randint(0, 2**31)),
            tol=1e-10, atol=1e-12,
            mutation=(0.5, 1.5), recombination=0.9,
            init='latinhypercube'
        )
        return result.x
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    best_bary_flat = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    initial_bary_list = []
    
    # Structured initializations
    n_structured = 5
    for i in range(n_structured):
        if time_left() < 30.0:
            break
        bary = _generate_structured_init(rng)
        initial_bary_list.append(bary)
    
    # Random initializations
    n_random = 8
    for i in range(n_random):
        if time_left() < 25.0:
            break
        bary = _generate_random_init(rng)
        initial_bary_list.append(bary)
    
    # Optimize each initialization with soft-min + annealing
    for start_bary in initial_bary_list:
        if time_left() < 15.0:
            break
        
        try:
            # Soft-min optimization with moderate beta
            refined = _l_bfgs_softmin(start_bary, beta=40.0, maxiter=150)
            score = _score_bary(refined)
            if score > best_score:
                best_score = score
                best_bary_flat = refined.copy()
        except Exception:
            pass
    
    # Polish best solution with beta annealing
    if best_bary_flat is not None and time_left() > 30.0:
        try:
            refined = _beta_anneal_refine(best_bary_flat)
            score = _score_bary(refined)
            if score > best_score:
                best_score = score
                best_bary_flat = refined.copy()
        except Exception:
            pass
    
    # Differential evolution global search
    if time_left() > 40.0:
        try:
            de_result = _differential_evolution_global(rng, maxiter=200, popsize=12)
            if de_result is not None:
                score = _score_bary(de_result)
                if score > best_score:
                    best_score = score
                    best_bary_flat = de_result.copy()
        except Exception:
            pass
    
    # Additional refinement from perturbed best solution
    if best_bary_flat is not None and time_left() > 25.0:
        for i in range(6):
            if time_left() < 12.0:
                break
            try:
                noise = rng.normal(0, 0.005, size=3 * N)
                perturbed = best_bary_flat + noise
                perturbed = _clamp_bary(perturbed.reshape(N, 3)).flatten()
                
                refined = _l_bfgs_softmin(perturbed, beta=60.0, maxiter=100)
                score = _score_bary(refined)
                if score > best_score:
                    best_score = score
                    best_bary_flat = refined.copy()
            except Exception:
                pass
    
    # Final polish
    if best_bary_flat is not None and time_left() > 10.0:
        try:
            refined = _l_bfgs_hard(best_bary_flat, maxiter=300)
            score = _score_bary(refined)
            if score > best_score:
                best_score = score
                best_bary_flat = refined.copy()
        except Exception:
            pass
    
    # Convert best barycentric to Cartesian
    if best_bary_flat is None:
        # Fallback: use structured init
        bary = _generate_structured_init(rng)
    else:
        bary = best_bary_flat.reshape(N, 3)
        bary = _clamp_bary(bary)
    
    pts = _bary_to_cart(bary)
    return np.array(pts)
