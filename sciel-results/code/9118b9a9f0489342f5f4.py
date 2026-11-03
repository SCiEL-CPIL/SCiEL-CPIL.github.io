"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the unit-base equilateral triangle

# Precompute indices for all triples to speed up area calculation
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _clamp_cart(points):
    """Clamp Cartesian points to be inside the triangle."""
    bary = _cart_to_bary(points)
    bary = _clamp_bary(bary)
    return _bary_to_cart(bary)

def _objective_bary(x):
    """Objective in barycentric coordinates: negative min triangle area.
    Barycentric coords (u,v) with u>=0, v>=0, u+v<=1 naturally constrain to triangle."""
    bary = x.reshape(N, 2)
    # Clamp to valid region
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _structured_init_bary(seed=42):
    """Generate diverse structured initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    
    # Pattern 1: Vertices + edge midpoints + centroid + interior
    patterns = []
    
    # Pattern A: Symmetric grid
    pA = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3],
        [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]
    ])
    patterns.append(pA)
    
    # Pattern B: Hexagonal-ish
    pB = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [0.33, 0.33],
        [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
    ])
    patterns.append(pB)
    
    # Pattern C: More spread out
    pC = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
        [0.33, 0.33],
        [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
    ])
    patterns.append(pC)
    
    # Pattern D: Random-ish structured
    pD = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [rng.uniform(0.3, 0.7), 0.0],
        [rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)],
        [0.0, rng.uniform(0.3, 0.7)],
        [1/3, 1/3],
        [rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)],
        [rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.3)],
        [rng.uniform(0.1, 0.3), rng.uniform(0.7, 0.9)],
        [rng.uniform(0.3, 0.7), rng.uniform(0.2, 0.5)]
    ])
    patterns.append(pD)
    
    # Choose a pattern and add small jitter
    idx = seed % len(patterns)
    bary = patterns[idx].copy()
    
    # Add small jitter
    jitter = rng.normal(0, 0.02, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    
    return bary

def _l_bfgs_optimize_bary(bary0, maxiter=500):
    """Run L-BFGS-B optimization in barycentric coordinates."""
    try:
        x0 = bary0.flatten()
        
        # Bounds: u in [0,1], v in [0,1] (u+v<=1 handled by clamping)
        bounds = [(0, 1)] * (2 * N)
        
        res = minimize(
            _objective_bary, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _nelder_mead_optimize_bary(bary0, maxiter=200):
    """Run Nelder-Mead optimization in barycentric coordinates."""
    try:
        x0 = bary0.flatten()
        res = minimize(
            _objective_bary, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _differential_evolution_global(seed=42, maxiter=50, popsize=15):
    """Run differential evolution for global optimization."""
    try:
        rng = np.random.RandomState(seed)
        
        # Bounds for barycentric coords
        bounds = [(0, 1)] * (2 * N)
        
        # Use a smaller population and fewer iterations for speed
        de_result = differential_evolution(
            _objective_bary,
            bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            updating='deferred'
        )
        
        bary_opt = _clamp_bary(de_result.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return None

def _simulated_annealing(bary0, max_iters=5000, t_start=0.1, t_end=0.001, seed=42):
    """Simulated annealing in barycentric coordinates with point-move neighborhood."""
    rng = np.random.RandomState(seed)
    bary = bary0.copy()
    points = _bary_to_cart(bary)
    current_score = _min_triangle_area_vectorized(points)
    best_bary = bary.copy()
    best_score = current_score
    
    n_triples = len(_get_indices())
    
    for it in range(max_iters):
        # Cooling schedule
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Choose a random point to move
        idx = rng.randint(N)
        
        # Generate a new position for that point
        new_bary = bary.copy()
        new_bary[idx] = bary[idx] + rng.normal(0, 0.05, size=2)
        new_bary = _clamp_bary(new_bary)
        
        new_points = _bary_to_cart(new_bary)
        new_score = _min_triangle_area_vectorized(new_points)
        
        # Accept if better or with probability exp((new - current) / t)
        if new_score >= current_score:
            bary = new_bary
            points = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_bary = bary.copy()
        else:
            delta = new_score - current_score
            if delta > 0:
                bary = new_bary
                points = new_points
                current_score = new_score
            else:
                accept_prob = np.exp(delta / t)
                if rng.random() < accept_prob:
                    bary = new_bary
                    points = new_points
                    current_score = new_score
    
    return best_bary, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_bary = None
    best_score = -np.inf
    
    def time_left():
        return deadline - time.time()
    
    def score_bary(bary):
        points = _bary_to_cart(bary)
        return _min_triangle_area_vectorized(points)
    
    # Phase 1: Try multiple structured initializations with local optimization
    initial_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in initial_seeds:
        if time_left() < 3.0:
            break
        
        bary_init = _structured_init_bary(seed=seed)
        score_init = score_bary(bary_init)
        
        # Local optimization with L-BFGS-B (faster than Nelder-Mead)
        if time_left() > 2.0:
            bary_opt = _l_bfgs_optimize_bary(bary_init, maxiter=300)
            score_opt = score_bary(bary_opt)
            if score_opt > score_init:
                score_init = score_opt
                bary_init = bary_opt
        
        if score_init > best_score:
            best_score = score_init
            best_bary = bary_init.copy()
    
    # Phase 2: Simulated annealing from best solution
    if time_left() > 10.0:
        sa_bary, sa_score = _simulated_annealing(
            best_bary, max_iters=3000, t_start=0.05, t_end=0.0005, seed=42
        )
        if sa_score > best_score:
            best_score = sa_score
            best_bary = sa_bary.copy()
            
            # Refine with local optimization
            if time_left() > 5.0:
                bary_refined = _l_bfgs_optimize_bary(best_bary, maxiter=500)
                score_refined = score_bary(bary_refined)
                if score_refined > best_score:
                    best_score = score_refined
                    best_bary = bary_refined.copy()
    
    # Phase 3: More simulated annealing restarts
    rng = np.random.RandomState(123)
    sa_restarts = 0
    max_sa_restarts = 10
    
    while sa_restarts < max_sa_restarts and time_left() > 5.0:
        sa_restarts += 1
        
        # Perturb the best solution for SA start
        pert = rng.normal(0, 0.03, size=(N, 2))
        bary_start = _clamp_bary(best_bary + pert)
        
        sa_bary, sa_score = _simulated_annealing(
            bary_start, max_iters=2000, t_start=0.03, t_end=0.0003, seed=42+sa_restarts
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_bary = sa_bary.copy()
            
            # Quick local refinement
            if time_left() > 3.0:
                bary_refined = _l_bfgs_optimize_bary(best_bary, maxiter=300)
                score_refined = score_bary(bary_refined)
                if score_refined > best_score:
                    best_score = score_refined
                    best_bary = bary_refined.copy()
    
    # Phase 4: Fine-tuning with very small perturbations
    if time_left() > 2.0:
        for ft in range(10):
            if time_left() < 1.0:
                break
            pert = rng.normal(0, 0.005, size=(N, 2))
            bary_ft = _clamp_bary(best_bary + pert)
            bary_opt = _l_bfgs_optimize_bary(bary_ft, maxiter=300)
            score_opt = score_bary(bary_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
    
    # Final fallback
    if best_bary is None:
        best_bary = _structured_init_bary(seed=42)
    
    return _bary_to_cart(best_bary)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
