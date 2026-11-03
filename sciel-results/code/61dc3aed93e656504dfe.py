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

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _objective_bary(x):
    """Objective in barycentric coordinates: negative min triangle area."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _generate_structured_inits(seed=42):
    """Generate diverse structured initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    inits = []
    
    # Pattern A: Symmetric grid (vertices, midpoints, centroid, etc.)
    pA = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3],
        [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]
    ])
    inits.append(pA.copy())
    
    # Pattern B: Hexagonal-ish
    pB = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [0.33, 0.33],
        [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
    ])
    inits.append(pB.copy())
    
    # Pattern C: Edge thirds + interior
    pC = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [1/3, 0.0], [2/3, 0.0],
        [0.0, 1/3], [0.0, 2/3],
        [1/3, 1/3], [2/3, 1/3],
        [1/6, 1/2], [5/6, 1/2]
    ])
    inits.append(pC.copy())
    
    # Pattern D: Random spread
    pD = np.zeros((N, 2))
    pD[0] = [0.0, 0.0]
    pD[1] = [1.0, 0.0]
    pD[2] = [0.0, 1.0]
    for i in range(3, N):
        u = rng.uniform(0.05, 0.95)
        v = rng.uniform(0.05, 0.95 - u)
        pD[i] = [u, v]
    inits.append(pD)
    
    # Add jitter to all
    for i in range(len(inits)):
        jitter = rng.normal(0, 0.02, size=(N, 2))
        inits[i] = _clamp_bary(inits[i] + jitter)
        
    return inits

def _nelder_mead_refine(bary0, maxiter=300):
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_bary = None
    best_score = -np.inf
    
    def time_left():
        return deadline - time.time()
        
    def score_bary(bary):
        points = _bary_to_cart(bary)
        return _min_triangle_area_vectorized(points)
        
    # Initialize with structured starts
    inits = _generate_structured_inits(seed=42)
    
    # Initialize best with the best initial
    for init in inits:
        s = score_bary(init)
        if s > best_score:
            best_score = s
            best_bary = init.copy()
            
    # Main loop: Alternate between Global Search (DE) and Local Refinement (NM)
    # We use DE to jump to new basins and NM to polish.
    
    # Phase 1: Global search with Differential Evolution
    # DE is good at finding the general region of the optimum.
    # We run multiple short DE instances with different seeds.
    
    de_seeds = [42, 123, 456, 789, 101, 202]
    
    for seed in de_seeds:
        if time_left() < 10.0:
            break
            
        try:
            # Bounds for barycentric coords: u in [0,1], v in [0,1]
            # Note: u+v<=1 is handled by clamping in the objective, 
            # but providing tighter bounds helps DE.
            # However, standard DE doesn't handle inequality constraints u+v<=1 natively.
            # We rely on the clamping in _objective_bary.
            bounds = [(0, 1)] * (2 * N)
            
            # Run DE with a modest population and iterations to stay fast
            de_result = differential_evolution(
                _objective_bary,
                bounds,
                seed=seed,
                maxiter=30,  # Keep it short
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,  # We do our own polishing with NM
                updating='deferred',
                init='sobol' if hasattr(differential_evolution, 'init') else 'latinhypercube'
            )
            
            if de_result.success or de_result.fun < best_score:
                bary_de = _clamp_bary(de_result.x.reshape(N, 2))
                score_de = score_bary(bary_de)
                
                if score_de > best_score:
                    best_score = score_de
                    best_bary = bary_de.copy()
                    
        except Exception:
            continue
            
        # Phase 2: Local Refinement with Nelder-Mead on the current best
        # NM is very effective for polishing non-smooth max-min problems.
        if time_left() > 5.0:
            bary_refined = _nelder_mead_refine(best_bary, maxiter=200)
            score_refined = score_bary(bary_refined)
            
            if score_refined > best_score:
                best_score = score_refined
                best_bary = bary_refined.copy()
                
    # Phase 3: Multi-start Local Refinement from Perturbed Best
    # If DE got stuck, try jumping out with random perturbations and refining.
    rng = np.random.RandomState(1000)
    restart_count = 0
    max_restarts = 50
    
    while restart_count < max_restarts and time_left() > 5.0:
        restart_count += 1
        
        # Perturb the best solution significantly
        # Scale perturbation based on time left? No, keep fixed but vary.
        scale = 0.05 + 0.05 * rng.rand()
        pert = rng.normal(0, scale, size=(N, 2))
        bary_start = _clamp_bary(best_bary + pert)
        
        # Refine with NM
        bary_opt = _nelder_mead_refine(bary_start, maxiter=150)
        score_opt = score_bary(bary_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_bary = bary_opt.copy()
            
    # Final Polish
    if time_left() > 2.0:
        bary_final = _nelder_mead_refine(best_bary, maxiter=300)
        score_final = score_bary(bary_final)
        if score_final > best_score:
            best_score = score_final
            best_bary = bary_final.copy()
            
    # Fallback
    if best_bary is None:
        best_bary = inits[0]
        
    return _bary_to_cart(best_bary)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
