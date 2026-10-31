"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

def _pairwise_dists_sq(pts):
    """Compute all pairwise squared distances."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    iu = np.triu_indices(n, 1)
    return dists_sq[iu]

def _score(pts):
    """Compute (max/min)^2."""
    d_sq = _pairwise_dists_sq(pts)
    d_sq = np.maximum(d_sq, 1e-12)
    return (np.max(d_sq) / np.min(d_sq))

def _soft_objective(pts, beta):
    """
    Smooth surrogate for the ratio of max to min squared distances.
    Uses log-sum-exp for soft max and soft min.
    Objective: soft_max(d_sq) / soft_min(d_sq)
    """
    d_sq = _pairwise_dists_sq(pts)
    d_sq = np.maximum(d_sq, 1e-12)
    
    # Soft max: (1/beta) * log(sum(exp(beta * x)))
    # To avoid overflow, subtract max
    max_d_sq = np.max(d_sq)
    soft_max_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
    
    # Soft min: - (1/beta) * log(sum(exp(-beta * x)))
    min_d_sq = np.min(d_sq)
    soft_min_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
    
    if soft_min_sq < 1e-10:
        return 1e10
        
    return soft_max_sq / soft_min_sq

def _init_hexagonal_cluster():
    """
    Create a hexagonal cluster of 16 points.
    Center + 6 in first ring + 9 in second ring? 
    Standard hex packing: 1, 7, 19...
    16 points: 1 center, 6 ring 1, 9 from ring 2?
    Or just take the 16 closest points in a hex grid.
    """
    pts = []
    # Generate a large hex grid
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    
    # Calculate distances to center and pick the 16 closest
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:N]
    return pts[idx]

def _init_rings():
    """Two rings: 8 inner, 8 outer."""
    inner_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    outer_angles = inner_angles + np.pi/8
    r_in = 1.0
    r_out = 1.8 # Adjusted for hex-like spacing
    inner = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    outer = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    return np.vstack([inner, outer])

def _init_grid():
    """4x4 grid, slightly perturbed to break symmetry if needed."""
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return pts

def _init_random(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(-2, 2, size=(N, D))

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_pts = None
    best_score = np.inf
    
    # Initial candidates
    candidates = [
        _init_hexagonal_cluster(),
        _init_rings(),
        _init_grid(),
        _init_random(1),
        _init_random(2),
        _init_random(3)
    ]
    
    # Score initial candidates
    for pts in candidates:
        sc = _score(pts)
        if sc < best_score:
            best_score = sc
            best_pts = pts.copy()
            
    # Optimization Loop
    # We will use Differential Evolution for global search on a subset of points
    # and L-BFGS-B for local polishing.
    
    # Since DE is stochastic and can be slow, we run it on the best few candidates
    # or just one good global search.
    
    # 1. Global Search with Differential Evolution
    # Bound the box. The optimal scale is arbitrary, so we bound to a reasonable range.
    bounds = [(-5, 5)] * (N * D)
    
    if time.time() < deadline - 30:
        try:
            # DE on the soft objective to make it smooth for the gradient-free DE? 
            # DE doesn't use gradients, so we can use the exact objective or soft.
            # Exact objective is fine for DE as it's just function evaluation.
            # However, the exact objective is non-smooth. DE handles this okay.
            # Let's use the soft objective with a moderate beta to guide DE towards smooth regions.
            beta_de = 10.0
            res_de = differential_evolution(
                lambda p: _soft_objective(p.reshape(N, D), beta_de),
                bounds=bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
                polish=False, # We will polish manually
                workers=1
            )
            pts_de = res_de.x.reshape(N, D)
            sc_de = _score(pts_de)
            if sc_de < best_score:
                best_score = sc_de
                best_pts = pts_de.copy()
        except Exception:
            pass

    # 2. Local Polishing with L-BFGS-B on Soft Objective
    # We polish the best points found so far, and also the initial candidates.
    
    def polish(pts, beta, method='L-BFGS-B'):
        if time.time() > deadline - 5:
            return pts, _score(pts)
        
        # Flatten
        x0 = pts.flatten()
        
        # Use soft objective for gradient-based methods
        def obj_func(x):
            return _soft_objective(x.reshape(N, D), beta)
            
        try:
            res = minimize(obj_func, x0, method=method, options={'maxiter': 500, 'ftol': 1e-12})
            pts_opt = res.x.reshape(N, D)
            sc_opt = _score(pts_opt)
            return pts_opt, sc_opt
        except Exception:
            return pts, _score(pts)
        
    # Polish the global best
    if best_pts is not None:
        # Try different betas for polishing
        for beta in [5.0, 20.0, 50.0]:
            pts_polished, sc_polished = polish(best_pts, beta)
            if sc_polished < best_score:
                best_score = sc_polished
                best_pts = pts_polished.copy()
                
            # Also try Powell on the exact objective from the polished point
            if time.time() < deadline - 5:
                try:
                    res_powell = minimize(
                        lambda x: _score(x.reshape(N, D)), 
                        pts_polished.flatten(), 
                        method='Powell', 
                        options={'maxiter': 200, 'maxfev': 2000}
                    )
                    pts_powell = res_powell.x.reshape(N, D)
                    sc_powell = _score(pts_powell)
                    if sc_powell < best_score:
                        best_score = sc_powell
                        best_pts = pts_powell.copy()
                except Exception:
                    pass

    # 3. If time permits, run a short DE from the best point (local DE?) or just more polishing.
    # Let's try a few random restarts of the local polish from the best point.
    if best_pts is not None and time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        for _ in range(5):
            # Perturb the best point slightly
            noise = rng.normal(0, 0.05, size=(N, D))
            pts_perturbed = best_pts + noise
            pts_polished, sc_polished = polish(pts_perturbed, 20.0)
            if sc_polished < best_score:
                best_score = sc_polished
                best_pts = pts_polished.copy()
                
            if time.time() > deadline - 5:
                break

    # Final check for distinctness
    if best_pts is None:
        best_pts = _init_hexagonal_cluster()
        
    d_check = pdist(best_pts, 'euclidean')
    if np.min(d_check) < 1e-8:
        # Fallback to a valid configuration
        best_pts = _init_hexagonal_cluster()

    return best_pts
# EVOLVE-BLOCK-END
