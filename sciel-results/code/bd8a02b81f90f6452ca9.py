import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from scipy.optimize import cma

N = 14
N_HALF = 7
DEADLINE = 138.0  # Safety margin before 150s limit

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()
N_TRIS = len(IDX_I)

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _objective(flat_points):
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    return -_score(pts)

def _objective_soft(flat_points, beta=100.0):
    """Soft-min objective: differentiable approximation of hull_area/min_area."""
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _objective_soft_multi_beta(flat_points, betas=None):
    """Multi-scale soft-min: average over multiple beta values for robust landscape."""
    if betas is None:
        betas = [30.0, 80.0, 200.0, 500.0]
    vals = []
    for b in betas:
        vals.append(_objective_soft(flat_points, beta=b))
    return np.mean(vals)

# --- Centrally symmetric parameterization ---
# For N=14, we use 7 independent vertex pairs (v_i, -v_i).
# Parameterize by 7 angles and 7 radii: v_i = r_i * (cos(theta_i), sin(theta_i))

def _sym_params_to_points(params):
    """Generate 14 points from 7 angles and 7 radii (centrally symmetric)."""
    angles = params[:7]
    radii = params[7:]
    radii = np.maximum(radii, 1e-8)
    half = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return np.vstack([half, -half])

def _objective_sym(params):
    """Hard objective for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _sym_params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    return -_score(points)

def _objective_sym_soft(params, beta=200.0):
    """Soft objective for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _sym_params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _objective_sym_soft_multi(params, betas=None):
    """Multi-scale soft objective for symmetric params."""
    if betas is None:
        betas = [80.0, 200.0, 500.0]
    vals = []
    for b in betas:
        vals.append(_objective_sym_soft(params, beta=b))
    return np.mean(vals)

def _generate_lattice_inits():
    """Generate initial points based on hexagonal lattice subsets."""
    inits = []
    # Hexagonal lattice points
    # We want 14 points. A regular hexagon of side 2 has 19 points.
    # Let's pick a subset that is roughly convex.
    # Or simply generate a grid and jitter.
    
    # Strategy: Start with a regular 14-gon but perturb it towards lattice-like spacing.
    # A better approach: Use a 4x4 grid, pick 14 points that form a convex shape, then deform.
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(pts)
    
    # 2. Hexagonal lattice subset
    # Generate points in hex lattice
    pts_hex = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = np.sqrt(3)/2 * j
            pts_hex.append([x, y])
    pts_hex = np.array(pts_hex)
    # Normalize to unit radius roughly
    max_r = np.max(np.linalg.norm(pts_hex, axis=1))
    pts_hex = pts_hex / max_r
    
    # Select 14 points that are closest to a circle of radius 0.8
    # Sort by distance from origin
    dists = np.linalg.norm(pts_hex, axis=1)
    # We want a mix of inner and outer to ensure convexity and spread
    # Let's just take the 14 points with distances in [0.5, 1.0]
    mask = (dists > 0.5) & (dists < 1.0)
    candidates = pts_hex[mask]
    if len(candidates) >= 14:
        # Sort by angle to ensure convex order
        angles_c = np.arctan2(candidates[:, 1], candidates[:, 0])
        idx_sorted = np.argsort(angles_c)
        pts_sel = candidates[idx_sorted[:14]]
        inits.append(pts_sel)
        
    # 3. Jittered lattice
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts_base = inits[0].copy()
        pts_jitter = pts_base + rng.normal(0, 0.1, size=(N, 2))
        inits.append(pts_jitter)
        
    return inits

def _normalize_points(pts):
    """Center and normalize points."""
    center = np.mean(pts, axis=0)
    pts = pts - center
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm
    return pts

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def remaining_time():
        return DEADLINE - (time.time() - start_time)

    # 1. CMA-ES on Symmetric Space
    # This is often the most effective method for this kind of problem.
    if remaining_time() > 40.0:
        # Initial guess: Regular heptagon based
        base_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        x0_sym = np.concatenate([base_angles, np.ones(7)])
        
        # CMA-ES settings
        opts = {
            'maxiter': 100,
            'seed': 42,
            'verbose': -9, # Suppress output
            'bounds': [(-np.pi, np.pi)] * 7 + [(0.1, 2.0)] * 7,
            'tolfun': 1e-9,
            'tolfun2': 1e-9
        }
        
        try:
            es = cma.CMAEvolutionStrategy(x0_sym, 0.5, opts)
            while not es.stop() and remaining_time() > 10.0:
                solutions = es.ask()
                fitnesses = []
                for sol in solutions:
                    pts = _sym_params_to_points(sol)
                    if np.any(~np.isfinite(pts)):
                        fitnesses.append(1e10)
                    else:
                        fitnesses.append(-_score(pts))
                es.tell(solutions, fitnesses)
                
                if es.result.xbest is not None:
                    pts_best = _sym_params_to_points(es.result.xbest)
                    score_best = -es.result.fbest
                    update_best(pts_best, score_best)
        except Exception:
            pass

    # 2. CMA-ES on General Space
    if remaining_time() > 30.0:
        # Use lattice-based initialization
        inits = _generate_lattice_inits()
        x0_gen = _normalize_points(inits[0]).flatten()
        
        # We need to handle bounds or normalization inside the objective if using CMA-ES directly
        # Since CMA-ES works in unbounded space, we can just let it roam and normalize inside objective?
        # No, normalization breaks the gradient/direction. 
        # Better: Constrain to a box or use the fact that score is scale invariant.
        # Score is scale invariant, so we can just let CMA-ES search in R^28.
        
        opts_gen = {
            'maxiter': 80,
            'seed': 123,
            'verbose': -9,
            'tolfun': 1e-9,
            'tolfun2': 1e-9
        }
        
        try:
            es_gen = cma.CMAEvolutionStrategy(x0_gen, 0.5, opts_gen)
            while not es_gen.stop() and remaining_time() > 10.0:
                solutions = es_gen.ask()
                fitnesses = []
                for sol in solutions:
                    pts = sol.reshape(N, 2)
                    if np.any(~np.isfinite(pts)):
                        fitnesses.append(1e10)
                    else:
                        # Normalize to avoid scaling issues? 
                        # Actually, score is invariant to scaling. 
                        # But if points collapse, hull area -> 0.
                        # Let's just compute score.
                        fitnesses.append(-_score(pts))
                es_gen.tell(solutions, fitnesses)
                
                if es_gen.result.xbest is not None:
                    pts_best = es_gen.result.xbest.reshape(N, 2)
                    score_best = -es_gen.result.fbest
                    update_best(pts_best, score_best)
        except Exception:
            pass

    # 3. Refine best found with Nelder-Mead (local polish)
    if best_points is not None and remaining_time() > 10.0:
        try:
            # Normalize best points for local search stability
            pts_norm = _normalize_points(best_points)
            res = minimize(_objective, pts_norm.flatten(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < 0:
                cand_pts = res.x.reshape(N, 2)
                cand_score = -res.fun
                update_best(cand_pts, cand_score)
        except Exception:
            pass

    # 4. If still time, try a few random restarts with CMA-ES on symmetric space
    if remaining_time() > 15.0:
        rng = np.random.default_rng(999)
        for i in range(3):
            if remaining_time() < 5.0:
                break
            # Random symmetric init
            angles = rng.uniform(0, 2*np.pi, 7)
            radii = rng.uniform(0.5, 1.5, 7)
            x0_rand = np.concatenate([angles, radii])
            
            opts_rand = {
                'maxiter': 30,
                'seed': 42 + i,
                'verbose': -9,
                'bounds': [(-np.pi, np.pi)] * 7 + [(0.1, 2.0)] * 7,
                'tolfun': 1e-8
            }
            try:
                es_rand = cma.CMAEvolutionStrategy(x0_rand, 0.3, opts_rand)
                while not es_rand.stop() and remaining_time() > 5.0:
                    solutions = es_rand.ask()
                    fitnesses = []
                    for sol in solutions:
                        pts = _sym_params_to_points(sol)
                        if np.any(~np.isfinite(pts)):
                            fitnesses.append(1e10)
                        else:
                            fitnesses.append(-_score(pts))
                    es_rand.tell(solutions, fitnesses)
                    
                    if es_rand.result.xbest is not None:
                        pts_best = _sym_params_to_points(es_rand.result.xbest)
                        score_best = -es_rand.result.fbest
                        update_best(pts_best, score_best)
            except Exception:
                pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
