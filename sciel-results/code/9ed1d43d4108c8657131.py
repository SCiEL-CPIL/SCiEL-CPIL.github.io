"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _pairwise_dists(pts):
    """Compute all pairwise squared distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    # Get upper triangle (exclude diagonal)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def _objective(params):
    """Objective: (max_d / min_d)^2 using smooth surrogate."""
    pts = params.reshape(N, D)
    sq_dists = _pairwise_dists(pts)
    # Smooth max and min using log-sum-exp
    # For squared distances, we want (max/min)^2
    # Let's work with distances directly for stability
    dists = np.sqrt(np.maximum(sq_dists, 1e-15))
    # Soft max via log-sum-exp
    beta = 20.0
    log_max = np.log(np.sum(np.exp(beta * dists))) / beta
    # Soft min: -log-sum-exp of -dists
    log_min = -np.log(np.sum(np.exp(-beta * dists))) / beta
    ratio_sq = (np.exp(log_max) / np.exp(log_min)) ** 2
    return ratio_sq

def _exact_score(pts):
    """Exact score: (max/min)^2"""
    sq_dists = _pairwise_dists(pts)
    min_d = np.min(sq_dists)
    max_d = np.max(sq_dists)
    if min_d < 1e-15:
        return np.inf
    return (max_d / min_d) ** 2

def _generate_initial_configs():
    """Generate multiple symmetric initial configurations."""
    configs = []
    
    # Config 1: Cube vertices + face centers + 2 extra
    # Cube: 8 vertices
    cube = np.array([[s, t, u] for s in [-1, 1] for t in [-1, 1] for u in [-1, 1]], dtype=float)
    # Scale to unit sphere
    cube = cube / np.linalg.norm(cube, axis=1, keepdims=True)
    # Add 6 face centers
    faces = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)
    pts1 = np.vstack([cube, faces])
    configs.append(pts1 / np.linalg.norm(pts1, axis=1, keepdims=True))
    
    # Config 2: Icosahedron (12 vertices) + 2 more
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 2 points opposite to minimize max distance
    extra = np.array([[1, 0, 0], [-1, 0, 0]], dtype=float)
    pts2 = np.vstack([ico, extra])
    configs.append(pts2 / np.linalg.norm(pts2, axis=1, keepdims=True))
    
    # Config 3: Two concentric squares rotated 45 degrees
    sq1 = np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]], dtype=float)
    sq2 = np.array([[1,-1,0],[-1,-1,0],[-1,1,0],[1,1,0]], dtype=float) * 0.7
    sq3 = np.array([[1,1,1],[-1,1,1],[-1,-1,1],[1,-1,1]], dtype=float) * 0.7
    sq4 = np.array([[1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,-1]], dtype=float) * 0.7
    pts3 = np.vstack([sq1, sq2, sq3, sq4])
    configs.append(pts3 / np.linalg.norm(pts3, axis=1, keepdims=True))
    
    # Config 4: Random on sphere
    np.random.seed(42)
    pts4 = np.random.randn(N, D)
    pts4 = pts4 / np.linalg.norm(pts4, axis=1, keepdims=True)
    configs.append(pts4)
    
    # Config 5: Cube + 6 more spread out
    cube5 = np.array([[s, t, u] for s in [-1, 1] for t in [-1, 1] for u in [-1, 1]], dtype=float)
    edge_mid = np.array([
        [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
        [1,1,0]/np.sqrt(2),[-1,1,0]/np.sqrt(2)
    ], dtype=float)
    pts5 = np.vstack([cube5, edge_mid])
    configs.append(pts5 / np.linalg.norm(pts5, axis=1, keepdims=True))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    best_pts = None
    best_score = np.inf
    
    configs = _generate_initial_configs()
    
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        init_params = init_pts.flatten()
        
        # Try multiple optimization methods and settings
        for method, options in [
            ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-10}),
            ('Nelder-Mead', {'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}),
        ]:
            if time.time() > deadline - 20:
                break
            
            try:
                result = minimize(
                    _objective,
                    init_params,
                    method=method,
                    options=options
                )
                if result.success or result.fun < best_score:
                    cand_pts = result.x.reshape(N, D)
                    # Check distinctness
                    sq_dists = _pairwise_dists(cand_pts)
                    if np.min(sq_dists) > 1e-10:
                        score = _exact_score(cand_pts)
                        if score < best_score:
                            best_score = score
                            best_pts = cand_pts
            except Exception:
                pass
        
        # Also try from the initial config directly
        sq_dists = _pairwise_dists(init_pts)
        if np.min(sq_dists) > 1e-10:
            score = _exact_score(init_pts)
            if score < best_score:
                best_score = score
                best_pts = init_pts
    
    # If we still have time, do a final polish on the best
    if best_pts is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                _objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12}
            )
            cand_pts = result.x.reshape(N, D)
            sq_dists = _pairwise_dists(cand_pts)
            if np.min(sq_dists) > 1e-10:
                score = _exact_score(cand_pts)
                if score < best_score:
                    best_score = score
                    best_pts = cand_pts
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return a reasonable configuration
        best_pts = configs[0]
    
    return best_pts
# EVOLVE-BLOCK-END
