import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 14
D = 3

def _pairwise_dists_sq(pts):
    # pts shape (N, D)
    # Return array of squared distances for all pairs i < j
    n = pts.shape[0]
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=-1)
    # Get upper triangle indices
    iu = np.triu_indices(n, 1)
    return dists_sq[iu]

def _objective(pts):
    # pts shape (N, D)
    # Center the points to remove translation invariance
    pts = pts - np.mean(pts, axis=0, keepdims=True)
    
    dists_sq = _pairwise_dists_sq(pts)
    
    # Avoid division by zero
    min_d_sq = np.min(dists_sq)
    max_d_sq = np.max(dists_sq)
    
    if min_d_sq < 1e-12:
        return 1e10
    
    return max_d_sq / min_d_sq

def _make_initial_random(seed):
    rng = np.random.RandomState(seed)
    # Random points in a cube, then center
    pts = rng.uniform(-1, 1, (N, D))
    pts -= np.mean(pts, axis=0, keepdims=True)
    return pts

def _make_initial_sphere(seed):
    rng = np.random.RandomState(seed)
    # Random points on sphere
    vecs = rng.normal(size=(N, D))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    pts = vecs / norms
    # Scale to unit sphere
    return pts

def _make_initial_symmetric(seed):
    # Icosahedron vertices (12) + 2 poles?
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Add 2 more points, e.g., on the z-axis
    # The icosahedron is already fairly symmetric. 
    # Let's just use 14 random points on sphere for the symmetric start to keep it simple, 
    # or perturb the icosahedron + 2 points.
    # Let's use 12 icosahedron points + 2 points at (0,0,1.2) and (0,0,-1.2)
    extra = np.array([[0, 0, 1.2], [0, 0, -1.2]])
    pts = np.vstack([verts, extra])
    pts -= np.mean(pts, axis=0, keepdims=True)
    
    # Add small random perturbation to break symmetry if needed, but Nelder-Mead can handle it
    rng = np.random.RandomState(seed)
    pts += rng.normal(0, 0.01, pts.shape)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_pts = None
    best_score = float('inf')
    
    # Generate a set of initial configurations
    initializers = [
        ("random", _make_initial_random),
        ("sphere", _make_initial_sphere),
        ("symmetric", _make_initial_symmetric)
    ]
    
    seeds = list(range(20)) # 20 seeds per initializer?
    # Total starts: 3 * 20 = 60. 
    # Each Nelder-Mead run on 42 dims might take a few seconds.
    # We need to be careful with the time budget.
    
    # Let's do fewer, higher quality starts.
    # 10 random, 10 sphere, 5 symmetric.
    
    tasks = []
    for i in range(10):
        tasks.append(("random", _make_initial_random, i))
    for i in range(10):
        tasks.append(("sphere", _make_initial_sphere, i))
    for i in range(5):
        tasks.append(("symmetric", _make_initial_symmetric, i))
        
    # Shuffle tasks to mix types
    np.random.shuffle(tasks)
    
    for name, init_func, seed in tasks:
        if time.time() > deadline:
            break
            
        x0 = init_func(seed)
        
        # Use Nelder-Mead
        # options: maxiter, xatol, fatol
        # We want it to converge reasonably fast.
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < best_score:
                # Check if it's valid (distinct points)
                pts = res.x
                pts = pts - np.mean(pts, axis=0, keepdims=True)
                dists_sq = _pairwise_dists_sq(pts)
                if np.min(dists_sq) > 1e-6:
                    score = res.fun
                    if score < best_score:
                        best_score = score
                        best_pts = pts
        except Exception:
            pass
            
    if best_pts is None:
        # Fallback to a simple grid if everything failed or timed out
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        best_pts -= np.mean(best_pts, axis=0, keepdims=True)

    return best_pts
