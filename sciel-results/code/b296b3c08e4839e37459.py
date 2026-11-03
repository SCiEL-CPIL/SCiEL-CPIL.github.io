import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import itertools
import math

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area_vec(pts):
    """Vectorized minimum triangle area computation."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    
    # Precompute all index combinations
    indices = list(itertools.combinations(range(N), 3))
    idx_array = np.array(indices, dtype=np.int32)
    p = pts[idx_array]  # (286, 3, 2)
    
    A = p[:, 0, :]
    B = p[:, 1, :]
    C = p[:, 2, :]
    
    AB = B - A
    AC = C - A
    cross = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return float(np.min(areas))

def _hull_area(pts):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Compute the score: min triangle area / hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_tri = _min_triangle_area_vec(pts)
    return min_tri / hull_a

def _negative_score(flat_pts):
    """Negative score for minimization."""
    pts = flat_pts.reshape(N, 2)
    s = _score(pts)
    return -s

def _run_de(start_time, deadline, seed=42):
    """Run Differential Evolution for global search."""
    # Define bounds. Since the problem is scale-invariant, we can restrict to a box.
    # [0, 1] x [0, 1] is a good canonical domain.
    bounds = [(0, 1) for _ in range(2 * N)]
    
    # DE parameters
    # popsize=15 -> 15*26 = 390 evaluations per iteration
    # maxiter=100 -> 39000 evaluations.
    # Each eval is ~286 triangles + hull. Fast enough.
    
    try:
        result = differential_evolution(
            _negative_score,
            bounds,
            seed=seed,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=False, # We will do our own polishing
            init='sobol', # Better initial distribution
            updateinterval=10,
            callback=lambda x, convergence: (time.time() > deadline - 10.0) # Stop if time is running out
        )
        
        best_pts_de = result.x.reshape(N, 2)
        return best_pts_de
        
    except Exception as e:
        print(f"DE failed: {e}")
        return None

def _polish_powell(pts_init, deadline):
    """Polish using Powell's method (derivative-free, robust)."""
    if time.time() > deadline - 5.0:
        return pts_init, _score(pts_init)
    
    flat_init = pts_init.flatten()
    
    try:
        result = minimize(
            _negative_score,
            flat_init,
            method='Powell',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        polished_pts = result.x.reshape(N, 2)
        polished_score = _score(polished_pts)
        return polished_pts, polished_score
    except Exception:
        return pts_init, _score(pts_init)

def _polish_nelder_mead(pts_init, deadline):
    """Polish using Nelder-Mead as a backup/alternative."""
    if time.time() > deadline - 5.0:
        return pts_init, _score(pts_init)
    
    flat_init = pts_init.flatten()
    
    try:
        result = minimize(
            _negative_score,
            flat_init,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        polished_pts = result.x.reshape(N, 2)
        polished_score = _score(polished_pts)
        return polished_pts, polished_score
    except Exception:
        return pts_init, _score(pts_init)

def _initial_guesses():
    """Generate a few diverse initial guesses to seed DE or use as fallbacks."""
    guesses = []
    # Halton
    def _halton_sequence(n, base):
        result = []
        f = 1.0
        for i in range(1, n + 1):
            x = 0.0
            j = i
            while j > 0:
                f /= base
                j, r = divmod(j, base)
                x += f * r
            result.append(x)
        return result

    xs = _halton_sequence(N, 2)
    ys = _halton_sequence(N, 3)
    guesses.append(np.column_stack([xs, ys]))
    
    # Grid
    side = 4 # 4x4 grid has 16 points, take 13
    points = []
    for i in range(side):
        for j in range(side):
            points.append([(i + 0.5) / side, (j + 0.5) / side])
            if len(points) >= N:
                break
        if len(points) >= N:
            break
    guesses.append(np.array(points[:N]))
    
    return guesses

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, use 145s
    deadline = start_time + 145.0
    
    best_pts = None
    best_score = 0.0
    
    # 1. Run Differential Evolution
    de_result = _run_de(start_time, deadline, seed=42)
    if de_result is not None:
        de_score = _score(de_result)
        if de_score > best_score:
            best_score = de_score
            best_pts = de_result
    
    # 2. Polish the best DE result
    if best_pts is not None and time.time() < deadline - 20.0:
        polished_pts, polished_score = _polish_powell(best_pts, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
            
        # Try Nelder-Mead on the Powell result if time allows
        if time.time() < deadline - 10.0:
            nm_pts, nm_score = _polish_nelder_mead(polished_pts, deadline)
            if nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts

    # 3. If DE failed or time remains, try multi-start local from guesses
    if best_pts is None or time.time() < deadline - 30.0:
        guesses = _initial_guesses()
        for guess in guesses:
            if time.time() > deadline - 15.0:
                break
            polished_pts, polished_score = _polish_powell(guess, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts

    if best_pts is not None:
        return best_pts
    
    # Fallback
    return _initial_guesses()[0]

# EVOLVE-BLOCK-END
