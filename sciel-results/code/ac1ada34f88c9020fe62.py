import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
from scipy.special import logsumexp
import time
import itertools

N = 13

def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Vectorized calculation for all pairs of edges from a base point is complex.
    # Given N=13, a triple loop is 286 iterations, which is very fast in Python.
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_true(params):
    """True score: min triangle area / hull area."""
    pts = params.reshape((N, 2))
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_a = _min_triangle_area(pts)
    return min_a / hull_a

def _score_smooth(params, beta=20.0):
    """Smoothed score for global optimization."""
    pts = params.reshape((N, 2))
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    
    areas = np.array([
        0.5 * abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - (pts[j,1]-pts[i,1])*(pts[k,0]-pts[i,0]))
        for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
    ])
    
    if len(areas) == 0:
        return 0.0
        
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # We want to maximize this, so we minimize its negative
    soft_min_val = -1.0 / beta * logsumexp(-beta * areas)
    
    return soft_min_val / hull_a

def _neg_score_true(params):
    return -_score_true(params)

def _neg_score_smooth(params, beta=20.0):
    return -_score_smooth(params, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    best_score = 0.0
    best_pts = None

    rng = np.random.default_rng(42)

    def update_best(pts, score):
        nonlocal best_score, best_pts
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    # 1. Initial structured guesses
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    update_best(pts_reg, _score_true(pts_reg.flatten()))

    # Fermat Spiral
    k = np.arange(N)
    r = np.sqrt(k + 0.5) * 0.6 # Scale to fit roughly in unit circle
    theta = k * 2 * np.pi / np.sqrt(5)
    pts_fermat = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    update_best(pts_fermat, _score_true(pts_fermat.flatten()))

    # Concentric Rings
    # Try different splits
    for inner_count in [3, 5, 6]:
        outer_count = N - inner_count
        if outer_count < 3: continue
        angles_in = np.linspace(0, 2 * np.pi, inner_count, endpoint=False)
        angles_out = np.linspace(0, 2 * np.pi, outer_count, endpoint=False) + np.pi / outer_count
        r_in = 0.4
        r_out = 1.0
        pts_in = np.column_stack([r_in * np.cos(angles_in), r_in * np.sin(angles_in)])
        pts_out = np.column_stack([r_out * np.cos(angles_out), r_out * np.sin(angles_out)])
        pts_ring = np.vstack([pts_in, pts_out])
        update_best(pts_ring, _score_true(pts_ring.flatten()))

    # Random starts
    for _ in range(50):
        pts_rand = rng.uniform(-1, 1, size=(N, 2))
        s = _score_true(pts_rand.flatten())
        update_best(pts_rand, s)

    if time.time() > deadline - 60:
        return best_pts

    # 2. Differential Evolution with Smoothed Objective
    # Use a beta that balances smoothness and accuracy
    beta_de = 20.0
    
    def de_objective(x):
        return _neg_score_smooth(x, beta_de)

    try:
        bounds = [(-2, 2)] * (2 * N)
        
        result = differential_evolution(
            de_objective,
            bounds=bounds,
            seed=42,
            maxiter=80,
            popsize=25,
            tol=1e-5,
            polish=True,
            init='sobol',
            updating='deferred'
        )
        
        if result.fun < -best_score:
            candidate_pts = result.x.reshape((N, 2))
            candidate_score = _score_true(candidate_pts)
            update_best(candidate_pts, candidate_score)
            
    except Exception:
        pass

    if time.time() > deadline - 30:
        return best_pts

    # 3. Local Refinement with Nelder-Mead on True Objective
    # Nelder-Mead is good for non-smooth functions
    if best_pts is not None:
        starts = [best_pts.flatten()]
        
        # Add a few perturbed starts
        for i in range(10):
            noise = rng.normal(0, 0.1, size=(N, 2))
            starts.append((best_pts + noise).flatten())
        
        # Also add the best initial guess if it was different
        # (Already covered by best_pts if it was the best)

        for start in starts:
            if time.time() > deadline - 2:
                break
                
            try:
                res = minimize(
                    _neg_score_true,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                
                if res.fun < -best_score:
                    candidate_pts = res.x.reshape((N, 2))
                    candidate_score = _score_true(candidate_pts)
                    update_best(candidate_pts, candidate_score)
                        
            except Exception:
                continue

    # 4. Final Polish: Simple Random Perturbation / SA-like
    # Sometimes NM gets stuck. Try small random steps.
    if best_pts is not None and time.time() < deadline - 5:
        current = best_pts.copy()
        current_score = best_score
        step_size = 0.05
        for _ in range(2000):
            if time.time() > deadline - 1:
                break
            # Random perturbation
            noise = rng.normal(0, step_size, size=(N, 2))
            candidate = current + noise
            cand_score = _score_true(candidate.flatten())
            if cand_score > current_score:
                current = candidate
                current_score = cand_score
                # Slightly increase step size if we improve? Or keep it small.
                # Let's keep it small to refine.
            else:
                # Sometimes shrink step size if no improvement for a while?
                # Simple approach: keep step size constant or decay slowly.
                pass
        
        update_best(current, current_score)

    return best_pts
