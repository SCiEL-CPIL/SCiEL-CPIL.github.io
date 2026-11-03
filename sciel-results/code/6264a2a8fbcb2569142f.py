"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools
import math

N = 13
NUM_TRI = len(list(itertools.combinations(range(N), 3)))

# EVOLVE-BLOCK-START

def _all_triangles_idx():
    """Precompute all triangle index combinations."""
    return np.array(list(itertools.combinations(range(N), 3)), dtype=np.int32)

_TRI_IDX = _all_triangles_idx()

def _min_triangle_area_vec(pts):
    """Vectorized minimum triangle area computation."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    p = pts[_TRI_IDX]  # (286, 3, 2)
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
    return -_score(pts)

def _soft_min_score(flat_pts, beta=50.0):
    """
    Smooth surrogate: soft-min of triangle areas divided by hull area.
    This is differentiable and allows gradient-based methods.
    """
    pts = flat_pts.reshape(N, 2)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    
    p = pts[_TRI_IDX]
    A = p[:, 0, :]
    B = p[:, 1, :]
    C = p[:, 2, :]
    AB = B - A
    AC = C - A
    cross = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * area)))
    # This approximates the min as beta -> infinity
    min_val = np.min(areas)
    shifted = areas - min_val
    soft_min = min_val - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
    
    return soft_min / hull_a

def _negative_soft_score(flat_pts, beta=50.0):
    return -_soft_min_score(flat_pts, beta)

def _halton_sequence(n, base):
    """Generate n Halton sequence values in [0, 1) for the given base."""
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

def _halton_points(n, seed=0):
    """Generate n well-spread 2D points using Halton sequences."""
    rng = np.random.RandomState(seed)
    xs = _halton_sequence(n, 2)
    ys = _halton_sequence(n, 3)
    jitter = rng.uniform(-0.01, 0.01, size=(n, 2))
    pts = np.column_stack([xs, ys]) + jitter
    return pts

def _jittered_grid(n, seed=42, spacing=0.1):
    """Generate a jittered grid of points."""
    rng = np.random.RandomState(seed)
    side = int(math.ceil(math.sqrt(n)))
    points = []
    for i in range(side):
        for j in range(side):
            if len(points) >= n:
                break
            x = (i + 0.5) / side + rng.uniform(-spacing/2, spacing/2)
            y = (j + 0.5) / side + rng.uniform(-spacing/2, spacing/2)
            points.append([x, y])
    return np.array(points[:n])

def _polish_lbfgs(pts_init, deadline, maxiter=300):
    """
    Polish using L-BFGS-B on the soft-min surrogate, then refine with Nelder-Mead.
    """
    if time.time() > deadline:
        return pts_init, _score(pts_init)
    
    flat_init = pts_init.flatten()
    best_pts = pts_init
    best_score = _score(pts_init)
    
    # Phase 1: L-BFGS-B on soft-min surrogate (fast, gradient-based)
    try:
        result = minimize(
            _negative_soft_score,
            flat_init,
            args=(50.0,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        polished_pts = result.x.reshape(N, 2)
        polished_score = _score(polished_pts)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead on the true score for final refinement
    if time.time() < deadline - 2.0:
        try:
            result2 = minimize(
                _negative_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            polished_pts2 = result2.x.reshape(N, 2)
            polished_score2 = _score(polished_pts2)
            if polished_score2 > best_score:
                best_score = polished_score2
                best_pts = polished_pts2
        except Exception:
            pass
    
    return best_pts, best_score

def _polish_nelder_mead(pts_init, deadline, maxiter=300):
    """Polish using Nelder-Mead optimization."""
    if time.time() > deadline:
        return pts_init, _score(pts_init)
    
    flat_init = pts_init.flatten()
    try:
        result = minimize(
            _negative_score,
            flat_init,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11}
        )
        polished_pts = result.x.reshape(N, 2)
        polished_score = _score(polished_pts)
        return polished_pts, polished_score
    except Exception:
        return pts_init, _score(pts_init)

def _differential_evolution(deadline, maxiter=50, popsize=10):
    """
    Global search using differential evolution on the soft-min surrogate.
    """
    if time.time() > deadline:
        return None, 0.0
    
    bounds = [(-0.5, 1.5)] * (2 * N)
    try:
        result = differential_evolution(
            _negative_soft_score,
            bounds,
            args=(30.0,),
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            seed=42,
            polish=False,
            init='halton'
        )
        polished_pts = result.x.reshape(N, 2)
        polished_score = _score(polished_pts)
        return polished_pts, polished_score
    except Exception:
        return None, 0.0

def _simulated_annealing(pts_init, deadline, max_iter=20000, T0=0.03, T_min=1e-6, cooling=0.9997):
    """Simulated annealing to maximize min triangle area / hull area."""
    pts = pts_init.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    current_score = best_score
    T = T0
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = rng.randint(N)
        delta = rng.normal(0, T * 2, size=2)
        new_pts = pts.copy()
        new_pts[idx] += delta
        new_pts[idx] = np.clip(new_pts[idx], -0.5, 1.5)
        
        new_score = _score(new_pts)
        
        if new_score >= current_score or rng.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Differential evolution for global search (if time permits)
    if deadline - start_time > 60:
        try:
            de_pts, de_score = _differential_evolution(deadline, maxiter=40, popsize=8)
            if de_pts is not None and de_score > best_score:
                best_score = de_score
                best_pts = de_pts
        except Exception:
            pass
    
    # Phase 2: Multi-start local optimization with diverse initializations
    initial_sets = []
    
    # Halton-based
    for seed in range(4):
        initial_sets.append(_halton_points(N, seed=seed))
    
    # Jittered grids
    for seed in [42, 123, 456]:
        initial_sets.append(_jittered_grid(N, seed=seed))
    
    # Random points
    rng = np.random.RandomState(999)
    for _ in range(3):
        initial_sets.append(rng.uniform(-0.1, 1.1, size=(N, 2)))
    
    # If we have a good point from DE, add it
    if best_pts is not None:
        initial_sets.insert(0, best_pts)
    
    # Process each initial set with L-BFGS-B + Nelder-Mead polish
    for i, init_pts in enumerate(initial_sets):
        if time.time() > deadline - 5.0:
            break
        
        try:
            polished_pts, polished_score = _polish_lbfgs(init_pts, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts
        except Exception:
            continue
    
    # Phase 3: Simulated annealing from best point if time remains
    if best_pts is not None and time.time() < deadline - 15.0:
        try:
            sa_pts, sa_score = _simulated_annealing(best_pts, deadline, max_iter=15000, T0=0.02, cooling=0.9998)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts
        except Exception:
            pass
    
    # Phase 4: Final polish
    if best_pts is not None and time.time() < deadline - 3.0:
        try:
            final_pts, final_score = _polish_nelder_mead(best_pts, deadline, maxiter=300)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts
        except Exception:
            pass
    
    if best_pts is not None:
        return best_pts
    
    # Fallback
    return _jittered_grid(N, seed=42)

# EVOLVE-BLOCK-END
