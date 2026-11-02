"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0


def _min_triangle_area(pts):
    """Vectorized minimum triangle area."""
    n = len(pts)
    if n < 3:
        return 0.0
    # Compute all cross products for triples
    # pts shape (n, 2)
    # For efficiency, use broadcasting on indices
    min_area = float('inf')
    for i in range(n - 2):
        vi = pts[i]
        rest = pts[i + 1:]
        # cross products of (rest[j] - vi) and (rest[k] - vi) for j < k
        m = len(rest)
        if m < 2:
            continue
        # Vectorized: for each pair (j, k) with j < k
        # Use outer product approach
        # rest[j] - vi and rest[k] - vi
        # cross = (rest[j,0]-vi[0])*(rest[k,1]-vi[1]) - (rest[j,1]-vi[1])*(rest[k,0]-vi[0])
        # = rest[j,0]*rest[k,1] - rest[j,1]*rest[k,0] - vi[0]*rest[k,1] + vi[1]*rest[k,0]
        # + vi[0]*rest[j,1] - vi[1]*rest[j,0]
        # Let's just do it in a loop over j for speed
        for j in range(m - 1):
            vj = rest[j] - vi
            vk = rest[j + 1:] - vi
            crosses = vj[0] * vk[:, 1] - vj[1] * vk[:, 0]
            areas = 0.5 * np.abs(crosses)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_area


def _neg_score(pts):
    return -_score(pts)


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _build_from_rings(ring_params):
    """
    Build 13 points from ring parameters.
    ring_params: list of (count, radius, phase) tuples.
    Each ring has `count` points evenly spaced at `radius` with angular offset `phase`.
    """
    pts = []
    for count, radius, phase in ring_params:
        if count == 0:
            continue
        if count == 1:
            angle = phase
            pts.append([radius * np.cos(angle), radius * np.sin(angle)])
        else:
            angles = phase + 2 * np.pi * np.arange(count) / count
            ring_pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
            pts.append(ring_pts)
    if not pts:
        return np.zeros((N, 2))
    return np.vstack(pts)


def _optimize_symmetric(ring_counts, max_iter=200, method='L-BFGS-B'):
    """
    Optimize ring parameters (radii and phases) for a given ring structure.
    ring_counts: list of integers summing to N.
    Returns (best_pts, best_score).
    """
    n_rings = len(ring_counts)
    
    # Parameters: for each ring, radius and phase
    # Total params: 2 * n_rings
    def _params_to_points(params):
        ring_params = []
        for i, count in enumerate(ring_counts):
            radius = abs(params[2 * i])  # radius must be non-negative
            phase = params[2 * i + 1]
            ring_params.append((count, radius, phase))
        return _build_from_rings(ring_params)
    
    def _objective(params):
        pts = _params_to_points(params)
        return -_score(pts)
    
    # Initial guess: equal radii, zero phases
    best_score = 0.0
    best_pts = None
    
    # Try multiple initializations
    np.random.seed(42)
    n_tries = 10
    
    for trial in range(n_tries):
        if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
            break
        
        if trial == 0:
            # Regular-ish: equal radii
            x0 = np.zeros(2 * n_rings)
            for i, count in enumerate(ring_counts):
                x0[2 * i] = 1.0 / (i + 1)  # decreasing radii
                x0[2 * i + 1] = 0.0
        elif trial == 1:
            # Single ring (if n_rings == 1)
            x0 = np.zeros(2 * n_rings)
            x0[0] = 1.0
        else:
            # Random initialization
            x0 = np.random.uniform(0.1, 1.5, size=2 * n_rings)
            # Ensure radii are positive
            x0[0::2] = np.abs(x0[0::2])
        
        try:
            result = minimize(
                _objective,
                x0,
                method=method,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            cand_pts = _params_to_points(result.x)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback
        best_pts = _regular_polygon(N)
        best_score = _score(best_pts)
    
    return best_pts, best_score


def _try_nelder_mead_symmetric(ring_counts, max_iter=300):
    """Nelder-Mead on symmetric parameters."""
    n_rings = len(ring_counts)
    
    def _params_to_points(params):
        ring_params = []
        for i, count in enumerate(ring_counts):
            radius = abs(params[2 * i])
            phase = params[2 * i + 1]
            ring_params.append((count, radius, phase))
        return _build_from_rings(ring_params)
    
    def _objective(params):
        pts = _params_to_points(params)
        return -_score(pts)
    
    best_score = 0.0
    best_pts = None
    
    np.random.seed(43)
    for trial in range(5):
        if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
            break
        
        x0 = np.random.uniform(0.1, 1.5, size=2 * n_rings)
        x0[0::2] = np.abs(x0[0::2])
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            cand_pts = _params_to_points(result.x)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _regular_polygon(N)
        best_score = _score(best_pts)
    
    return best_pts, best_score


def construct():
    """Construct 13 points maximizing min triangle area / convex hull area."""
    
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)
    
    # Try various symmetric ring structures
    # 13 = 1 + 12, 1 + 6 + 6, 7 + 6, 13, 2 + 11, 3 + 10, 4 + 9, 5 + 8, 6 + 7
    ring_structures = [
        [13],           # single ring
        [1, 12],        # center + 12-gon
        [1, 6, 6],      # center + two 6-gons
        [7, 6],         # 7-gon + 6-gon
        [2, 11],        # 2 points + 11-gon
        [3, 10],        # 3 points + 10-gon
        [4, 9],         # 4 points + 9-gon
        [5, 8],         # 5 points + 8-gon
        [6, 7],         # 6-gon + 7-gon
    ]
    
    for ring_counts in ring_structures:
        if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
            break
        
        # L-BFGS-B optimization
        cand_pts, cand_score = _optimize_symmetric(ring_counts, max_iter=150, method='L-BFGS-B')
        if cand_score > best_score:
            best_score = cand_score
            best_pts = cand_pts
        
        # Nelder-Mead refinement
        if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
            break
        cand_pts, cand_score = _try_nelder_mead_symmetric(ring_counts, max_iter=200)
        if cand_score > best_score:
            best_score = cand_score
            best_pts = cand_pts
    
    # Final fine-tuning: perturb the best solution and re-optimize
    if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
        np.random.seed(44)
        for _ in range(3):
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                break
            # Small perturbation
            perturbed = best_pts + np.random.normal(0, 0.01, size=(N, 2))
            # Re-optimize as free points using Nelder-Mead
            def _neg_score_free(x):
                return -_score(x.reshape(N, 2))
            
            try:
                result = minimize(
                    _neg_score_free,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                cand_pts = result.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
            except Exception:
                pass
    
    # Final check
    if _score(best_pts) <= 0:
        best_pts = _regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
