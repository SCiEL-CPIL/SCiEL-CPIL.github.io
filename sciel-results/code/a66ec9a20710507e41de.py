"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _get_triangles_indices(n):
    return np.array(list(combinations(range(n), 3)))

_TRI_INDICES = _get_triangles_indices(N)

def _score(points):
    """
    Compute score: min triangle area / convex hull area.
    points: (N, 2) array.
    """
    if len(points) < 3:
        return 0.0
    
    # Vectorized triangle area calculation
    p1 = points[_TRI_INDICES[:, 0]]
    p2 = points[_TRI_INDICES[:, 1]]
    p3 = points[_TRI_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_tri = np.min(areas)
    
    if min_tri < 1e-10:
        return 0.0

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 1e-12:
        return 0.0
        
    return min_tri / hull_area

def _make_points_from_rings(radii, phases, counts):
    """
    Construct points from a list of rings.
    radii: list of radii
    phases: list of phases
    counts: list of number of points in each ring
    """
    points = []
    for r, ph, k in zip(radii, phases, counts):
        if k == 0:
            continue
        angles = 2 * np.pi * np.arange(k) / k + ph
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        points.append(np.column_stack((xs, ys)))
    if not points:
        return np.zeros((N, 2))
    return np.vstack(points)

def _optimize_counts(counts):
    """
    Optimize radii and phases for a given partition of points into rings.
    counts: tuple of ints summing to N.
    """
    n_rings = len(counts)
    
    def neg_score(params):
        # Unpack params
        radii = params[:n_rings]
        phases = params[n_rings:]
        
        # Ensure radii are positive
        radii = np.clip(radii, 1e-6, 2.0)
        
        pts = _make_points_from_rings(radii, phases, counts)
        
        # Normalize so that max radius is 1.0
        max_r = np.max(np.linalg.norm(pts, axis=1))
        if max_r < 1e-6:
            return 0.0
        pts = pts / max_r
        
        sc = _score(pts)
        return -sc

    best_sc = 0.0
    best_params = None
    
    # Initial guesses
    np.random.seed(42)
    
    # Determine number of starts based on complexity
    # 2 rings: 4 params. 3 rings: 6 params.
    n_starts = 50 if n_rings == 2 else 20
    
    for _ in range(n_starts):
        p0 = []
        for i, k in enumerate(counts):
            # Radius: log-uniform between 0.1 and 1.0
            r = np.exp(np.random.uniform(np.log(0.1), np.log(1.0)))
            p0.append(r)
        for i, k in enumerate(counts):
            ph = np.random.uniform(0, 2 * np.pi / k)
            p0.append(ph)
            
        p0 = np.array(p0)
        
        try:
            res = minimize(neg_score, p0, method='Powell',
                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
            sc = -res.fun
            if sc > best_sc:
                best_sc = sc
                best_params = res.x
        except Exception:
            pass
            
    # Add specific grid search for 2 rings to catch known good structures
    if n_rings == 2:
        k1, k2 = counts
        # Grid over r1 and phase1
        # r2 is fixed to 1.0 in this sub-search? No, we optimize both.
        # But we can fix r2=1.0 and optimize r1, ph1, ph2.
        # Let's just do a small grid on r1 and ph1, setting r2=1, ph2=0.
        for r1 in np.linspace(0.2, 0.8, 5):
            for ph1 in np.linspace(0, 2 * np.pi / k1, 3, endpoint=False):
                p0 = np.array([1.0, r1, 0.0, ph1])
                try:
                    res = minimize(neg_score, p0, method='Powell',
                                   options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-7})
                    sc = -res.fun
                    if sc > best_sc:
                        best_sc = sc
                        best_params = res.x
                except Exception:
                    pass

    if best_params is None:
        return 0.0, None

    # Reconstruct points
    radii = best_params[:n_rings]
    phases = best_params[n_rings:]
    pts = _make_points_from_rings(radii, phases, counts)
    max_r = np.max(np.linalg.norm(pts, axis=1))
    if max_r < 1e-6:
        return 0.0, None
    pts = pts / max_r
    
    return best_sc, pts

def construct():
    """
    Construct 13 points by optimizing ring configurations.
    """
    best_score = 0.0
    best_points = None
    
    # Partitions of 13 into rings
    partitions = []
    
    # 1 ring
    partitions.append((13,))
    
    # 2 rings
    for i in range(1, 7):
        partitions.append((i, 13 - i))
        
    # 3 rings: Select a subset of likely good partitions to save time
    # (1, 5, 7), (2, 4, 7), (2, 5, 6), (3, 3, 7), (3, 4, 6), (4, 4, 5)
    partitions.extend([
        (1, 5, 7),
        (2, 4, 7),
        (2, 5, 6),
        (3, 3, 7),
        (3, 4, 6),
        (4, 4, 5)
    ])
    
    # Try all partitions
    for counts in partitions:
        sc, pts = _optimize_counts(counts)
        if pts is not None and sc > best_score:
            best_score = sc
            best_points = pts
            
    if best_points is None:
        # Fallback: regular 13-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
