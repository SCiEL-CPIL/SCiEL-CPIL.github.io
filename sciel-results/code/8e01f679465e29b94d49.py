import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _fourier_to_points(params):
    """
    Convert Fourier parameters to 13 points on a convex polygon.
    params: [r0, a1, phi1, a2, phi2]
    Support function h(theta) = r0 + a1*cos(theta + phi1) + a2*cos(2*theta + phi2)
    Vertices are intersections of tangent lines at theta_k = 2*pi*k/N.
    """
    r0, a1, phi1, a2, phi2 = params
    
    # Ensure r0 is large enough to keep polygon convex and away from origin
    # h(theta) > 0 for all theta
    # min h is roughly r0 - a1 - a2
    # We enforce this in the objective or bounds
    
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Calculate support function values and normals
    h_vals = r0 + a1 * np.cos(thetas + phi1) + a2 * np.cos(2 * thetas + phi2)
    normals = np.column_stack([np.cos(thetas), np.sin(thetas)])
    
    # If any h_val is <= 0, the polygon is not strictly convex or encloses origin incorrectly
    # We return a penalty score or invalid points
    if np.any(h_vals <= 1e-6):
        return None
        
    # Find vertices by intersecting consecutive lines
    # Line i: normal[i] . x = h[i]
    # Line i+1: normal[i+1] . x = h[i+1]
    
    vertices = np.zeros((N, 2))
    for i in range(N):
        n1 = normals[i]
        n2 = normals[(i + 1) % N]
        h1 = h_vals[i]
        h2 = h_vals[(i + 1) % N]
        
        # Solve [n1; n2] x = [h1; h2]
        A = np.array([n1, n2])
        b = np.array([h1, h2])
        
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-9:
            # Parallel lines, shouldn't happen with evenly spaced angles
            return None
            
        x = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
        y = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det
        vertices[i] = [x, y]
        
    return vertices

def _fourier_objective(params):
    """
    Objective for Differential Evolution on Fourier parameters.
    Returns negative score.
    """
    points = _fourier_to_points(params)
    if points is None:
        return 1e10
        
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _local_polish(points, deadline):
    """Multi-method local polish."""
    best_pts = points.copy()
    best_s = _score(points)
    
    # Nelder-Mead with adaptive options
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_pts = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    return best_pts, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # --- NEW STRUCTURAL APPROACH: Fourier Parameterized Convex Hull ---
    
    # Parameters: r0, a1, phi1, a2, phi2
    # Bounds:
    # r0: [0.5, 2.0] (must be > a1+a2 to ensure convexity)
    # a1: [0.0, 0.5]
    # phi1: [0, 2*pi]
    # a2: [0.0, 0.5]
    # phi2: [0, 2*pi]
    
    bounds = [
        (0.5, 2.0),   # r0
        (0.0, 0.5),   # a1
        (0.0, 2 * np.pi), # phi1
        (0.0, 0.5),   # a2
        (0.0, 2 * np.pi)  # phi2
    ]
    
    # Run Differential Evolution on the low-dimensional Fourier space
    if time.time() < deadline - 30:
        try:
            # Use a decent popsize and iterations for global search in 5D
            result = differential_evolution(
                _fourier_objective,
                bounds,
                maxiter=100,
                popsize=20,
                mutation=(0.5, 1.0),
                recombination=0.7,
                tol=1e-8,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            if result.fun < 1e9:
                # Convert best Fourier params to points
                best_pts_fourier = _fourier_to_points(result.x)
                if best_pts_fourier is not None:
                    _update_best(best_pts_fourier)
                    print(f"Fourier DE result: score={best_score:.6f}", flush=True)
                    
                    # Polish the Cartesian points found
                    if time.time() < deadline - 10:
                        polished_pts, polished_s = _local_polish(best_pts_fourier, deadline)
                        _update_best(polished_pts)
                        print(f"Polished Fourier result: score={best_score:.6f}", flush=True)
                        
        except Exception as e:
            print(f"Fourier DE failed: {e}", flush=True)

    # --- FALLBACK / ENHANCEMENT: Standard Cartesian Search ---
    # If the Fourier approach didn't find a good solution or we have time,
    # run a few standard Cartesian optimizations from diverse starts.
    
    if time.time() < deadline - 20:
        # Generate a few diverse starts
        rng = np.random.default_rng(12345)
        
        # 1. Regular 13-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])
        _update_best(pts_reg)
        
        # 2. Random perturbations of regular polygon
        for i in range(5):
            if time.time() > deadline - 15: break
            noise = rng.random((N, 2)) * 0.2 - 0.1
            pts_pert = pts_reg + noise
            _update_best(pts_pert)
            
            # Quick local polish
            if time.time() < deadline - 10:
                try:
                    polished_pts, polished_s = _local_polish(pts_pert, deadline)
                    _update_best(polished_pts)
                except Exception:
                    pass

    # Final high-precision polish on the best found so far
    if best_points is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
