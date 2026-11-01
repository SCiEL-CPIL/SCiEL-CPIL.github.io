"""Improved program: place 14 points in 3D to minimize (max/min pairwise distance)^2."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 14
D = 3
DEADLINE = 145.0  # safety margin before 150s kill

def _pairwise_dists_sq(pts):
    """Return all pairwise squared distances (upper triangle), shape (n_pairs,)."""
    n = len(pts)
    idx_i, idx_j = np.triu_indices(n, k=1)
    diffs = pts[idx_i] - pts[idx_j]
    return np.sum(diffs * diffs, axis=1)

def _score(pts):
    """Compute (max_dist / min_dist)^2."""
    d2 = _pairwise_dists_sq(pts)
    dmin = np.min(d2)
    dmax = np.max(d2)
    if dmin < 1e-30:
        return 1e12
    return (dmax / dmin) ** 2

def _get_16cell_14pts():
    """
    Get 14 points in 4D from the 16-cell structure.
    The 16-cell has 8 vertices at (±1,0,0,0), (0,±1,0,0), (0,0,±1,0), (0,0,0,±1).
    We use all 8 vertices plus 6 face centers of the 16-cell (which are at 
    (±1/√2, ±1/√2, 0, 0) and permutations) to get 14 points.
    """
    # 8 vertices of 16-cell
    verts = np.array([
        [1,0,0,0], [-1,0,0,0],
        [0,1,0,0], [0,-1,0,0],
        [0,0,1,0], [0,0,-1,0],
        [0,0,0,1], [0,0,0,-1]
    ], dtype=float)
    
    # 6 face centers: these are at (±1/√2, ±1/√2, 0, 0) and permutations
    # There are 6 pairs of opposite axes, and for each pair we have 4 sign combinations
    # But we only need 6 more points. Let's use the 6 "edge midpoints" of the 16-cell
    # which lie in 2D planes.
    # Actually, let's use 6 points that are midpoints of edges connecting non-adjacent vertices
    # or use the 6 face centers of the 8 tetrahedral cells.
    
    # Simpler: use 6 points at (1/√2, 1/√2, 0, 0) and sign/permutation variants
    # to get 6 distinct points with good symmetry
    s = 1.0 / np.sqrt(2.0)
    extra = np.array([
        [s, s, 0, 0],
        [s, -s, 0, 0],
        [0, s, s, 0],
        [0, s, -s, 0],
        [0, 0, s, s],
        [0, 0, s, -s]
    ], dtype=float)
    
    pts_4d = np.vstack([verts, extra])
    return pts_4d

def _project_4d_to_3d(pts_4d, M):
    """Project 4D points to 3D using 3x4 matrix M."""
    return pts_4d @ M.T

def _objective_projection(M_flat):
    """Objective: minimize (max/min)^2 for projected points."""
    M = M_flat.reshape(3, 4)
    pts_4d = _get_16cell_14pts()
    pts_3d = _project_4d_to_3d(pts_4d, M)
    
    d2 = _pairwise_dists_sq(pts_3d)
    dmin = np.min(d2)
    dmax = np.max(d2)
    if dmin < 1e-20:
        return 1e12
    return (dmax / dmin) ** 2

def _optimize_projection(M_init, max_iter=5000, tol=1e-10):
    """Optimize the 3x4 projection matrix."""
    M_flat_init = M_init.flatten()
    result = minimize(_objective_projection, M_flat_init, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
    return result.x.reshape(3, 4)

def _optimize_from_start(start_pts, max_iter=5000, tol=1e-10):
    """Run L-BFGS-B from a starting point in coordinate space."""
    x0 = start_pts.flatten()
    result = minimize(_objective, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
    return result.x.reshape(-1, D)

def _objective(pts_flat):
    """Objective for optimizer: minimize (max/min)^2 with penalty for near-coincident points."""
    pts = pts_flat.reshape(-1, D)
    d2 = _pairwise_dists_sq(pts)
    dmin = np.min(d2)
    dmax = np.max(d2)
    if dmin < 1e-20:
        return 1e12
    ratio_sq = (dmax / dmin) ** 2
    # Penalty for any pair too close (encourages distinctness)
    penalty = 0.0
    for v in d2:
        if v < 1e-6:
            penalty += (1e-6 - v) ** 2 * 1e6
    return ratio_sq + penalty

def _tetrahedron_init():
    """Create a regular tetrahedron with edge length 1, plus midpoints of edges."""
    v0 = np.array([1, 1, 1], dtype=float)
    v1 = np.array([1, -1, -1], dtype=float)
    v2 = np.array([-1, 1, -1], dtype=float)
    v3 = np.array([-1, -1, 1], dtype=float)
    scale = 1.0 / (2.0 * np.sqrt(2.0))
    verts = np.array([v0, v1, v2, v3]) * scale
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    mids = np.array([(verts[i] + verts[j]) / 2.0 for i, j in edges])
    faces = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    face_centers = np.array([(verts[i]+verts[j]+verts[k])/3.0 for i,j,k in faces])
    pts = np.vstack([verts, mids, face_centers])
    return pts

def _random_init():
    """Random initialization in unit cube."""
    return np.random.rand(N, D)

def _fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci spiral."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    sphere_pts = np.zeros((n, 3))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(1 - z*z)
        theta = golden_angle * i
        sphere_pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return sphere_pts

def construct():
    t_start = time.time()
    best_pts = None
    best_score = 1e12
    
    # Strategy 1: Optimize projection of 16-cell from 4D to 3D
    # This is a 12-dimensional optimization (3x4 matrix) vs 42-dimensional
    pts_4d = _get_16cell_14pts()
    
    # Try multiple initial projections
    projection_starts = []
    
    # Identity-like projection (drop 4th coordinate)
    M1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=float)
    projection_starts.append(M1)
    
    # Random projections
    for _ in range(5):
        M_rand = np.random.randn(3, 4)
        # Normalize rows
        for i in range(3):
            norm = np.linalg.norm(M_rand[i])
            if norm > 1e-10:
                M_rand[i] /= norm
        projection_starts.append(M_rand)
    
    # Orthogonal projection (SVD-based)
    U, s, Vt = np.linalg.svd(pts_4d, full_matrices=False)
    M_ortho = Vt[:3, :].T  # 3x4
    projection_starts.append(M_ortho)
    
    for i, M_init in enumerate(projection_starts):
        if time.time() - t_start > DEADLINE - 40:
            break
        try:
            M_opt = _optimize_projection(M_init, max_iter=3000, tol=1e-10)
            pts_3d = _project_4d_to_3d(pts_4d, M_opt)
            sc = _score(pts_3d)
            if sc < best_score:
                best_score = sc
                best_pts = pts_3d.copy()
        except Exception:
            pass
    
    # Strategy 2: Direct coordinate optimization from various starts
    starts = []
    starts.append(_tetrahedron_init())
    starts.append(_fibonacci_sphere(N))
    for _ in range(3):
        starts.append(_random_init())
    
    # Also add the best projection result as a start
    if best_pts is not None:
        starts.append(best_pts.copy())
    
    for i, start in enumerate(starts):
        if time.time() - t_start > DEADLINE - 20:
            break
        try:
            opt_pts = _optimize_from_start(start, max_iter=2000, tol=1e-9)
            sc = _score(opt_pts)
            if sc < best_score:
                best_score = sc
                best_pts = opt_pts
        except Exception:
            pass
    
    # Final polish
    if best_pts is not None and time.time() - t_start < DEADLINE - 5:
        try:
            polished = _optimize_from_start(best_pts, max_iter=3000, tol=1e-12)
            sc = _score(polished)
            if sc < best_score:
                best_score = sc
                best_pts = polished
        except Exception:
            pass
    
    # Ensure distinctness
    if best_pts is None:
        best_pts = _tetrahedron_init()
    
    d2 = _pairwise_dists_sq(best_pts)
    if np.min(d2) < 1e-10:
        for i in range(N):
            for j in range(i+1, N):
                if np.sum((best_pts[i] - best_pts[j])**2) < 1e-10:
                    best_pts[j] += np.random.randn(D) * 1e-5
    
    return best_pts
