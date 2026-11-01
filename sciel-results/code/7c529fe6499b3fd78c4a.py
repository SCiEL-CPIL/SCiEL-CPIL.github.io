"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    dists = pdist(points)
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d <= 1e-12:
        return 1e10
    return (max_d / min_d) ** 2

def _smooth_objective(points_flat, beta=10.0):
    """
    Smooth approximation of (max/min)^2 using soft max/min.
    """
    points = points_flat.reshape(N, D)
    dists = pdist(points)
    d_safe = np.maximum(dists, 1e-15)
    
    # Soft max using log-sum-exp
    max_approx = beta * np.log(np.sum(np.exp(d_safe / beta)))
    # Soft min using negative log-sum-exp
    min_approx = -beta * np.log(np.sum(np.exp(-d_safe / beta)))
    
    if min_approx <= 1e-12:
        return 1e10
    
    return (max_approx / min_approx) ** 2

def _generate_fibonacci_sphere(n):
    """Generate n points on a sphere using the Fibonacci sphere method."""
    points = np.zeros((n, D))
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1 - (i + 0.5) * 2.0 / n
        r = np.sqrt(1 - z * z)
        theta = phi * i
        points[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return points

def _generate_icosahedron_based():
    """Start from icosahedron (12 vertices) and add 2 points."""
    # Icosahedron vertices
    t = (1 + np.sqrt(5)) / 2
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ]
    pts = np.array(verts, dtype=float)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Add 2 points at optimal positions (try poles first)
    poles = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
    return np.vstack([pts, poles])

def _generate_tetrahedral_lattice():
    """Generate points using a distorted tetrahedral lattice."""
    pts = []
    # 4 tetrahedron vertices
    t = np.sqrt(2/3)
    tetra = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ]) * t
    pts.extend([p for p in tetra])
    
    # Add 10 more points distributed symmetrically
    rng = np.random.default_rng(42)
    for _ in range(10):
        p = rng.standard_normal(3)
        p /= np.linalg.norm(p)
        pts.append(p)
    
    return np.array(pts)

def _initialization_candidates():
    """Generate diverse initial configurations."""
    candidates = []
    
    # 1. Fibonacci Sphere
    candidates.append(_generate_fibonacci_sphere(N))
    
    # 2. Icosahedron + 2 poles
    candidates.append(_generate_icosahedron_based())
    
    # 3. Tetrahedral lattice
    candidates.append(_generate_tetrahedral_lattice())
    
    # 4. Random Sphere (more diverse seeds)
    for seed in range(12):
        rng_i = np.random.default_rng(seed * 7919 + 13)
        pts = rng_i.standard_normal((N, D))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts)
    
    # 5. 2 Poles + 12 on equator
    pts = np.zeros((N, D))
    pts[0] = [0, 0, 1]
    pts[1] = [0, 0, -1]
    for i in range(12):
        theta = 2 * np.pi * i / 12
        pts[2+i] = [np.cos(theta), np.sin(theta), 0]
    candidates.append(pts)
    
    # 6. 2 Poles + 6 at z=0.5 + 6 at z=-0.5
    pts = np.zeros((N, D))
    pts[0] = [0, 0, 1]
    pts[1] = [0, 0, -1]
    r_half = np.sqrt(1 - 0.5**2)
    for i in range(6):
        theta = 2 * np.pi * i / 6
        pts[2+i] = [r_half * np.cos(theta), r_half * np.sin(theta), 0.5]
    for i in range(6):
        theta = 2 * np.pi * i / 6 + np.pi/6
        pts[8+i] = [r_half * np.cos(theta), r_half * np.sin(theta), -0.5]
    candidates.append(pts)
    
    # 7. 3 rings of 4 points at different z-levels + 2 poles
    pts = []
    pts.append([0, 0, 1])
    pts.append([0, 0, -1])
    z_levels = [0.5, 0.0, -0.5]
    for ring_idx, z in enumerate(z_levels):
        r = np.sqrt(1 - z**2)
        offset = ring_idx * np.pi / 4
        for i in range(4):
            theta = 2 * np.pi * i / 4 + offset
            pts.append([r * np.cos(theta), r * np.sin(theta), z])
    candidates.append(np.array(pts))
    
    return candidates

def _refine_with_nelder_mead(points, maxiter=5000):
    """Refine points using Nelder-Mead on the true objective."""
    x0 = points.flatten()
    try:
        res = minimize(
            lambda x: _compute_score(x.reshape(N, D)),
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        return res.x.reshape(N, D)
    except Exception:
        return points

def _refine_with_lbfgsb(points, beta=10.0, maxiter=1000):
    """Refine points using L-BFGS-B on smooth objective."""
    x0 = points.flatten()
    try:
        res = minimize(
            lambda x: _smooth_objective(x, beta=beta),
            x0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        return res.x.reshape(N, D)
    except Exception:
        return points

def construct():
    """
    Construct 14 points in 3D space to minimize the squared ratio of
    maximum to minimum pairwise distance.
    """
    best_score = 1e10
    best_points = None
    
    candidates = _initialization_candidates()
    
    # Phase 1: Try L-BFGS-B with smooth objective from each candidate
    for i, init_pts in enumerate(candidates):
        refined = _refine_with_lbfgsb(init_pts, beta=10.0, maxiter=500)
        score = _compute_score(refined)
        if score < best_score:
            best_score = score
            best_points = refined
    
    # Phase 2: Polish best result with Nelder-Mead
    if best_points is not None:
        refined = _refine_with_nelder_mead(best_points, maxiter=10000)
        score = _compute_score(refined)
        if score < best_score:
            best_score = score
            best_points = refined
    
    # Phase 3: Try Nelder-Mead from top candidates
    candidate_scores = []
    for i, init_pts in enumerate(candidates):
        score = _compute_score(init_pts)
        candidate_scores.append((score, i, init_pts))
    
    candidate_scores.sort()
    
    for score, idx, init_pts in candidate_scores[:5]:
        if idx == 0:
            continue
        refined = _refine_with_nelder_mead(init_pts, maxiter=5000)
        new_score = _compute_score(refined)
        if new_score < best_score:
            best_score = new_score
            best_points = refined
    
    # Phase 4: Final polish
    if best_points is not None:
        refined = _refine_with_nelder_mead(best_points, maxiter=20000)
        score = _compute_score(refined)
        if score < best_score:
            best_score = score
            best_points = refined
    
    # Fallback
    if best_points is None:
        best_points = np.random.randn(N, D)
        
    return best_points
# EVOLVE-BLOCK-END
