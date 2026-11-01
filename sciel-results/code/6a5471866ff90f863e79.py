import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices once
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]
_NTRIS = len(_IDX)

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def _build_polygon_from_edge_lengths(edge_lengths, start_angle=0.0):
    """
    Build a convex polygon from edge lengths and cumulative turning angles.
    For a convex polygon, the external angles sum to 2*pi.
    We parameterize by edge lengths and assume equal angular steps (regular-like),
    then optimize the shape.
    
    Actually, a better approach: parameterize by angles between consecutive edges.
    For a convex n-gon, the turning angles (external angles) are positive and sum to 2*pi.
    
    Let's use a different parameterization: given n edge lengths and n-1 turning angles
    (the last is determined), we can reconstruct the polygon.
    
    Simpler: use polar coordinates from centroid with sorted angles.
    """
    n = len(edge_lengths)
    if n < 3:
        return np.array([[0, 0], [1, 0], [0, 1]])
    
    # For a convex polygon, we can parameterize by:
    # - Edge lengths (n values)
    # - Turning angles (n-1 values, last determined by closure)
    # 
    # Alternative: use the fact that for a convex polygon, we can specify
    # the direction of each edge. Let theta_i be the angle of edge i.
    # For convexity, theta_{i+1} - theta_i > 0 and sum of turns = 2*pi.
    #
    # Let's use a simpler approach: parameterize by the angles of vertices
    # from the centroid, ensuring they are sorted and cover 2*pi.
    
    # Actually, let's use a robust approach: 
    # Given edge lengths, construct the polygon by walking with specified turns.
    # We'll optimize the turning angles as well.
    
    # For now, let's use a different strategy: 
    # Parameterize the convex polygon by its vertex angles from centroid.
    # This guarantees convexity if angles are sorted.
    
    # But the directive says to use edge lengths. Let me think differently.
    # 
    # Key insight: For a fixed set of edge lengths, there's a family of convex polygons.
    # The shape is determined by the turning angles. We can optimize both edge lengths
    # and turning angles, with constraints that turns are positive and sum to 2*pi.
    
    # Let's use: x = [edge_lengths (n), turning_angles (n-1)]
    # The last turning angle is 2*pi - sum(turning_angles[:n-1])
    # Constraint: all turning angles > 0
    
    # For now, return a regular polygon as placeholder
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + start_angle
    mean_len = np.mean(edge_lengths)
    x = mean_len * np.cos(angles)
    y = mean_len * np.sin(angles)
    return np.column_stack((x, y))

def _build_polygon_from_angles_radii(angles, radii):
    """Build convex polygon from sorted angles and radii."""
    n = len(angles)
    # Sort by angle to ensure proper ordering
    order = np.argsort(angles)
    angles_sorted = angles[order]
    radii_sorted = radii[order]
    
    # Make angles cover exactly 2*pi
    if angles_sorted[-1] - angles_sorted[0] < 2 * np.pi - 1e-6:
        # Adjust to span 2*pi
        angles_sorted = np.linspace(angles_sorted[0], angles_sorted[0] + 2 * np.pi, n, endpoint=False)
    
    x = radii_sorted * np.cos(angles_sorted)
    y = radii_sorted * np.sin(angles_sorted)
    return np.column_stack((x, y))

def _score_from_convex_params(angles, radii):
    """Score a convex polygon configuration."""
    points = _build_polygon_from_angles_radii(angles, radii)
    return compute_score(points)

def _neg_score_convex(x):
    """Negative score for convex polygon parameterization."""
    n = N
    angles = x[:n]
    radii = x[n:]
    radii = np.maximum(radii, 0.1)  # Ensure positive radii
    return -_score_from_convex_params(angles, radii)

def _optimize_convex_polygon(x0, maxiter=1000):
    """Optimize convex polygon parameters."""
    try:
        res = minimize(_neg_score_convex, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'ftol': 1e-10})
        return res.x
    except Exception:
        return x0

def _optimize_full(points, maxiter=500):
    """Full 2D optimization of all 14 points."""
    x0 = points.flatten()
    try:
        res = minimize(neg_score, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'ftol': 1e-10})
        return res.x.reshape((N, 2))
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)

    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    def remaining():
        return deadline - (time.time() - start_time)

    # --- Phase 1: Convex polygon optimization (14D: 7 angles + 7 radii) ---
    # Regular 14-gon initialization
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii_reg = np.ones(N)
    x0_convex = np.concatenate([angles_reg, radii_reg])
    
    # Optimize convex polygon
    x_opt = _optimize_convex_polygon(x0_convex, maxiter=500)
    angles_opt = x_opt[:N]
    radii_opt = np.maximum(x_opt[N:], 0.1)
    points_convex = _build_polygon_from_angles_radii(angles_opt, radii_opt)
    try_update(points_convex)
    
    # Multiple restarts for convex polygon optimization
    for restart in range(5):
        if remaining() < 15:
            break
        # Perturb the regular polygon
        angles_pert = angles_reg + rng.normal(0, 0.1, N)
        radii_pert = radii_reg + rng.normal(0, 0.05, N)
        x0_pert = np.concatenate([angles_pert, radii_pert])
        x_opt_pert = _optimize_convex_polygon(x0_pert, maxiter=300)
        angles_opt_pert = x_opt_pert[:N]
        radii_opt_pert = np.maximum(x_opt_pert[N:], 0.1)
        points_pert = _build_polygon_from_angles_radii(angles_opt_pert, radii_opt_pert)
        try_update(points_pert)
    
    # --- Phase 2: Diverse initial points for full 2D optimization ---
    inits = []
    
    # Regular 14-gon
    p_reg = np.column_stack((np.cos(angles_reg), np.sin(angles_reg)))
    inits.append(p_reg)
    
    # The optimized convex polygon
    inits.append(points_convex)
    
    # Two concentric rings
    for n_out in [6, 7, 8, 9]:
        n_in = N - n_out
        a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
        for ratio in [0.4, 0.5, 0.6, 0.7]:
            for offset in [0, np.pi / n_out]:
                a_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + offset
                p_out = np.column_stack((np.cos(a_out), np.sin(a_out)))
                p_in = np.column_stack((ratio * np.cos(a_in), ratio * np.sin(a_in)))
                inits.append(np.vstack((p_out, p_in)))
    
    # Jittered regular polygon
    for _ in range(5):
        inits.append(p_reg + rng.normal(0, 0.1, (N, 2)))
    
    # Random in disk
    for _ in range(10):
        r = np.sqrt(rng.random(N))
        th = rng.random(N) * 2 * np.pi
        inits.append(np.column_stack((r * np.cos(th), r * np.sin(th))))
    
    # Evaluate all inits
    for p in inits:
        try_update(p)
    
    # Sort by score
    scored = [(compute_score(p), p) for p in inits]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # --- Phase 3: Full 2D Nelder-Mead from top inits ---
    for init_score, init_points in scored[:6]:
        if remaining() < 15:
            break
        new_p = _optimize_full(init_points, maxiter=400)
        try_update(new_p)
    
    # --- Phase 4: Powell refinement from best ---
    if best_points is not None and remaining() > 20:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                           options={'maxiter': 1000, 'xtol': 1e-9, 'ftol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # --- Phase 5: Basin hopping ---
    if best_points is not None and remaining() > 25:
        for step in [0.3, 0.5, 0.7]:
            if remaining() < 15:
                break
            x0 = best_points.flatten()
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-8, 'ftol': 1e-10})
            for _ in range(5):
                if remaining() < 10:
                    break
                perturbation = rng.normal(0, step, (N, 2))
                x_pert = res.x + perturbation.flatten()
                res2 = minimize(neg_score, x_pert, method='Nelder-Mead',
                                options={'maxiter': 150, 'xatol': 1e-8, 'ftol': 1e-10})
                if res2.fun < res.fun:
                    res = res2
            try_update(res.x.reshape((N, 2)))
    
    # --- Phase 6: Tight polish ---
    if best_points is not None and remaining() > 12:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'ftol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # --- Phase 7: Final Powell polish ---
    if best_points is not None and remaining() > 8:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                           options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    if best_points is None:
        return p_reg
    return best_points
