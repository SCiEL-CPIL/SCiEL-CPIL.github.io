import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, minimize as local_minimize
from scipy.spatial.transform import Rotation
from itertools import product

N = 14
D = 3

# Precompute pairwise indices for speed
_IU = np.triu_indices(N, k=1)

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    return sq_dists[_IU]

def _regular_tetrahedron():
    """Return vertices of a regular tetrahedron centered at origin, unit circumradius."""
    phi = (1 + np.sqrt(5)) / 2
    # Regular tetrahedron vertices (alternating cube vertices)
    verts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    return verts

def _apply_rotation_xyz(v, ax, ay, az):
    """Apply Euler rotations (XYZ order) to point v."""
    # Rotation around X
    cx, sx = np.cos(ax), np.sin(ax)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    # Rotation around Y
    cy, sy = np.cos(ay), np.sin(ay)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    # Rotation around Z
    cz, sz = np.cos(az), np.sin(az)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    R = Rz @ Ry @ Rx
    return v @ R.T

def _two_tetrahedra_points(params):
    """
    Generate 14 points as two concentric tetrahedra (8 points) + 6 additional points.
    Actually, let's use: 7 points on inner sphere + 7 on outer sphere,
    or better: use the 14-point structure as two interpenetrating sets.
    
    For 14 points: Let's use 7+7 configuration where each set of 7 is
    approximated by a tetrahedron (4) + 3 extra points, or use a different decomposition.
    
    Actually, a cleaner approach: use 14 points = two sets of 7.
    Each set of 7 can be parameterized as a regular hexagon + 1 center, but that's 7D each.
    
    Let me use: 14 points on two concentric spheres.
    Inner sphere: 7 points approximated as regular heptagon in a plane + rotation
    Outer sphere: 7 points similarly
    
    Parameters: [r_inner, r_outer, rot_inner(3), rot_outer(3), tilt_inner, tilt_outer]
    = 9 parameters
    """
    r_in = params[0]
    r_out = params[1]
    rot_in = params[2:5]  # ax, ay, az for inner
    rot_out = params[5:8]  # ax, ay, az for outer
    tilt = params[8]  # tilt angle between the two rings
    
    # Generate 7 points on a circle (regular heptagon) in xy-plane
    angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(7)])
    
    # Inner ring: scaled and rotated
    inner_pts = circle_pts * r_in
    # Apply tilt: rotate around x-axis by tilt angle
    ct, st = np.cos(tilt), np.sin(tilt)
    Rx = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
    inner_pts = inner_pts @ Rx.T
    # Apply Euler rotation
    inner_pts = _apply_rotation_xyz(inner_pts, rot_in[0], rot_in[1], rot_in[2])
    
    # Outer ring: scaled and rotated with opposite tilt
    outer_pts = circle_pts * r_out
    outer_pts = outer_pts @ Rx.T  # same tilt matrix
    outer_pts = _apply_rotation_xyz(outer_pts, rot_out[0], rot_out[1], rot_out[2])
    
    return np.vstack([inner_pts, outer_pts])

def _two_tetrahedra_objective(params, tau=100.0):
    """Objective for two-ring parameterization."""
    points = _two_tetrahedra_points(params)
    
    # Center
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    points = points / np.sqrt(mean_sq_dist)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    log_sq_dists = np.log(sq_dists)
    
    m_max_log = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max_log))
    soft_max_log = m_max_log + np.log(np.sum(exp_max)) / tau
    
    m_min_log = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min_log))
    soft_min_log = m_min_log - np.log(np.sum(exp_min)) / tau
    
    return soft_max_log - soft_min_log

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def objective_log_ratio(x, tau=50.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    Minimizing this minimizes the ratio of max to min distance.
    Includes scale normalization to remove scale invariance.
    """
    points = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Normalize to fix scale: make the mean squared distance = 1
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    points = points / np.sqrt(mean_sq_dist)
    
    # Recompute squared distances after normalization
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max of log(squared distances)
    log_sq_dists = np.log(sq_dists)
    m_max_log = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max_log))
    soft_max_log = m_max_log + np.log(np.sum(exp_max)) / tau
    
    # Soft min of log(squared distances)
    m_min_log = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min_log))
    soft_min_log = m_min_log - np.log(np.sum(exp_min)) / tau
    
    # Objective is the difference (log ratio)
    obj = soft_max_log - soft_min_log
    
    return obj

def objective_epigraph(x, tau=50.0):
    """
    Epigraph formulation: minimize t subject to constraints.
    Uses smooth approximation of constraints.
    x = [points_flat, t] where t is the log ratio.
    """
    n_params = N * D
    points = x[:n_params].reshape(N, D)
    t = x[n_params]
    
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Normalize to fix scale: make the mean squared distance = 1
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    points = points / np.sqrt(mean_sq_dist)
    
    # Recompute squared distances after normalization
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    log_sq_dists = np.log(sq_dists)
    
    # Soft max of log(squared distances)
    m_max_log = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max_log))
    soft_max_log = m_max_log + np.log(np.sum(exp_max)) / tau
    
    # Soft min of log(squared distances)
    m_min_log = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min_log))
    soft_min_log = m_min_log - np.log(np.sum(exp_min)) / tau
    
    # Penalty if soft_max - soft_min > t
    violation = soft_max_log - soft_min_log - t
    penalty = np.maximum(0.0, violation) ** 2
    
    return t + 100.0 * penalty

def construct():
    start_time = time.time()
    timeout = 148.0  # Safety margin before 150s
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        # Check for distinct points
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Icosahedron + 2 points at symmetric positions
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Try several configurations for the 2 extra points
    # Configuration A: center + one on sphere
    p_center = np.array([0.0, 0.0, 0.0])
    p_sphere1 = np.array([1.0, 0.0, 0.0])
    full_pts_A = np.vstack([ico, p_center, p_sphere1])
    candidates.append(full_pts_A.flatten())
    
    # Configuration B: two points on sphere at opposite poles (but ico already has poles)
    # Use points that are not too close to ico vertices
    p1_b = np.array([0.5, 0.5, 0.5])
    p1_b = p1_b / np.linalg.norm(p1_b)
    p2_b = -p1_b
    full_pts_B = np.vstack([ico, p1_b, p2_b])
    candidates.append(full_pts_B.flatten())
    
    # Random perturbations of icosahedron-based configs
    for i in range(10):
        base = ico.copy()
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        full_pts = np.vstack([base, p1, p2])
        # Random rotation
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())

    # 2. Fibonacci sphere with 14 points
    def fibonacci_sphere(n, rng):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * i / (n - 1)
            radius = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        pts = np.array(points)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        return pts
    
    for i in range(8):
        pts = fibonacci_sphere(N, rng)
        candidates.append(pts.flatten())

    # 3. Random spherical points with better distribution
    for i in range(15):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        # Random rotation
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 4. Uniform random in cube
    for i in range(5):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())

    # 5. Try a few more structured starting points
    # Points on a sphere with some jitter
    for i in range(5):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        # Add small random jitter
        pts += 0.1 * rng.normal(size=(N, D))
        candidates.append(pts.flatten())
        
    # 6. Two rings of 7 points each
    for i in range(3):
        ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        ring2_angles = ring1_angles + np.pi/7
        ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles), np.zeros(7)])
        ring2 = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles), np.full(7, 0.5)])
        pts = np.vstack([ring1, ring2])
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())
    
    # 7. Structured two-ring parameterization: global search in 9D
    # Parameters: [r_in, r_out, rot_in(3), rot_out(3), tilt]
    bounds_2ring = [
        (0.1, 1.5),  # r_in
        (0.5, 2.5),  # r_out
        (-np.pi, np.pi),  # rot_in ax
        (-np.pi, np.pi),  # rot_in ay
        (-np.pi, np.pi),  # rot_in az
        (-np.pi, np.pi),  # rot_out ax
        (-np.pi, np.pi),  # rot_out ay
        (-np.pi, np.pi),  # rot_out az
        (-np.pi/2, np.pi/2)  # tilt
    ]
    
    # Run differential evolution on the 9D parameter space
    if time.time() - start_time < timeout - 30:
        try:
            de_result = differential_evolution(
                _two_tetrahedra_objective,
                bounds=bounds_2ring,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                update_interval=1,
                polish=True,
                init='sobol',
                tau=100.0
            )
            if de_result.success or de_result.fun < 1e10:
                pts = _two_tetrahedra_points(de_result.x)
                update_best(pts)
                candidates.append(de_result.x)
                # Also add the full point configuration as a candidate for further refinement
                candidates.append(pts.flatten())
        except Exception:
            pass
    
    # Local refinement of the best two-ring configuration
    if time.time() - start_time < timeout - 20:
        # Try multiple starting points for Nelder-Mead in the 9D space
        start_params_list = []
        # Symmetric starting points
        for r_in in [0.3, 0.5, 0.7, 1.0]:
            for r_out in [1.0, 1.5, 2.0]:
                for tilt in [0.0, np.pi/4, np.pi/3]:
                    for rot_offset in [0.0, np.pi/7]:
                        p = np.array([
                            r_in, r_out,
                            0.0, 0.0, 0.0,
                            rot_offset, 0.0, 0.0,
                            tilt
                        ])
                        start_params_list.append(p)
        
        # Random starting points
        for _ in range(20):
            p = np.array([
                rng.uniform(0.2, 1.2),
                rng.uniform(0.8, 2.0),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-np.pi/2, np.pi/2)
            ])
            start_params_list.append(p)
        
        for p0 in start_params_list:
            if time.time() - start_time > timeout - 10:
                break
            try:
                res = local_minimize(
                    _two_tetrahedra_objective,
                    p0,
                    method='Nelder-Mead',
                    args=(100.0,),
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                pts = _two_tetrahedra_points(res.x)
                update_best(pts)
                candidates.append(pts.flatten())
            except Exception:
                pass

    # Optimization Loop
    n_restarts = len(candidates)
    
    # We will do multiple passes with increasing tau
    taus = [10.0, 50.0, 200.0, 500.0, 1000.0, 2000.0]
    
    # First, do a quick L-BFGS with low tau on all candidates
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 12:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_log_ratio, 
                x0, 
                method='L-BFGS-B', 
                args=(taus[0],),
                options={'maxiter': 60, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Second, do a more precise L-BFGS with higher tau on all candidates
    for tau in taus[1:]:
        if time.time() - start_time > timeout - 6:
            break
        for i in range(n_restarts):
            if time.time() - start_time > timeout - 3:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_log_ratio, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Final high-precision polish: take the best few candidates and refine with very high tau
    # Sort candidates by their current raw score
    scored_candidates = []
    for i in range(n_restarts):
        pts = candidates[i].reshape(N, D)
        sc = raw_score(pts)
        scored_candidates.append((sc, i))
    scored_candidates.sort()
    
    # Take top 8 candidates for final polish
    top_candidates = [scored_candidates[j][1] for j in range(min(8, len(scored_candidates)))]
    
    for idx in top_candidates:
        if time.time() - start_time > timeout - 2:
            break
        x0 = candidates[idx]
        # Use very high tau for precise optimization
        try:
            res = minimize(
                objective_log_ratio, 
                x0, 
                method='L-BFGS-B', 
                args=(taus[-1],),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[idx] = res.x
        except Exception:
            pass
    
    # If we still have time, do a few more random restarts with high tau
    while time.time() - start_time < timeout - 3:
        x0 = rng.uniform(-1, 1, size=(N * D))
        try:
            res = minimize(
                objective_log_ratio, 
                x0, 
                method='L-BFGS-B', 
                args=(taus[-1],),
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
