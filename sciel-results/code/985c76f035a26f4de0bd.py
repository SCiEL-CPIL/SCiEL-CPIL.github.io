import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    max_sq = np.max(sq_dists)
    return (max_sq / min_sq)

def normalize_points(points):
    """
    Center and normalize points to unit mean squared norm.
    This removes translational and scale invariance.
    """
    points = points - np.mean(points, axis=0, keepdims=True)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def objective_smooth(x, tau=100.0):
    """
    Smooth approximation of the log ratio of max to min squared distance.
    Uses Log-Sum-Exp for soft max and soft min.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    # Soft min
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    # Minimize log(soft_max) - log(soft_min)
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    return obj_val

def fibonacci_sphere(n, rng):
    """Generate n points on a sphere using Fibonacci lattice with random rotation."""
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

def d3d_symmetric_points(r1, r2, r3, h):
    """
    Generate 14 points using D_{3d} symmetry with 4 independent radial/height parameters.
    
    Configuration:
    - 6 points on a hexagonal ring at radius r1, height -h
    - 6 points on a hexagonal ring at radius r2, height +h (offset by 30 degrees)
    - 2 points on the z-axis at heights +/- r3
    
    Total: 6 + 6 + 2 = 14
    """
    # 6 points on hexagonal ring 1 at height -h
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    pts1 = np.column_stack([
        r1 * np.cos(angles1),
        r1 * np.sin(angles1),
        np.full(6, -h)
    ])
    
    # 6 points on hexagonal ring 2 at height +h, offset by 30 degrees (pi/6)
    angles2 = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False)
    pts2 = np.column_stack([
        r2 * np.cos(angles2),
        r2 * np.sin(angles2),
        np.full(6, h)
    ])
    
    # 2 points on z-axis
    pts3 = np.array([[0, 0, -r3], [0, 0, r3]])
    
    return np.vstack([pts1, pts2, pts3])

def d3d_objective(params):
    """Objective for D_{3d} symmetric configuration. Returns raw score."""
    r1, r2, r3, h = params
    if r1 < 0 or r2 < 0 or h < 0:
        return 1e10
    pts = d3d_symmetric_points(r1, r2, r3, h)
    return raw_score(pts)

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # --- Strategy 1: D_{3d} symmetry-constrained optimization ---
    # This exploits the high symmetry of the problem structure.
    # Grid search over the 4D parameter space (r1, r2, r3, h)
    r1_vals = np.linspace(0.5, 2.0, 8)
    r2_vals = np.linspace(0.5, 2.0, 8)
    r3_vals = np.linspace(0.2, 2.0, 8)
    h_vals = np.linspace(0.2, 1.5, 6)
    
    best_sym_score = np.inf
    best_sym_params = None
    
    for r1 in r1_vals:
        if time.time() - start_time > timeout - 40:
            break
        for r2 in r2_vals:
            for r3 in r3_vals:
                for h in h_vals:
                    try:
                        score = d3d_objective([r1, r2, r3, h])
                        if score < best_sym_score:
                            best_sym_score = score
                            best_sym_params = [r1, r2, r3, h]
                    except Exception:
                        pass
    
    # Refine with Nelder-Mead around the best grid point
    if best_sym_params is not None:
        try:
            res_sym = minimize(
                d3d_objective,
                best_sym_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_sym.x is not None:
                pts_sym = d3d_symmetric_points(*res_sym.x)
                update_best(pts_sym)
        except Exception:
            pass
    
    # Also try several random initial points for Nelder-Mead
    for _ in range(5):
        if time.time() - start_time > timeout - 30:
            break
        r1_init = rng.uniform(0.5, 2.0)
        r2_init = rng.uniform(0.5, 2.0)
        r3_init = rng.uniform(0.2, 2.0)
        h_init = rng.uniform(0.2, 1.5)
        try:
            res_rand = minimize(
                d3d_objective,
                [r1_init, r2_init, r3_init, h_init],
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_rand.x is not None:
                pts_rand = d3d_symmetric_points(*res_rand.x)
                update_best(pts_rand)
        except Exception:
            pass
    
    # --- Strategy 2: Coordinate-space optimization with diverse initializations ---
    
    # Initial candidates
    candidates = []
    
    # 1. Icosahedron + 2 points (various placements)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    for i in range(5):
        pts = ico.copy()
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        full_pts = np.vstack([pts, p1, p2])
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())
    
    # 2. Fibonacci sphere with 14 points
    for i in range(5):
        pts = fibonacci_sphere(N, rng)
        candidates.append(pts.flatten())
    
    # 3. Random spherical points
    for i in range(5):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())
    
    # 4. Uniform random in cube
    for i in range(3):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())
    
    # 5. Two rings of 7 points each (approximate D_{3d})
    for i in range(3):
        ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        ring2_angles = ring1_angles + np.pi/7
        ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles), np.zeros(7)])
        ring2 = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles), np.full(7, 0.5)])
        pts = np.vstack([ring1, ring2])
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())
    
    # 6. Cuboctahedron vertices (12 points) + 2 random
    perms = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
    ])
    cubo = perms / np.linalg.norm(perms, axis=1, keepdims=True)
    for i in range(5):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([cubo, p1.reshape(1, D), p2.reshape(1, D)])
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())
    
    # 7. Concentric shells: 8 on inner shell, 6 on outer shell
    inner = fibonacci_sphere(8, rng) * 0.7
    outer = fibonacci_sphere(6, rng) * 1.0
    shell_pts = np.vstack([inner, outer])
    shell_pts = shell_pts / np.linalg.norm(shell_pts, axis=1, keepdims=True)
    candidates.append(shell_pts.flatten())
    
    for i in range(3):
        inner_pert = inner + rng.normal(size=(8, D)) * 0.1
        outer_pert = outer + rng.normal(size=(6, D)) * 0.1
        shell_pert = np.vstack([inner_pert, outer_pert])
        shell_pert = shell_pert / np.linalg.norm(shell_pert, axis=1, keepdims=True)
        candidates.append(shell_pert.flatten())
    
    # If we have a good symmetric point, add it as a candidate
    if best_points is not None:
        candidates.append(best_points.flatten())
    
    n_restarts = len(candidates)
    
    # Tau schedule: start low for smooth landscape, increase for precision
    # Extended range from inspiration
    tau_schedule = [1.0, 5.0, 15.0, 50.0, 150.0, 500.0, 1500.0, 5000.0]
    
    # Phase 1: Quick optimization with low tau on all candidates
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 20:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[0],),
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 2: Progressive tau increase with more iterations
    for tau_idx in range(1, len(tau_schedule)):
        if time.time() - start_time > timeout - 10:
            break
        tau = tau_schedule[tau_idx]
        max_iter = 100 + tau_idx * 40
        for i in range(n_restarts):
            if time.time() - start_time > timeout - 5:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_smooth, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Final polish: use the best points found and do a high-precision coordinate optimization
    if best_points is not None and time.time() - start_time < timeout - 2:
        try:
            res_final = minimize(
                objective_smooth, 
                best_points.flatten(), 
                args=(10000.0,),
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            if res_final.x is not None:
                pts_final = res_final.x.reshape(N, D)
                update_best(pts_final)
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
