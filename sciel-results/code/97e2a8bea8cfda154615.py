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

def objective_smooth(x, tau=10.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    Minimizing this minimizes the ratio of max to min distance.
    """
    points = x.reshape(N, D)
    
    # Center the points
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Normalize scale to fix gauge: sum of squared norms = 1
    # This is a common constraint for spherical codes, but we are in 3D space.
    # However, the ratio is scale invariant. Fixing scale prevents drift.
    norms_sq = np.sum(points**2, axis=1)
    sum_norms_sq = np.sum(norms_sq)
    if sum_norms_sq < 1e-12:
        return 1e10
    points = points / np.sqrt(sum_norms_sq)
    
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Softmax for log max
    max_sq = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - max_sq))
    log_sum_exp_max = max_sq + np.log(np.sum(exp_max)) / tau
    
    # Softmin for log min
    min_sq = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - min_sq))
    log_sum_exp_min = min_sq - np.log(np.sum(exp_min)) / tau
    
    # Objective: log(max) - log(min)
    obj = log_sum_exp_max - log_sum_exp_min
    
    return obj

def generate_icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    return ico

def generate_fibonacci_sphere(n, rng):
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        radius = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    pts = np.array(points)
    # Random rotation
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        # Check for duplicate points
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
    
    # 1. Icosahedron + 2 points
    # The icosahedron has 12 vertices. We need 2 more.
    # A good guess is to place them at the centers of opposite faces? Or just random on sphere?
    # Let's try placing them at the north and south poles, but rotate the icosahedron so poles are not vertices.
    ico = generate_icosahedron()
    # Rotate ico so that a face is aligned with z-axis?
    # Actually, just adding 2 random points on the sphere to the ico and letting the optimizer fix them is a good strategy.
    for _ in range(5):
        p1 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1, p2])
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())
        
    # 2. Fibonacci sphere
    for _ in range(5):
        pts = generate_fibonacci_sphere(N, rng)
        candidates.append(pts.flatten())
        
    # 3. Random points on sphere
    for _ in range(5):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())
        
    # 4. Random points in cube (might find non-spherical solutions, though unlikely to be optimal for this metric)
    for _ in range(3):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())

    # Optimization Loop with tau annealing
    # Tau controls the sharpness of the max/min approximation.
    # Low tau: smooth, easy to optimize, but might not capture the exact max/min.
    # High tau: sharp, close to true objective, but harder to optimize.
    
    tau_schedule = [5.0, 20.0, 50.0, 100.0, 200.0]
    
    # Phase 1: Coarse optimization
    for i in range(len(candidates)):
        if time.time() - start_time > timeout - 15:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[0],),
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 2: Progressive refinement
    for tau_idx in range(1, len(tau_schedule)):
        if time.time() - start_time > timeout - 5:
            break
        tau = tau_schedule[tau_idx]
        max_iter = 100 + tau_idx * 50
        for i in range(len(candidates)):
            if time.time() - start_time > timeout - 3:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_smooth, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 3: High-precision polish on the best candidates
    if time.time() - start_time < timeout - 5:
        scored_candidates = []
        for i in range(len(candidates)):
            pts = candidates[i].reshape(N, D)
            score = raw_score(pts)
            scored_candidates.append((score, i))
        scored_candidates.sort()
        
        # Polish the top 5 candidates
        for score, i in scored_candidates[:5]:
            if time.time() - start_time > timeout - 2:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_smooth, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau_schedule[-1],),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 4: Additional random restarts if time remains
    restart_count = 0
    while time.time() - start_time < timeout - 3:
        x0 = rng.uniform(-1, 1, size=(N * D))
        # Normalize initial guess to sphere to help convergence
        pts0 = x0.reshape(N, D)
        pts0 = pts0 / np.linalg.norm(pts0, axis=1, keepdims=True)
        x0 = pts0.flatten()
        
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[2],),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                
                # If this is good, do a high-precision polish
                if raw_score(pts) < best_score * 1.05 and time.time() - start_time < timeout - 2:
                    try:
                        res2 = minimize(
                            objective_smooth, 
                            res.x, 
                            method='L-BFGS-B', 
                            args=(tau_schedule[-1],),
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                        )
                        if res2.x is not None:
                            pts2 = res2.x.reshape(N, D)
                            update_best(pts2)
                    except Exception:
                        pass
        except Exception:
            pass
        restart_count += 1

    if best_points is None:
        # Fallback
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
