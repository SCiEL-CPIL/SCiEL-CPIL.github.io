import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def normalize_points(points):
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def objective_direct(x, penalty_lambda=1.0, t_min=0.0, t_max=np.inf):
    """
    Direct ratio objective with optional QCQP-style penalties.
    Minimizes max_sq_dist / min_sq_dist, with penalties for distances
    outside [t_min, t_max].
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-10)
    
    min_d = np.min(sq_dists)
    max_d = np.max(sq_dists)
    
    # Core ratio objective
    ratio = max_d / min_d
    
    # QCQP penalties: push distances into [t_min, t_max]
    penalty = 0.0
    if t_min > 0:
        viol_min = np.maximum(t_min - sq_dists, 0.0)
        penalty += np.sum(viol_min ** 2)
    if t_max < np.inf:
        viol_max = np.maximum(sq_dists - t_max, 0.0)
        penalty += np.sum(viol_max ** 2)
    
    return ratio + penalty_lambda * penalty

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, D):
            return
        if not np.all(np.isfinite(pts)):
            return
        pts = normalize_points(pts)
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random points on sphere
    for _ in range(15):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
    
    # 2. Fibonacci sphere (14 points)
    def fibonacci_sphere(n, rng):
        points = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            z = 1.0 - 2.0 * i / (n - 1)
            r = np.sqrt(max(0.0, 1.0 - z * z))
            theta = golden_angle * i
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    fib_pts = fibonacci_sphere(N, rng)
    candidates.append(fib_pts.flatten())
    
    # 3. Icosahedron + 2 symmetric poles
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym.flatten())
    
    # Random perturbations of ico + poles
    for _ in range(5):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())
    
    # 4. Cuboctahedron vertices (12 points) + 2
    perms = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
    ])
    cubo = perms / np.linalg.norm(perms, axis=1, keepdims=True)
    for _ in range(5):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([cubo, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())
    
    # Optimization: for each candidate, do direct optimization then QCQP refinement
    n_restarts = len(candidates)
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 20:
            break
        
        x0 = candidates[i].copy()
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            current_x = x0
            
            # Phase 1: Direct ratio optimization (no penalties)
            res = minimize(
                objective_direct,
                current_x,
                args=(0.0, 0.0, np.inf),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.x is not None:
                current_x = res.x
                pts_current = current_x.reshape(N, D)
                update_best(pts_current)
            
            # Phase 2: QCQP-style refinement with tightening bounds
            if time.time() - start_time < timeout - 15:
                pts_current = current_x.reshape(N, D)
                pts_current = normalize_points(pts_current)
                sq_dists = get_pairwise_sq_dists(pts_current)
                
                t_min = np.min(sq_dists) * 0.995
                t_max = np.max(sq_dists) * 1.005
                
                # Iteratively tighten bounds with increasing penalty
                penalty_lambdas = [0.1, 0.5, 1.0, 2.0, 5.0]
                
                for lam in penalty_lambdas:
                    if time.time() - start_time > timeout - 10:
                        break
                    
                    # Tighten bounds further
                    for _ in range(2):
                        if time.time() - start_time > timeout - 10:
                            break
                        
                        res = minimize(
                            objective_direct,
                            current_x,
                            args=(lam, t_min, t_max),
                            method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        
                        if res.x is not None:
                            current_x = res.x
                            pts_current = current_x.reshape(N, D)
                            pts_current = normalize_points(pts_current)
                            sq_dists = get_pairwise_sq_dists(pts_current)
                            
                            t_min = np.min(sq_dists) * 0.999
                            t_max = np.max(sq_dists) * 1.001
                            
                            update_best(pts_current)
                            
        except Exception:
            pass
    
    # Final high-precision polish on best point
    if best_points is not None and time.time() - start_time < timeout - 5:
        try:
            x_polish = best_points.flatten()
            pts_polish = best_points.copy()
            sq_dists = get_pairwise_sq_dists(pts_polish)
            t_min = np.min(sq_dists) * 0.9999
            t_max = np.max(sq_dists) * 1.0001
            
            res = minimize(
                objective_direct,
                x_polish,
                args=(10.0, t_min, t_max),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res.x is not None:
                pts_final = res.x.reshape(N, D)
                update_best(pts_final)
        except Exception:
            pass

    if best_points is None:
        best_points = normalize_points(candidates[0].reshape(N, D))
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
