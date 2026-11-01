import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.spatial.transform import Rotation

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2  # 91

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
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def objective_smooth(x, tau=50.0, scale_reg=0.01):
    """
    Smooth approximation of the log ratio of max to min squared distance.
    Includes a small regularization to fix the scale.
    """
    points = x.reshape(N, D)
    points = points - np.mean(points, axis=0, keepdims=True)
    
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
    
    obj = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    total_sq_norm = np.sum(points**2)
    reg = scale_reg * (total_sq_norm - 1.0)**2
    
    return obj + reg

def objective_epigraph(x, scale_reg=0.01):
    """
    Epigraph formulation: minimize t subject to all sq_dists <= t * max_sq and >= t * min_sq?
    Actually, we want to minimize ratio = max/min.
    Let t be the ratio.
    We want max(sq) / min(sq) <= t.
    This is equivalent to:
    max(sq) <= t * min(sq)
    
    This is still non-convex and non-smooth.
    Let's stick to the log-sum-exp soft ratio but perhaps with a different parameterization.
    
    Alternative: Minimize sum of log(sq_dists) - log(min) ? No.
    
    Let's try a different objective:
    Minimize variance of log(sq_dists).
    If all distances are equal, variance is 0.
    This encourages all distances to be close to each other.
    """
    points = x.reshape(N, D)
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    log_dists = np.log(sq_dists)
    mean_log = np.mean(log_dists)
    var_log = np.var(log_dists)
    
    # Add a term to prevent collapse to a point
    total_sq_norm = np.sum(points**2)
    reg = scale_reg * (total_sq_norm - 1.0)**2
    
    return var_log + reg

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    for i in range(10):
        pts = ico.copy()
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        full_pts = np.vstack([pts, p1, p2])
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())
        
    # 2. Fibonacci sphere
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

    for i in range(10):
        pts = fibonacci_sphere(N, rng)
        candidates.append(pts.flatten())

    # 3. Random spherical points
    for i in range(20):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 4. Uniform random in cube
    for i in range(10):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())
        
    # 5. Two rings
    for i in range(5):
        ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        ring2_angles = ring1_angles + np.pi/7
        ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles), np.zeros(7)])
        ring2 = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles), np.full(7, 0.5)])
        pts = np.vstack([ring1, ring2])
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # Optimization Loop
    
    # Phase 1: Refine initial candidates with variance of log distances objective
    # This is a smoother objective than the ratio
    for i in range(len(candidates)):
        if time.time() - start_time > timeout - 30:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_epigraph, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 2: Differential Evolution with the variance objective
    bounds = [(-1.0, 1.0)] * (N * D)
    
    try:
        de_result = differential_evolution(
            objective_epigraph, 
            bounds, 
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-4,
            init='sobol',
            polish=False
        )
        
        if de_result.x is not None:
            pts = de_result.x.reshape(N, D)
            update_best(pts)
            
            # Refine DE result with the soft ratio objective (higher precision)
            for tau in [10.0, 50.0, 200.0]:
                if time.time() - start_time > timeout - 10:
                    break
                try:
                    res = minimize(
                        objective_smooth, 
                        de_result.x, 
                        method='L-BFGS-B', 
                        args=(tau,),
                        options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7}
                    )
                    if res.x is not None:
                        pts = res.x.reshape(N, D)
                        update_best(pts)
                        de_result.x = res.x
                except Exception:
                    pass
    except Exception:
        pass

    # Phase 3: Basin Hopping on the best candidates
    # Use the best points found so far as the starting point for basin hopping
    if best_points is not None:
        x0 = best_points.flatten()
        
        # Define a step function for basin hopping
        def step(x):
            x = x.copy()
            x += 0.1 * rng.normal(size=x.shape)
            return x
        
        # Use the soft ratio objective for basin hopping
        def bh_objective(x):
            return objective_smooth(x, tau=50.0)
        
        try:
            bh_result = basinhopping(
                bh_objective, 
                x0, 
                stepfunc=step, 
                niter=20, 
                minenergy_iters=5,
                T=1.0,
                seed=42
            )
            if bh_result.x is not None:
                pts = bh_result.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass

    # Phase 4: Final polish with high tau
    if best_points is not None:
        x0 = best_points.flatten()
        for tau in [100.0, 500.0, 1000.0]:
            if time.time() - start_time > timeout - 5:
                break
            try:
                res = minimize(
                    objective_smooth, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    x0 = res.x
            except Exception:
                pass

    if best_points is None:
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
