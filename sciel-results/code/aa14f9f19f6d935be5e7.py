import numpy as np
import time
from scipy.optimize import minimize

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
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def normalize_points(points):
    """
    Center and normalize points to unit mean squared norm.
    """
    points = points - np.mean(points, axis=0)
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

def objective_epigraph(x, t, tau=100.0):
    """
    Epigraph formulation: minimize t subject to all pairwise sq dists in [1/t, t].
    We minimize t + penalty for violations.
    Actually, let's use a direct formulation:
    Minimize t such that 1/t <= d_ij^2 <= t for all i<j.
    This is equivalent to minimizing max(t, max(d_ij^2)/min(d_ij^2))... no.
    
    Let's use the formulation:
    Minimize t
    s.t. d_ij^2 >= 1/t
         d_ij^2 <= t
    
    Since t > 0, this is equivalent to minimizing t where t >= max(d_ij^2) and t <= 1/min(d_ij^2).
    Wait, if d_ij^2 <= t and d_ij^2 >= 1/t, then 1/t <= min(d_ij^2) <= max(d_ij^2) <= t.
    So t^2 >= max/min, i.e. t >= sqrt(ratio).
    Minimizing t is equivalent to minimizing sqrt(ratio), which is equivalent to minimizing ratio.
    
    We can solve this with SLSQP or just use a penalty method.
    Let's use a smooth penalty:
    Objective: t + sum(max(1/t - d_ij^2, 0)^2 * lambda) + sum(max(d_ij^2 - t, 0)^2 * lambda)
    But t is a variable. The input x should include t?
    The current interface expects x of length N*D.
    We can add t as an extra variable, making length N*D + 1.
    """
    # x[:N*D] are points, x[N*D] is t
    pts = x[:N*D].reshape(N, D)
    t = x[N*D]
    
    if t < 1e-6:
        t = 1e-6
    
    pts = normalize_points(pts)
    sq_dists = get_pairwise_sq_dists(pts)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Penalty for d^2 < 1/t
    viol_low = np.maximum(1.0/t - sq_dists, 0)**2
    # Penalty for d^2 > t
    viol_high = np.maximum(sq_dists - t, 0)**2
    
    lam = 100.0
    return t + lam * (np.sum(viol_low) + np.sum(viol_high))

def objective_direct(x):
    """
    Direct non-smooth objective for SLSQP or other methods that can handle it.
    Minimizes log(max) - log(min) or just ratio.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return 1e10
    return np.log(np.max(sq_dists)) - np.log(np.min(sq_dists))

def fibonacci_sphere(n, seed=42):
    """Generate n points on a sphere using Fibonacci lattice."""
    rng = np.random.default_rng(seed)
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = np.zeros((n, D))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(max(0, 1 - z**2))
        theta = golden_angle * i + rng.uniform(0, 2*np.pi) * 0.01
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
        points[i, 2] = z
    return points

def construct():
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        pts = normalize_points(pts)
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Fibonacci sphere
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # 2. Random points on sphere
    for _ in range(10):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 3. Icosahedron + 2 random points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Symmetric candidate: Ico + Poles
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym.flatten())
    
    for _ in range(5):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())

    # Optimization Loop
    n_restarts = len(candidates)
    
    # Phase 1: Smooth Log-Ratio Annealing
    taus = [5.0, 20.0, 50.0, 100.0, 500.0, 2000.0]
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 20:
            break
            
        x0 = candidates[i].copy()
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            current_x = x0
            
            # Phase 1: Standard smooth optimization
            for tau in taus:
                if time.time() - start_time > timeout - 20:
                    break
                
                res = minimize(
                    objective_smooth, 
                    current_x, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.x is not None:
                    current_x = res.x
                    pts_current = current_x.reshape(N, D)
                    update_best(pts_current)
            
            # Phase 2: Epigraph refinement with SLSQP
            if time.time() - start_time < timeout - 10:
                pts_current = current_x.reshape(N, D)
                pts_current = normalize_points(pts_current)
                sq_dists = get_pairwise_sq_dists(pts_current)
                
                # Initial t estimate: sqrt(max/min)
                t0 = np.sqrt(np.max(sq_dists) / np.min(sq_dists))
                
                x_epigraph = np.concatenate([pts_current.flatten(), [t0]])
                
                # Bounds for t
                bounds = [(None, None)] * (N * D) + [(1.0, 10.0)]
                
                res = minimize(
                    objective_epigraph, 
                    x_epigraph, 
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                
                if res.x is not None:
                    pts_epigraph = res.x[:N*D].reshape(N, D)
                    update_best(pts_epigraph)
            
            # Phase 3: Direct non-smooth polish with SLSQP
            if time.time() - start_time < timeout - 5:
                pts_current = current_x.reshape(N, D)
                pts_current = normalize_points(pts_current)
                
                res = minimize(
                    objective_direct, 
                    pts_current.flatten(), 
                    method='SLSQP',
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                
                if res.x is not None:
                    pts_direct = res.x.reshape(N, D)
                    update_best(pts_direct)

        except Exception:
            pass
    
    # Final high-precision polish pass on best point
    if best_points is not None and time.time() - start_time < timeout - 5:
        try:
            x_polish = best_points.flatten()
            # Use very high tau for precise max/min
            res = minimize(
                objective_smooth, 
                x_polish, 
                method='L-BFGS-B', 
                args=(5000.0,),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts_polish = res.x.reshape(N, D)
                update_best(pts_polish)
                
            # Another SLSQP pass on the result
            if time.time() - start_time < timeout - 2:
                res = minimize(
                    objective_direct, 
                    best_points.flatten(), 
                    method='SLSQP',
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res.x is not None:
                    pts_polish2 = res.x.reshape(N, D)
                    update_best(pts_polish2)
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
