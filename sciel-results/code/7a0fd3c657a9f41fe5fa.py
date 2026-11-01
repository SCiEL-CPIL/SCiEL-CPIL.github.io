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

def objective_ratio(x, tau_min=50.0, tau_max=50.0):
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau_max * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau_max
    
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau_min * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau_min
    
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    return obj_val

def objective_qcqp(x, d_max, penalty):
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_min = np.min(sq_dists)
    tau_min = 100.0
    exp_min = np.exp(-tau_min * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min)) / tau_min
    
    m_max = np.max(sq_dists)
    tau_max = 100.0
    exp_max = np.exp(tau_max * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max)) / tau_max
    
    violation = max(0.0, soft_max - d_max)
    penalty_term = penalty * violation * violation
    
    obj_val = -soft_min + penalty_term
    return obj_val

def objective_epigraph(x, t, tau=50.0):
    """
    Epigraph formulation: minimize t subject to d_ij <= t for all pairs.
    This is equivalent to minimizing the max squared distance.
    We combine with a term to push min up.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max constraint: sum(exp(tau*(d_ij - t)))
    # We want this to be small, so we add it as a penalty or constraint.
    # Actually, let's just minimize t + penalty * (soft_max - t)^2 + lambda * (-soft_min)
    # Wait, the epigraph formulation usually introduces an auxiliary variable.
    # Here t is the auxiliary variable representing the max distance.
    # We want to minimize t - soft_min? No, we want to minimize the ratio.
    # Let's stick to the ratio but use a different smooth approximation.
    
    # Actually, let's try a different strategy:
    # Minimize log(soft_max) - log(soft_min) is the current approach.
    # Let's try minimizing soft_max / soft_min directly? No, that's harder.
    
    # Let's implement a proper epigraph for the ratio.
    # Minimize r s.t. d_max <= r * d_min
    # This is non-convex.
    
    # Instead, let's use the fact that minimizing log(r) is minimizing log(d_max) - log(d_min).
    # We already do that.
    
    # Let's try a different smooth min/max.
    # Power mean: M_p(x) = (sum x^p / n)^(1/p)
    # For p -> inf, M_p -> max. For p -> -inf, M_p -> min.
    
    p = 50.0
    mean_max = (np.mean(sq_dists**p))**(1.0/p)
    mean_min = (np.mean(sq_dists**(-p)))**(-1.0/p)
    
    obj = np.log(mean_max) - np.log(mean_min)
    return obj

def construct():
    start_time = time.time()
    timeout = 145.0
    
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
    
    # 1. Generate Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # 2. Initial candidates
    candidates = []
    
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym.flatten())
    
    for i in range(10):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())
        
    for i in range(10):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    for i in range(5):
        pts = rng.normal(size=(N, D))
        candidates.append(pts.flatten())
        
    def fibonacci_sphere(n, seed=42):
        rng_fib = np.random.default_rng(seed)
        golden_angle = np.pi * (3 - np.sqrt(5))
        points = np.zeros((n, D))
        for i in range(n):
            z = 1 - 2 * i / (n - 1)
            r = np.sqrt(max(0, 1 - z**2))
            theta = golden_angle * i + rng_fib.uniform(0, 2*np.pi) * 0.01
            points[i, 0] = r * np.cos(theta)
            points[i, 1] = r * np.sin(theta)
            points[i, 2] = z
        return points
    
    fib_pts = fibonacci_sphere(N, seed=42)
    candidates.append(fib_pts.flatten())
    
    inner = fibonacci_sphere(8, seed=100) * 0.7
    outer = fibonacci_sphere(6, seed=200) * 1.0
    shell_pts = np.vstack([inner, outer])
    shell_pts = shell_pts / np.linalg.norm(shell_pts, axis=1, keepdims=True)
    candidates.append(shell_pts.flatten())

    # 3. Optimization Loop
    taus_ratio = [10.0, 50.0, 200.0, 1000.0]
    
    max_restarts = 100
    
    for i in range(max_restarts):
        if time.time() - start_time > timeout - 25:
            break
            
        if i < len(candidates):
            x0 = candidates[i].copy()
        else:
            if rng.random() > 0.5:
                pts = rng.normal(size=(N, D))
                pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            else:
                pts = rng.normal(size=(N, D))
            x0 = pts.flatten()
            
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            current_x = x0
            
            # Phase 1: Ratio Annealing
            for tau in taus_ratio:
                if time.time() - start_time > timeout - 20:
                    break
                
                res = minimize(
                    objective_ratio, 
                    current_x, 
                    method='L-BFGS-B', 
                    args=(tau, tau),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.x is not None:
                    current_x = res.x
                    pts_current = current_x.reshape(N, D)
                    update_best(pts_current)
                    
            # Phase 2: QCQP Refinement
            pts_current = current_x.reshape(N, D)
            pts_current = normalize_points(pts_current)
            sq_dists = get_pairwise_sq_dists(pts_current)
            d_max = np.max(sq_dists)
            
            d_max_curr = np.max(sq_dists)
            
            penalties = [10.0, 100.0, 1000.0]
            
            for pen in penalties:
                if time.time() - start_time > timeout - 15:
                    break
                
                res_qcqp = minimize(
                    objective_qcqp, 
                    current_x, 
                    method='L-BFGS-B', 
                    args=(d_max_curr, pen),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res_qcqp.x is not None:
                    current_x = res_qcqp.x
                    pts_qcqp = current_x.reshape(N, D)
                    pts_qcqp = normalize_points(pts_qcqp)
                    update_best(pts_qcqp)
                    
                    sq_dists_new = get_pairwise_sq_dists(pts_qcqp)
                    d_max_curr = np.max(sq_dists_new)
                    
        except Exception:
            pass
    
    # 4. Final high-precision polish pass on best point
    if best_points is not None and time.time() - start_time < timeout - 5:
        try:
            x_polish = best_points.flatten()
            res = minimize(
                objective_ratio, 
                x_polish, 
                method='L-BFGS-B', 
                args=(5000.0, 5000.0),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts_polish = res.x.reshape(N, D)
                update_best(pts_polish)
                
                pts_polish = normalize_points(pts_polish)
                sq_dists = get_pairwise_sq_dists(pts_polish)
                d_max_polish = np.max(sq_dists)
                
                res_qcqp_final = minimize(
                    objective_qcqp, 
                    x_polish, 
                    method='L-BFGS-B', 
                    args=(d_max_polish, 10000.0),
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res_qcqp_final.x is not None:
                    pts_final = res_qcqp_final.x.reshape(N, D)
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
