import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3
PAIRS = N * (N - 1) // 2

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
    Center points to remove translational invariance.
    """
    points = points - np.mean(points, axis=0)
    return points

def _pairwise_grad(points):
    """
    Compute gradient of pairwise squared distances w.r.t. points.
    Returns array of shape (N, D, PAIRS) where each slice is grad w.r.t. one point.
    """
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    
    diff = points[i_idx] - points[j_idx]  # (PAIRS, D)
    
    grad = np.zeros((N, D, PAIRS))
    for k in range(PAIRS):
        i = i_idx[k]
        j = j_idx[k]
        d = 2.0 * diff[k]
        grad[i, :, k] = d
        grad[j, :, k] = -d
    
    return grad

def objective_and_grad(x, tau=50.0):
    """
    Smooth log-ratio objective with analytical gradient.
    Minimizes log(soft_max) - log(soft_min) using log-sum-exp.
    """
    points = x.reshape(N, D).copy()
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-10)
    
    # Soft max via log-sum-exp
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    sum_exp_max = np.sum(exp_max)
    log_sum_exp_max = m_max + np.log(sum_exp_max) / tau
    
    # Soft min via log-sum-exp
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    sum_exp_min = np.sum(exp_min)
    log_sum_exp_min = m_min - np.log(sum_exp_min) / tau
    
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    # Gradient computation
    w_max = exp_max / sum_exp_max
    w_min = exp_min / sum_exp_min
    
    grad_sq = _pairwise_grad(points)
    
    grad_max = (1.0 / tau) * np.einsum('k,ndk->nd', w_max, grad_sq)
    grad_min = -(1.0 / tau) * np.einsum('k,ndk->nd', w_min, grad_sq)
    
    grad_obj = (1.0 / log_sum_exp_max) * grad_max - (1.0 / log_sum_exp_min) * grad_min
    
    grad_flat = grad_obj.reshape(-1)
    
    return obj_val, grad_flat

def _fibonacci_sphere(n, rng):
    """Generate n points on a sphere using Fibonacci spiral."""
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    pts = np.column_stack([x, y, z])
    return pts

def _icosahedron():
    """Return 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    return ico

def _cuboctahedron():
    """Return 12 vertices of a cuboctahedron."""
    perms = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
    ])
    return perms / np.linalg.norm(perms, axis=1, keepdims=True)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
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
    
    def time_ok():
        return time.time() < deadline - 2.0
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Fibonacci sphere
    pts_fib = _fibonacci_sphere(N, rng)
    candidates.append(pts_fib.flatten())
    
    # 2. Icosahedron + 2 random
    ico = _icosahedron()
    for _ in range(5):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())
    
    # 3. Cuboctahedron + 2 random
    cubo = _cuboctahedron()
    for _ in range(5):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([cubo, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())
    
    # 4. Random points on sphere
    for _ in range(10):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
    
    # 5. Random points in ball
    for _ in range(5):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        pts = pts / norms * rng.uniform(0.3, 1.0, size=(N, 1))
        candidates.append(pts.flatten())
    
    n_restarts = len(candidates)
    tau_schedule = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    
    for i in range(n_restarts):
        if not time_ok():
            break
        
        x0 = candidates[i].copy().reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            current_x = x0
            
            for tau in tau_schedule:
                if not time_ok():
                    break
                
                res = minimize(
                    objective_and_grad,
                    current_x,
                    args=(tau,),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 500}
                )
                
                if res.x is not None and np.all(np.isfinite(res.x)):
                    current_x = res.x
                    pts_current = current_x.reshape(N, D)
                    update_best(pts_current)
                    
        except Exception:
            continue
    
    # Final high-precision polish
    if best_points is not None and time_ok():
        try:
            x_polish = best_points.flatten()
            for tau in [5000.0, 20000.0]:
                if not time_ok():
                    break
                res = minimize(
                    objective_and_grad,
                    x_polish,
                    args=(tau,),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 1000}
                )
                if res.x is not None and np.all(np.isfinite(res.x)):
                    x_polish = res.x
                    pts_polish = x_polish.reshape(N, D)
                    update_best(pts_polish)
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
