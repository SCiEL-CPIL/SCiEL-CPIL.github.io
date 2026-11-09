import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590

    def score(pts):
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def coulomb_obj_and_grad(x):
        """Coulomb repulsion: minimize sum of 1/d_i^2 (smooth, well-spread)"""
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        norms = np.linalg.norm(pts, axis=1)
        max_norm = np.max(norms)
        if max_norm < 1e-12:
            return np.inf, np.zeros_like(x)
        pts = pts / max_norm
        
        dists = pdist(pts)
        # Use squared distances for numerical stability
        d2 = dists ** 2
        d2 = np.maximum(d2, 1e-10)
        
        val = np.sum(1.0 / d2)
        
        # Gradient: d/dx_i sum_j 1/|x_i - x_j|^2 = sum_j -2*(x_i - x_j)/|x_i - x_j|^4
        grad = np.zeros((n, d))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                d2_val = d2[idx]
                coeff = -2.0 / (d2_val ** 2)
                grad[i] += coeff * diff
                grad[j] -= coeff * diff
                idx += 1
        
        return val, grad.flatten()

    def objective_and_grad(x, beta):
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        norms = np.linalg.norm(pts, axis=1)
        max_norm = np.max(norms)
        if max_norm < 1e-12:
            return np.inf, np.zeros_like(x)
        pts = pts / max_norm
        
        dists = pdist(pts)
        
        # Soft min and max via log-sum-exp
        neg_d = -beta * dists
        max_neg_d = np.max(neg_d)
        log_sum_exp_neg = max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d)))
        soft_min = -log_sum_exp_neg / beta
        
        pos_d = beta * dists
        max_pos_d = np.max(pos_d)
        log_sum_exp_pos = max_pos_d + np.log(np.sum(np.exp(pos_d - max_pos_d)))
        soft_max = log_sum_exp_pos / beta
        
        if soft_max < 1e-12:
            return np.inf, np.zeros_like(x)
            
        val = -soft_min / soft_max
        
        # Weights for soft min and soft max
        exp_neg = np.exp(neg_d - max_neg_d)
        sum_exp_neg = np.sum(exp_neg)
        w_min = exp_neg / sum_exp_neg
        
        exp_pos = np.exp(pos_d - max_pos_d)
        sum_exp_pos = np.sum(exp_pos)
        w_max = exp_pos / sum_exp_pos
        
        # Gradient of R = soft_min / soft_max
        # dR/dx = sum_i [ (w_min_i * soft_max - w_max_i * soft_min) / soft_max^2 ] * d(d_i)/dx
        
        grad = np.zeros((n, d))
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                w_m = w_min[idx]
                w_x = w_max[idx]
                dist = dists[idx]
                if dist < 1e-12:
                    idx += 1
                    continue
                diff = pts[i] - pts[j]
                coeff = (w_m * soft_max - w_x * soft_min) / (soft_max * soft_max)
                
                grad[i] += coeff * diff / dist
                grad[j] -= coeff * diff / dist
                idx += 1
        
        # Objective is -R, so gradient is -grad_R
        grad = -grad.flatten()
        
        return val, grad

    def make_objective(beta):
        def obj(x):
            val, _ = objective_and_grad(x, beta)
            return val
        return obj

    def make_grad(beta):
        def grad(x):
            _, g = objective_and_grad(x, beta)
            return g
        return grad

    def make_coulomb_obj():
        def obj(x):
            val, _ = coulomb_obj_and_grad(x)
            return val
        return obj

    def make_coulomb_grad():
        def grad(x):
            _, g = coulomb_obj_and_grad(x)
            return g
        return grad

    best_points = None
    best_score = -np.inf

    # Generate initial configurations
    initial_configs = []
    
    # 1. Icosahedron + 2 points on axis
    phi = (1 + np.sqrt(5)) / 2
    ico12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico12 = ico12 / np.linalg.norm(ico12, axis=1, keepdims=True)
    
    cfg1 = np.vstack([ico12, np.array([[0, 0, 1.2], [0, 0, -1.2]])])
    initial_configs.append(cfg1)
    
    # 2. Random perturbations of icosahedron
    rng = np.random.RandomState(42)
    for i in range(20):
        pert = ico12 + rng.randn(12, 3) * 0.1
        extra = rng.randn(2, 3) * 0.5
        cfg = np.vstack([pert, extra])
        initial_configs.append(cfg)
        
    # 3. Fibonacci sphere
    def fibonacci_sphere(n_pts):
        indices = np.arange(0, n_pts, dtype=float) + 0.5
        phi_f = np.arccos(1 - 2 * indices / n_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi_f)
        y = np.sin(theta) * np.sin(phi_f)
        z = np.cos(phi_f)
        return np.column_stack([x, y, z])

    fib14 = fibonacci_sphere(14)
    initial_configs.append(fib14)
    
    # 4. Random uniform on sphere
    for i in range(8):
        pts = rng.randn(n, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        initial_configs.append(pts)

    # Optimization strategy: 
    # 1. Coulomb repulsion to get well-spread initial config
    # 2. Multi-stage L-BFGS-B with increasing beta (soft min/max)
    # 3. Polish with Nelder-Mead on the exact objective
    
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    coulomb_obj = make_coulomb_obj()
    coulomb_grad = make_coulomb_grad()
    
    for i, init in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        current_pts = init.copy()
        
        # Stage 1: Coulomb repulsion optimization (smooth, helps spread points)
        if time.time() < deadline - 25:
            try:
                result_coulomb = minimize(
                    coulomb_obj,
                    current_pts.flatten(),
                    jac=coulomb_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_pts = result_coulomb.x.reshape((n, d))
            except Exception:
                pass
        
        # Stage 2: Multi-stage optimization with increasing beta
        for beta in betas:
            if time.time() > deadline - 15:
                break
            
            init_flat = current_pts.flatten()
            
            try:
                obj_func = make_objective(beta)
                grad_func = make_grad(beta)
                
                result = minimize(
                    obj_func,
                    init_flat,
                    jac=grad_func,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                current_pts = result.x.reshape((n, d))
                
            except Exception:
                pass
        
        # Evaluate and update best
        s = score(current_pts)
        if s > best_score:
            best_score = s
            best_points = current_pts.copy()
        
        # Stage 3: Polish with Nelder-Mead on exact objective
        if time.time() < deadline - 5:
            def obj_nm(x):
                pts_nm = x.reshape((n, d))
                return -score(pts_nm)
            
            try:
                result_nm = minimize(
                    obj_nm,
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                pts_nm = result_nm.x.reshape((n, d))
                s_nm = score(pts_nm)
                if s_nm > best_score:
                    best_score = s_nm
                    best_points = pts_nm.copy()
                    
            except Exception:
                pass

    if best_points is None:
        best_points = initial_configs[0]
        
    # Final normalization
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100

    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))

    return best_points
