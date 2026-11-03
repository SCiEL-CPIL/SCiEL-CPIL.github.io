"""Rewritten program: hexagonal lattice init + epigraph L-BFGS-B optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
DEADLINE = 140.0  # seconds, safety margin before 150s limit


def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return N closest to center."""
    best_points = None
    for spacing in [0.10, 0.102, 0.104, 0.106, 0.108, 0.11, 0.112, 0.114]:
        k = int(1.0 / spacing) + 4
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 0.98:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind='stable')[:N]
            pts = points[idx]
            if best_points is None:
                best_points = pts
    
    if best_points is None:
        best_points = np.random.randn(N, 2) * 0.5
    
    return best_points[:N]


def _compute_radius(points):
    """Compute the minimum radius: min of half min distance and min distance to boundary."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(1.0 - radii)
    return min(min_dist / 2.0, min_boundary)


def _epigraph_objective(x, soft_beta=50.0):
    """
    Epigraph formulation: minimize -t + penalty for constraints violated.
    x = [x_1, y_1, ..., x_N, y_N, t]
    Constraints: |c_i - c_j| >= 2t for all i<j, and |c_i| + t <= 1
    We use a soft penalty: for each constraint g(x) <= 0, add soft_beta * max(0, g(x))^2
    """
    n = N
    pts = x[:2*n].reshape(n, 2)
    t = x[2*n]
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Constraint violations: dists[i,j] - 2t >= 0
    pair_viol = 2*t - dists  # positive when violated
    
    # Boundary violations: |c_i| + t - 1 >= 0
    radii = np.hypot(pts[:, 0], pts[:, 1])
    bound_viol = radii + t - 1.0  # positive when violated
    
    # Soft penalty using smooth max(0, x)^2 approximation
    # Use: soft_max(x) = (x + soft_beta*x^2/2) / (1 + soft_beta*x) for x>0, 0 otherwise
    # Simpler: use sigmoid-like: (x + sqrt(x^2 + eps))/2 squared
    eps = 1e-6
    soft_pair = (pair_viol + np.sqrt(pair_viol**2 + eps)) / 2.0
    soft_bound = (bound_viol + np.sqrt(bound_viol**2 + eps)) / 2.0
    
    penalty = np.sum(soft_pair**2) + np.sum(soft_bound**2)
    
    # Objective: minimize -t + penalty
    return -t + penalty


def _epigraph_gradient(x, soft_beta=50.0):
    """Gradient of the epigraph objective."""
    n = N
    pts = x[:2*n].reshape(n, 2)
    t = x[2*n]
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    pair_viol = 2*t - dists
    radii = np.hypot(pts[:, 0], pts[:, 1])
    bound_viol = radii + t - 1.0
    
    eps = 1e-6
    # Gradient of soft_max(x)^2 = 2*soft_max(x) * sigmoid(x)
    # soft_max(x) = (x + sqrt(x^2+eps))/2
    # d(soft_max)/dx = (1 + x/sqrt(x^2+eps))/2 = (sqrt(x^2+eps) + x)/(2*sqrt(x^2+eps))
    
    sqrt_pair = np.sqrt(pair_viol**2 + eps)
    soft_pair_val = (pair_viol + sqrt_pair) / 2.0
    sig_pair = (sqrt_pair + pair_viol) / (2.0 * sqrt_pair)
    
    sqrt_bound = np.sqrt(bound_viol**2 + eps)
    soft_bound_val = (bound_viol + sqrt_bound) / 2.0
    sig_bound = (sqrt_bound + bound_viol) / (2.0 * sqrt_bound)
    
    grad = np.zeros_like(x)
    
    # Gradient w.r.t. t: -1 + 2*sum(soft_pair * sig_pair) + 2*sum(soft_bound * sig_bound)
    # But pair_viol depends on t: d(pair_viol)/dt = 2
    # So d(soft_pair^2)/dt = 2*soft_pair * sig_pair * 2 = 4*soft_pair*sig_pair
    # And d(soft_bound^2)/dt = 2*soft_bound * sig_bound * 1 = 2*soft_bound*sig_bound
    d_obj_dt = -1.0 + 4.0 * np.sum(soft_pair_val * sig_pair) + 2.0 * np.sum(soft_bound_val * sig_bound)
    grad[2*n] = d_obj_dt
    
    # Gradient w.r.t. points
    # d(soft_pair^2)/d(pts_i) involves d(dists)/d(pts_i)
    # d(dists[i,j])/d(pts_i) = (pts_i - pts_j) / dists[i,j]
    # d(pair_viol)/d(pts_i) = -d(dists[i,j])/d(pts_i) for j != i
    
    for i in range(n):
        # Sum over all j != i
        for j in range(n):
            if i == j:
                continue
            d_ij = dists[i, j]
            if d_ij < 1e-10:
                continue
            dir_ij = (pts[i] - pts[j]) / d_ij
            # d(soft_pair^2)/d(pts_i) = 2*soft_pair * sig_pair * (-dir_ij)
            grad[2*i] += -2.0 * soft_pair_val[i, j] * sig_pair[i, j] * dir_ij[0]
            grad[2*i+1] += -2.0 * soft_pair_val[i, j] * sig_pair[i, j] * dir_ij[1]
        
        # Boundary gradient
        r_i = radii[i]
        if r_i > 1e-10:
            dir_bound = pts[i] / r_i
            grad[2*i] += 2.0 * soft_bound_val[i] * sig_bound[i] * dir_bound[0]
            grad[2*i+1] += 2.0 * soft_bound_val[i] * sig_bound[i] * dir_bound[1]
    
    return grad


def _lbfgs_optimize(points, max_time=60.0):
    """Use L-BFGS-B to optimize the epigraph formulation."""
    n = N
    start_time = time.time()
    
    # Initial t estimate
    r_est = _compute_radius(points)
    t0 = max(r_est * 0.95, 0.01)
    
    x0 = np.concatenate([points.flatten(), [t0]])
    
    best_points = points.copy()
    best_r = r_est
    
    # Run L-BFGS-B with time limit
    def objective_and_grad(x):
        if time.time() - start_time > max_time:
            raise TimeoutError()
        val = _epigraph_objective(x)
        grad = _epigraph_gradient(x)
        return val, grad
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': 2000,
                'maxfun': 5000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50
            }
        )
        
        if result.success or result.fun < 0:
            x_opt = result.x
            pts_opt = x_opt[:2*n].reshape(n, 2)
            r_opt = _compute_radius(pts_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_points = pts_opt.copy()
    except (TimeoutError, Exception) as e:
        pass
    
    return best_points, best_r


def construct():
    """Construct 300 circle centers in the unit disk to maximize minimum radius."""
    start_time = time.time()
    
    # Step 1: Initialize with hexagonal lattice
    points = _hex_lattice_init()
    
    # Step 2: L-BFGS-B optimization on epigraph formulation
    remaining = DEADLINE - (time.time() - start_time) - 10.0
    if remaining > 5.0:
        points, best_r = _lbfgs_optimize(points, max_time=remaining)
    else:
        best_r = _compute_radius(points)
    
    # Step 3: Final verification and cleanup
    radii = np.hypot(points[:, 0], points[:, 1])
    mask = radii > 0.9999
    if np.any(mask):
        points[mask] *= 0.9999 / radii[mask]
    
    # Ensure exactly N points
    if len(points) != N:
        if len(points) < N:
            while len(points) < N:
                p = np.random.randn(2) * 0.5
                if np.hypot(p[0], p[1]) < 0.9:
                    points = np.vstack([points, p])
        else:
            points = points[:N]
    
    return points[:N]
