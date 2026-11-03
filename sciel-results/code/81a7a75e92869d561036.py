import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(points):
    """Calculate the packing radius for a set of points."""
    if len(points) != N:
        return -np.inf
    if not np.all(np.isfinite(points)):
        return -np.inf
    
    # Distance to boundary
    radii = np.linalg.norm(points, axis=1)
    boundary_dist = 1.0 - radii
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair_dist)

def _hexagonal_lattice_init():
    """Generate a hexagonal lattice initialization."""
    best_points = None
    best_score = -np.inf
    
    for spacing in np.linspace(0.20, 0.30, 20):
        points = []
        max_k = int(1.0 / spacing) + 2
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = i * spacing + (j % 2) * spacing / 2.0
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            continue
        
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists, kind='stable')[:N]
        points = points[indices]
        
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points
            
    if best_points is None:
        best_points = np.random.uniform(-0.8, 0.8, (N, 2))
        
    return best_points

def _log_barrier_optimize(points0, r0, mu_init=0.01, max_iter=500, time_budget=60.0):
    """
    Optimize using log-barrier formulation where r is an explicit variable.
    
    Minimize: -r + mu * [sum_i log(1/r - (1-|c_i|)) + sum_{i<j} log(1/r - |c_i-c_j|/2)]
    Subject to: 1/r - (1-|c_i|) > 0 and 1/r - |c_i-c_j|/2 > 0
    
    This is equivalent to maximizing r subject to all constraints, with barrier penalty.
    """
    start_time = time.time()
    
    # Precompute pair indices
    pair_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    n_pairs = len(pair_indices)
    
    def objective_and_grad(x):
        """
        x = [r, x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}]
        """
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return 1e10, np.zeros_like(x)
        
        r = x[0]
        if r <= 1e-6:
            return 1e10, np.zeros_like(x)
        
        points = x[1:].reshape(-1, 2)
        
        # Boundary violations: 1/r - (1 - |c_i|) must be > 0
        radii = np.linalg.norm(points, axis=1)
        boundary_slacks = 1.0 / r - (1.0 - radii)
        
        # Pairwise violations: 1/r - |c_i - c_j| / 2 must be > 0
        i_idx, j_idx = pair_indices[:, 0], pair_indices[:, 1]
        diff = points[i_idx] - points[j_idx]
        pair_dists = np.sqrt(np.sum(diff**2, axis=1))
        pair_slacks = 1.0 / r - pair_dists / 2.0
        
        # If any slack is non-positive, we're infeasible
        if np.any(boundary_slacks <= 0) or np.any(pair_slacks <= 0):
            # Heavy penalty
            min_slack = min(np.min(boundary_slacks), np.min(pair_slacks))
            return 1e6 - min_slack, np.zeros_like(x)
        
        # Log barrier objective: -r + mu * (sum log(boundary_slacks) + sum log(pair_slacks))
        mu = mu_init * (0.99 ** int(elapsed * 10))  # Decay mu over time
        
        log_barrier = np.sum(np.log(boundary_slacks)) + np.sum(np.log(pair_slacks))
        obj = -r + mu * log_barrier
        
        # Gradient
        grad = np.zeros_like(x)
        
        # d(-r)/dr = -1
        # d(mu * log(s))/dr = mu * (1/s) * ds/dr
        # s_boundary_i = 1/r - (1 - |c_i|)
        # ds_boundary_i/dr = -1/r^2
        # s_pair_ij = 1/r - |c_i - c_j|/2
        # ds_pair_ij/dr = -1/r^2
        
        # d(mu * log(s))/dr = mu * (1/s) * (-1/r^2)
        # = -mu / (r^2 * s)
        
        grad_r = -1.0 - mu * (np.sum(1.0 / (r**2 * boundary_slacks)) + np.sum(1.0 / (r**2 * pair_slacks)))
        grad[0] = grad_r
        
        # For points:
        # d(s_boundary_i)/dx_i = d(1/r - (1 - |c_i|))/dx_i = d|c_i|/dx_i = c_i / |c_i|
        # d(s_pair_ij)/dx_i = -d|c_i - c_j|/dx_i / 2 = -(c_i - c_j) / (2 * |c_i - c_j|)
        # d(s_pair_ij)/dx_j = -(c_j - c_i) / (2 * |c_i - c_j|)
        
        # d(mu * log(s))/dx = mu * (1/s) * ds/dx
        
        # Boundary part
        unit_vecs = np.zeros_like(points)
        mask = radii > 1e-8
        unit_vecs[mask] = points[mask] / radii[mask][:, np.newaxis]
        
        # d(s_boundary_i)/dx_i = unit_vecs[i]
        # d(mu * log(s_boundary_i))/dx_i = mu / s_boundary_i * unit_vecs[i]
        grad_points = np.zeros_like(points)
        grad_points += mu * (1.0 / boundary_slacks)[:, np.newaxis] * unit_vecs
        
        # Pairwise part
        for k in range(n_pairs):
            i, j = pair_indices[k]
            s = pair_slacks[k]
            d = pair_dists[k]
            if d < 1e-8:
                continue
            
            diff_ij = points[i] - points[j]
            # d(s_pair_ij)/dx_i = -diff_ij / (2 * d)
            # d(s_pair_ij)/dx_j = -diff_ij / (2 * d) (since c_j - c_i = -diff_ij, so -(c_j-c_i)/(2d) = diff_ij/(2d))
            # Wait: s_pair_ij = 1/r - |c_i - c_j|/2
            # ds_pair_ij/dx_i = -d|c_i - c_j|/dx_i / 2 = -(c_i - c_j) / (2 * |c_i - c_j|) = -diff_ij / (2*d)
            # ds_pair_ij/dx_j = -d|c_i - c_j|/dx_j / 2 = -(c_j - c_i) / (2 * |c_i - c_j|) = diff_ij / (2*d)
            
            grad_i = -diff_ij / (2.0 * d)
            grad_j = diff_ij / (2.0 * d)
            
            grad_points[i] += mu / s * grad_i
            grad_points[j] += mu / s * grad_j
        
        grad[1:] = grad_points.ravel()
        
        return obj, grad
    
    # Initial x: [r, points...]
    x0 = np.concatenate([[r0], points0.ravel()])
    
    # Bounds: r in (0, 1], points in [-1.1, 1.1]
    bounds = [(1e-6, 1.0)] + [(-1.1, 1.1)] * (2 * N)
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
        
        r_opt = res.x[0]
        points_opt = res.x[1:].reshape(-1, 2)
        score = _score(points_opt)
        
        return points_opt, score, res
    except Exception as e:
        return points0, _score(points0), None

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    # 1. Initialize with hex lattice
    points0 = _hexagonal_lattice_init()
    best_points = points0
    best_score = _score(points0)
    
    # Initial r estimate
    r0 = best_score * 0.95  # Start slightly below current best
    
    # 2. Log-barrier optimization
    try:
        points_opt, score, res = _log_barrier_optimize(points0, r0, mu_init=0.05, max_iter=800, time_budget=45.0)
        if score > best_score:
            best_score = score
            best_points = points_opt
    except Exception:
        pass
    
    # 3. If time permits, refine with smaller mu
    if time.time() < deadline - 15.0:
        r1 = best_score * 0.98
        try:
            points_opt2, score2, res2 = _log_barrier_optimize(best_points, r1, mu_init=0.01, max_iter=600, time_budget=30.0)
            if score2 > best_score:
                best_score = score2
                best_points = points_opt2
        except Exception:
            pass
    
    # 4. Final polish if time permits
    if time.time() < deadline - 5.0:
        r2 = best_score * 0.99
        try:
            points_opt3, score3, res3 = _log_barrier_optimize(best_points, r2, mu_init=0.001, max_iter=400, time_budget=10.0)
            if score3 > best_score:
                best_score = score3
                best_points = points_opt3
        except Exception:
            pass
    
    # Ensure validity
    if not np.all(np.isfinite(best_points)) or len(best_points) != N:
        best_points = _hexagonal_lattice_init()
        
    return best_points

# EVOLVE-BLOCK-END
