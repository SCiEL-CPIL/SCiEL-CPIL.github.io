"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_radius(points):
    """Compute the achievable common radius for given centers."""
    n = len(points)
    # Min distance between any pair / 2
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
    # Min distance to boundary
    min_boundary = np.min(1.0 - np.linalg.norm(points, axis=1))
    return min(min_pair / 2.0, min_boundary)


def _initial_hex_grid(n):
    """Generate initial hexagonal grid points, select best n."""
    spacing = 2.0 / (np.sqrt(n) + 1.5)
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii <= 1.0 - spacing / 2.0 - 0.001
        points = points[mask]
        radii = radii[mask]
        if len(points) >= n:
            # Select points closest to center
            idx = np.argsort(radii, kind='stable')[:n]
            return points[idx]
        spacing *= 0.97


def _repulsive_force_objective(x, n, repulsion_power=4.0, boundary_weight=100.0):
    """
    Objective function based on repulsive forces.
    Minimizes the sum of inverse distances between all pairs plus boundary penalties.
    This is a smooth objective that encourages spreading points apart.
    """
    points = x.reshape(n, 2)
    
    # Compute all pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diffs**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Repulsive force: sum of 1/d^repulsion_power for all pairs
    # Use a small epsilon to avoid division by zero
    eps = 1e-10
    repulsive = np.sum(1.0 / (dists_sq + eps) ** (repulsion_power / 2.0))
    
    # Boundary penalty: for points close to or outside the boundary
    norms = np.sqrt(np.sum(points**2, axis=1))
    boundary_violations = np.maximum(0.0, norms - 1.0)
    boundary_penalty = boundary_weight * np.sum(boundary_violations**2)
    
    return repulsive + boundary_penalty


def _repulsive_force_gradient(x, n, repulsion_power=4.0, boundary_weight=100.0):
    """Gradient of the repulsive force objective."""
    points = x.reshape(n, 2)
    grad = np.zeros(2 * n)
    
    # Compute all pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diffs**2, axis=2)
    np.fill_diagonal(dists_sq, 0.0)
    
    eps = 1e-10
    repulsion_power_half = repulsion_power / 2.0
    
    # Gradient of repulsive term: -repulsion_power * (diff) / (d^2 + eps)^(repulsion_power/2 + 1)
    # For each pair (i,j), the contribution to grad[i] is -repulsion_power * (points[i] - points[j]) / (d_ij^2 + eps)^(repulsion_power/2 + 1)
    # And to grad[j] is +repulsion_power * (points[i] - points[j]) / (d_ij^2 + eps)^(repulsion_power/2 + 1)
    
    # Vectorized computation
    inv_dists = 1.0 / (dists_sq + eps) ** (repulsion_power_half + 1.0)
    
    # For each point i, sum over all j != i
    # grad[i] = -repulsion_power * sum_j (points[i] - points[j]) * inv_dists[i,j]
    # Note: the factor is repulsion_power, not 2*repulsion_power, because of the chain rule
    
    # Actually, let's be more careful:
    # f = sum_{i<j} 1/d_ij^p where p = repulsion_power
    # df/d(points[i]) = sum_{j != i} -p * (points[i] - points[j]) / d_ij^(p+2)
    
    # Using the vectorized form:
    weighted_diffs = inv_dists * diffs  # shape (n, n, 2)
    grad_repulsive = -repulsion_power * np.sum(weighted_diffs, axis=1)
    
    # Boundary gradient
    norms = np.sqrt(np.sum(points**2, axis=1))
    norms_safe = np.where(norms > 1e-10, norms, 1e-10)
    boundary_grad = np.zeros_like(points)
    mask = norms > 1.0
    boundary_grad[mask] = 2.0 * boundary_weight * (norms[mask] - 1.0) * (points[mask] / norms_safe[mask])
    
    # Combine gradients
    grad[:n] = grad_repulsive[:, 0] + boundary_grad[:, 0]
    grad[n:] = grad_repulsive[:, 1] + boundary_grad[:, 1]
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Phase 1: Initial hexagonal grid
    points = _initial_hex_grid(N)
    best_radius = _compute_radius(points)
    best_points = points.copy()
    
    # Phase 2: Optimize using repulsive force objective with L-BFGS-B
    # This is a smooth, unconstrained optimization that naturally keeps points inside
    # due to the boundary penalty
    
    x0 = best_points.flatten()
    
    # Run L-BFGS-B optimization
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            result = minimize(
                _repulsive_force_objective, x0,
                args=(N,),
                method='L-BFGS-B',
                jac=lambda x: _repulsive_force_gradient(x, N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < _repulsive_force_objective(x0, N):
                new_points = result.x.reshape(N, 2)
                # Check if all points are inside the unit circle
                norms = np.linalg.norm(new_points, axis=1)
                if np.all(norms <= 1.0 + 1e-9):
                    new_radius = _compute_radius(new_points)
                    if new_radius > best_radius:
                        best_points = new_points
                        best_radius = new_radius
        except Exception:
            pass
    
    # Phase 3: SLSQP refinement with the original min-radius objective
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            x0 = best_points.flatten()
            
            def obj(x):
                pts = x.reshape(N, 2)
                norms = np.linalg.norm(pts, axis=1)
                if np.any(norms > 1.0 - 1e-10):
                    return 1e6 + np.sum(norms[norms > 1.0 - 1e-10])
                
                # Compute min pair distance
                min_pair = np.inf
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d < min_pair:
                            min_pair = d
                
                min_boundary = np.min(1.0 - norms)
                radius = min(min_pair / 2.0, min_boundary)
                return -radius
            
            def obj_grad(x):
                pts = x.reshape(N, 2)
                norms = np.linalg.norm(pts, axis=1)
                
                min_pair = np.inf
                min_pair_idx = None
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d < min_pair:
                            min_pair = d
                            min_pair_idx = (i, j)
                
                min_boundary = np.min(1.0 - norms)
                min_boundary_idx = np.argmin(1.0 - norms)
                
                radius = min(min_pair / 2.0, min_boundary)
                
                grad = np.zeros(2 * N)
                if min_pair / 2.0 <= min_boundary:
                    # Gradient w.r.t. the closest pair
                    i, j = min_pair_idx
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    if d > 1e-12:
                        unit = diff / d
                        grad[2*i:2*i+2] = -unit
                        grad[2*j:2*j+2] = unit
                else:
                    # Gradient w.r.t. the point closest to boundary
                    i = min_boundary_idx
                    p = pts[i]
                    norm = np.linalg.norm(p)
                    if norm > 1e-12:
                        unit = p / norm
                        grad[2*i:2*i+2] = -unit
                
                return grad
            
            # Constraints: keep points inside unit circle
            constraints = []
            for i in range(N):
                def make_constraint(i):
                    def con(x):
                        p = x.reshape(N, 2)[i]
                        return 1.0 - np.linalg.norm(p)
                    return con
                constraints.append({'type': 'ineq', 'fun': make_constraint(i)})
            
            # Bounds
            bounds = [(-1.0, 1.0)] * (2 * N)
            
            result = minimize(
                obj, x0, method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if result.fun < 0:
                new_points = result.x.reshape(N, 2)
                new_radius = _compute_radius(new_points)
                if new_radius > best_radius:
                    best_points = new_points
                    best_radius = new_radius
        except Exception:
            pass
    
    # Phase 4: Final refinement if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            x0 = best_points.flatten()
            
            def obj(x):
                pts = x.reshape(N, 2)
                norms = np.linalg.norm(pts, axis=1)
                if np.any(norms > 1.0 - 1e-10):
                    return 1e6 + np.sum(norms[norms > 1.0 - 1e-10])
                
                min_pair = np.inf
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d < min_pair:
                            min_pair = d
                
                min_boundary = np.min(1.0 - norms)
                radius = min(min_pair / 2.0, min_boundary)
                return -radius
            
            def obj_grad(x):
                pts = x.reshape(N, 2)
                norms = np.linalg.norm(pts, axis=1)
                
                min_pair = np.inf
                min_pair_idx = None
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d < min_pair:
                            min_pair = d
                            min_pair_idx = (i, j)
                
                min_boundary = np.min(1.0 - norms)
                min_boundary_idx = np.argmin(1.0 - norms)
                
                radius = min(min_pair / 2.0, min_boundary)
                
                grad = np.zeros(2 * N)
                if min_pair / 2.0 <= min_boundary:
                    i, j = min_pair_idx
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    if d > 1e-12:
                        unit = diff / d
                        grad[2*i:2*i+2] = -unit
                        grad[2*j:2*j+2] = unit
                else:
                    i = min_boundary_idx
                    p = pts[i]
                    norm = np.linalg.norm(p)
                    if norm > 1e-12:
                        unit = p / norm
                        grad[2*i:2*i+2] = -unit
                
                return grad
            
            constraints = []
            for i in range(N):
                def make_constraint(i):
                    def con(x):
                        p = x.reshape(N, 2)[i]
                        return 1.0 - np.linalg.norm(p)
                    return con
                constraints.append({'type': 'ineq', 'fun': make_constraint(i)})
            
            bounds = [(-1.0, 1.0)] * (2 * N)
            
            result = minimize(
                obj, x0, method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if result.fun < 0:
                new_points = result.x.reshape(N, 2)
                new_radius = _compute_radius(new_points)
                if new_radius > best_radius:
                    best_points = new_points
                    best_radius = new_radius
        except Exception:
            pass
    
    # Validate final result
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(~np.isfinite(best_points)) or np.any(norms > 1.0 + 1e-9):
        # Fallback to hex grid
        best_points = _initial_hex_grid(N)
    
    return best_points
# EVOLVE-BLOCK-END
