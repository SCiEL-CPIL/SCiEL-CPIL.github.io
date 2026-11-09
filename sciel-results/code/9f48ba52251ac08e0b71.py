"""Pack 50 equal circles in a unit circle; maximize common radius."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    boundary_min = 1.0 - np.max(np.hypot(centers[:, 0], centers[:, 1]))
    
    # Use scipy.spatial for efficient pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(centers)
    min_pair_dist = np.min(d) if len(d) > 0 else np.inf
    
    return min(boundary_min, min_pair_dist / 2.0)


def _initial_placement(n, method='hex'):
    """Generate initial placement of n points in unit disk."""
    if method == 'hex':
        # Hexagonal lattice
        # Estimate spacing for n points in unit disk
        area = np.pi
        spacing = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
        
        points = []
        max_r = 1.2
        k_max = int(np.ceil(max_r / spacing)) + 1
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - 0.01:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) > n:
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind='stable')[:n]
            points = points[idx]
        elif len(points) < n:
            # Fallback: random points
            points = np.random.uniform(-0.9, 0.9, (n, 2))
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii > 0.9
            if np.any(mask):
                points[mask] *= (0.9 / radii[mask])[:, None]
        
        return points[:n]
    else:
        # Random placement
        points = np.random.uniform(-0.85, 0.85, (n, 2))
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii > 0.85
        if np.any(mask):
            points[mask] *= (0.85 / radii[mask])[:, None]
        return points[:n]


def _soft_min_radius_and_grad(x, beta, n=N):
    """Soft-min objective with gradient, using log-sum-exp.
    Objective: -soft_min of all constraints (boundary and pairwise)."""
    centers = x.reshape(n, 2)
    
    # Boundary values: 1 - |c_i|
    boundary_vals = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    
    # Pairwise distances / 2
    from scipy.spatial.distance import pdist
    pair_dists = pdist(centers) / 2.0
    
    # All values
    all_vals = np.concatenate([boundary_vals, pair_dists])
    n_boundary = n
    n_pairs = len(pair_dists)
    
    # Soft min via log-sum-exp
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted))) + beta * min_val
    soft_min = -log_sum_exp / beta
    
    # Weights for gradient
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient of soft min
    grad = np.zeros((n, 2))
    
    # Boundary terms: d/dx_i (1 - |c_i|) = -c_i / |c_i|
    for i in range(n):
        r = np.hypot(centers[i, 0], centers[i, 1])
        if r > 1e-10:
            grad[i, 0] += weights[i] * (-centers[i, 0] / r)
            grad[i, 1] += weights[i] * (-centers[i, 1] / r)
    
    # Pairwise terms: need to map pair indices back to point indices
    # pdist returns distances in order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[n_boundary + idx]
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.hypot(dx, dy)
            if d > 1e-10:
                grad[i, 0] += w * (dx / d)
                grad[i, 1] += w * (dy / d)
                grad[j, 0] -= w * (dx / d)
                grad[j, 1] -= w * (dy / d)
            idx += 1
    
    # Objective is -soft_min, so gradient is -grad
    return -soft_min, -grad.reshape(-1)


def _repulsive_force_objective_and_grad(x, n=N):
    """Repulsive force objective: minimize sum of 1/d_ij^2 + boundary penalty."""
    centers = x.reshape(n, 2)
    obj = 0.0
    grad = np.zeros((n, 2))
    
    # Boundary penalty: sum of max(0, |c_i| - 0.99)^2
    for i in range(n):
        r = np.hypot(centers[i, 0], centers[i, 1])
        if r > 0.99:
            obj += (r - 0.99)**2
            if r > 1e-10:
                grad[i, 0] += 2 * (r - 0.99) * centers[i, 0] / r
                grad[i, 1] += 2 * (r - 0.99) * centers[i, 1] / r
    
    # Repulsive forces between pairs
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.hypot(dx, dy)
            if d > 1e-6:
                obj += 1.0 / (d**2)
                # d/dx_i (1/d^2) = -2*(x_i - x_j)/d^4
                grad[i, 0] += -2 * dx / (d**4)
                grad[i, 1] += -2 * dy / (d**4)
                grad[j, 0] += 2 * dx / (d**4)
                grad[j, 1] += 2 * dy / (d**4)
    
    return obj, grad.reshape(-1)


def _feasibility_violation(x, r, n=N):
    """Compute constraint violations for given radius r.
    Violations: for each pair i<j, need |c_i - c_j| >= 2r, and for each i, |c_i| <= 1-r."""
    centers = x.reshape(n, 2)
    grad = np.zeros((n, 2))
    max_viol = 0.0
    
    # Boundary violations: |c_i| - (1-r)
    for i in range(n):
        cx, cy = centers[i]
        dist = np.hypot(cx, cy)
        viol = dist - (1.0 - r)
        if viol > max_viol:
            max_viol = viol
        if viol > 0:
            if dist > 1e-10:
                grad[i, 0] += cx / dist
                grad[i, 1] += cy / dist
    
    # Pairwise violations: 2r - d_ij
    from scipy.spatial.distance import pdist
    pair_dists = pdist(centers)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = pair_dists[idx]
            viol = 2.0 * r - d
            if viol > max_viol:
                max_viol = viol
            if viol > 0 and d > 1e-10:
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                grad[i, 0] -= dx / d
                grad[i, 1] -= dy / d
                grad[j, 0] += dx / d
                grad[j, 1] += dy / d
            idx += 1
    
    return max_viol, grad.reshape(-1)


def _try_radius_feasibility(r, x0, max_iter=300, ftol=1e-14):
    """Try to find a feasible configuration for radius r starting from x0."""
    n = N
    
    def obj_and_grad(x):
        viol, grad = _feasibility_violation(x, r, n)
        return viol, grad
    
    try:
        result = minimize(
            lambda x: obj_and_grad(x)[0],
            x0.copy(),
            jac=lambda x: obj_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-10, 'disp': False}
        )
        
        if result.x is not None:
            final_viol, _ = _feasibility_violation(result.x, r, n)
            if final_viol < 1e-7:
                return True, result.x.reshape(n, 2)
    except Exception:
        pass
    
    return False, None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Multiple initializations and optimization strategies
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Try multiple initializations with soft-min optimization
    initializations = [
        ('hex', 1.0),
        ('hex', 0.95),
        ('random', 1.0),
        ('random', 0.9),
    ]
    
    for method, scale in initializations:
        if time.time() > deadline - 30.0:
            break
        
        centers = _initial_placement(N, method)
        if scale != 1.0:
            centers *= scale
        
        # Optimize with soft-min
        x0 = centers.flatten()
        try:
            result = minimize(
                lambda x: _soft_min_radius_and_grad(x, 50.0)[0],
                x0,
                jac=lambda x: _soft_min_radius_and_grad(x, 50.0)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            if result.x is not None:
                new_centers = result.x.reshape(N, 2)
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
        except Exception:
            pass
    
    if best_centers is None:
        # Fallback
        best_centers = _initial_placement(N, 'hex')
        best_radius = _compute_radius(best_centers)
    
    # Phase 2: Binary search on radius
    r_low = best_radius
    r_high = min(0.15, 1.0 / np.sqrt(N / np.pi))
    
    for _ in range(8):
        if time.time() > deadline - 15.0:
            break
        
        r_mid = (r_low + r_high) / 2.0
        success, new_centers = _try_radius_feasibility(r_mid, best_centers.flatten())
        
        if success:
            r_low = r_mid
            best_centers = new_centers
            best_radius = _compute_radius(best_centers)
        else:
            r_high = r_mid
    
    # Phase 3: Final polish with higher beta
    if time.time() < deadline - 5.0:
        try:
            x0 = best_centers.flatten()
            result2 = minimize(
                lambda x: _soft_min_radius_and_grad(x, 100.0)[0],
                x0,
                jac=lambda x: _soft_min_radius_and_grad(x, 100.0)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            if result2.x is not None:
                new_centers = result2.x.reshape(N, 2)
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_centers = new_centers
                    best_radius = new_radius
        except Exception:
            pass
    
    # Phase 4: Local perturbations
    if time.time() < deadline - 2.0:
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            perturbed = best_centers + np.random.normal(0, 0.0003, (N, 2))
            radii = np.hypot(perturbed[:, 0], perturbed[:, 1])
            mask = radii > (1.0 - best_radius)
            if np.any(mask):
                perturbed[mask] *= ((1.0 - best_radius) / radii[mask])[:, None]
            
            r = _compute_radius(perturbed)
            if r > best_radius:
                best_centers = perturbed
                best_radius = r
    
    return best_centers
# EVOLVE-BLOCK-END
