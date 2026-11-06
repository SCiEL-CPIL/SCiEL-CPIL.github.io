"""Rewritten program for the suite task; stronger packing algorithm."""
N = 100
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    """
    Returns array of shape (N, 3): centres of N equal spheres inside cube [-1/2, 1/2]^3.
    Uses a multi-restart gradient-based optimization with penalty terms to maximize
    the minimum distance (which determines the sphere radius).
    """
    deadline = time.time() + 110  # safety margin before 120s limit

    def compute_radius(points):
        """Compute the maximum common radius for given centers."""
        # Distance from each point to nearest boundary
        boundary_dists = []
        for p in points:
            d = min(0.5 - abs(p[0]), 0.5 - abs(p[1]), 0.5 - abs(p[2]))
            boundary_dists.append(d)
        min_boundary = min(boundary_dists)

        # Minimum pairwise distance / 2
        min_pair_dist = float('inf')
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_pair_dist:
                    min_pair_dist = d

        return min(min_boundary, min_pair_dist / 2.0)

    def objective_and_grad(points):
        """
        Objective: negative of a smooth surrogate for the minimum distance.
        We maximize the minimum of:
          - pairwise distances / 2
          - distance to boundary
        
        Use a soft-min (log-sum-exp based) to get a smooth approximation.
        Actually, let's use a different approach: minimize the maximum violation
        or use a penalty method.
        
        Better approach: We want to maximize r such that all constraints are satisfied.
        Equivalent to minimizing the maximum of:
          - (r * 2 - d_ij) for all pairs
          - (r * 2 - 2 * boundary_dist_i) for all i
        
        But we can also just directly optimize for maximum minimum distance.
        
        Let's use: minimize -soft_min(log of distances)
        Or simpler: use a power mean with negative exponent.
        
        Actually, the cleanest approach: 
        f(points) = -min_i min_k (0.5 - |c_ik|, min_j d_ij/2)
        
        To make this smooth, use a soft-min.
        """
        n = len(points)
        
        # Compute all pairwise distances
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    d[i, j] = np.linalg.norm(points[i] - points[j])
        
        # Distances to boundary
        boundary = np.zeros(n)
        for i in range(n):
            boundary[i] = min(0.5 - abs(points[i][0]), 
                              0.5 - abs(points[i][1]), 
                              0.5 - abs(points[i][2]))
        
        # All relevant distances: pairwise/2 and boundary distances
        # We want to maximize the minimum of these
        # Collect all values
        all_vals = []
        for i in range(n):
            all_vals.append(boundary[i])
        for i in range(n):
            for j in range(i + 1, n):
                all_vals.append(d[i, j] / 2.0)
        
        all_vals = np.array(all_vals)
        
        # Soft-min using log-sum-exp with negative exponent
        # soft_min(x) = -log(sum(exp(-x_i / t))) * t
        t = 0.01  # temperature for soft-min
        neg_vals = -all_vals / t
        # For numerical stability
        max_neg = np.max(neg_vals)
        log_sum = np.log(np.sum(np.exp(neg_vals - max_neg))) + max_neg
        soft_min_val = -log_sum * t
        
        # Gradient: we need to compute d(soft_min)/d(points)
        # This is complex, so let's use finite differences or a simpler penalty.
        
        # Actually, let's use a simpler approach: minimize a penalty function
        # that penalizes when any distance is too small.
        
        # Return the negative soft-min as objective (we want to maximize soft-min)
        return -soft_min_val, None  # We'll use numerical gradient

    def objective(points):
        """Objective function to minimize (negative of radius proxy)."""
        n = len(points)
        
        # Compute all pairwise distances / 2
        pair_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_dists.append(np.linalg.norm(points[i] - points[j]) / 2.0)
        
        # Distances to boundary
        bound_dists = []
        for i in range(n):
            bound_dists.append(min(0.5 - abs(points[i][0]), 
                                   0.5 - abs(points[i][1]), 
                                   0.5 - abs(points[i][2])))
        
        all_vals = np.array(pair_dists + bound_dists)
        
        # Use a soft-min to get smooth approximation
        t = 0.02
        neg_vals = -all_vals / t
        max_neg = np.max(neg_vals)
        log_sum = np.log(np.sum(np.exp(neg_vals - max_neg))) + max_neg
        soft_min_val = -log_sum * t
        
        return -soft_min_val

    def objective_grad(points):
        """Compute gradient of objective numerically."""
        n = len(points)
        grad = np.zeros_like(points)
        h = 1e-7
        
        f0 = objective(points)
        for i in range(n):
            for k in range(3):
                points[i, k] += h
                f1 = objective(points)
                points[i, k] -= h
                grad[i, k] = (f1 - f0) / h
        
        return f0, grad

    # Best solution tracking
    best_points = None
    best_radius = 0.0

    # Multiple random restarts
    np.random.seed(42)
    
    for restart in range(30):  # Up to 30 restarts
        if time.time() > deadline:
            break
        
        # Initialize: random points in cube
        points = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        # Try to optimize
        x0 = points.flatten()
        
        def obj_flat(x):
            p = x.reshape(N, 3)
            # Clip to stay in bounds during optimization
            p = np.clip(p, -0.49, 0.49)
            return objective(p)
        
        def grad_flat(x):
            p = x.reshape(N, 3)
            p = np.clip(p, -0.49, 0.49)
            f, g = objective_grad(p)
            return g.flatten()
        
        # Use L-BFGS-B with bounds
        bounds = [(-0.49, 0.49)] * (N * 3)
        
        try:
            result = minimize(obj_flat, x0, method='L-BFGS-B', 
                             jac=grad_flat, bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < (best_radius if best_points is not None else 1.0):
                final_points = result.x.reshape(N, 3)
                final_points = np.clip(final_points, -0.49, 0.49)
                r = compute_radius(final_points)
                
                if r > best_radius:
                    best_radius = r
                    best_points = final_points.copy()
                    
        except Exception:
            pass
    
    # If no valid result, return grid
    if best_points is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_points
