import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_centers = None
    best_radius = -1.0
    
    # Smooth surrogate for maximizing the minimum distance
    # We want to maximize r. 
    # Let d_k be the set of all pairwise distances and distances to boundary.
    # We want to maximize min(d_k).
    # This is equivalent to minimizing -min(d_k).
    # A smooth approximation is - (1/alpha) * log(sum(exp(-alpha * d_k)))
    # Wait, log-sum-exp approximates the max. 
    # log(sum(exp(alpha * d_k))) approximates max(d_k).
    # We want to maximize the MIN distance.
    # So we want to minimize the MAX "deficit".
    # Alternatively, we can just minimize the sum of squared overlaps for a fixed r, 
    # but we want to find the optimal r.
    
    # A better smooth objective for maximizing the minimum distance:
    # Minimize J(x) = (1/alpha) * log( sum_{k} exp(-alpha * d_k) )
    # As alpha -> inf, this approaches -min(d_k).
    # So minimizing J is equivalent to maximizing min(d_k).
    
    def objective(params):
        # Check time
        if time.time() > deadline:
            return 1e10 # Penalize heavily if we run out of time
        
        centers = params.reshape(N, 2)
        
        # Calculate pairwise distances
        # d[i,j] = ||c_i - c_j||
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Calculate distances to boundary (1 - |c_i|)
        radii = np.linalg.norm(centers, axis=1)
        boundary_dists = 1.0 - radii
        
        # Combine all distances
        # We want the minimum of all these.
        # Using log-sum-exp with negative sign to approximate min.
        # min(d) = - max(-d)
        # max(-d) ≈ (1/alpha) log sum exp(alpha * (-d))
        # So min(d) ≈ -(1/alpha) log sum exp(-alpha * d)
        
        # We need to handle the boundary distances as well.
        # Let's concatenate the upper triangle of dists and boundary_dists.
        
        # To avoid memory issues or slow loops, we can compute the log-sum-exp carefully.
        # But N=50 is small.
        
        d_pairs = dists[~np.eye(N, dtype=bool)] # Flatten upper triangle? No, just all i<j
        # Actually, dists is symmetric. We just need i<j.
        iu = np.triu_indices(N, 1)
        d_pairs = dists[iu]
        
        all_dists = np.concatenate([d_pairs, boundary_dists])
        
        # Clip distances to avoid overflow in exp
        # We want to maximize the minimum.
        # If a distance is very small, we want a large penalty.
        # Objective to minimize: -min(all_dists)
        # Approximation: (1/alpha) * log(sum(exp(-alpha * d)))
        # If d is small, exp(-alpha*d) is large (close to 1 if d=0, small if d large).
        # Wait.
        # If d -> 0, exp(-alpha*0) = 1.
        # If d -> large, exp(-alpha*d) -> 0.
        # Sum is dominated by the smallest d.
        # log(sum) ≈ log(exp(-alpha * d_min)) = -alpha * d_min.
        # (1/alpha) * log(sum) ≈ -d_min.
        # So minimizing this objective minimizes -d_min, which maximizes d_min.
        
        alpha = 100.0 # Higher alpha makes it closer to the exact min
        # To prevent overflow, subtract max before exp?
        # max_val = np.max(-alpha * all_dists)
        # log_sum = max_val + np.log(np.sum(np.exp(-alpha * all_dists - max_val)))
        # obj = -log_sum / alpha
        
        # Simpler: just use the sum of squares of "violations" against a target?
        # No, the log-sum-exp is robust.
        
        # Let's use a slightly different approach: 
        # Minimize the sum of squared distances that are below a threshold?
        # No, the log-sum-exp is a standard smooth min.
        
        # Implementation of smooth min:
        # m = min(d)
        # approx_m = -(1/alpha) * log(sum(exp(-alpha * d)))
        
        # To avoid numerical issues:
        # Let y = -alpha * all_dists
        # max_y = np.max(y)
        # log_sum = max_y + np.log(np.sum(np.exp(y - max_y)))
        # smooth_min = -log_sum / alpha
        
        y = -alpha * all_dists
        max_y = np.max(y)
        if not np.isfinite(max_y):
            return 1e10
        log_sum = max_y + np.log(np.sum(np.exp(y - max_y)))
        smooth_min = -log_sum / alpha
        
        return -smooth_min # We want to maximize smooth_min, so minimize -smooth_min

    def score(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        radii = np.linalg.norm(centers, axis=1)
        min_bound = np.min(1.0 - radii)
        return min(min_pair / 2.0, min_bound)

    # Initializations
    inits = []
    
    # 1. Center + Rings
    # Try to place points on concentric rings
    for num_rings in [5, 6, 7]:
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # Rings
        for k in range(1, num_rings):
            # Distribute points evenly on ring k
            # Radius r_k
            r_k = k * 0.15 # Rough spacing
            # Number of points on ring k
            # Circumference = 2 * pi * r_k
            # Spacing ~ 0.3
            n_k = max(1, int(2 * np.pi * r_k / 0.3))
            # Ensure total N
            # This is a rough heuristic. Let's just generate a grid and pick.
            pass
        
    # Better Init: Hexagonal Lattice (as in seed) but with jitter
    spacing = 0.25
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    if len(points) >= N:
        # Pick N points with largest radius? Or just first N?
        # Sort by radius to keep center dense?
        idx = np.argsort(np.linalg.norm(points, axis=1))
        inits.append(points[idx[:N]].copy())
        
    # 2. Random Init
    np.random.seed(42)
    for _ in range(5):
        # Random points in disk
        r = np.sqrt(np.random.random(N))
        theta = np.random.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)

    # 3. Grid Init
    # 10x10 grid
    grid_x = np.linspace(-0.8, 0.8, 10)
    grid_y = np.linspace(-0.8, 0.8, 10)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    # Filter inside unit circle
    mask = np.linalg.norm(grid_pts, axis=1) <= 0.9
    grid_pts = grid_pts[mask]
    if len(grid_pts) >= N:
        inits.append(grid_pts[:N].copy())

    # Run optimization for each init
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Add small jitter to avoid symmetry issues
        init = init + np.random.normal(0, 0.01, init.shape)
        
        # Use COBYLA or L-BFGS-B?
        # L-BFGS-B requires gradients. We don't have them easily for the smooth min.
        # COBYLA is derivative-free and works well for small N.
        # Or use 'Nelder-Mead'?
        # Let's try 'COBYLA' with a reasonable maxiter.
        
        # Constraints: None needed if we use the penalty in the objective, 
        # but we must ensure points stay roughly in the disk.
        # The objective penalizes going outside, but might not be strong enough initially.
        # We can add a hard constraint or a large penalty term.
        
        # Let's add a penalty for being outside the disk to the objective.
        def penalized_objective(params):
            centers = params.reshape(N, 2)
            radii = np.linalg.norm(centers, axis=1)
            # Penalty for |c| > 1
            outside = np.maximum(0, radii - 1.0)
            penalty = 100.0 * np.sum(outside**2)
            
            # Check time
            if time.time() > deadline:
                return 1e10
                
            base_obj = objective(params)
            return base_obj + penalty

        try:
            res = minimize(penalized_objective, init.ravel(), method='COBYLA', 
                           options={'maxiter': 500, 'rhobeg': 0.1})
            
            if res.success or res.fun < 1e9:
                centers = res.x.reshape(N, 2)
                # Verify validity
                if np.all(np.isfinite(centers)):
                    r = score(centers)
                    if r > best_radius:
                        best_radius = r
                        best_centers = centers.copy()
        except Exception:
            pass

    # If no valid center found, return a fallback (e.g., simple grid)
    if best_centers is None:
        # Fallback to the first init
        best_centers = inits[0]
        
    # Final Polish:
    # If we have time, do a few more iterations of local search on the best solution.
    if time.time() < deadline - 10:
        # Try to refine best_centers
        # Use a smaller step size Nelder-Mead
        try:
            res = minimize(penalized_objective, best_centers.ravel(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < 1e9:
                centers = res.x.reshape(N, 2)
                r = score(centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = centers.copy()
        except Exception:
            pass

    return best_centers
