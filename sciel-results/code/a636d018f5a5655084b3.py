import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import math

N = 16
D = 2

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Precompute indices for upper triangle of pairwise distances
    iu = np.triu_indices(N, 1)
    num_pairs = N * (N - 1) // 2

    def get_dists_sq(pts):
        """Compute squared pairwise distances for the upper triangle."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        return dists_sq[iu]

    def exact_objective(params):
        """
        Exact objective: (max_d / min_d)^2.
        params: flat array of length 2*(N-2), representing points 2..N-1.
        Points 0 and 1 are fixed at (0,0) and (1,0).
        """
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        d_sq = get_dists_sq(pts)
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    def soft_objective(params, beta=50.0):
        """
        Smooth surrogate using log-sum-exp for soft-max and soft-min.
        Returns (soft_max_d / soft_min_d)^2.
        """
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        d_sq = get_dists_sq(pts)
        
        # Numerical stability: shift by max
        max_val = np.max(d_sq)
        # Soft max of d_sq
        log_sum_max = np.log(np.sum(np.exp(beta * (d_sq - max_val))))
        soft_max_sq = max_val + (1.0 / beta) * log_sum_max
        
        # Soft min of d_sq
        min_val = np.min(d_sq)
        # We want -log(sum(exp(-beta * d_sq))) / beta
        # = -log(sum(exp(-beta * (d_sq - min_val))) * exp(-beta * min_val)) / beta
        # = -log(sum(exp(-beta * (d_sq - min_val)))) / beta + min_val
        log_sum_min = np.log(np.sum(np.exp(-beta * (d_sq - min_val))))
        soft_min_sq = min_val - (1.0 / beta) * log_sum_min
        
        if soft_min_sq < 1e-12:
            return 1e10
            
        soft_max = np.sqrt(soft_max_sq)
        soft_min = np.sqrt(soft_min_sq)
        
        return (soft_max / soft_min)**2

    def make_hex_init():
        """Generate initial parameters from a hexagonal lattice subset."""
        # Generate a large hex lattice
        R = 6
        pts = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                pts.append((x, y))
        
        pts = np.array(pts)
        # Normalize to roughly unit scale
        pts = pts / 5.0 
        
        # Center
        pts -= np.mean(pts, axis=0)
        
        # Select 16 points closest to origin
        dists_from_origin = np.sum(pts**2, axis=1)
        sorted_indices = np.argsort(dists_from_origin)
        selected = pts[sorted_indices[:N]]
        
        # Normalize so that min distance is 1 (fixed by transformation later, but good for init)
        # Actually, we need to transform this to our parameter space.
        
        # Compute pairwise distances to find min
        d_sq = get_dists_sq(selected)
        min_d = np.sqrt(np.min(d_sq))
        
        # Scale so min_d = 1
        if min_d > 1e-8:
            selected = selected / min_d
        
        # Find the pair with min distance
        min_idx = np.argmin(d_sq)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1:
            i0, i1 = i1, i0
            
        p0 = selected[i0]
        p1 = selected[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            return np.random.randn(2 * (N - 2)) * 0.1
            
        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (selected - p0) @ R_mat.T / norm_v
        
        # Reorder so i0 is 0, i1 is 1
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        transformed = transformed[perm]
        
        return transformed[2:].flatten()

    def make_random_init():
        """Generate a random initial configuration."""
        # Random points in a disk
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
        # Ensure distinct
        # Simple jitter if collisions (unlikely)
        
        # Normalize min dist to 1
        d_sq = get_dists_sq(pts)
        min_d = np.sqrt(np.min(d_sq))
        if min_d > 1e-8:
            pts = pts / min_d
            
        # Find min pair
        min_idx = np.argmin(d_sq)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1:
            i0, i1 = i1, i0
            
        p0 = pts[i0]
        p1 = pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            return np.random.randn(2 * (N - 2)) * 0.1
            
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        transformed = transformed[perm]
        
        return transformed[2:].flatten()

    def make_circle_init():
        """Points on a circle with perturbation."""
        theta = 2 * np.pi * np.arange(N) / N
        r = np.ones(N) + 0.1 * np.random.randn(N)
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
        d_sq = get_dists_sq(pts)
        min_d = np.sqrt(np.min(d_sq))
        if min_d > 1e-8:
            pts = pts / min_d
            
        min_idx = np.argmin(d_sq)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1:
            i0, i1 = i1, i0
            
        p0 = pts[i0]
        p1 = pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            return np.random.randn(2 * (N - 2)) * 0.1
            
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        transformed = transformed[perm]
        
        return transformed[2:].flatten()

    best_params = None
    best_score = np.inf

    # Strategy 1: Hex lattice initialization + Local Optimization
    init_hex = make_hex_init()
    
    # Phase 1a: Soft optimization from hex
    if time.time() < deadline - 20:
        try:
            res = minimize(soft_objective, init_hex, args=(30.0,), method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6})
            score = exact_objective(res.x)
            if score < best_score:
                best_score = score
                best_params = res.x
        except Exception:
            pass

    # Phase 1b: Exact optimization from hex result
    if best_params is not None and time.time() < deadline - 10:
        try:
            res = minimize(exact_objective, best_params, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            score = exact_objective(res.x)
            if score < best_score:
                best_score = score
                best_params = res.x
        except Exception:
            pass

    # Strategy 2: Differential Evolution (Global Search)
    # Use a subset of the time budget for DE
    de_time_limit = 40.0
    if time.time() < deadline - de_time_limit:
        try:
            # Bounds: points should be within a reasonable radius. 
            # Max dist is what we minimize, but it shouldn't be huge.
            # Let's bound coordinates to [-10, 10]
            bounds = [(-10, 10)] * (2 * (N - 2))
            
            # Use a wrapper to enforce time limit if possible, or just rely on maxiter/popsize
            # DE can be slow. Let's use a modest population.
            res_de = differential_evolution(
                soft_objective, bounds, args=(20.0,), 
                maxiter=20, popsize=15, tol=1e-6, 
                seed=42, polish=False # We will polish manually
            )
            
            # Evaluate exact
            score_de = exact_objective(res_de.x)
            if score_de < best_score:
                best_score = score_de
                best_params = res_de.x
        except Exception:
            pass

    # Strategy 3: Multi-start Local Optimization from Random/Circle inits
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        if i % 2 == 0:
            init = make_random_init()
        else:
            init = make_circle_init()
            
        # Soft then Exact
        try:
            # Short soft run
            res_soft = minimize(soft_objective, init, args=(50.0,), method='L-BFGS-B', 
                                options={'maxiter': 200, 'ftol': 1e-10})
            
            # Exact polish
            res_exact = minimize(exact_objective, res_soft.x, method='L-BFGS-B', 
                                 options={'maxiter': 300, 'ftol': 1e-12})
            
            score = exact_objective(res_exact.x)
            if score < best_score:
                best_score = score
                best_params = res_exact.x
        except Exception:
            pass

    # Final High-Precision Polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            # Very tight tolerance
            res_final = minimize(exact_objective, best_params, method='L-BFGS-B', 
                                 options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            score = exact_objective(res_final.x)
            if score < best_score:
                best_score = score
                best_params = res_final.x
        except Exception:
            pass

    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    # Verify distinctness and return
    d_sq = get_dists_sq(final_pts)
    if np.min(d_sq) < 1e-8:
        # Fallback to hex if something went wrong
        init_hex = make_hex_init()
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = init_hex.reshape(-1, 2)

    return final_pts

# EVOLVE-BLOCK-END
