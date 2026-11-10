import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

N = 300

def construct():
    t0 = time.time()
    # 150s limit. Keep a large safety margin (15s) to avoid timeout.
    DEADLINE = t0 + 130.0

    def elapsed():
        return time.time() - t0

    def score(points):
        """
        Calculate the maximum radius for the given centers.
        points: (N, 2) array.
        """
        if points.shape[0] < 2:
            return 0.0
        
        # Use cdist for faster pairwise distance calculation
        dist = cdist(points, points)
        
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dist, np.inf)
        
        min_dist = np.min(dist)
        
        # Calculate maximum distance from origin
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        
        # Radius is limited by both pair distances and boundary
        r_pair = 0.5 * min_dist
        r_bound = 1.0 - max_norm
        
        return min(r_pair, r_bound)

    def gen_hex_seed():
        """
        Generate a hexagonal lattice seed.
        """
        # Estimate initial spacing
        # r ~ sqrt(0.9 / N)
        r_est = np.sqrt(0.9 / N)
        spacing = 2.0 * r_est
        
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        # If we have more than N points, select the ones closest to the center
        if len(pts) > N:
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii)
            pts = pts[idx[:N]]
        elif len(pts) < N:
            # Shrink spacing and retry
            while len(pts) < N:
                spacing *= 0.95
                k = int(1.0 / spacing) + 3
                pts = []
                for j in range(-k, k + 1):
                    for i in range(-k, k + 1):
                        x = (i + 0.5 * (j % 2)) * spacing
                        y = j * spacing * np.sqrt(3.0) / 2.0
                        if x*x + y*y <= 1.0:
                            pts.append([x, y])
                pts = np.array(pts)
                if len(pts) > N:
                    radii = np.linalg.norm(pts, axis=1)
                    idx = np.argsort(radii)
                    pts = pts[idx[:N]]
                elif len(pts) == N:
                    break
        
        return pts

    # Initial Seed
    best_pts = gen_hex_seed()
    best_score = score(best_pts)
    
    # --- Ring Model Setup ---
    
    # Assign points to rings based on radius
    radii = np.linalg.norm(best_pts, axis=1)
    sorted_indices = np.argsort(radii)
    sorted_radii = radii[sorted_indices]
    sorted_points = best_pts[sorted_indices]
    
    # Determine ring boundaries
    # Heuristic: Look for gaps in the sorted radii
    # Or just fix a number of rings. For N=300, ~15 rings is a good guess.
    # Let's try to detect rings by clustering radii.
    # Simple approach: Divide into K rings.
    # Estimate K: The number of rings is roughly sqrt(N) / 2?
    # For hex packing, number of points in ring k is ~ 6k.
    # Sum_{k=1}^K 6k = 3 K(K+1).
    # 300 = 3 K^2 => K ~ 10.
    # Let's try K from 10 to 20 and pick the one that fits best or just use a fixed K=15.
    # Let's use a dynamic assignment based on gaps.
    
    rings = []
    current_ring = [sorted_indices[0]]
    for i in range(1, N):
        r_prev = sorted_radii[i-1]
        r_curr = sorted_radii[i]
        # If the jump is significant relative to the average spacing, start a new ring
        # Average spacing in radius is roughly 2*r_est
        # Let's use a threshold
        if r_curr - r_prev > 0.5 * (2.0 * np.sqrt(0.9/N)):
            rings.append(current_ring)
            current_ring = [sorted_indices[i]]
        else:
            current_ring.append(sorted_indices[i])
    rings.append(current_ring)
    
    # If too many rings, merge the smallest ones? Or just cap it.
    # If too few, split the largest?
    # For robustness, let's force a specific number of rings if the detection is poor.
    # Let's stick with the detected rings, but ensure at least 5 and at most 30.
    if len(rings) < 5:
        # Fallback: quantile split
        K = 15
        bins = np.linspace(0, N, K+1).astype(int)
        rings = [sorted_indices[bins[i]:bins[i+1]] for i in range(K) if len(sorted_indices[bins[i]:bins[i+1]]) > 0]
    elif len(rings) > 30:
        # Merge rings
        # This is complex. Let's just use the first 30? No, that's bad.
        # Let's just use the detected rings. It's usually fine for hex seed.
        pass

    n_rings = len(rings)
    
    # Parameters: [r_0, theta_0, r_1, theta_1, ...]
    # Bounds: r in [0, 1], theta in [-pi, pi]
    
    def build_points_from_params(params):
        pts = np.zeros((N, 2))
        idx = 0
        for k in range(n_rings):
            r_k = params[2*k]
            th_k = params[2*k+1]
            ring_indices = rings[k]
            m = len(ring_indices)
            for j, point_idx in enumerate(ring_indices):
                angle = th_k + 2.0 * np.pi * j / m
                pts[point_idx, 0] = r_k * np.cos(angle)
                pts[point_idx, 1] = r_k * np.sin(angle)
        return pts

    def objective_ring(params):
        pts = build_points_from_params(params)
        # Penalty if outside
        norms = np.linalg.norm(pts, axis=1)
        pen = np.sum(np.maximum(0, norms - 1.0)**2)
        s = score(pts)
        return -s + 1e-3 * pen

    # Initial parameters from seed
    init_params = np.zeros(2 * n_rings)
    for k in range(n_rings):
        ring_indices = rings[k]
        p = best_pts[ring_indices]
        init_params[2*k] = np.linalg.norm(p, axis=1).mean()
        if len(p) > 1:
            angs = np.arctan2(p[:, 1], p[:, 0])
            # Unwrap angles to avoid mean of -pi and pi being 0
            angs = np.unwrap(angs)
            # Align with the first point in the ring to determine offset
            # Actually, we just want the mean angle
            c = np.mean(np.cos(angs))
            s_ = np.mean(np.sin(angs))
            init_params[2*k+1] = np.arctan2(s_, c)
        else:
            init_params[2*k+1] = np.arctan2(p[0, 1], p[0, 0])

    # Optimization
    bounds = []
    for k in range(n_rings):
        bounds.append((0.0, 1.0))
        bounds.append((-np.pi, np.pi))

    current_params = init_params.copy()
    
    # 1. Global Optimization with Differential Evolution
    # DE is good for non-convex problems but can be slow.
    # We have a 130s budget.
    # Let's run DE for a portion of the time.
    if elapsed() < DEADLINE - 40:
        try:
            # tol and maxiter control convergence
            # popsize 15 is default
            de_res = differential_evolution(
                objective_ring,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=True, # Use L-BFGS-B for polish
                init='latinhypercube'
            )
            if de_res.success or de_res.fun < objective_ring(init_params):
                current_params = de_res.x
                cand_pts = build_points_from_params(current_params)
                cand_score = score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass

    # 2. Local Optimization with L-BFGS-B on ring parameters
    # This refines the solution found by DE (or the seed if DE failed)
    if elapsed() < DEADLINE - 10:
        try:
            # Use the best params found so far
            # We need to map best_pts back to params?
            # It's safer to just use current_params which is updated by DE.
            # If DE didn't run or failed, current_params is init_params.
            
            # Let's do a few restarts of L-BFGS-B with small noise
            for _ in range(3):
                if elapsed() > DEADLINE - 10:
                    break
                # Add small noise
                noisy_params = current_params + np.random.normal(0, 0.01, current_params.shape)
                # Clip angles
                for i in range(n_rings):
                    noisy_params[2*i+1] = (noisy_params[2*i+1] + np.pi) % (2*np.pi) - np.pi
                    noisy_params[2*i] = np.clip(noisy_params[2*i], 0, 1)
                
                res = minimize(
                    objective_ring,
                    noisy_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res.x is not None:
                    cand_pts = build_points_from_params(res.x)
                    if np.all(np.isfinite(cand_pts)):
                        cand_score = score(cand_pts)
                        if cand_score > best_score:
                            best_score = cand_score
                            best_pts = cand_pts
                            current_params = res.x
                            
        except Exception:
            pass

    # 3. Final Polish on Full Coordinates
    # The ring model might be slightly restrictive.
    # We can take the best points from the ring model and do a few L-BFGS-B steps on the full 600D space.
    # This can break the ring symmetry if it helps.
    if elapsed() < DEADLINE - 5:
        try:
            x0 = best_pts.flatten()
            # Use a very tight tolerance and few iterations
            res = minimize(
                lambda x: -score(x.reshape(N, 2)),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                cand_pts = res.x.reshape(N, 2)
                if np.all(np.isfinite(cand_pts)):
                    cand_score = score(cand_pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts
        except Exception:
            pass

    # Final validity check
    if not np.all(np.isfinite(best_pts)):
        best_pts = gen_hex_seed()
        
    rr = np.linalg.norm(best_pts, axis=1)
    m = rr > 1.0
    if np.any(m):
        best_pts[m] = best_pts[m] / rr[m, None]

    return best_pts
