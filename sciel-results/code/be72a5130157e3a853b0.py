import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    max_time = 115.0  # Safety margin for 120s limit
    
    # --- Initialization: Concentric Rings ---
    # We model the 300 points as lying on K concentric rings.
    # This reduces the search space from 600 dims to 2*K dims.
    # The known optimal packing for N=300 is not strictly rings, but rings
    # provide a very good global structure that is hard to reach from random
    # hex lattices due to topological constraints.
    
    # Heuristic for number of rings:
    # In a hex packing, area per point ~ sqrt(3)/2 * (2r)^2 = 2*sqrt(3)*r^2.
    # Total area pi. N * 2*sqrt(3)*r^2 <= pi -> r <= sqrt(pi / (2*sqrt(3)*N)).
    # r ~ 0.053.
    # Circumference of ring k is 2*pi*R_k.
    # Number of points on ring k ~ 2*pi*R_k / (2*r) = pi*R_k/r.
    # Sum n_k = N.
    # Let's just pick a reasonable K and distribute points proportional to radius.
    
    # A common heuristic is K ~ sqrt(N). sqrt(300) ~ 17.
    # Let's try a few K values or just fix a good one.
    # For N=300, K=16-20 is typical.
    
    K = 17
    
    # Initial radii: roughly uniform distribution of area.
    # R_k ~ sqrt((k+0.5)/K)
    radii = np.sqrt((np.arange(K) + 0.5) / K)
    
    # Initial angular offsets: 0
    angles = np.zeros(K)
    
    def get_points_from_params(params):
        """
        params: [r_1, r_2, ..., r_K, a_1, a_2, ..., a_K]
        Returns (N, 2) array of points.
        """
        r_vals = params[:K]
        a_vals = params[K:]
        
        # Clip radii to [0, 1]
        r_vals = np.clip(r_vals, 0.0, 1.0)
        
        # Calculate number of points per ring
        # n_k proportional to r_k
        # Ensure at least 1 point per ring if r_k > 0
        weights = r_vals
        # Avoid division by zero or very small weights
        weights[weights < 1e-6] = 1e-6
        weights = weights / np.sum(weights)
        
        n_points = np.zeros(K, dtype=int)
        # Allocate N points proportional to weights
        raw_counts = weights * N
        n_points = np.floor(raw_counts).astype(int)
        
        # Distribute remaining points
        remainder = N - np.sum(n_points)
        if remainder > 0:
            # Give to rings with largest fractional part
            fracs = raw_counts - np.floor(raw_counts)
            order = np.argsort(-fracs)
            for i in range(remainder):
                n_points[order[i % K]] += 1
        
        # Ensure no ring has 0 points if we can avoid it, but sum must be N.
        # If a ring has 0 points, its radius/angle don't matter, but let's keep it simple.
        
        points = []
        for k in range(K):
            n_k = n_points[k]
            if n_k == 0:
                continue
            r_k = r_vals[k]
            a_k = a_vals[k]
            # Angles for points on this ring
            thetas = a_k + 2 * np.pi * np.arange(n_k) / n_k
            x = r_k * np.cos(thetas)
            y = r_k * np.sin(thetas)
            points.append(np.column_stack((x, y)))
        
        if not points:
            # Fallback: random
            return np.random.randn(N, 2) * 0.5
        
        points = np.vstack(points)
        
        # Ensure exactly N points (safety check)
        if len(points) < N:
            # Pad with center or random if logic failed
            diff = N - len(points)
            pad = np.zeros((diff, 2))
            points = np.vstack([points, pad])
        elif len(points) > N:
            points = points[:N]
            
        return points

    def calc_score(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
        return min(min_pair, min_bound)

    def negative_score(params):
        """Objective function for minimization."""
        x = get_points_from_params(params)
        return -calc_score(x)

    # --- Stage 1: Global Search on Ring Parameters ---
    # We will run several Nelder-Mead optimizations from different starting points.
    
    best_x = None
    best_r = 0.0
    
    # Initial run
    init_params = np.concatenate([radii, angles])
    
    # Try a few random perturbations of the initial ring structure
    num_restarts = 5
    found_improvement = False
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time - 30:
            break
            
        if restart == 0:
            p0 = init_params.copy()
        else:
            # Perturb radii and angles
            p0 = init_params.copy()
            p0[:K] += np.random.randn(K) * 0.1
            p0[K:] += np.random.randn(K) * 0.5
            p0[:K] = np.clip(p0[:K], 0.0, 1.0)
        
        try:
            res = minimize(
                negative_score, 
                p0, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            x_res = get_points_from_params(res.x)
            r_res = calc_score(x_res)
            
            if r_res > best_r:
                best_r = r_res
                best_x = x_res
                found_improvement = True
        except Exception:
            continue

    if best_x is None:
        # Fallback to hex lattice if ring search failed completely
        s_init = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
        points = []
        R = 1.2
        k = int(R / s_init) + 2
        for j in range(-k, k + 1):
            y = j * s_init * np.sqrt(3) / 2
            x_offset = (j % 2) * s_init / 2
        for i in range(-k, k + 1):
            x = i * s_init + x_offset
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            points = np.random.randn(N, 2) * 0.5
        else:
            radii_pts = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(radii_pts)
            points = points[indices[:N]]
        best_x = points
        best_r = calc_score(best_x)

    # --- Stage 2: Local Polish in Full 600D Space ---
    # The ring solution is a good local minimum, but not global.
    # We use a repulsive force model to push points apart and into the boundary.
    # This allows breaking the ring symmetry if beneficial.
    
    x = best_x.copy()
    
    # Force parameters
    # Repulsion: 1/d^2 (softened)
    # Attraction to boundary: for points close to boundary? No, usually we just want max min distance.
    # A simple gradient ascent on the min distance is difficult due to non-smoothness.
    # Instead, we use a "soft" potential: V = sum 1/d_ij^2 + sum 1/(1-|c_i|)^2 (if inside)
    # Minimizing V maximizes distances.
    
    def compute_forces(x):
        forces = np.zeros_like(x)
        
        # Pairwise repulsion
        # To be fast, we can use a cutoff or just vectorize.
        # N=300, N^2 = 90,000 ops. Vectorized is fine.
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1e-6) # Avoid div by 0
        
        # Force F_ij = k * (x_i - x_j) / dist^3  (direction is diff, magnitude 1/d^2)
        # Actually grad(1/d) = -1/d^2 * (x_i-x_j)/d = -diff/d^3.
        # We want to MINIMIZE potential, so we move AGAINST gradient.
        # Potential U = sum 1/d. Grad U = -sum diff/d^3.
        # Force (negative grad) = sum diff/d^3.
        
        # Mask out self
        mask = ~np.eye(N, dtype=bool)
        
        # d^3 = (d^2)^1.5
        d_cubed = np.power(dists_sq, 1.5)
        
        # Force matrix: F[i,j] = (x_i - x_j) / d_ij^3
        # Note: diff[i,j] = x_i - x_j
        force_matrix = diff / d_cubed[:, :, np.newaxis]
        
        # Sum forces on each point
        # F_i = sum_j F_ij
        pair_forces = np.sum(force_matrix, axis=1)
        
        # Boundary repulsion
        # Potential V_b = sum 1/(1 - |c_i|)^2 ? No, if |c_i| > 1, it's invalid.
        # We want to push points OUT if they are too close to center? No, we want to push them to boundary?
        # Actually, the constraint is min(1-|c_i|).
        # If a point is far from boundary, it's not the bottleneck.
        # If a point is close to boundary, it's the bottleneck.
        # We want to INCREASE 1-|c_i|. So we want to DECREASE |c_i|.
        # So force should be INWARD for points close to boundary?
        # Wait. If we increase the radius r, the points must move apart.
        # If a point is at the boundary, it can't move out. It must move in?
        # No, if it moves in, it creates space for others?
        # Let's think about the objective: Maximize r.
        # If min_dist is the bottleneck, we push points apart.
        # If boundary is the bottleneck, we push points INWARD?
        # If we push a boundary point inward, its distance to boundary increases, but its distance to neighbors might decrease.
        # Actually, usually the optimal packing has many points on the boundary.
        # If we push them inward, we might allow the whole cluster to expand?
        # No, the container is fixed.
        
        # Let's use a simple heuristic:
        # If 1 - |c_i| is small (point near boundary), apply a small inward force to relieve pressure?
        # Or, apply an outward force?
        # If we apply outward force, they hit the wall.
        # Let's stick to pairwise repulsion and a weak "centering" or "boundary" term only if needed.
        # Actually, for circle packing in a circle, the boundary constraint is active for many points.
        # A common trick is to add a small attraction to the boundary for points that are not the closest to the boundary?
        # No, let's just use pairwise repulsion. The boundary constraint is handled by projection.
        
        # However, pure repulsion might collapse the structure to the center if not balanced.
        # But we start from a good ring structure.
        
        # Add a weak force to push points towards the boundary if they are "too central"?
        # No, that's not right.
        
        # Let's just use pairwise repulsion and project to circle.
        
        return pair_forces

    # Optimization loop
    alpha = 0.01
    best_r_local = best_r
    
    while time.time() - start_time < max_time - 5:
        forces = compute_forces(x)
        
        # Normalize forces to prevent explosion
        f_norm = np.linalg.norm(forces)
        if f_norm > 1e-8:
            forces = forces / f_norm
            
        # Step
        x_new = x + alpha * forces
        
        # Project to unit circle
        norms = np.linalg.norm(x_new, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            x_new[out_mask] = x_new[out_mask] / norms[out_mask, np.newaxis]
            
        # Check score
        r_new = calc_score(x_new)
        
        if r_new > best_r_local:
            best_r_local = r_new
            best_x = x_new.copy()
            x = x_new
            # Increase step slightly if improving
            alpha = min(alpha * 1.05, 0.05)
        else:
            # Decrease step
            alpha *= 0.9
            if alpha < 0.001:
                # Random kick
                x += np.random.randn(N, 2) * 0.01
                norms = np.linalg.norm(x, axis=1)
                out_mask = norms > 1.0
                if np.any(out_mask):
                    x[out_mask] = x_new[out_mask] / norms[out_mask, np.newaxis]
                alpha = 0.01
                r_new = calc_score(x)
                if r_new > best_r_local:
                    best_r_local = r_new
                    best_x = x.copy()

    # Final validation
    final_r = calc_score(best_x)
    if final_r > best_r:
        best_r = final_r
        
    return best_x
