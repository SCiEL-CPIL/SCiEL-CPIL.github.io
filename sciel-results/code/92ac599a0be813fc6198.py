import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Helper to calculate radius
    def calc_radius(points):
        # points shape (N, 2)
        # Check boundary
        dists_to_origin = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.min(dists_to_origin)
        
        # Check pairwise distances
        # Use broadcasting for pairwise distances
        # d_ij = ||p_i - p_j||
        # To save memory/time, we can compute in chunks or use scipy.spatial.distance
        # For N=300, (300, 300) matrix is small.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists) / 2.0
        
        return min(min_boundary, min_pairwise)

    # Helper to calculate energy
    def energy(x):
        points = x.reshape(N, 2)
        # Boundary term
        dists_to_origin = np.linalg.norm(points, axis=1)
        # Avoid division by zero or negative
        # If point is outside, penalty is huge
        # We want to keep points inside.
        # Let's use a soft barrier.
        # If dist > 1, penalty is (dist - 1)^2
        # If dist < 1, penalty is 1/(1-dist)^2 ? No, that pushes to center.
        # We want to push away from boundary if close, and away from each other.
        
        # Pairwise repulsion
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0) # Avoid div by 0
        # Use inverse square for pairwise
        # Sum over i<j
        # To avoid double counting and diagonal, we can sum all and divide by 2, but diagonal is 1.
        # Better: use upper triangle indices
        i, j = np.triu_indices(N, k=1)
        pair_dists_sq = dists_sq[i, j]
        # Energy term: sum(1/d^2)
        e_pair = np.sum(1.0 / pair_dists_sq)
        
        # Boundary term: sum(1/(1-r)^2) for r<1, else (r-1)^2
        r = dists_to_origin
        # Clip to avoid issues
        # If r >= 1, penalty = (r-1)^2 * large_factor
        # If r < 1, penalty = 1/(1-r)^2
        # This creates a strong repulsion from the boundary.
        
        # Actually, a simpler objective is often better:
        # Maximize the minimum distance.
        # But energy minimization is a good proxy.
        
        # Let's refine the boundary term.
        # We want points to be as far from boundary as possible? No, just inside.
        # The pairwise repulsion will push them inwards if the boundary is a hard constraint.
        # But L-BFGS-B doesn't have hard constraints easily without bounds.
        # We can use bounds: x in [-1, 1], y in [-1, 1].
        # And the energy function penalizes being close to the circle edge.
        
        # Boundary energy:
        # If r > 1: (r-1)^2 * 1000
        # If r <= 1: 0? No, if we don't penalize being close to boundary, they might stick to it.
        # But pairwise repulsion will push them apart. If they are pushed to the edge,
        # we need to penalize crossing the edge.
        # Let's use: e_bound = sum( (max(0, r-1))^2 * 100 )
        # This allows them to be on the boundary, but penalizes going out.
        # To maximize radius, we want them to be as far from each other as possible.
        # The pairwise term does that.
        
        e_bound = np.sum(np.maximum(0.0, r - 1.0)**2) * 100.0
        
        return e_pair + e_bound

    def energy_grad(x):
        points = x.reshape(N, 2)
        grad = np.zeros_like(x)
        
        # Pairwise gradient
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        dists_sq_safe = np.where(dists_sq < 1e-12, 1e-12, dists_sq)
        
        # d/dp_i (1/d_ij^2) = -2 * (p_i - p_j) / d_ij^4
        # Sum over j != i
        # Factor = -2 * diff / dists_sq_safe**2
        # But we only want i<j in the energy sum?
        # Energy = sum_{i<j} 1/d_ij^2
        # Gradient w.r.t p_i: sum_{j!=i} d/dp_i (1/d_ij^2)
        # d/dp_i (1/d_ij^2) = -2 * (p_i - p_j) / d_ij^4
        # Note: (p_i - p_j) / d_ij^4 = diff[i,j] / dists_sq_safe[i,j]**2
        
        # Let's compute the force on each point.
        # Force on i from j is -2 * (p_i - p_j) / d_ij^4
        # Total force on i = sum_{j != i} -2 * (p_i - p_j) / d_ij^4
        
        # We can compute this using matrix operations.
        # diff[i,j] is p_i - p_j.
        # inv_d4 = 1.0 / (dists_sq_safe**2)
        # grad_pair[i] = -2 * sum_j (diff[i,j] * inv_d4[i,j])
        
        inv_d4 = 1.0 / (dists_sq_safe**2)
        # Set diagonal to 0 to ignore self-interaction
        np.fill_diagonal(inv_d4, 0.0)
        
        # grad_pair[i] = -2 * (diff[i, :] * inv_d4[i, :]).sum(axis=1)
        grad_pair = -2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
        
        # Boundary gradient
        r = np.linalg.norm(points, axis=1)
        # d/dr (max(0, r-1)^2 * 100)
        # If r > 1: 2 * (r-1) * 100 * dr/dx
        # dr/dx = p / r
        mask = r > 1.0
        if np.any(mask):
            grad_bound = np.zeros_like(points)
            # Only for points outside
            r_masked = r[mask]
            p_masked = points[mask]
            # Gradient of 100 * (r-1)^2 is 200 * (r-1) * (p/r)
            factor = 200.0 * (r_masked - 1.0) / r_masked
            grad_bound[mask] = p_masked * factor[:, np.newaxis]
        else:
            grad_bound = np.zeros_like(points)
            
        grad = (grad_pair + grad_bound).flatten()
        return grad

    # Initial configuration: Hexagonal lattice
    # Estimate spacing
    # Area = pi. N=300. r ~ 0.05. Spacing ~ 0.1.
    # Let's try a few spacings.
    
    best_radius = 0.0
    best_points = None
    
    # Generate hex lattice
    def get_hex_points(spacing):
        points = []
        # Range
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                points.append([x, y])
        points = np.array(points)
        # Filter inside unit circle (with margin)
        r = np.linalg.norm(points, axis=1)
        # Margin = spacing/2 to ensure we can fit
        mask = r <= 1.0 - spacing/2.0
        points = points[mask]
        # Sort by radius to pick central ones
        points = points[np.argsort(np.linalg.norm(points, axis=1))]
        # Take N points
        if len(points) >= N:
            return points[:N]
        else:
            return None

    # Try different initial spacings
    spacings = [0.1, 0.09, 0.11, 0.08, 0.12]
    
    for sp in spacings:
        if time.time() > deadline:
            break
            
        init_points = get_hex_points(sp)
        if init_points is None:
            continue
            
        # Add small random noise to break symmetry
        init_points = init_points + np.random.normal(0, 0.001, init_points.shape)
        
        # Optimize
        try:
            res = minimize(
                energy,
                init_points.flatten(),
                jac=energy_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            opt_points = res.x.reshape(N, 2)
            
            # CRITICAL FIX: Project any points outside the unit circle back inside
            dists = np.linalg.norm(opt_points, axis=1)
            outside = dists > 1.0
            if np.any(outside):
                opt_points[outside] = opt_points[outside] / dists[outside, np.newaxis] * (1.0 - 1e-9)
            
            rad = calc_radius(opt_points)
            
            if rad > best_radius:
                best_radius = rad
                best_points = opt_points
                    
        except Exception as e:
            pass
            
    # If we didn't find anything, return the initial hex points (valid)
    if best_points is None:
        init_points = get_hex_points(0.1)
        if init_points is not None:
            best_points = init_points
        else:
            # Fallback: simple grid
            best_points = np.random.uniform(-0.9, 0.9, (N, 2))
            
    # Ensure final result is valid
    dists = np.linalg.norm(best_points, axis=1)
    outside = dists > 1.0
    if np.any(outside):
        best_points[outside] = best_points[outside] / dists[outside, np.newaxis] * (1.0 - 1e-9)
        
    return best_points
