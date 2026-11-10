# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Place 300 equal circles inside a unit circle using a ring-based representation.
    """
    
    N = 300

    def get_points_from_params(params, n_per_ring):
        n_rings = len(n_per_ring)
        points = np.zeros((N, 2))
        points[0] = [0.0, 0.0]
        idx = 1
        for i in range(1, n_rings):
            n_i = n_per_ring[i]
            if n_i == 0:
                continue
            r_i = params[2*(i-1)]
            phi_i = params[2*(i-1) + 1]
            angles = phi_i + 2 * np.pi * np.arange(n_i) / n_i
            points[idx:idx+n_i, 0] = r_i * np.cos(angles)
            points[idx:idx+n_i, 1] = r_i * np.sin(angles)
            idx += n_i
        return points

    def compute_min_radius(points):
        """
        Compute the maximum radius r such that circles are disjoint and inside.
        """
        # Distance to boundary
        dists_center = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.max(dists_center)
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        return min(min_pair / 2.0, min_boundary)

    def objective(params, n_per_ring):
        points = get_points_from_params(params, n_per_ring)
        r = compute_min_radius(points)
        return -r

    def local_refine(points, n_iter=200, init_step=0.005):
        """
        Simple local search to maximize min-distance.
        """
        current_r = compute_min_radius(points)
        step_size = init_step
        
        for _ in range(n_iter):
            # Compute nearest neighbor for each point
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            nearest_indices = np.argmin(dists, axis=1)
            nearest_dists = dists[np.arange(N), nearest_indices]
            
            # Force: vector away from nearest neighbor
            vectors = points - points[nearest_indices]
            norms = nearest_dists
            norms[norms == 0] = 1.0
            forces = vectors / norms[:, np.newaxis]
            
            # Boundary force
            dists_center = np.linalg.norm(points, axis=1)
            boundary_forces = np.zeros_like(points)
            # Push inward if close to boundary
            close_to_boundary = dists_center > 0.9
            for i in np.where(close_to_boundary)[0]:
                boundary_forces[i] = -points[i] / dists_center[i]
            
            # Update
            new_points = points + step_size * (forces + 0.5 * boundary_forces)
            
            # Ensure inside boundary
            new_dists = np.linalg.norm(new_points, axis=1)
            mask = new_dists > 1.0
            if np.any(mask):
                new_points[mask] = new_points[mask] / new_dists[mask][:, np.newaxis]
            
            new_r = compute_min_radius(new_points)
            
            if new_r > current_r:
                points = new_points
                current_r = new_r
            else:
                step_size *= 0.9
                if step_size < 1e-6:
                    break
                    
        return points, current_r

    best_r = 0.0
    best_points = None

    # Search over number of rings
    for n_rings in range(8, 16):
        # Generate multiple candidate distributions
        candidates = []
        
        # 1. Linear: n_i ~ i
        c_lin = 2.0 * (N - 1) / ((n_rings - 1) * n_rings)
        n_lin = [1] + [max(1, int(round(c_lin * i))) for i in range(1, n_rings)]
        # Adjust sum
        diff = N - sum(n_lin)
        for j in range(len(n_lin) - 1, 0, -1):
            if diff == 0: break
            if diff > 0:
                n_lin[j] += 1; diff -= 1
            else:
                if n_lin[j] > 1:
                    n_lin[j] -= 1; diff += 1
        if sum(n_lin) == N:
            candidates.append(n_lin)

        # 2. Sqrt: n_i ~ sqrt(i)
        c_sq = (N - 1) / sum([np.sqrt(i) for i in range(1, n_rings)])
        n_sq = [1] + [max(1, int(round(c_sq * np.sqrt(i)))) for i in range(1, n_rings)]
        diff = N - sum(n_sq)
        for j in range(len(n_sq) - 1, 0, -1):
            if diff == 0: break
            if diff > 0:
                n_sq[j] += 1; diff -= 1
            else:
                if n_sq[j] > 1:
                    n_sq[j] -= 1; diff += 1
        if sum(n_sq) == N:
            candidates.append(n_sq)

        for n_per_ring in candidates:
            init_params = np.zeros(2 * (n_rings - 1))
            for i in range(1, n_rings):
                init_params[2*(i-1)] = i / (n_rings - 1) * 0.95
                init_params[2*(i-1) + 1] = 0.0

            try:
                res = minimize(objective, init_params, args=(n_per_ring,),
                               method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
                
                r_val = -res.fun
                points = get_points_from_params(res.x, n_per_ring)
                
                # Local refinement
                refined_points, refined_r = local_refine(points)
                
                if refined_r > best_r:
                    best_r = refined_r
                    best_points = refined_points
                    
            except Exception:
                pass

    if best_points is None:
        # Fallback: Hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    return best_points

# EVOLVE-BLOCK-END
