"""Ring-based packing for 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def construct():
    """
    Place 300 equal circles inside a unit circle using a ring-based representation.
    Optimizes ring radii and angular offsets for various ring counts and distributions.
    """
    
    def compute_min_radius(params, n_per_ring):
        """
        params: array of length 2*(n_rings-1) [r_1, phi_1, r_2, phi_2, ...]
        n_per_ring: list of circle counts per ring, n_per_ring[0] is center (1)
        Returns: max possible radius r such that circles fit.
        """
        n_rings = len(n_per_ring)
        
        # Build points
        points = np.zeros((N, 2))
        points[0] = [0.0, 0.0]
        
        idx = 1
        for i in range(1, n_rings):
            n_i = n_per_ring[i]
            if n_i == 0: continue
            r_i = params[2*(i-1)]
            phi_i = params[2*(i-1) + 1]
            angles = phi_i + 2 * np.pi * np.arange(n_i) / n_i
            points[idx:idx+n_i, 0] = r_i * np.cos(angles)
            points[idx:idx+n_i, 1] = r_i * np.sin(angles)
            idx += n_i
            
        # Ensure all points are within unit circle
        dists = np.linalg.norm(points, axis=1)
        if np.any(dists >= 1.0):
            return -1.0
            
        # Calculate min distance to boundary
        r_boundary = 1.0 - np.max(dists)
        
        # Calculate min pairwise distance
        # Using cKDTree for efficiency
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        min_pair_dist = np.min(distances[:, 1])
        
        r_pair = min_pair_dist / 2.0
        
        return min(r_boundary, r_pair)

    def objective(params, n_per_ring):
        r = compute_min_radius(params, n_per_ring)
        return -r

    best_r = 0.0
    best_points = None
    
    # Search over number of rings
    # The number of rings K typically scales with sqrt(N) or similar, but for N=300, 10-20 is a good range.
    for n_rings in range(6, 21):
        # We need to distribute N-1 circles into n_rings-1 rings.
        # Heuristic: n_i proportional to radius R_i.
        # Assume R_i is linearly spaced: R_i = i * (1-r_est) / (n_rings-1)
        # r_est ~ 0.05
        # n_i ~ 2 * pi * R_i / (2 * r_est) = pi * R_i / r_est
        
        # Let's try a few distribution strategies for each n_rings
        strategies = [
            "linear",  # n_i proportional to i
            "quadratic", # n_i proportional to i^1.5 or similar? No, area is pi*r^2, circumference is 2*pi*r.
                         # Number of circles in annulus of width dr is ~ (2*pi*r * dr) / (pi*r^2) ? No.
                         # Number of circles in ring i is proportional to circumference r_i.
                         # So n_i proportional to i is the standard heuristic.
        ]
        
        for strategy in strategies:
            if strategy == "linear":
                # n_i = c * i
                # Sum n_i = N-1
                # c * (n_rings-1)*n_rings/2 = N-1
                c = 2.0 * (N - 1) / ((n_rings - 1) * n_rings)
                n_per_ring = [1]
                for i in range(1, n_rings):
                    n_i = max(1, int(round(c * i)))
                    n_per_ring.append(n_i)
            elif strategy == "quadratic":
                # Try n_i proportional to i^1.2 ?
                c = (N - 1) / np.sum([i**1.2 for i in range(1, n_rings)])
                n_per_ring = [1]
                for i in range(1, n_rings):
                    n_i = max(1, int(round(c * i**1.2)))
                    n_per_ring.append(n_i)

            # Adjust to sum exactly N
            total = sum(n_per_ring)
            diff = N - total
            # Distribute diff to the largest rings to minimize impact on packing density
            if diff != 0:
                # Sort indices by n_i descending, add/subtract 1
                indices = list(range(1, n_rings))
                indices.sort(key=lambda x: n_per_ring[x], reverse=(diff > 0))
                for idx in indices:
                    if diff == 0: break
                    if diff > 0:
                        n_per_ring[idx] += 1
                        diff -= 1
                    else:
                        if n_per_ring[idx] > 1:
                            n_per_ring[idx] -= 1
                            diff += 1
            
            # Initial guess for parameters
            # Radii roughly proportional to i
            init_params = np.zeros(2 * (n_rings - 1))
            for i in range(1, n_rings):
                init_params[2*(i-1)] = i / n_rings * 0.98
                init_params[2*(i-1) + 1] = 0.0
            
            # Optimize
            # Use Powell method which is robust for this non-smooth objective
            try:
                result = minimize(objective, init_params, args=(n_per_ring,),
                                 method='Powell', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                
                r_val = -result.fun
                if r_val > best_r:
                    best_r = r_val
                    # Reconstruct points
                    points = np.zeros((N, 2))
                    points[0] = [0.0, 0.0]
                    idx = 1
                    for i in range(1, n_rings):
                        n_i = n_per_ring[i]
                        if n_i == 0: continue
                        r_i = result.x[2*(i-1)]
                        phi_i = result.x[2*(i-1) + 1]
                        angles = phi_i + 2 * np.pi * np.arange(n_i) / n_i
                        points[idx:idx+n_i, 0] = r_i * np.cos(angles)
                        points[idx:idx+n_i, 1] = r_i * np.sin(angles)
                        idx += n_i
                    best_points = points
            except Exception:
                pass
    
    # Fallback if something went wrong
    if best_points is None:
        # Simple hexagonal lattice fallback
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
