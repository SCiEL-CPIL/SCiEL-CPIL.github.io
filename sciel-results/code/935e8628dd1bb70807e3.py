"""Ring-based packing for 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time

def construct():
    """
    Place 300 equal circles inside a unit circle using a ring-based representation.
    """
    
    def compute_min_distance(params, n_per_ring, n_rings):
        """
        Compute the minimum radius for a given set of ring parameters.
        params: [r_1, phi_1, ..., r_{K-1}, phi_{K-1}]
        """
        points = np.zeros((N, 2))
        points[0] = [0.0, 0.0]
        
        idx = 1
        for i in range(1, n_rings):
            n_i = n_per_ring[i]
            r_i = params[2*(i-1)]
            phi_i = params[2*(i-1) + 1]
            # Clip radius to be positive and reasonable
            r_i = np.clip(r_i, 0.0, 1.0)
            angles = phi_i + 2 * np.pi * np.arange(n_i) / n_i
            points[idx:idx+n_i, 0] = r_i * np.cos(angles)
            points[idx:idx+n_i, 1] = r_i * np.sin(angles)
            idx += n_i
        
        # Check all points are within unit circle
        dists = np.linalg.norm(points, axis=1)
        if np.any(dists >= 1.0):
            return -1.0
        
        # Distance to boundary
        min_dist = 1.0 - np.max(dists)
        
        # Pairwise distances
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        min_nn = np.min(distances[:, 1])
        min_dist = min(min_dist, min_nn / 2.0)
        
        return min_dist

    def negative_min_distance(params, n_per_ring, n_rings):
        return -compute_min_distance(params, n_per_ring, n_rings)

    best_r = 0.0
    best_points = None
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer

    # Generate candidate ring distributions
    candidates = []
    
    # Pattern 1: Linear distribution n_i ~ i
    for n_rings in range(6, 14):
        # Target sum N-1
        # n_i = c * i
        c = 2.0 * (N - 1) / ((n_rings - 1) * n_rings)
        n_per_ring = [1]
        for i in range(1, n_rings):
            n_i = max(1, int(round(c * i)))
            n_per_ring.append(n_i)
        # Adjust
        total = sum(n_per_ring)
        diff = N - total
        if diff != 0:
            # Add/subtract from the largest ring
            n_per_ring[-1] += diff
            if n_per_ring[-1] < 1:
                # Distribute
                for j in range(n_rings-2, 0, -1):
                    if diff > 0:
                        n_per_ring[j] += 1
                        diff -= 1
                    elif diff < 0:
                        n_per_ring[j] -= 1
                        diff += 1
                    if diff == 0: break
        if sum(n_per_ring) == N and all(n >= 1 for n in n_per_ring):
            candidates.append(n_per_ring)

    # Pattern 2: Hexagonal-like n_i ~ 6i
    for n_rings in range(6, 12):
        n_per_ring = [1]
        for i in range(1, n_rings):
            n_i = max(1, int(round(6.0 * i)))
            n_per_ring.append(n_i)
        total = sum(n_per_ring)
        if total < N:
            # Add remaining to outer rings
            diff = N - total
            # Distribute diff to the last few rings
            for j in range(n_rings-1, 0, -1):
                add = min(diff, 5)
                n_per_ring[j] += add
                diff -= add
                if diff == 0: break
        elif total > N:
            # Remove from outer rings
            diff = total - N
            for j in range(n_rings-1, 0, -1):
                rem = min(diff, n_per_ring[j]-1)
                n_per_ring[j] -= rem
                diff -= rem
                if diff == 0: break
        if sum(n_per_ring) == N and all(n >= 1 for n in n_per_ring):
            # Avoid duplicates
            if n_per_ring not in candidates:
                candidates.append(n_per_ring)
                
    # Pattern 3: Uniform-ish
    for n_rings in range(10, 20):
        base = (N - 1) // (n_rings - 1)
        rem = (N - 1) % (n_rings - 1)
        n_per_ring = [1]
        for i in range(1, n_rings):
            n_i = base + (1 if i <= rem else 0)
            n_per_ring.append(n_i)
        if n_per_ring not in candidates:
            candidates.append(n_per_ring)

    # Optimize each candidate
    for n_per_ring in candidates:
        if time.time() - start_time > time_limit:
            break
            
        n_rings = len(n_per_ring)
        dim = 2 * (n_rings - 1)
        
        # Smart initialization
        init_params = np.zeros(dim)
        # Estimate r based on packing
        # r_i ~ i * 2 * r_est
        # Let's guess r_est ~ 0.05
        r_est = 0.05
        for i in range(1, n_rings):
            # Approximate radius
            init_params[2*(i-1)] = i * 2.0 * r_est * 0.9
            init_params[2*(i-1) + 1] = 0.0
            
        # Multi-start Nelder-Mead
        best_local_r = -1.0
        best_local_x = None
        
        # Try a few random angular offsets
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
                
            x0 = init_params.copy()
            if attempt > 0:
                # Randomize angles
                for i in range(1, n_rings):
                    x0[2*(i-1) + 1] = np.random.uniform(0, 2*np.pi)
                # Slight randomization of radii
                x0[0::2] *= np.random.uniform(0.9, 1.1, size=n_rings-1)
            
            try:
                result = minimize(negative_min_distance, x0, args=(n_per_ring, n_rings),
                                 method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9})
                r_val = -result.fun
                if r_val > best_local_r:
                    best_local_r = r_val
                    best_local_x = result.x
            except Exception:
                pass
        
        if best_local_r > best_r:
            best_r = best_local_r
            # Reconstruct points
            points = np.zeros((N, 2))
            points[0] = [0.0, 0.0]
            idx = 1
            for i in range(1, n_rings):
                n_i = n_per_ring[i]
                r_i = best_local_x[2*(i-1)]
                phi_i = best_local_x[2*(i-1) + 1]
                angles = phi_i + 2 * np.pi * np.arange(n_i) / n_i
                points[idx:idx+n_i, 0] = r_i * np.cos(angles)
                points[idx:idx+n_i, 1] = r_i * np.sin(angles)
                idx += n_i
            best_points = points
            
    # Fallback if something went wrong
    if best_points is None or best_r < 0.04:
        # Simple hexagonal lattice fallback
        spacing = 0.1
        while True:
            k = int(1.0 / spacing) + 2
            points_list = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                        points_list.append((x, y))
            points = np.array(points_list)
            if len(points) >= N:
                # Sort by radius to keep center filled
                radii = np.hypot(points[:, 0], points[:, 1])
                indices = np.argsort(radii)
                return points[indices[:N]]
            spacing *= 0.95
            if spacing < 0.01:
                break
    
    return best_points
# EVOLVE-BLOCK-END
