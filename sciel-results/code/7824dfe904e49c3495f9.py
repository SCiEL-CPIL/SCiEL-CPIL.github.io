"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct centers for 50 equal circles inside the unit circle.
    Uses a hexagonal grid initialization followed by Simulated Annealing
    to maximize the minimum distance between circles and to the boundary.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    def calc_radius(centers):
        """
        Compute the maximum valid radius for the given centers.
        Returns 0.0 if invalid (out of bounds or non-finite).
        """
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        
        # Boundary constraint: min distance from center to boundary
        norms = np.linalg.norm(centers, axis=1)
        r_boundary = 1.0 - np.max(norms)
        if r_boundary <= 0:
            return 0.0

        # Pairwise constraint: min distance between any two centers / 2
        # Vectorized calculation of pairwise distances
        # diff[i, j, :] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        r_pair = min_pair_dist / 2.0
        
        return min(r_boundary, r_pair)

    def generate_hex_init():
        """
        Generate a valid initial configuration using a hexagonal grid.
        """
        # Estimate initial spacing. 
        # Area of unit circle = pi. 
        # Hex cell area for spacing s is s^2 * sqrt(3)/2.
        # N * s^2 * sqrt(3)/2 <= pi => s <= sqrt(2*pi / (N*sqrt(3)))
        # For N=50, s ~ sqrt(6.28 / 86.6) ~ sqrt(0.0725) ~ 0.27
        # Let's try a range of spacings to ensure we get at least N points.
        
        best_points = None
        # Try decreasing spacing until we get enough points
        for spacing in np.linspace(0.3, 0.1, 30):
            k = int(1.0 / spacing) + 2
            # Generate grid points
            xs, ys = np.meshgrid(
                np.arange(-k, k + 1) * spacing,
                np.arange(-k, k + 1) * spacing * np.sqrt(3) / 2.0
            )
            # Offset odd rows
            for j in range(-k, k + 1):
                if j % 2 != 0:
                    xs[:, j + k] += spacing / 2.0
            
            points = np.column_stack((xs.ravel(), ys.ravel()))
            norms = np.linalg.norm(points, axis=1)
            
            # Filter points that are safely inside (margin for radius)
            # We don't know the exact radius yet, but it's roughly spacing/2 * something
            # Using 1.0 - spacing is a safe conservative filter
            mask = norms <= 1.0 - spacing
            points = points[mask]
            
            if len(points) >= N:
                # Sort by distance to center to pick the most central N points
                idx = np.argsort(norms[mask])
                best_points = points[idx[:N]]
                break
        
        if best_points is None:
            # Fallback: random points inside unit circle
            angles = np.random.uniform(0, 2*np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.9
            best_points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        return best_points

    # 1. Initialize
    current_centers = generate_hex_init()
    current_r = calc_radius(current_centers)
    best_centers = current_centers.copy()
    best_r = current_r
    
    if current_r <= 0:
        # Fallback to simple grid if init failed
        spacing = 0.2
        pts = []
        for x in np.arange(-1, 1, spacing):
            for y in np.arange(-1, 1, spacing):
                if x*x + y*y < 1:
                    pts.append([x, y])
        if len(pts) >= N:
            current_centers = np.array(pts[:N])
            current_r = calc_radius(current_centers)
            best_centers = current_centers.copy()
            best_r = current_r
        else:
            return np.zeros((N, 2)) # Should not happen

    # 2. Simulated Annealing
    T = 0.05  # Initial temperature
    T_min = 1e-6
    alpha = 0.99995 # Cooling rate
    step_scale = 0.05 # Scale for random moves
    
    # Pre-allocate for speed
    # We can't easily pre-allocate the distance matrix in a loop without overhead, 
    # but N=50 is small.
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        # Pick a random index
        i = np.random.randint(N)
        
        # Generate a random move
        # Scale the move by current temperature to allow larger moves early on
        move_scale = step_scale * (T / 0.05) # Normalize so at T=0.05 scale is step_scale
        delta = np.random.normal(0, move_scale, 2)
        
        new_centers = current_centers.copy()
        new_centers[i] += delta
        
        # Check validity quickly: if any point goes out, reject
        # Quick check: norm of new point
        new_norm = np.linalg.norm(new_centers[i])
        if new_norm > 1.0:
            # Reject immediately
            continue
            
        new_r = calc_radius(new_centers)
        
        # Accept if better, or with probability exp((new_r - current_r) / T) if worse
        if new_r > current_r:
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Metropolis criterion
            # Avoid log of 0 or negative
            diff_r = new_r - current_r
            if diff_r >= 0:
                accept = True
            else:
                # If T is very small, exp is 0
                if T < 1e-8:
                    accept = False
                else:
                    # Use log to avoid underflow
                    # log_prob = diff_r / T
                    # accept if log(random) < log_prob
                    # random is uniform [0, 1), log is (-inf, 0]
                    # So we need log(random) < diff_r / T
                    # Which is random < exp(diff_r / T)
                    # We can just compute exp if T is not too small
                    prob = np.exp(diff_r / T)
                    accept = np.random.random() < prob
            
            if accept:
                current_centers = new_centers
                current_r = new_r
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cool down
        T *= alpha
        
        # Restart if stuck
        if no_improve_count > max_no_improve:
            # Reheat
            T = max(T, 0.01)
            no_improve_count = 0
            # Optional: Perturb current state from best state
            # current_centers = best_centers + np.random.normal(0, 0.01, (N, 2))
            # current_r = calc_radius(current_centers)

    return best_centers

# EVOLVE-BLOCK-END
