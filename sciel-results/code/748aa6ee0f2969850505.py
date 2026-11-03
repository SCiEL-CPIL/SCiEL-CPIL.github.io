import numpy as np
import time
import scipy.spatial.distance as dist

N = 300

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # 1. Initial Configuration: Hexagonal Lattice
    # Estimate spacing. Area of unit circle is pi. 300 circles.
    # Approx area per circle = pi/300. Radius approx sqrt(pi/300)/2 ? No, radius r. Area pi*r^2.
    # 300 * pi * r^2 <= pi => r <= 1/sqrt(300) ~ 0.0577.
    # Hexagonal spacing s = 2r.
    # Let's start with a slightly larger spacing to ensure no overlaps, then let SA tighten.
    # Or start with a standard hexagonal grid and select best 300.
    
    def get_hex_initial():
        # Try to generate a hexagonal grid that covers the unit disk
        # and pick the 300 points closest to the center that fit.
        # Spacing s.
        # Let's try s such that we get roughly 300 points.
        # Number of points in hex grid of radius R_grid ~ (3/2) * (R_grid/s)^2 ?
        # Let's just iterate s.
        
        best_points = None
        best_r = 0
        
        for s in np.linspace(0.05, 0.10, 50):
            # Generate grid
            k = int(1.0 / s) + 3
            rows = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3.0) / 2.0
                    rows.append((x, y))
            pts = np.array(rows)
            dists = np.hypot(pts[:, 0], pts[:, 1])
            # Keep points strictly inside (with margin s/2 to avoid initial overlap with boundary)
            # Actually, we want them inside.
            mask = dists < 1.0 - s/2.0
            pts = pts[mask]
            if len(pts) < N:
                continue
            
            # Sort by distance to center, take closest N
            order = np.argsort(dists[mask])
            pts = pts[order[:N]]
            
            # Check validity and radius
            # Vectorized check
            D = dist.cdist(pts, pts)
            np.fill_diagonal(D, np.inf)
            min_dist = np.min(D)
            min_boundary = 1.0 - np.min(np.hypot(pts[:, 0], pts[:, 1]))
            r = min(min_dist / 2.0, min_boundary)
            
            if r > best_r:
                best_r = r
                best_points = pts.copy()
        
        if best_points is None:
            # Fallback: random
            best_points = np.random.rand(N, 2) * 2.0 - 1.0
            # Normalize to be inside
            norms = np.hypot(best_points[:, 0], best_points[:, 1])
            best_points = best_points / norms[:, np.newaxis] * 0.9
        
        return best_points

    points = get_hex_initial()
    
    # 2. Simulated Annealing
    # Objective: Maximize r = min(min_pair_dist/2, min_boundary_dist)
    
    def get_radius(pts):
        # Compute minimum pairwise distance
        # For N=300, cdist is fast
        D = dist.cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_dist = np.min(D)
        
        # Compute minimum distance to boundary
        norms = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = 1.0 - np.min(norms)
        
        return min(min_dist / 2.0, min_boundary)

    current_r = get_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # SA Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.999  # Cooling factor
    max_iter = 500000  # Max iterations
    
    # We need to move faster. Vectorized radius calc is ~1ms.
    # 500k iters * 1ms = 500s. Too slow.
    # We need to reduce iterations or speed up.
    # N=300. cdist is O(N^2). 90k ops.
    # In Python, loop overhead is high.
    # Let's aim for ~100,000 iterations max.
    
    # Better: Use a simpler energy?
    # No, cdist is vectorized in C. It should be fast.
    # Let's measure.
    
    iter_count = 0
    while T > T_min and iter_count < max_iter:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Save old position
        old_pos = points[idx].copy()
        
        # Perturb
        # Step size proportional to T?
        # Radius is ~0.05. Step size should be similar scale.
        step_size = 0.01 * T / 0.01 # Just use T scaled?
        # Let's use a fixed small step for SA, controlled by T acceptance.
        # Or adaptive step.
        # Standard: step ~ sqrt(T) or just T?
        # Let's try step_size = 0.05 * T / 0.01 ? No.
        # Let's use a step size that allows exploration.
        # Initial T=0.01. Radius ~0.05.
        # Let's try delta = np.random.normal(0, 0.01, 2) * (T / 0.01)
        
        delta = np.random.normal(0, 0.02, 2) * (T / 0.01)
        new_pos = old_pos + delta
        
        # Constraint: Must stay inside?
        # If it goes outside, radius becomes negative or small.
        # The objective function handles it: 1 - norm.
        # If norm > 1, 1-norm < 0.
        # We want to maximize r. So negative r is bad.
        
        points[idx] = new_pos
        
        new_r = get_radius(points)
        
        # Acceptance
        if new_r > current_r:
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            # Boltzmann probability
            # We maximize, so energy E = -r
            # Delta E = -new_r - (-current_r) = current_r - new_r
            # P = exp(-Delta E / T) = exp((new_r - current_r) / T)
            prob = np.exp((new_r - current_r) / T)
            if np.random.rand() < prob:
                current_r = new_r
            else:
                # Revert
                points[idx] = old_pos
        
        # Cool
        T *= alpha
        iter_count += 1
        
        # Periodic restart? Or just keep going.
        # If stuck, we can increase T slightly? No, standard SA cools.
        
    # 3. Local Polish
    # Use the best points found.
    # Do a steepest ascent or small random steps with very low T.
    
    points = best_points.copy()
    current_r = get_radius(points)
    
    # Polish phase: 50,000 iterations with very small steps
    T_polish = 1e-5
    for i in range(50000):
        if time.time() - start_time > time_limit - 1:
            break
            
        idx = np.random.randint(N)
        old_pos = points[idx].copy()
        
        # Very small step
        delta = np.random.normal(0, 1e-4, 2)
        new_pos = old_pos + delta
        points[idx] = new_pos
        
        new_r = get_radius(points)
        
        if new_r > current_r:
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            points[idx] = old_pos
            
        # Simple cooling for polish
        T_polish *= 0.999
        
    # Final check and return
    # Ensure no NaNs or invalid values
    if not np.all(np.isfinite(best_points)):
        return points # Fallback to last valid? Or initial?
        
    return best_points

# EVOLVE-BLOCK-END
