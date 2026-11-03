"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def compute_radius(centers):
        """Compute the valid radius for a given set of centers."""
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        if min_dist_to_wall < 0:
            return 0.0
        n = len(centers)
        min_pair_dist = np.inf
        for i in range(n):
            diffs = centers[i+1:] - centers[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            if len(dists) > 0:
                d = np.min(dists)
                if d < min_pair_dist:
                    min_pair_dist = d
        if not np.isfinite(min_pair_dist):
            return 0.0
        r = min(min_dist_to_wall, min_pair_dist / 2.0)
        return max(0.0, r)

    def generate_hex_lattice(n_rows, n_cols_per_row, jitter=0.0):
        """Generate a hexagonal lattice inside the unit square."""
        pts = []
        if n_rows <= 0:
            return np.array([])
        # Compute spacing to fit in unit square
        # Width: (n_cols - 1) * dx + margin; Height: (n_rows - 1) * dy + margin
        # For hex: dx = d, dy = d * sqrt(3)/2
        # We want to fill the square as much as possible
        # Try to fit n_cols in width 1.0 and n_rows in height 1.0
        # d_x = 1.0 / (n_cols - 1 + 1) = 1/n_cols (roughly)
        # d_y = 1.0 / (n_rows - 1 + 1) = 1/n_rows
        # For hex: d_y = d_x * sqrt(3)/2, so d_x = d_y * 2/sqrt(3)
        
        # Let's parameterize by d (center-to-center distance)
        # Width needed: (n_cols - 1) * d + d = n_cols * d (for even rows)
        # Actually with offset: even rows have n_cols, odd rows have n_cols-1
        # Width = (n_cols - 1) * d + d = n_cols * d for full rows
        # Height = (n_rows - 1) * d * sqrt(3)/2 + d
        
        # Solve for d that fits:
        # n_cols * d <= 1.0  =>  d <= 1/n_cols
        # (n_rows - 1) * d * sqrt(3)/2 + d <= 1.0  =>  d * ((n_rows-1)*sqrt(3)/2 + 1) <= 1.0
        # d <= 1 / ((n_rows-1)*sqrt(3)/2 + 1)
        
        d_max_x = 1.0 / n_cols
        d_max_h = 1.0 / ((n_rows - 1) * math.sqrt(3) / 2 + 1)
        d = min(d_max_x, d_max_h)
        
        # Center the lattice in the square
        total_width = (n_cols - 1) * d + d  # = n_cols * d for full rows
        total_height = (n_rows - 1) * d * math.sqrt(3) / 2 + d
        
        x_offset = -0.5 + d / 2
        y_offset = -0.5 + d / 2
        
        for row in range(n_rows):
            y = y_offset + row * d * math.sqrt(3) / 2
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = x_offset
            else:
                n_in_row = n_cols - 1
                x_start = x_offset + d / 2
            
            for col in range(n_in_row):
                x = x_start + col * d
                if jitter > 0:
                    x += np.random.uniform(-jitter, jitter)
                    y += np.random.uniform(-jitter, jitter)
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) > N:
            pts = pts[:N]
        elif len(pts) < N:
            # Pad with points near the edges
            while len(pts) < N:
                x = np.random.uniform(-0.5 + d/2, 0.5 - d/2)
                y = np.random.uniform(-0.5 + d/2, 0.5 - d/2)
                pts = np.vstack([pts, [[x, y]]])
        
        return pts

    def force_directed_refinement(centers, max_iters=200, dt=0.001):
        """Apply force-directed repulsion to expand the packing."""
        centers = centers.copy()
        n = len(centers)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion (inverse square)
            for i in range(n):
                for j in range(i+1, n):
                    diff = centers[i] - centers[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-12:
                        # Random perturbation for coincident points
                        diff = np.random.randn(2) * 0.001
                        dist_sq = np.dot(diff, diff)
                    dist = np.sqrt(dist_sq)
                    # Repulsion force: k / dist^2 in direction of diff
                    # k is a scaling factor
                    k = 0.0001
                    force_mag = k / (dist_sq * dist)  # k / dist^3 for unit vector * k/dist^2
                    forces[i] += force_mag * diff
                    forces[j] -= force_mag * diff
            
            # Boundary repulsion
            for i in range(n):
                for dim in range(2):
                    if centers[i, dim] > 0.5:
                        forces[i, dim] -= (centers[i, dim] - 0.5) * 10.0
                    elif centers[i, dim] < -0.5:
                        forces[i, dim] += (-0.5 - centers[i, dim]) * 10.0
            
            # Apply forces with damping
            centers += dt * forces
            # Clamp to square
            centers = np.clip(centers, -0.5, 0.5)
        
        return centers

    def nelder_mead_polish(centers, max_iter=500):
        """Polish using Nelder-Mead on the soft-min objective."""
        centers = centers.copy()
        
        def objective(x):
            c = x.reshape(-1, 2)
            # Soft-min of pairwise distances and wall distances
            # Use a large beta for near-hard min
            beta = 100.0
            
            # Wall distances
            wall_dists = 0.5 - np.abs(c)
            min_wall = np.min(wall_dists)
            
            # Pairwise distances
            min_pair = np.inf
            for i in range(N):
                diffs = c[i+1:] - c[i]
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                if len(dists) > 0:
                    d = np.min(dists)
                    if d < min_pair:
                        min_pair = d
            
            if not np.isfinite(min_pair):
                return -1.0
            
            r = min(min_wall, min_pair / 2.0)
            return -r
        
        x0 = centers.flatten()
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
            new_centers = result.x.reshape(-1, 2)
            # Clamp to square
            new_centers = np.clip(new_centers, -0.5, 0.5)
            return new_centers
        except Exception:
            return centers

    def coordinate_ascent_polish(centers, max_iter=30):
        """Final polish with coordinate ascent."""
        centers = centers.copy()
        for iteration in range(max_iter):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                orig = centers[i].copy()
                best_pos = orig
                best_r = compute_radius(centers)
                
                # Try moving in small steps in 8 directions
                for dx in [-0.001, 0, 0.001]:
                    for dy in [-0.001, 0, 0.001]:
                        if dx == 0 and dy == 0:
                            continue
                        new_center = orig + np.array([dx, dy])
                        if new_center[0] < -0.5 or new_center[0] > 0.5:
                            continue
                        if new_center[1] < -0.5 or new_center[1] > 0.5:
                            continue
                        centers[i] = new_center
                        r = compute_radius(centers)
                        if r > best_r:
                            best_r = r
                            best_pos = new_center.copy()
                
                if not np.allclose(best_pos, orig):
                    centers[i] = best_pos
                    improved = True
            
            if not improved:
                break
        
        return centers

    # Main strategy:
    # 1. Generate multiple hexagonal lattice initializations
    # 2. Apply force-directed refinement to each
    # 3. Polish the best with Nelder-Mead
    # 4. Final coordinate ascent polish

    best_centers = None
    best_radius = 0.0

    # Try different hexagonal configurations
    for n_rows in range(12, 18):
        if time.time() > deadline - 60:
            break
        n_cols = math.ceil(N / n_rows)
        
        for jitter in [0.0, 0.002, 0.005]:
            if time.time() > deadline - 60:
                break
            
            centers = generate_hex_lattice(n_rows, n_cols, jitter=jitter)
            if len(centers) != N:
                continue
            
            # Force-directed refinement
            centers_fd = force_directed_refinement(centers, max_iters=100, dt=0.001)
            r = compute_radius(centers_fd)
            
            if r > best_radius:
                best_radius = r
                best_centers = centers_fd.copy()

    # Also try square grid as alternative
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    centers_fd = force_directed_refinement(grid, max_iters=100, dt=0.001)
    r = compute_radius(centers_fd)
    if r > best_radius:
        best_radius = r
        best_centers = centers_fd.copy()

    # Polish the best configuration with Nelder-Mead
    if best_centers is not None and time.time() < deadline - 30:
        polished = nelder_mead_polish(best_centers, max_iter=300)
        r = compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()

    # Final coordinate ascent polish
    if best_centers is not None and time.time() < deadline - 5:
        polished = coordinate_ascent_polish(best_centers, max_iter=20)
        r = compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()

    # Fallback
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
# EVOLVE-BLOCK-END
