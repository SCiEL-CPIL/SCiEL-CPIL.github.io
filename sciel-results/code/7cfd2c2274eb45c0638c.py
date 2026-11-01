"""Improved circle packing: force-directed layout with hexagonal init and local refinement."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Initialize with hexagonal packing
    def hex_init():
        # Use a hexagonal grid pattern
        # For 50 circles, we can use a pattern like:
        # Row 0: 8 circles
        # Row 1: 7 circles (offset)
        # Row 2: 8 circles
        # Row 3: 7 circles (offset)
        # Row 4: 8 circles
        # Row 5: 7 circles (offset)
        # Row 6: 5 circles (partial)
        # Total: 8+7+8+7+8+7+5 = 50
        
        pts = []
        row_widths = [8, 7, 8, 7, 8, 7, 5]
        x_spacing = 0.125  # 1/8 for full rows
        y_spacing = x_spacing * np.sqrt(3) / 2
        
        y_offset = 0
        for row_idx, width in enumerate(row_widths):
            y = y_offset
            if row_idx % 2 == 1:
                x_start = -0.5 + x_spacing / 2 + x_spacing / 2
            else:
                x_start = -0.5 + x_spacing / 2
            
            for i in range(width):
                x = x_start + i * x_spacing
                pts.append([x, y])
            
            y_offset += y_spacing
        
        # Center the pattern
        pts = np.array(pts)
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()
        pts[:, 0] -= cx
        pts[:, 1] -= cy
        
        # Scale to fit within [-0.5, 0.5]^2
        max_ext = max(np.abs(pts).max(axis=0))
        if max_ext > 0.45:
            pts *= 0.45 / max_ext
        
        return pts

    def compute_radius(pts):
        """Compute the valid radius for given points."""
        if len(pts) < 2:
            return 0.5
        # Minimum distance between pairs
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_dist = dists.min()
        # Minimum distance to boundary
        bounds = 0.5 - np.abs(pts)
        min_bound = bounds.min()
        return min(min_dist / 2.0, min_bound)

    def force_directed(pts, max_iters=50000, dt_init=0.01, dt_end=1e-6, alpha=10.0):
        """Force-directed layout to maximize minimum distance."""
        pts = pts.copy()
        n = len(pts)
        best_pts = pts.copy()
        best_radius = compute_radius(pts)
        
        for it in range(max_iters):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Repulsive forces between pairs
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-10:
                        # Random nudge
                        diff = np.random.randn(2) * 1e-6
                        dist = np.linalg.norm(diff)
                    # Force magnitude: repulsive, stronger when close
                    force_mag = alpha / (dist ** 3 + 1e-8)
                    force = force_mag * diff / dist
                    forces[i] += force
                    forces[j] -= force
            
            # Attractive force toward center (weak, to keep points in square)
            for i in range(n):
                forces[i] -= 0.01 * pts[i]
            
            # Boundary forces: push points away from walls
            for i in range(n):
                for k in range(2):
                    if pts[i, k] > 0.5 - 0.05:
                        forces[i, k] -= 1.0 * (pts[i, k] - (0.5 - 0.05))
                    elif pts[i, k] < -0.5 + 0.05:
                        forces[i, k] += 1.0 * ((-0.5 + 0.05) - pts[i, k])
            
            # Update positions with simulated annealing
            temperature = dt_init * (dt_end / dt_init) ** (it / max_iters)
            step = np.random.randn(n, 2) * temperature
            new_pts = pts + forces * dt_init * 0.01 + step
            
            # Clip to square
            new_pts = np.clip(new_pts, -0.5, 0.5)
            
            new_radius = compute_radius(new_pts)
            if new_radius > best_radius or np.random.rand() < np.exp((new_radius - best_radius) / (temperature + 1e-10)):
                best_pts = new_pts.copy()
                best_radius = new_radius
            
            pts = new_pts
        
        return best_pts, best_radius

    def local_refine(pts, max_iters=20000):
        """Local refinement using gradient ascent on minimum distance."""
        pts = pts.copy()
        n = len(pts)
        best_pts = pts.copy()
        best_radius = compute_radius(pts)
        
        for it in range(max_iters):
            elapsed = time.time() - start_time
            if elapsed > deadline - 3:
                break
            
            # Find the critical pair and boundary constraint
            dists = cdist(pts, pts)
            np.fill_diagonal(dists, np.inf)
            
            min_dist = dists.min()
            min_bound = (0.5 - np.abs(pts)).min()
            
            if min_dist / 2.0 < min_bound:
                # Critical: pair distance
                idx = np.unravel_index(np.argmin(dists), dists.shape)
                i, j = idx
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    continue
                grad_i = diff / dist
                grad_j = -diff / dist
            else:
                # Critical: boundary
                idx = np.unravel_index(np.argmin(0.5 - np.abs(pts)), (n, 2))
                i, k = idx
                grad_i = np.zeros(2)
                grad_i[k] = -np.sign(pts[i, k]) if abs(pts[i, k]) > 1e-10 else 1.0
                grad_j = None
            
            # Gradient step
            lr = 1e-4 / (1 + it * 1e-5)
            if grad_j is not None:
                pts[i] += lr * grad_i
                pts[j] += lr * grad_j
            else:
                pts[i] += lr * grad_i
            
            # Clip to square
            pts = np.clip(pts, -0.5, 0.5)
            
            new_radius = compute_radius(pts)
            if new_radius > best_radius:
                best_pts = pts.copy()
                best_radius = new_radius
        
        return best_pts, best_radius

    # Main search
    best_pts = None
    best_radius = 0

    # Multiple random restarts with hexagonal init
    n_restarts = 10
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break

        # Add random perturbation to hex init
        pts = hex_init()
        if restart > 0:
            noise_scale = 0.02 * restart
            pts += np.random.randn(N, 2) * noise_scale
            pts = np.clip(pts, -0.48, 0.48)

        # Force-directed layout
        pts_fd, radius_fd = force_directed(pts, max_iters=20000)
        
        if radius_fd > best_radius:
            best_radius = radius_fd
            best_pts = pts_fd.copy()

        # Local refinement
        pts_ref, radius_ref = local_refine(pts_fd, max_iters=10000)
        
        if radius_ref > best_radius:
            best_radius = radius_ref
            best_pts = pts_ref.copy()

    # Final polish
    if best_pts is not None:
        final_pts, final_radius = local_refine(best_pts, max_iters=5000)
        if final_radius > best_radius:
            best_radius = final_radius
            best_pts = final_pts

    if best_pts is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]

    return best_pts
