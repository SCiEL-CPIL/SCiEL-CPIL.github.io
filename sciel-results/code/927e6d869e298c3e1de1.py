"""Circle packing in perimeter-4 rectangle using Voronoi-based inradius optimization."""
import numpy as np
from scipy.spatial import Voronoi
import numpy as np

N = 21


def _compute_inradii(centers, W, H):
    """
    For each circle center, compute the inradius of its Voronoi cell clipped to [0,W]x[0,H].
    The inradius is the minimum distance from the center to the boundary of its cell.
    """
    n = len(centers)
    if n < 3:
        # Fallback: just min distance to walls
        radii = np.zeros(n)
        for i in range(n):
            radii[i] = min(centers[i, 0], W - centers[i, 0], centers[i, 1], H - centers[i, 1])
        return radii

    try:
        vor = Voronoi(centers)
    except Exception:
        # Fallback if Voronoi fails
        radii = np.zeros(n)
        for i in range(n):
            radii[i] = min(centers[i, 0], W - centers[i, 0], centers[i, 1], H - centers[i, 1])
        return radii

    radii = np.zeros(n)
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if not region or -1 in region:
            # Unbounded region; distance to walls
            radii[i] = min(centers[i, 0], W - centers[i, 0], centers[i, 1], H - centers[i, 1])
            continue

        min_dist = float('inf')
        # Distance to walls
        min_dist = min(min_dist, centers[i, 0], W - centers[i, 0], centers[i, 1], H - centers[i, 1])

        # Distance to each edge of the Voronoi cell
        for j in region:
            edge = vor.ridge_points[j]
            if edge is None or -1 in edge:
                continue
            p1 = vor.vertices[edge[0]]
            p2 = vor.vertices[edge[1]]
            # Distance from center to line segment p1-p2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-12:
                dist = np.sqrt((centers[i, 0] - p1[0]) ** 2 + (centers[i, 1] - p1[1]) ** 2)
            else:
                t = ((centers[i, 0] - p1[0]) * dx + (centers[i, 1] - p1[1]) * dy) / seg_len_sq
                t = max(0.0, min(1.0, t))
                proj_x = p1[0] + t * dx
                proj_y = p1[1] + t * dy
                dist = np.sqrt((centers[i, 0] - proj_x) ** 2 + (centers[i, 1] - proj_y) ** 2)
            min_dist = min(min_dist, dist)

        radii[i] = min_dist

    return radii


def construct():
    # Rectangle with perimeter 4: 2*(W+H) = 4 => W+H = 2
    # We'll optimize W as well, but start with a reasonable aspect ratio
    # For 21 circles, a wider rectangle might be better. Let's try W=1.4, H=0.6 initially
    # Actually, let's use a grid-like initial placement and optimize.

    # Initial aspect ratio: try several and pick best, or just optimize W too.
    # Let's fix W+H=2 and optimize W in [0.5, 1.5] along with centers.

    # Start with a grid-like arrangement
    # 21 circles: maybe 5x5 grid with some removed, or 7x3
    # Let's use a 7x3 grid for initial placement
    cols, rows = 7, 3
    W0 = 1.4
    H0 = 0.6
    centers = np.zeros((N, 2))
    for i in range(N):
        c = i % cols
        r = i // cols
        centers[i, 0] = (c + 0.5) / cols * W0
        centers[i, 1] = (r + 0.5) / rows * H0

    # Optimize W as well
    W = W0
    H = 2.0 - W

    # Repulsion-based optimization
    # Force model: each pair of circles repels if their inradii sum exceeds distance
    # We want to maximize sum of inradii, so we use gradient-free or gradient-based approach
    # Use simple iterative repulsion + wall attraction

    np.random.seed(42)
    best_sum = 0.0
    best_centers = centers.copy()
    best_W = W

    for iteration in range(2000):
        H = 2.0 - W
        radii = _compute_inradii(centers, W, H)
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_W = W

        # Compute forces
        forces = np.zeros_like(centers)

        # Wall forces: push circles away from walls if they're too close
        for i in range(N):
            # Left wall
            if centers[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - centers[i, 0]) * 0.5
            # Right wall
            if centers[i, 0] > W - radii[i]:
                forces[i, 0] -= (centers[i, 0] - (W - radii[i])) * 0.5
            # Bottom wall
            if centers[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - centers[i, 1]) * 0.5
            # Top wall
            if centers[i, 1] > H - radii[i]:
                forces[i, 1] -= (centers[i, 1] - (H - radii[i])) * 0.5

        # Pairwise repulsion: if two circles overlap (distance < r_i + r_j), push apart
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-10:
                    # Random small perturbation
                    dx = np.random.randn() * 1e-5
                    dy = np.random.randn() * 1e-5
                    dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = (min_dist - dist)
                    fx = (dx / dist) * overlap
                    fy = (dy / dist) * overlap
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy

        # Also add a gentle force to maximize inradii: push circles toward the "center" of their Voronoi cell
        # This is approximated by moving toward the centroid of the Voronoi region
        # For simplicity, use a small attraction toward the rectangle center for under-utilized space
        center_x = W / 2.0
        center_y = H / 2.0
        for i in range(N):
            # Mild attraction to center if far from it
            forces[i, 0] += (center_x - centers[i, 0]) * 0.001
            forces[i, 1] += (center_y - centers[i, 1]) * 0.001

        # Update positions
        lr = 0.05
        centers += lr * forces

        # Clamp to rectangle
        for i in range(N):
            centers[i, 0] = max(0.01, min(W - 0.01, centers[i, 0]))
            centers[i, 1] = max(0.01, min(H - 0.01, centers[i, 1]))

        # Slowly adjust W to find better aspect ratio
        # Try small perturbations to W
        if iteration % 100 == 0:
            W_perturb = W + np.random.randn() * 0.02
            W_perturb = max(0.5, min(1.5, W_perturb))
            H_perturb = 2.0 - W_perturb
            # Quick check: scale centers to new rectangle
            centers_scaled = centers.copy()
            centers_scaled[:, 0] *= W_perturb / W
            centers_scaled[:, 1] *= H_perturb / H
            radii_test = _compute_inradii(centers_scaled, W_perturb, H_perturb)
            if np.sum(radii_test) > current_sum * 0.95:
                W = W_perturb
                H = 2.0 - W
                centers = centers_scaled

    # Final refinement: local search on centers with fixed best_W
    W = best_W
    H = 2.0 - W
    centers = best_centers.copy()

    for iteration in range(1000):
        radii = _compute_inradii(centers, W, H)
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_W = W

        # Gradient-free: try small random perturbations
        improved = False
        for _ in range(50):
            trial = centers.copy()
            idx = np.random.randint(N)
            trial[idx, 0] += np.random.randn() * 0.01
            trial[idx, 1] += np.random.randn() * 0.01
            trial[idx, 0] = max(0.01, min(W - 0.01, trial[idx, 0]))
            trial[idx, 1] = max(0.01, min(H - 0.01, trial[idx, 1]))
            trial_radii = _compute_inradii(trial, W, H)
            trial_sum = np.sum(trial_radii)
            if trial_sum > current_sum:
                centers = trial
                current_sum = trial_sum
                improved = True
                break

        if not improved and iteration > 500:
            break

    # Final result
    W = best_W
    H = 2.0 - W
    centers = best_centers
    radii = _compute_inradii(centers, W, H)

    # Ensure disjoint interiors: shrink radii slightly if needed
    # Check all pairs
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j]:
                # Shrink both
                excess = radii[i] + radii[j] - dist
                shrink = excess / 2.0 + 1e-6
                radii[i] = max(1e-6, radii[i] - shrink)
                radii[j] = max(1e-6, radii[j] - shrink)

    # Also ensure circles are inside rectangle
    for i in range(N):
        radii[i] = min(radii[i], centers[i, 0], W - centers[i, 0], centers[i, 1], H - centers[i, 1])
        radii[i] = max(1e-6, radii[i])

    result = np.column_stack([centers[:, 0], centers[:, 1], radii])
    return result
