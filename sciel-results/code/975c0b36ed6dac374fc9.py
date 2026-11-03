"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct 300 equal circle centers inside the unit disk to maximize the minimum
    common radius. Uses a multi-start simulated annealing approach with a hexagonal
    lattice initialization.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Target radius based on known best
    target_r = 0.0531542991469

    def hex_init(n):
        """Generate a hexagonal lattice packing of n points in the unit disk."""
        # Estimate spacing for n points in unit disk
        # Area of disk = pi, area per point ~ pi/n
        # Hex spacing s: each point occupies area ~ s^2 * sqrt(3)/2
        # So s ~ sqrt(2*pi/(n*sqrt(3)))
        s = np.sqrt(2 * np.pi / (n * np.sqrt(3))) * 0.95

        # Generate hex lattice points
        points = []
        # For hex lattice, points are at:
        # (i*s + j*s/2, j*s*sqrt(3)/2) for integers i, j
        # with appropriate offsets
        max_coord = int(1.0 / s) + 2

        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if x * x + y * y <= (1.0 - s / 2) ** 2:
                    points.append([x, y])

        points = np.array(points)
        if len(points) < n:
            # If not enough points, reduce spacing
            s *= 0.9
            points = []
            max_coord = int(1.0 / s) + 2
            for j in range(-max_coord, max_coord + 1):
                for i in range(-max_coord, max_coord + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3) / 2
                    if x * x + y * y <= (1.0 - s / 2) ** 2:
                        points.append([x, y])
            points = np.array(points)

        # Select n points closest to center
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            return points[idx]
        else:
            # Fallback: random points
            angles = np.random.uniform(0, 2 * np.pi, n)
            radii = np.sqrt(np.random.uniform(0, 1, n))
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def compute_score(pts):
        """Compute the minimum common radius for a set of points."""
        n = len(pts)
        # Distance to boundary
        dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)

        # Pairwise distances
        if n < 2:
            return np.min(dist_boundary)

        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)

        return min(min_pair_dist / 2, np.min(dist_boundary))

    def anneal(pts, time_budget):
        """Run simulated annealing to improve the packing."""
        n = len(pts)
        best_pts = pts.copy()
        best_score = compute_score(pts)
        current_pts = pts.copy()
        current_score = best_score

        # Initial temperature
        T = 0.001
        cooling = 0.999

        steps = 0
        while time.time() < time_budget:
            # Randomly perturb one or two points
            if np.random.random() < 0.5:
                idx = np.random.randint(n)
                delta = np.random.normal(0, T, size=2)
                new_pts = current_pts.copy()
                new_pts[idx] += delta
            else:
                idx1, idx2 = np.random.choice(n, 2, replace=False)
                delta1 = np.random.normal(0, T, size=2)
                delta2 = np.random.normal(0, T, size=2)
                new_pts = current_pts.copy()
                new_pts[idx1] += delta1
                new_pts[idx2] += delta2

            # Check boundary
            if np.any(np.linalg.norm(new_pts, axis=1) > 1.0):
                new_score = 0.0
            else:
                new_score = compute_score(new_pts)

            # Accept or reject
            if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
                current_pts = new_pts
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()

            T *= cooling
            steps += 1

            # Early termination if we've hit the target
            if best_score >= target_r:
                break

        return best_pts, best_score

    # Multi-start annealing
    best_pts = None
    best_score = 0.0
    n_starts = 5

    for start in range(n_starts):
        if time.time() > deadline - 10:
            break

        # Generate initial configuration
        if start == 0:
            pts = hex_init(N)
        else:
            # Perturb the best found so far
            if best_pts is not None:
                pts = best_pts + np.random.normal(0, 0.01, size=(N, 2))
            else:
                pts = hex_init(N)

        # Ensure points are within unit disk
        norms = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                pts[i] = pts[i] / norms[i]

        # Run annealing with time budget
        time_budget = deadline - 15
        pts, score = anneal(pts, time_budget)

        if score > best_score:
            best_score = score
            best_pts = pts

        if best_score >= target_r:
            break

    # Final check: ensure all points are valid
    if best_pts is None:
        best_pts = hex_init(N)

    norms = np.linalg.norm(best_pts, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            best_pts[i] = best_pts[i] / norms[i] * 0.99

    return best_pts
# EVOLVE-BLOCK-END
