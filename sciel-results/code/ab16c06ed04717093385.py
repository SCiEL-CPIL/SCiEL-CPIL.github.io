"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # Grid initialization
    m = int(math.ceil(math.sqrt(N)))  # 14 for N=200
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()

    # Compute radius efficiently using vectorized operations
    def compute_radius(c):
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)

        # Boundary distances
        dist_x = 0.5 - np.abs(c[:, 0])
        dist_y = 0.5 - np.abs(c[:, 1])
        min_boundary = np.min(np.minimum(dist_x, dist_y))

        return min(min_pair_dist / 2.0, min_boundary)

    def optimize_force_directed(centers, max_time):
        """Physics-based annealing: repulsive forces + boundary constraints."""
        centers = centers.copy()
        n = len(centers)

        # Parameters
        repulsion_power = 4  # Force ~ 1/d^4
        repulsion_strength = 0.01
        boundary_strength = 0.1
        initial_temp = 0.01
        cooling_rate = 0.999
        min_temp = 1e-6
        max_step = 0.05

        temp = initial_temp
        best_radius = compute_radius(centers)
        best_centers = centers.copy()

        while time.time() < max_time:
            # Compute forces
            forces = np.zeros_like(centers)

            # Pairwise repulsive forces (vectorized)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, 1.0)  # avoid division by zero

            # Force magnitude: k / d^power, direction: along diff
            # F_ij = k * (x_i - x_j) / d^power
            force_magnitudes = repulsion_strength / (dists ** repulsion_power)
            np.fill_diagonal(force_magnitudes, 0.0)

            # Sum forces: F_i = sum_j F_ij * (x_i - x_j) / d_ij
            # Actually: F_i = sum_j k * (x_i - x_j) / d_ij^power
            # = sum_j force_magnitudes[i,j] * diff[i,j] / dists[i,j]
            # But diff[i,j] = x_i - x_j, and we want force on i from j to be away from j
            # So F_i += k * (x_i - x_j) / d^power
            # = force_magnitudes[i,j] * (x_i - x_j) / d_ij
            # Wait: force_magnitudes already includes 1/d^power, so:
            # F_i = sum_j force_magnitudes[i,j] * (x_i - x_j) / d_ij
            # But that's not right. Let me redo:
            # F on i due to j: k * (x_i - x_j) / |x_i - x_j|^power
            # = k * (x_i - x_j) / d^power
            # So the vector is (x_i - x_j) * k / d^power
            # = diff[i,j] * force_magnitudes[i,j] / dists[i,j]  ... no
            # force_magnitudes[i,j] = k / d^power
            # So vector = diff[i,j] * k / d^power / d = diff[i,j] * k / d^(power+1)
            # That's wrong. Let me just compute directly:
            # F_i = sum_{j != i} k * (x_i - x_j) / |x_i - x_j|^power
            # = sum_{j != i} k * diff[i,j] / dists[i,j]^power
            # So the contribution from j to i is: k * diff[i,j] / dists[i,j]^power
            # = diff[i,j] * (k / dists[i,j]^power)
            # = diff[i,j] * force_magnitudes[i,j]  (since force_magnitudes = k/d^power)
            # But we need to divide by d to get the unit direction... no, diff[i,j] is already the vector
            # diff[i,j] = x_i - x_j, which has magnitude d.
            # So the force vector on i from j is: k * (x_i - x_j) / d^power = k * diff[i,j] / d^power
            # = diff[i,j] * (k / d^power) = diff[i,j] * force_magnitudes[i,j]
            # But that gives a force proportional to d * 1/d^power = 1/d^(power-1), which is what we want.
            # Wait no: (x_i - x_j) / |x_i - x_j|^power is a vector of magnitude 1/d^(power-1)
            # That's correct for an inverse-power force.

            # So: F_i = sum_j diff[i,j] * force_magnitudes[i,j]
            # But force_magnitudes[i,j] = k / d^power, and diff[i,j] has magnitude d
            # So the contribution has magnitude d * k / d^power = k / d^(power-1)
            # That's the correct inverse-power force.

            # Actually, I realize I'm overcomplicating. Let me just do:
            # For each pair (i,j), force on i is: k * (x_i - x_j) / |x_i - x_j|^power
            # Vectorized:
            # forces[i] += sum_j k * (x_i - x_j) / dists[i,j]^power
            # = sum_j diff[i,j] * k / dists[i,j]^power
            # = sum_j diff[i,j] * force_magnitudes[i,j]

            # But this double-counts: force on i from j and force on j from i are both included
            # in the sum for i and j respectively. That's correct.

            # Let me compute: forces = np.sum(diff * force_magnitudes[:, :, np.newaxis], axis=1)
            # But diff[i,j] = x_i - x_j, and we want force on i to be in direction away from j
            # So force on i from j should be in direction (x_i - x_j), which is diff[i,j]. Correct.

            # However, this is O(N^2) memory which for N=200 is 40000 floats, fine.
            forces_pair = np.sum(diff * force_magnitudes[:, :, np.newaxis], axis=1)

            # Boundary forces: push points away from boundaries
            # For x: if x > 0, push left; if x < 0, push right
            # Force = -boundary_strength * sign(x) * (0.5 - |x|)^-something
            # Simpler: quadratic penalty near boundary
            # F_x = -boundary_strength * (x / (0.5 - |x|)) if |x| < 0.5
            # Or just: F_x = -boundary_strength * x / max(eps, 0.5 - |x|)
            # Let's use: F_x = -boundary_strength * x / (0.5 - |x| + 1e-6)
            # This pushes points toward center, stronger near boundary

            eps = 1e-6
            dist_to_boundary_x = 0.5 - np.abs(centers[:, 0])
            dist_to_boundary_y = 0.5 - np.abs(centers[:, 1])

            forces[:, 0] += forces_pair[:, 0] - boundary_strength * centers[:, 0] / (dist_to_boundary_x + eps)
            forces[:, 1] += forces_pair[:, 1] - boundary_strength * centers[:, 1] / (dist_to_boundary_y + eps)

            # Add random perturbation (annealing)
            if temp > min_temp:
                forces += temp * np.random.randn(n, 2)

            # Limit step size
            force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
            max_force = max_step
            scale = np.minimum(1.0, max_force / (force_mags + eps))
            forces *= scale

            # Update positions
            centers += forces

            # Clip to square (with small margin to stay strictly inside)
            margin = 1e-9
            centers[:, 0] = np.clip(centers[:, 0], -0.5 + margin, 0.5 - margin)
            centers[:, 1] = np.clip(centers[:, 1], -0.5 + margin, 0.5 - margin)

            # Cool down
            temp *= cooling_rate

            # Track best
            current_radius = compute_radius(centers)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()

        return best_centers

    # Run optimization
    centers = optimize_force_directed(centers, deadline - 5.0)

    # Final polish: local search on the best configuration
    # Try small random perturbations and keep if they improve
    best_radius = compute_radius(centers)
    best_centers = centers.copy()

    polish_time = deadline - 2.0
    while time.time() < polish_time:
        # Pick a random subset of points to perturb
        n_perturb = max(1, N // 10)
        idx = np.random.choice(N, n_perturb, replace=False)
        perturbation = 0.005 * np.random.randn(n_perturb, 2)

        trial = best_centers.copy()
        trial[idx] += perturbation
        trial[:, 0] = np.clip(trial[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        trial[:, 1] = np.clip(trial[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)

        trial_radius = compute_radius(trial)
        if trial_radius > best_radius:
            best_radius = trial_radius
            best_centers = trial.copy()

    return best_centers

# EVOLVE-BLOCK-END
