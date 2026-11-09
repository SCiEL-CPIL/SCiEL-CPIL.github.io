"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _score(pts):
    # pts: (N,3) array of centres in [-0.5, 0.5]^3
    # Returns the achievable radius (min of half-pairwise-distances and wall clearances)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall)
    return float(min(min_pair, min_wall))

def _fibonacci_sphere(n, scale):
    # Deterministic well-separated initial points via Fibonacci lattice in 3D
    pts = np.empty((n, 3))
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n):
        # Fibonacci sphere mapping
        phi = math.acos(1.0 - 2.0 * (i + 0.5) / n)
        theta = 2.0 * math.pi * i / golden
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        pts[i] = [x, y, z]
    # Scale to fit inside the unit cube [-0.5, 0.5]^3 with margin
    pts *= scale
    # Add a small deterministic jitter to break symmetries
    rng = np.random.default_rng(42)
    pts += rng.uniform(-0.001, 0.001, size=pts.shape)
    # Clamp to be strictly inside
    pts = np.clip(pts, -0.45, 0.45)
    return pts

def _initial_grid(n):
    # Fallback: cubic grid
    m = int(math.ceil(n ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill

    # Start with Fibonacci lattice initialization
    pts = _fibonacci_sphere(N, 0.35)
    best_pts = pts.copy()
    best_r = _score(best_pts)

    # Iterative repulsion optimization
    # Each iteration: compute forces pushing spheres apart, apply small step
    step_size = 0.01
    momentum = 0.5
    velocity = np.zeros_like(pts)

    iteration = 0
    while time.time() < deadline:
        # Compute pairwise distances
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)

        # Force: push apart based on inverse distance (stronger when close)
        # Use a soft repulsion: F = 1/d^2 direction
        force = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                dist = d[i, j]
                if dist < 1e-10:
                    continue
                # Repulsion magnitude
                mag = 1.0 / (dist * dist)
                force[i] += mag * diff / dist
                force[j] -= mag * diff / dist

        # Wall forces: push inward if too close to boundary
        for i in range(N):
            for k in range(3):
                if pts[i, k] > 0.4:
                    force[i, k] -= 10.0 * (pts[i, k] - 0.4)
                elif pts[i, k] < -0.4:
                    force[i, k] += 10.0 * (-0.4 - pts[i, k])

        # Update with momentum
        velocity = momentum * velocity + step_size * force
        pts_new = pts + velocity

        # Clamp to cube
        pts_new = np.clip(pts_new, -0.499, 0.499)

        # Check if improvement
        r_new = _score(pts_new)
        if r_new > best_r:
            best_r = r_new
            best_pts = pts_new.copy()
            # Increase step slightly on improvement
            step_size = min(step_size * 1.05, 0.05)
        else:
            # Decay step size
            step_size = max(step_size * 0.95, 0.001)

        pts = pts_new
        iteration += 1

        # Periodic check: if step is too small, try a random restart nudge
        if step_size < 0.002 and iteration % 100 == 0:
            # Add small random perturbation to escape local minima
            rng = np.random.default_rng(iteration)
            pts += rng.uniform(-0.01, 0.01, size=pts.shape)
            pts = np.clip(pts, -0.499, 0.499)
            step_size = 0.01

    # Final: ensure best_pts is valid
    best_pts = np.clip(best_pts, -0.5, 0.5)
    return best_pts
# EVOLVE-BLOCK-END
