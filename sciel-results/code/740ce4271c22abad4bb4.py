"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np


def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n, seed=0.0):
    """Generate n points on the sphere using a Fibonacci spiral with an offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng=None):
    """Generate n random points on the unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _lloyd_sweep(points, rng, temperature=0.0):
    """
    One Lloyd-style sweep: for each point, replace it with the centroid
    of all other points projected onto the sphere. Optionally add noise.
    This is a simple coordinate-descent that minimizes pairwise log-energy.
    """
    n = len(points)
    new_pts = points.copy()
    for i in range(n):
        # Centroid of all points except i
        others = np.delete(points, i, axis=0)
        centroid = np.mean(others, axis=0)
        norm_c = np.linalg.norm(centroid)
        if norm_c > 1e-15:
            new_pts[i] = centroid / norm_c
        else:
            # Fallback: keep old point or random
            if temperature > 0:
                new_pts[i] = _random_sphere(1, rng)[0]
            else:
                new_pts[i] = points[i]
    
    if temperature > 0:
        # Add small random perturbation to escape local minima
        noise = rng.standard_normal((n, 3)) * temperature
        new_pts = new_pts + noise
        new_pts = _normalize(new_pts)
    
    return new_pts


def _anneal_lloyd(points, rng, n_sweeps=200, temp_start=0.1, temp_end=0.001):
    """
    Run multiple Lloyd sweeps with annealed temperature.
    """
    pts = points.copy()
    for s in range(n_sweeps):
        # Linear annealing
        t = temp_start * (temp_end / temp_start) ** (s / max(n_sweeps - 1, 1))
        pts = _lloyd_sweep(pts, rng, temperature=t)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_energy = float('inf')
    best_points = _fibonacci_sphere(N)

    # 1. Evaluate initial Fibonacci
    try:
        e = _log_energy(best_points)
        if e < best_energy:
            best_energy = e
    except Exception:
        pass

    # Refine the Fibonacci start with Lloyd + annealing
    rng = np.random.default_rng(42)
    refined = _anneal_lloyd(best_points, rng, n_sweeps=300, temp_start=0.05, temp_end=0.001)
    try:
        e = _log_energy(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 20:
        return _normalize(best_points)

    # 2. Portfolio of random restarts with Lloyd + annealing
    num_restarts = 100

    for k in range(num_restarts):
        if time.time() > deadline - 10:
            break

        # Random start
        pts = _random_sphere(N, rng)

        # Quick Lloyd refinement with annealing
        try:
            refined_pts = _anneal_lloyd(pts, rng, n_sweeps=150, temp_start=0.08, temp_end=0.001)
            e = _log_energy(refined_pts)

            if e < best_energy:
                best_energy = e
                best_points = refined_pts

                # If it's good, refine more with lower temperature
                if time.time() < deadline - 15:
                    further_refined = _anneal_lloyd(refined_pts, rng, n_sweeps=200, temp_start=0.02, temp_end=0.0005)
                    e2 = _log_energy(further_refined)
                    if e2 < best_energy:
                        best_energy = e2
                        best_points = further_refined
        except Exception:
            continue

    # 3. Final polish if time permits
    if time.time() < deadline - 5:
        try:
            final_pts = _anneal_lloyd(best_points, rng, n_sweeps=500, temp_start=0.01, temp_end=0.0001)
            e = _log_energy(final_pts)
            if e < best_energy:
                best_energy = e
                best_points = final_pts
        except Exception:
            pass

    return _normalize(best_points)

# EVOLVE-BLOCK-END
