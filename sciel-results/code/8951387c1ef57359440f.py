"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
from math import log, fsum
import time

# EVOLVE-BLOCK-START
import numpy as np
from math import log, fsum
import time
import random

N = 150
DEADLINE_FRACTION = 0.90  # safety margin

def _log_energy_vec(pts):
    """Vectorized log energy for fast evaluation."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Avoid log(0) and self-interaction
    np.fill_diagonal(dists, 1.0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Clip very small distances to avoid inf/nan issues in log, though we check validity later
    d = np.maximum(d, 1e-15)
    return np.sum(-np.log(d))

def _log_energy_exact(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return 1e30  # penalty for coincident points
            terms.append(-log(d))
    return fsum(terms)

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def _apply_random_rotation(pts, rng, angle):
    """Apply a small random rotation to a subset of points or all points."""
    # Generate a random rotation matrix
    # Using QR decomposition of a random matrix
    if rng.random() < 0.5:
        # Rotate all points slightly
        M = rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(M)
        d = np.sign(np.diag(R))
        Q = Q * d
        pts_new = pts @ Q.T
    else:
        # Rotate a random subset (e.g., 10-30 points) to allow local rearrangement
        k = rng.integers(10, 30)
        idx = rng.choice(N, k, replace=False)
        M = rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(M)
        d = np.sign(np.diag(R))
        Q = Q * d
        pts_new = pts.copy()
        pts_new[idx] = pts[idx] @ Q.T
    return _normalize(pts_new)

def _apply_pair_swap(pts, rng):
    """Swap two random points (effectively a 180 degree rotation of the pair relative to each other? No, just swap indices).
    Actually, swapping indices doesn't change the set of points, so it doesn't change energy.
    Instead, let's do a 'kick': move two points slightly in random directions."""
    i, j = rng.choice(N, 2, replace=False)
    pts_new = pts.copy()
    # Add small random noise to these two points
    noise_i = rng.standard_normal(3) * 0.1
    noise_j = rng.standard_normal(3) * 0.1
    pts_new[i] = pts_new[i] + noise_i
    pts_new[j] = pts_new[j] + noise_j
    return _normalize(pts_new)

def _move_single_point(pts, rng, step_size):
    """Move one random point in a random direction on the sphere."""
    i = rng.integers(0, N)
    pts_new = pts.copy()
    # Random tangent vector
    v = rng.standard_normal(3)
    v = v - np.dot(v, pts[i]) * pts[i]
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-9:
        v = v / v_norm
        # Rotate point around v by angle step_size? Or just add step_size * v
        # Adding vector and normalizing is a small geodesic step
        pts_new[i] = pts_new[i] + step_size * v
    return _normalize(pts_new)

def construct():
    start_time = time.time()
    total_time = 150.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_spiral(N)
    current_pts = best_pts.copy()
    best_energy = _log_energy_vec(best_pts)
    current_energy = best_energy
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_min = 1e-6
    cooling_rate = 0.9995
    
    T = T_start
    steps = 0
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Generate a perturbation
        r = rng.random()
        if r < 0.3:
            # Small global rotation
            new_pts = _apply_random_rotation(current_pts, rng, 0.01)
        elif r < 0.6:
            # Local subset rotation
            new_pts = _apply_random_rotation(current_pts, rng, 0.05)
        elif r < 0.8:
            # Move single point
            new_pts = _move_single_point(current_pts, rng, 0.05)
        else:
            # Pair kick
            new_pts = _apply_pair_swap(current_pts, rng)
        
        new_energy = _log_energy_vec(new_pts)
        delta_e = new_energy - current_energy
        
        # Acceptance probability
        if delta_e < 0:
            accept = True
        else:
            # Metropolis criterion
            if T > T_min:
                accept = rng.random() < np.exp(-delta_e / T)
            else:
                accept = False
        
        if accept:
            current_pts = new_pts
            current_energy = new_energy
            no_improve_steps = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
        else:
            no_improve_steps += 1
            
        # Cooling
        if steps % 100 == 0:
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        steps += 1
        
        # Restart if stuck for too long
        if no_improve_steps > max_no_improve:
            # Perturb best solution to restart
            current_pts = best_pts + rng.standard_normal((N, 3)) * 0.1
            current_pts = _normalize(current_pts)
            current_energy = _log_energy_vec(current_pts)
            T = T_start * 0.5
            no_improve_steps = 0
            
    # Final refinement with a quick local optimization on the best point found
    # Use a few steps of gradient descent or simple coordinate descent if time permits
    if time.time() < deadline - 2.0:
        # Simple local polish: try moving each point slightly in gradient direction
        # Gradient of E = sum -log(d_ij) w.r.t x_i is sum (x_i - x_j) / d^2
        pts = best_pts.copy()
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            # Compute gradient
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            mask = dists > 1e-9
            inv_d2 = np.zeros_like(dists)
            inv_d2[mask] = 1.0 / (dists[mask] ** 2)
            np.fill_diagonal(inv_d2, 0.0)
            grads = np.sum(diff * inv_d2[:, np.newaxis, :], axis=1)
            
            # Project to tangent space
            for i in range(N):
                g = grads[i]
                p = pts[i]
                grads[i] = g - np.dot(g, p) * p
            
            # Update
            step_size = 0.01
            new_pts = pts + step_size * grads
            new_pts = _normalize(new_pts)
            
            new_e = _log_energy_vec(new_pts)
            if new_e < _log_energy_vec(pts):
                pts = new_pts
            else:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        final_e = _log_energy_vec(pts)
        if final_e < best_energy:
            best_pts = pts
            best_energy = final_e

    # Ensure validity
    # Check for any points closer than 1e-9
    valid = True
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        valid = False
    
    if not valid:
        # Fallback to Fibonacci if invalid
        return _fibonacci_spiral(N)
        
    return best_pts

# EVOLVE-BLOCK-END
