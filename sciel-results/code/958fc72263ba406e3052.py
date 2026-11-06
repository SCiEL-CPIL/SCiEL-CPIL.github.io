"""
Program for packing 50 equal circles in a unit circle.
Strategy: Physics-based simulation (Molecular Dynamics) with Verlet integration.
Uses repulsive forces and a soft-wall constraint to maximize the minimum distance.
Includes Simulated Annealing to escape local minima.
"""
import numpy as np
import time
import random

N = 50
# Safety margin: The evaluator kills at 150s. We aim to finish by 145s.
DEADLINE = time.time() + 145.0


def compute_radius(points):
    """Compute the packing radius for given centres."""
    points = np.asarray(points, dtype=np.float64)
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(boundary_dists.min(), min_pair_dist / 2.0)


def seed_hexagonal():
    """Generate a hexagonal lattice packing as a seed."""
    spacing = 0.2
    points_list = []
    for j in range(-5, 6):
        for i in range(-5, 6):
            x = i * spacing + (0.5 * spacing if j % 2 else 0)
            y = j * spacing * np.sqrt(3) / 2
            if np.sqrt(x * x + y * y) < 1.0:
                points_list.append([x, y])
    
    points = np.array(points_list)
    if len(points) < N:
        # Fallback if lattice is too small (should not happen with these params)
        return np.random.uniform(-0.8, 0.8, size=(N, 2))
        
    # Sort by distance from center to prefer central points
    dists = np.linalg.norm(points, axis=1)
    points = points[np.argsort(dists)]
    
    # Select N points with largest minimum spacing (Greedy)
    selected = [points[0]]
    while len(selected) < N:
        sel = np.array(selected)
        min_dists = np.min(np.linalg.norm(points[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
        selected.append(points[np.argmax(min_dists)])
    return np.array(selected)


def seed_random():
    """Generate a random valid packing."""
    # Try random points with rejection
    for _ in range(1000):
        points = np.random.uniform(-1, 1, size=(N, 2))
        norms = np.linalg.norm(points, axis=1)
        max_norm = norms.max()
        if max_norm > 0:
            points *= (0.9 / max_norm)
        r = compute_radius(points)
        if r > 0.08:
            break
    return points


def simulate_annealing(initial_points, max_time=10.0):
    """
    Molecular Dynamics simulation with Simulated Annealing.
    Forces:
    1. Repulsion between circles: F ~ 1/d^2 (strong when close)
    2. Soft wall: F ~ (1 - r)^2 (pushes back if outside)
    """
    points = initial_points.copy()
    velocities = np.zeros_like(points)
    
    # Parameters
    k_rep = 50.0      # Repulsion strength
    k_wall = 100.0    # Wall strength
    dt = 0.005        # Time step
    T_start = 0.5     # Initial temperature
    T_end = 0.001     # Final temperature
    cooling_rate = 0.999
    
    T = T_start
    start_time = time.time()
    best_points = points.copy()
    best_radius = compute_radius(points)
    
    while time.time() < start_time + max_time and T > T_end:
        if time.time() > DEADLINE - 5.0:
            break
            
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Force magnitude: k / d^2
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-6, 1e-6, dists)
        force_mag = k_rep / (dists_safe ** 2)
        np.fill_diagonal(force_mag, 0.0)
        
        # Force vector = magnitude * unit_vector
        # unit_vector = diff / dists
        unit_vecs = diff / dists_safe[:, :, np.newaxis]
        forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Wall constraint: if |x| > 1, push inwards
        norms = np.linalg.norm(points, axis=1)
        wall_violation = norms - 1.0
        wall_force_mag = np.where(wall_violation > 0, k_wall * wall_violation, 0.0)
        # Direction: -x / |x|
        wall_dirs = points / np.where(norms < 1e-6, 1e-6, norms)[:, np.newaxis]
        forces -= wall_force_mag[:, np.newaxis] * wall_dirs
        
        # Verlet integration
        # v_new = v_old + a*dt + noise
        # x_new = x_old + v_new*dt
        
        # Noise (Simulated Annealing)
        noise = np.random.normal(0, T, size=(N, 2))
        
        velocities += forces * dt
        velocities += noise
        
        # Damping to help convergence
        velocities *= 0.99
        
        points += velocities * dt
        
        # Hard constraint: if any point is way outside, pull it back
        norms = np.linalg.norm(points, axis=1)
        too_far = norms > 1.0
        if np.any(too_far):
            points[too_far] *= 0.999 / norms[too_far]
        
        # Update best
        current_radius = compute_radius(points)
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        T *= cooling_rate
        
    return best_points, best_radius


def local_refinement(points, n_steps=500, step_size=0.01):
    """Fast coordinate descent local refinement."""
    points = points.copy()
    current_r = compute_radius(points)
    
    for step in range(n_steps):
        if time.time() > DEADLINE - 5.0:
            break
        i = random.randint(0, N - 1)
        best_new_r = current_r
        best_new_point = points[i].copy()
        
        # Try 8 neighbors + current
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0:
                    continue
                new_point = points[i].copy()
                new_point[0] += dx
                new_point[1] += dy
                if np.linalg.norm(new_point) >= 1.0:
                    continue
                points[i] = new_point
                new_r = compute_radius(points)
                if new_r > best_new_r:
                    best_new_r = new_r
                    best_new_point = new_point.copy()
                else:
                    points[i] = points[i] - np.array([dx, dy])
        
        if best_new_r > current_r:
            points[i] = best_new_point
            current_r = best_new_r
    return points


def search_portfolio():
    """
    Portfolio search:
    1. Hexagonal seed -> MD -> Refine
    2. Random seeds -> MD -> Refine
    Keep the best.
    """
    best_points = None
    best_radius = 0.0
    
    # 1. Hexagonal Start
    seed_hex = seed_hexagonal()
    r_hex = compute_radius(seed_hex)
    
    # Run MD from hex
    hex_optimized, r_hex_opt = simulate_annealing(seed_hex, max_time=30.0)
    if r_hex_opt > best_radius:
        best_radius = r_hex_opt
        best_points = hex_optimized.copy()
    
    # 2. Random Starts
    n_random_seeds = 5
    for i in range(n_random_seeds):
        if time.time() > DEADLINE - 20.0:
            break
        seed = seed_random()
        r_seed = compute_radius(seed)
        
        # Only run MD if seed is decent, or if we have lots of time
        if r_seed > 0.1 or i < 2:
            opt_points, r_opt = simulate_annealing(seed, max_time=20.0)
            if r_opt > best_radius:
                best_radius = r_opt
                best_points = opt_points.copy()
    
    # 3. Final Refinement on best
    if best_points is not None and time.time() < DEADLINE - 5.0:
        best_points = local_refinement(best_points, n_steps=1000, step_size=0.005)
        best_radius = compute_radius(best_points)
        
    # 4. Second pass refinement with smaller steps
    if best_points is not None and time.time() < DEADLINE - 2.0:
        best_points = local_refinement(best_points, n_steps=500, step_size=0.001)
        best_radius = compute_radius(best_points)
        
    if best_points is None:
        # Fallback
        best_points = seed_hexagonal()
        
    return best_points


def construct():
    """Construct centres of N equal circles inside the unit circle."""
    global DEADLINE
    DEADLINE = time.time() + 145.0
    
    np.random.seed(42)
    random.seed(42)
    
    points = search_portfolio()
    
    # Final validation and clipping
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1)
    for i in range(N):
        if norms[i] >= 1.0:
            points[i] *= 0.999 / norms[i]
    
    # Ensure finite values
    if not np.all(np.isfinite(points)):
        points = seed_hexagonal()
        
    return points


if __name__ == "__main__":
    points = construct()
    print("Final radius:", compute_radius(points))
