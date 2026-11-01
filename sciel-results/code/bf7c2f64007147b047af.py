"""
Packing 100 equal spheres in a unit cube.
Strategy: Simulated Annealing with Force-Directed Refinement.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 145.0  # Safety margin from 150s limit

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls: min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|) for each point
    # Actually, distance to nearest wall for point (x,y,z) in [-0.5, 0.5] is:
    # min(0.5-x, 0.5+x, 0.5-y, 0.5+y, 0.5-z, 0.5+z) = 0.5 - max(|x|, |y|, |z|)
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    if min_wall < 0:
        return 0.0

    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall, min_pair_dist / 2.0)

def _force_directed_refinement(centers, max_iter=200, tol=1e-10):
    """
    Refine centers using gradient descent on a potential energy function.
    Energy is high when spheres overlap or are too close to walls.
    """
    centers = centers.copy()
    current_energy = float('inf')
    
    for _ in range(max_iter):
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Wall forces: push away from walls if too close
        # We want to maximize min distance to wall.
        # Let's use a soft penalty for being close to walls.
        # d_i = 0.5 - max(|c_i|)
        # If d_i < r_target, push in. But we don't have r_target fixed.
        # Instead, let's just maximize the minimum distance.
        # A simpler approach for refinement: 
        # 1. Calculate all pairwise distances and wall distances.
        # 2. Identify the "bottleneck" distance (the minimum).
        # 3. Move spheres apart from the bottleneck.
        
        # Alternative: Minimize a potential energy U = sum(1/d^2) for close pairs + wall penalties.
        # Let's use a specific potential: 
        # U = sum_{i<j} (1/d_ij^2) * (if d_ij < 2*r_est) + sum_i (1/w_i^2) * (if w_i < r_est)
        # This is tricky without a fixed r.
        
        # Better approach for refinement: 
        # Use the objective: maximize r.
        # This is equivalent to minimizing the "energy" of violations.
        # Let's define a target radius r_target based on current best estimate or just use the current min distance.
        
        # Let's use a generic potential that pushes spheres apart and away from walls.
        # U = sum_{i<j} max(0, (2*r_est - d_ij)^2) + sum_i max(0, (r_est - w_i)^2)
        # We need an estimate of r. Let's use the current computed radius.
        
        r_est = _compute_radius(centers)
        if r_est <= 0:
             return centers, 0.0
            
        # Compute pairwise distances
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        
        # Forces from pairs
        # For pairs where d < 2*r_est, force is proportional to (2*r_est - d) * unit_vec
        # We want to increase d.
        # Force on i due to j: k * (2*r_est - d_ij) * (c_i - c_j)/d_ij
        
        # Vectorized calculation
        # We only care about pairs where dists < 2*r_est
        mask = dists < 2 * r_est
        # Set diagonal to 0 to avoid self-interaction
        np.fill_diagonal(mask, False)
        
        # Calculate unit vectors
        # diff = centers[:, None, :] - centers[None, :, :]
        # This is N x N x 3, which is 100x100x3 = 30000 floats. Manageable.
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_safe = np.where(dists > 1e-8, dists, 1.0)
        unit_vecs = diffs / dists_safe[:, :, np.newaxis]
        
        # Force magnitude
        # F_ij = k * (2*r_est - d_ij) if d_ij < 2*r_est else 0
        force_mag = np.where(mask, (2 * r_est - dists), 0.0)
        
        # Total force on i is sum of forces from all j
        # F_i = sum_j F_ij * unit_vec_ij
        forces = np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Wall forces
        # If w_i < r_est, push towards center.
        # Direction: -sign(c_i) component-wise? No, towards center is -c_i.
        # But we need to handle the max(|x|,|y|,|z|) derivative.
        # Gradient of max(|x|,|y|,|z|) is sign of the max component.
        # So gradient of w_i = - sign(c_i[argmax]).
        # Force should be opposite to gradient of penalty.
        # Penalty is (r_est - w_i)^2. Grad penalty = 2(r_est - w_i) * (-grad w_i).
        # grad w_i = - sign(c_i[argmax]).
        # So Grad penalty = -2(r_est - w_i) * sign(c_i[argmax]).
        # Force = -Grad penalty = 2(r_est - w_i) * sign(c_i[argmax]).
        
        max_abs_idx = np.argmax(np.abs(centers), axis=1)
        signs = np.sign(centers[np.arange(N), max_abs_idx])
        
        wall_force_mag = np.where(wall_dists < r_est, 2 * (r_est - wall_dists), 0.0)
        
        # Apply wall forces to the specific coordinate
        for i in range(N):
            if wall_force_mag[i] > 0:
                forces[i, max_abs_idx[i]] += wall_force_mag[i] * signs[i]
        
        # Update centers
        # Simple gradient ascent on radius (or descent on energy)
        # We are maximizing r, so we move in direction of increasing distances.
        # The forces calculated above push spheres apart and away from walls.
        # So we add forces * step_size.
        
        # Adaptive step size
        step_size = 0.01 * r_est
        # Normalize forces to prevent huge jumps
        norm_forces = np.linalg.norm(forces, axis=1, keepdims=True)
        norm_forces = np.where(norm_forces > 0, norm_forces, 1.0)
        normalized_forces = forces / norm_forces
        
        centers += step_size * normalized_forces
        
        # Keep centers in bounds
        centers = np.clip(centers, -0.5, 0.5)
        
        # Check convergence (optional, can be expensive)
        # new_r = _compute_radius(centers)
        # if abs(new_r - current_energy) < tol: break
        
    return centers, _compute_radius(centers)

def _simulated_annealing(centers, time_limit, seed=42):
    """
    Run Simulated Annealing to improve the packing.
    """
    rng = np.random.default_rng(seed)
    current_centers = centers.copy()
    current_r = _compute_radius(current_centers)
    
    best_centers = current_centers.copy()
    best_r = current_r
    
    # Initial temperature
    T = 0.05
    cooling_rate = 0.995
    
    # We will perform steps until time runs out
    start_time = time.time()
    
    while time.time() < time_limit:
        # Generate a random perturbation
        # Perturb a random subset of spheres
        n_perturb = rng.integers(1, 10)
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        # Random displacement
        # Scale displacement with T
        displacements = rng.normal(0, T, (n_perturb, 3))
        
        new_centers = current_centers.copy()
        new_centers[indices] += displacements
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_r = _compute_radius(new_centers)
        
        # Acceptance criterion
        if new_r > current_r:
            # Always accept improvement
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
        else:
            # Accept worse with probability exp((new_r - current_r) / T)
            # Since we are maximizing, delta = new_r - current_r < 0
            # prob = exp(delta / T)
            delta = new_r - current_r
            if delta < -0.01 * T: # Optimization: avoid exp underflow
                pass
            else:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_centers = new_centers
                    current_r = new_r
        
        # Cool down
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6
            
        # Periodic refinement
        # Every 1000 steps, do a quick refinement to snap to local optimum
        # This is expensive, so we do it less frequently
        if rng.random() < 0.01:
            refined, r_ref = _force_directed_refinement(current_centers, max_iter=50)
            if r_ref > current_r:
                current_centers = refined
                current_r = r_ref
                if current_r > best_r:
                    best_r = current_r
                    best_centers = current_centers.copy()
        
    return best_centers, best_r

def _init_grid(n=100, seed=42):
    nx, ny, nz = 5, 5, 4
    margin = 0.05
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers[:n]

def _init_fcc_like(n=100, seed=42):
    # FCC lattice points in a cube
    # Points at corners, face centers, and body center (if BCC)
    # FCC: corners + face centers
    # Let's generate a grid and add FCC offsets
    centers = []
    # 4x4x4 grid of FCC unit cells?
    # Simpler: Random jittered grid
    return _init_grid(n, seed)

def _init_random(n=100, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, (n, 3))

def construct():
    start_time = time.time()
    
    # List of initializations
    inits = [
        _init_grid(N, seed=1),
        _init_random(N, seed=2),
        _init_grid(N, seed=3) # Different jitter
    ]
    
    best_overall_centers = None
    best_overall_r = 0.0
    
    # Allocate time to each initialization
    # We have ~145s total.
    # Let's do 3 runs, each with ~40s budget.
    time_per_run = 40.0
    
    for i, init in enumerate(inits):
        if time.time() > DEADLINE - 5.0:
            break
            
        run_deadline = start_time + (i + 1) * time_per_run
        run_deadline = min(run_deadline, DEADLINE - 5.0)
        
        # Run SA
        c_opt, r_opt = _simulated_annealing(init, run_deadline, seed=100+i)
        
        if r_opt > best_overall_r:
            best_overall_r = r_opt
            best_overall_centers = c_opt.copy()
            
    # Final polish on the best result
    if best_overall_centers is not None and time.time() < DEADLINE - 2.0:
        refined, r_ref = _force_directed_refinement(best_overall_centers, max_iter=100)
        if r_ref > best_overall_r:
            best_overall_r = r_ref
            best_overall_centers = refined.copy()
            
    # Fallback
    if best_overall_centers is None:
        best_overall_centers = _init_grid(N)
        best_overall_r = _compute_radius(best_overall_centers)
        
    return best_overall_centers
