import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 120.0  # 30s safety margin

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances: min distance from any point to any wall
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    if min_wall < -1e-9:
        return 0.0
    min_wall = max(0.0, min_wall)
    
    # Pairwise distances
    # Use broadcasting for memory efficiency if N is small, but N=50 is small enough for full matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist_sq = np.min(dists_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(min_wall, min_dist / 2.0)
    return r

def _hilbert_map(n_points):
    """
    Generate initial points using a 3D Hilbert curve logic.
    This ensures space-filling initialization.
    """
    # Simple approximation of Hilbert curve ordering for initialization
    # We map 0..1 to a space-filling pattern
    # For 50 points, a 4x4x4 grid with jitter is often good, 
    # but Hilbert ensures better connectivity.
    
    # Let's use a deterministic quasi-random sequence (Halton) which is often better 
    # for initial packing than Hilbert in high dimensions due to uniformity,
    # but the prompt suggests Hilbert. Let's stick to a robust space-filling init.
    # Actually, a 5x5x5 grid subset or Halton is very robust.
    # Let's use Halton sequence for better uniformity than random, and similar properties to Hilbert.
    
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f = f / base
            r = r + f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, n_points + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        points.append([x * 0.9 + 0.05, y * 0.9 + 0.05, z * 0.9 + 0.05]) # Scale to [0.05, 0.95] then shift to [-0.45, 0.45]
        
    pts = np.array(points) - 0.5
    return pts

def _repulsion_energy(x, power=2):
    """
    Energy function: Sum of inverse distances to a given power.
    Lower energy means more spread out.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances squared
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Prevent div by zero
    
    # We want to minimize 1/d^power
    # To avoid numerical issues with very small distances, add epsilon
    eps = 1e-6
    dists_sq = dists_sq + eps
    
    # Calculate sum of 1/sqrt(dists_sq)^power = 1/dists_sq^(power/2)
    # If power=2, energy = sum(1/dists_sq)
    # If power=4, energy = sum(1/dists_sq^2)
    
    energy = np.sum(1.0 / (dists_sq ** (power / 2.0)))
    
    # Wall penalty: penalize going outside or too close to edge?
    # L-BFGS-B handles bounds. But a soft penalty helps gradient flow near bounds.
    # However, for max radius, we WANT points near walls if constrained by walls.
    # The repulsion will push them to walls naturally.
    # So no wall penalty in energy, rely on bounds.
    
    return energy

def _optimize_stage(centers, power, max_time, time_budget):
    """
    Run L-BFGS-B for a specific repulsion power.
    """
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def objective(x):
        if time.time() - start_time > time_budget:
            raise TimeoutError
        return _repulsion_energy(x, power)
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 2000,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxfun': 5000,
                'disp': False
            }
        )
        return res.x.reshape(N, 3)
    except Exception:
        return centers

def _sa_polish(centers, time_budget):
    """
    Simulated Annealing polish on the exact radius metric.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    current_centers = centers.copy()
    current_radius = best_radius
    
    rng = np.random.RandomState(42)
    
    T_start = 0.005
    T_end = 0.00001
    # Estimate steps based on time
    # Each step is fast, but let's cap iterations
    max_steps = 50000
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
            
        frac = step / max_steps
        T = T_start * (T_end / T_start) ** frac
        
        i = rng.randint(N)
        
        # Adaptive step size: smaller as T drops
        step_size = 0.01 * (1.0 - 0.8 * frac)
        delta = rng.uniform(-step_size, step_size, 3)
        
        new_centers = current_centers.copy()
        new_centers[i] += delta
        new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius >= current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = new_centers.copy()
        else:
            # Accept worse moves with probability exp(-delta_E / T)
            # delta_E is negative
            delta_e = new_radius - current_radius
            # Scale delta_e to be comparable to T
            # Radius changes are small, T is small
            prob = np.exp(delta_e / (T * 0.01)) 
            if rng.random() < prob:
                current_centers = new_centers
                current_radius = new_radius

    return best_centers, best_radius

def construct():
    start_time = time.time()
    
    # Time allocation
    # We will run 3 restarts.
    # Each restart: Init -> Opt(2) -> Opt(4) -> Opt(8) -> SA
    # Allocate time evenly among restarts, keeping a global safety margin.
    
    num_restarts = 3
    time_per_restart = (DEADLINE - (time.time() - start_time)) / num_restarts
    time_per_restart = max(1.0, time_per_restart - 5.0) # 5s global safety per restart? No, just ensure positive.
    
    best_global_radius = 0.0
    best_global_centers = None
    
    for r_idx in range(num_restarts):
        if time.time() - start_time > DEADLINE - 5.0:
            break
            
        # 1. Initialize
        # Use different starting points for each restart
        if r_idx == 0:
            centers = _hilbert_map(N)
        else:
            # Jitter the hilbert points or use a different seed
            rng = np.random.RandomState(100 + r_idx)
            centers = _hilbert_map(N) + rng.uniform(-0.05, 0.05, (N, 3))
            centers = np.clip(centers, -0.5, 0.5)
            
        # 2. Global Optimization Stages
        # Stage 1: Power 2 (Soft)
        t_alloc = time_per_restart * 0.25
        t_start = time.time()
        centers = _optimize_stage(centers, power=2, max_iter=1000, time_budget=t_alloc)
        
        # Stage 2: Power 4 (Medium)
        t_alloc = time_per_restart * 0.25
        t_start = time.time()
        centers = _optimize_stage(centers, power=4, max_iter=1000, time_budget=t_alloc)
        
        # Stage 3: Power 8 (Hard)
        t_alloc = time_per_restart * 0.25
        t_start = time.time()
        centers = _optimize_stage(centers, power=8, max_iter=1000, time_budget=t_alloc)
        
        # Stage 4: SA Polish
        t_alloc = time_per_restart * 0.25
        centers, radius = _sa_polish(centers, t_alloc)
        
        # Check if this restart beat the global best
        if radius > best_global_radius:
            best_global_radius = radius
            best_global_centers = centers.copy()
            
    # Final return
    if best_global_centers is None:
        # Fallback
        return _hilbert_map(N)
        
    return best_global_centers

# Ensure function name and signature match
# The prompt asks to keep the function names and signatures of the seed program.
# The seed program has `construct()`.

