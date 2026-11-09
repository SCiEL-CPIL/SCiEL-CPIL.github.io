"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math

def _fibonacci_points(n, seed_offset=0.0):
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_dists = np.clip(pair_dists, 1e-15, None)
    return -np.sum(np.log(pair_dists))

def _simulated_annealing(pts0, t_start=100.0, t_end=0.1, max_time=100.0):
    """
    Simulated Annealing on the sphere.
    - Uses a geometric random walk (Gaussian perturbations projected to tangent space).
    - Accepts worse moves with probability exp(-delta_E / T).
    - Adaptive cooling.
    """
    pts = pts0.copy()
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    t = t_start
    step_size = 0.1
    
    t_start_time = time.time()
    
    while time.time() - t_start_time < max_time:
        # Generate a random perturbation in tangent space
        # Random vector in R3
        delta = np.random.normal(0, step_size, size=(N, 3))
        
        # Project to tangent space at each point: v - (v . x) x
        dots = np.sum(delta * pts, axis=1, keepdims=True)
        delta_tangent = delta - dots * pts
        
        # Proposed new points
        new_pts = pts + delta_tangent
        new_pts = _normalize(new_pts)
        
        # Check validity (no points too close)
        # We can skip full energy calc if points are too close to save time, 
        # but let's just compute energy and check min dist.
        # For speed, we compute energy first.
        new_energy = _log_energy(new_pts)
        
        # Check min distance
        diff = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)
        iu = np.triu_indices(N, k=1)
        min_dist = np.min(dists[iu])
        
        if min_dist < 1e-9:
            # Reject invalid move
            t *= 0.99
            continue
            
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e < 0:
            # Better
            pts = new_pts
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
        else:
            # Worse, accept with probability exp(-delta_e / T)
            # Avoid overflow/underflow
            if t > 0:
                prob = math.exp(-delta_e / t)
                if np.random.random() < prob:
                    pts = new_pts
                    current_energy = new_energy
        
        # Cooling
        # Geometric cooling is standard
        t *= 0.995
        
        # Adapt step size based on acceptance rate? 
        # Simple heuristic: if t is low, reduce step size.
        # Also, if we are stuck (no improvement for a long time), we might want to kick.
        # But simple SA with good initial T usually works well.
        
        # If time is running out, we can break early if t is very small
        if t < 1e-6:
            break
            
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # 1. Initial candidate: Fibonacci spiral (very good starting point for sphere packing)
    fib_pts = _fibonacci_points(N)
    fib_energy = _log_energy(fib_pts)
    best_pts = fib_pts
    best_energy = fib_energy
    
    # 2. Run Simulated Annealing from the Fibonacci start
    # We allocate most of the time to this.
    # SA is stochastic, so we might run multiple restarts if time permits, 
    # but a long run from a good start is often better than many short runs.
    
    remaining_time = deadline - start_time - 5.0
    if remaining_time > 10.0:
        try:
            sa_pts, sa_energy = _simulated_annealing(fib_pts, max_time=remaining_time)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts
        except Exception:
            pass
            
    # 3. If we still have time, try a random restart with SA
    if time.time() < deadline - 15.0:
        np.random.seed(42)
        random_pts = np.random.normal(0, 1, size=(N, 3))
        random_pts = _normalize(random_pts)
        rand_energy = _log_energy(random_pts)
        
        # Only proceed if random start isn't catastrophically bad (it usually is)
        # But SA can escape bad starts.
        remaining = deadline - time.time() - 5.0
        if remaining > 10.0:
            try:
                sa_rand_pts, sa_rand_energy = _simulated_annealing(random_pts, t_start=50.0, max_time=remaining)
                if sa_rand_energy < best_energy:
                    best_energy = sa_rand_energy
                    best_pts = sa_rand_pts
            except Exception:
                pass

    # Final validation
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        return _fibonacci_points(N)
    
    # Ensure no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    if np.min(pair_dists) < 1e-9:
        return _fibonacci_points(N)
        
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
