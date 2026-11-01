"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math

def _initial_fibonacci():
    """Generate initial Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy(points):
    """Calculate total logarithmic energy."""
    # Calculate pairwise distances
    # points shape (N, 3)
    # dists shape (N, N)
    # dists[i, j] = ||points[i] - points[j]||
    
    # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal and upper triangle to avoid self-interaction and double counting
    mask = ~np.eye(N, dtype=bool)
    # We only want i < j
    mask = np.triu(mask, k=1)
    
    # Avoid log(0)
    safe_dists = np.where(mask, dists, 1.0)
    
    # Energy term: log(1/d) = -log(d)
    terms = -np.log(safe_dists)
    terms = np.where(mask, terms, 0.0)
    
    return np.sum(terms)

def _delta_energy(points, i, new_pi):
    """Calculate change in energy if point i is moved to new_pi."""
    # Distances from i to all other points
    # points shape (N, 3)
    # new_pi shape (3,)
    
    # Old distances
    diff_old = points[i] - points
    dists_old = np.linalg.norm(diff_old, axis=1)
    
    # New distances
    diff_new = new_pi - points
    dists_new = np.linalg.norm(diff_new, axis=1)
    
    # Exclude self (index i)
    dists_old[i] = 1.0
    dists_new[i] = 1.0
    
    # Calculate delta
    # E_old_part = sum_{j != i} -log(dists_old[j])
    # E_new_part = sum_{j != i} -log(dists_new[j])
    # Delta = E_new_part - E_old_part
    
    delta = np.sum(-np.log(dists_new)) - np.sum(-np.log(dists_old))
    
    return delta

def _tangent_vector(point, rng):
    """Generate a random unit vector in the tangent plane at 'point'."""
    # Random vector
    v = rng.normal(size=3)
    # Project to tangent plane: v_t = v - (v . p) p
    dot = np.dot(v, point)
    v_t = v - dot * point
    norm = np.linalg.norm(v_t)
    if norm < 1e-10:
        # Rarely happens, try again or use a fixed orthogonal vector
        # Simple fallback
        if abs(point[0]) < 0.9:
            v_t = np.array([1, 0, 0])
        else:
            v_t = np.array([0, 1, 0])
        v_t = v_t - np.dot(v_t, point) * point
        norm = np.linalg.norm(v_t)
    return v_t / norm

def construct():
    rng = np.random.default_rng(42)
    
    # Initial configuration
    points = _initial_fibonacci()
    
    # Simulated Annealing parameters
    T_start = 0.1
    T_end = 1e-6
    n_steps = 50000  # Number of SA iterations
    # Geometric cooling
    cool_factor = (T_end / T_start) ** (1.0 / n_steps)
    
    current_energy = _energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Pre-compute some things if possible, but delta_energy is O(N)
    
    for step in range(n_steps):
        T = T_start * (cool_factor ** step)
        
        # Select random point
        i = rng.integers(0, N)
        p_i = points[i]
        
        # Generate move
        # Step size can decrease slightly or stay constant relative to T
        # A common heuristic is step ~ sqrt(T) or similar, but let's keep it simple
        # We want to explore locally.
        # Let's use a fixed small step size for the tangent move, maybe scaled by T?
        # If T is high, we accept bad moves. If T is low, we reject them.
        # The step size itself determines the scale of exploration.
        # Let's try step_size = 0.1 * T + 0.01
        step_size = 0.05 + 0.5 * T 
        
        v_t = _tangent_vector(p_i, rng)
        new_pi = p_i + step_size * v_t
        new_pi = new_pi / np.linalg.norm(new_pi)
        
        # Calculate delta energy
        delta = _delta_energy(points, i, new_pi)
        
        # Acceptance criterion
        if delta < 0 or rng.random() < math.exp(-delta / T):
            # Accept move
            points[i] = new_pi
            current_energy += delta
            
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
                
    # Final check and projection
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
