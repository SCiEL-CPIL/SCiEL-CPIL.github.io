"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on unit sphere."""
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask out self-pairs (diagonal)
    mask = ~np.eye(n, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Use upper triangle to avoid double counting
    triu_indices = np.triu_indices(n, k=1)
    d = dist[triu_indices]
    energy = -np.sum(np.log(d))
    
    # Gradient: d/dx_i [log(1/|x_i - x_j|)] = (x_i - x_j) / |x_i - x_j|^2
    # For each point i, gradient is sum over j != i of (x_i - x_j) / |x_i - x_j|^2
    # But since we're minimizing -sum log(d), the gradient of -log(d) w.r.t x_i is:
    # d/dx_i [-log(|x_i - x_j|)] = -(x_i - x_j)/|x_i - x_j|^2
    # Wait, let me be careful. Energy E = sum_{i<j} log(1/|x_i-x_j|) = -sum_{i<j} log(|x_i-x_j|)
    # dE/dx_i = -sum_{j != i} d/dx_i log(|x_i-x_j|) = -sum_{j != i} (x_i-x_j)/|x_i-x_j|^2
    # But for j < i, the term is log(1/|x_j-x_i|) = -log(|x_j-x_i|), and d/dx_i of -log(|x_j-x_i|) = (x_i-x_j)/|x_i-x_j|^2
    # So actually: dE/dx_i = sum_{j<i} (x_i-x_j)/|x_i-x_j|^2 - sum_{j>i} (x_i-x_j)/|x_i-x_j|^2
    # Hmm, let me think again. For pair (i,j) with i<j, term is -log(|x_i-x_j|).
    # d/dx_i [-log(|x_i-x_j|)] = -(x_i-x_j)/|x_i-x_j|^2
    # d/dx_j [-log(|x_i-x_j|)] = -(x_j-x_i)/|x_i-x_j|^2 = (x_i-x_j)/|x_i-x_j|^2
    # So for point i, gradient contribution from all j != i:
    # If j > i: -(x_i-x_j)/|x_i-x_j|^2
    # If j < i: (x_i-x_j)/|x_i-x_j|^2
    # Combined: sum_{j != i} (x_j - x_i) / |x_i - x_j|^2 ... no wait.
    # Let me just compute it directly.
    
    # For each i, grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # This is because d/dx_i [-log(|x_i-x_j|)] = -(x_i-x_j)/|x_i-x_j|^2 = (x_j-x_i)/|x_i-x_j|^2
    # And this applies for all j != i (regardless of whether j < i or j > i, since the 
    # energy term for pair (min(i,j), max(i,j)) contributes to both points).
    
    # Vectorized computation
    # diff[i,j] = x_i - x_j
    # We want sum_j (x_j - x_i) / |x_i - x_j|^2 = -sum_j (x_i - x_j) / |x_i - x_j|^2
    # = -sum_j diff[i,j] / dist[i,j]^2
    
    # Avoid division by zero on diagonal
    dist_safe = np.where(mask, dist, 1.0)
    inv_dist_sq = np.where(mask, 1.0 / (dist_safe ** 2), 0.0)
    
    # grad[i] = -sum_j (x_i - x_j) / |x_i - x_j|^2 = -sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=42):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _optimize_single(pts, time_budget=10.0):
    """Optimize a single starting configuration using L-BFGS-B with time limit."""
    start_time = time.time()
    best_pts = _project_to_sphere(pts.copy())
    best_energy = _energy_and_grad(best_pts)[0]
    
    def objective(x):
        pts_x = _project_to_sphere(x.reshape(-1, 3))
        return _energy_and_grad(pts_x)[0]
    
    def gradient(x):
        pts_x = _project_to_sphere(x.reshape(-1, 3))
        e, g = _energy_and_grad(pts_x)
        # Project gradient to tangent space
        g = g - np.sum(g * pts_x, axis=1, keepdims=True) * pts_x
        return g.flatten()
    
    x0 = best_pts.flatten()
    
    while time.time() - start_time < time_budget:
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 1000, 'maxfun': 5000, 'gtol': 1e-10})
            if res.success or res.fun < best_energy:
                new_pts = _project_to_sphere(res.x.reshape(-1, 3))
                new_energy = _energy_and_grad(new_pts)[0]
                if new_energy < best_energy:
                    best_pts = new_pts
                    best_energy = new_energy
            x0 = res.x
        except Exception:
            break
        
        # Check if we've converged
        if time.time() - start_time >= time_budget:
            break
    
    return best_pts, best_energy


def _simulated_annealing(pts, time_budget=15.0, T0=0.01, T1=1e-6, alpha=0.999):
    """Simulated annealing on the sphere."""
    start_time = time.time()
    best_pts = _project_to_sphere(pts.copy())
    best_energy = _energy_and_grad(best_pts)[0]
    current_pts = best_pts.copy()
    current_energy = best_energy
    
    T = T0
    n = len(current_pts)
    
    while time.time() - start_time < time_budget:
        # Generate a random perturbation
        idx = np.random.randint(n)
        # Random direction in tangent space
        rand_dir = np.random.randn(3)
        rand_dir -= np.dot(rand_dir, current_pts[idx]) * current_pts[idx]
        norm = np.linalg.norm(rand_dir)
        if norm > 1e-12:
            rand_dir /= norm
        
        # Step size proportional to temperature
        step_size = T * 0.1
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + step_size * rand_dir
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _energy_and_grad(new_pts)[0]
        delta_e = new_energy - current_energy
        
        # Accept or reject
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            current_pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_pts = current_pts.copy()
                best_energy = current_energy
        
        T *= alpha
        if T < T1:
            break
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Phase 1: Generate diverse starting points
    starts = []
    starts.append(_fibonacci_sphere(N))
    
    for seed in range(1, 10):
        if time.time() > deadline - 30.0:
            break
        starts.append(_random_sphere(N, seed=seed))
    
    # Phase 2: Optimize each starting point with limited time
    best_pts = starts[0]
    best_energy = _energy_and_grad(best_pts)[0]
    
    time_per_start = max(5.0, (deadline - start_time - 20.0) / max(1, len(starts)))
    
    for i, s in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        
        remaining = deadline - 20.0 - time.time()
        if remaining < 3.0:
            break
        
        pts, energy = _optimize_single(s, time_budget=min(time_per_start, remaining))
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Phase 3: Simulated annealing from best point
    if time.time() < deadline - 10.0:
        sa_budget = min(10.0, deadline - 10.0 - time.time())
        if sa_budget > 2.0:
            sa_pts, sa_energy = _simulated_annealing(best_pts, time_budget=sa_budget)
            if sa_energy < best_energy:
                best_pts = sa_pts
                best_energy = sa_energy
    
    # Phase 4: Final polish with L-BFGS-B
    if time.time() < deadline - 2.0:
        polish_budget = deadline - 2.0 - time.time()
        if polish_budget > 1.0:
            final_pts, final_energy = _optimize_single(best_pts, time_budget=polish_budget)
            if final_energy < best_energy:
                best_pts = final_pts
                best_energy = final_energy
    
    return _project_to_sphere(best_pts)
# EVOLVE-BLOCK-END
