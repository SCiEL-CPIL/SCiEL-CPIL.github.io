"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _pairwise_distances(centers):
    """Compute all pairwise distances between centers (N, N matrix)."""
    # centers: (N, 3)
    # Use broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist

def _compute_radius(centers):
    """Compute the valid radius for a given set of centers."""
    # Min distance between any pair
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Min distance to boundary
    # Cube is [-0.5, 0.5]^3
    min_boundary_dist = np.min(0.5 - np.abs(centers))
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _energy_and_gradient(centers, r):
    """
    Hard squared-violation energy: penalize overlaps and boundary violations.
    Returns (energy, gradient).
    
    Energy = sum_{i<j} max(0, r*2 - |c_i - c_j|)^2 + sum_i sum_k max(0, r - (0.5 - |c_ik|))^2
    """
    N = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise overlap penalty
    for i in range(N):
        for j in range(i + 1, N):
            d = centers[i] - centers[j]
            dist = np.sqrt(np.sum(d ** 2))
            if dist < 2.0 * r:
                # Violation: 2*r - dist
                violation = 2.0 * r - dist
                energy += violation ** 2
                # Gradient w.r.t. c_i: 2 * violation * (d / dist)
                # Gradient w.r.t. c_j: -2 * violation * (d / dist)
                if dist > 1e-12:
                    factor = 2.0 * violation / dist
                    grad[i] += factor * d
                    grad[j] -= factor * d
    
    # Boundary penalty
    for i in range(N):
        for k in range(3):
            coord = centers[i, k]
            dist_to_boundary = 0.5 - abs(coord)
            if dist_to_boundary < r:
                violation = r - dist_to_boundary
                energy += violation ** 2
                # d/dcoord of (r - (0.5 - |coord|))^2
                # = 2*(r - (0.5 - |coord|)) * d/dcoord(0.5 - |coord|)
                # d/dcoord(0.5 - |coord|) = -sign(coord)
                if abs(coord) > 1e-12:
                    sign = 1.0 if coord > 0 else -1.0
                    grad[i, k] += 2.0 * violation * (-sign)
                else:
                    # At origin, subgradient; use 0
                    pass
    
    return energy, grad

def _local_optimize(centers, r, max_iter=500, tol=1e-10):
    """
    Local optimization using L-BFGS-B to maximize the minimum clearance.
    We minimize the energy for a given r, but actually we want to find a feasible
    configuration for radius r. Instead, let's directly optimize to maximize r.
    
    Alternative approach: Use a penalty method where we try to achieve radius r
    and minimize violations. If energy is near 0, we can increase r.
    
    Actually, let's use a different strategy: minimize a function that encourages
    large minimum distances.
    """
    # We'll use a simulated annealing / local search approach instead
    # for better robustness.
    pass

def _simulated_annealing(centers, r, steps=5000, T0=0.01, T_end=1e-6):
    """
    Simulated annealing to find a feasible configuration for radius r.
    Returns improved centers.
    """
    N = len(centers)
    best_centers = centers.copy()
    best_energy, _ = _energy_and_gradient(centers, r)
    
    T = T0
    cooling = (T_end / T0) ** (1.0 / steps)
    
    for step in range(steps):
        T *= cooling
        
        # Random perturbation
        candidate = centers.copy()
        # Move a random sphere by a small random amount
        idx = np.random.randint(N)
        delta = np.random.randn(3) * 0.01
        candidate[idx] += delta
        
        # Clip to stay in cube (with some margin)
        candidate = np.clip(candidate, -0.5, 0.5)
        
        cand_energy, _ = _energy_and_gradient(candidate, r)
        
        # Accept if better, or with probability exp(-delta_E / T)
        if cand_energy < best_energy:
            centers = candidate
            best_centers = candidate.copy()
            best_energy = cand_energy
        else:
            delta_E = cand_energy - best_energy
            if delta_E < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_E / max(T, 1e-12))
            if np.random.random() < accept_prob:
                centers = candidate
    
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Initial configuration: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Compute initial radius
    current_radius = _compute_radius(centers)
    best_radius = current_radius
    best_centers = centers.copy()
    
    # Strategy: incrementally increase radius and optimize
    # Start with a slightly larger radius and try to find feasible config
    r = current_radius
    
    # First, do a local optimization to improve the initial configuration
    # Use gradient-based method to maximize minimum distance
    
    def objective(x):
        """Objective to minimize: negative of minimum clearance."""
        c = x.reshape(-1, 3)
        # Clip to cube
        c = np.clip(c, -0.5, 0.5)
        r = _compute_radius(c)
        return -r
    
    # Use L-BFGS-B to maximize radius
    x0 = centers.flatten()
    
    # Multiple restarts with different initial perturbations
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb from best found so far
            x_start = best_centers.flatten() + np.random.randn(N * 3) * 0.005
            x_start = np.clip(x_start, -0.5, 0.5)
        
        try:
            result = minimize(
                objective, x_start,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        c_opt = result.x.reshape(-1, 3)
        c_opt = np.clip(c_opt, -0.5, 0.5)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = c_opt.copy()
        
        # If we've hit the deadline, break
        if time.time() > deadline - 10.0:
            break
    
    # Additional local refinement with simulated annealing at the best radius
    if time.time() < deadline - 20.0:
        # Try to push radius slightly higher
        r_target = best_radius * 1.001
        sa_centers = _simulated_annealing(best_centers, r_target, steps=2000, T0=0.005, T_end=1e-7)
        r_sa = _compute_radius(sa_centers)
        if r_sa > best_radius:
            best_radius = r_sa
            best_centers = sa_centers
    
    # Final check and return
    final_radius = _compute_radius(best_centers)
    if final_radius > best_radius:
        best_radius = final_radius
    
    # Ensure all centers are within the cube
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    # Final validation
    final_r = _compute_radius(best_centers)
    if not np.isfinite(final_r) or final_r <= 0:
        # Fallback to initial grid
        return grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
