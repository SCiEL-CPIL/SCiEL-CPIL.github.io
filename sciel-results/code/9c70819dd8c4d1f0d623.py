"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Constructs the centers of 300 circles inside a unit circle.
    Uses Simulated Annealing with a repulsive force model to maximize the minimum distance.
    """
    
    # 1. Initialization
    # Use a random initialization with a minimum distance constraint to avoid bad starts
    def initialize(n):
        # Try a few random inits and pick the one with lowest energy
        best_pos = None
        best_energy = np.inf
        
        for _ in range(5):
            # Random points in unit circle
            angles = np.random.uniform(0, 2 * np.pi, n)
            radii = np.sqrt(np.random.uniform(0, 1, n))
            pos = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
            
            # Calculate initial energy
            energy = calc_total_energy(pos)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        
        return best_pos

    def calc_total_energy(pos):
        # E = sum_{i<j} 1/d_ij^2 + sum_i 1/(1-r_i)^2
        # Vectorized calculation for total energy
        # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # This is O(N^2) memory and time, which is 300^2 = 90k, fine.
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dists_sq, 1.0)
        
        # Pairwise repulsion: sum over i<j
        # We sum all i!=j and divide by 2
        pair_energy = np.sum(1.0 / dists_sq) / 2.0
        
        # Boundary repulsion
        radii = np.linalg.norm(pos, axis=1)
        # Avoid division by zero if point is exactly at 1
        boundary_dist = 1.0 - radii
        boundary_dist = np.maximum(boundary_dist, 1e-6)
        boundary_energy = np.sum(1.0 / boundary_dist**2)
        
        return pair_energy + boundary_energy

    def calc_delta_energy(pos, idx, new_pos_idx):
        """
        Calculate change in energy when moving point idx to new_pos_idx.
        """
        old_point = pos[idx]
        new_point = new_pos_idx
        
        # Interactions with all other points
        others = np.delete(pos, idx, axis=0)
        
        # Old distances
        old_dists_sq = np.sum((others - old_point)**2, axis=1)
        old_dists_sq = np.maximum(old_dists_sq, 1e-12)
        
        # New distances
        new_dists_sq = np.sum((others - new_point)**2, axis=1)
        new_dists_sq = np.maximum(new_dists_sq, 1e-12)
        
        # Pair energy change
        old_pair = np.sum(1.0 / old_dists_sq)
        new_pair = np.sum(1.0 / new_dists_sq)
        
        # Boundary energy change
        old_r = np.linalg.norm(old_point)
        new_r = np.linalg.norm(new_point)
        
        old_b_dist = max(1.0 - old_r, 1e-6)
        new_b_dist = max(1.0 - new_r, 1e-6)
        
        old_bound = 1.0 / old_b_dist**2
        new_bound = 1.0 / new_b_dist**2
        
        return (new_pair + new_bound) - (old_pair + old_bound)

    # 2. Simulated Annealing
    pos = initialize(N)
    
    T = 0.1
    T_min = 1e-6
    cooling_rate = 0.9995
    n_iter = 200000
    step_size = 0.05
    
    best_pos = pos.copy()
    best_energy = calc_total_energy(pos)
    
    for i in range(n_iter):
        # Pick random index
        idx = np.random.randint(N)
        
        # Propose move
        delta = np.random.normal(0, step_size, 2)
        new_point = pos[idx] + delta
        
        # Keep inside circle (soft constraint handled by energy, but hard constraint for validity)
        r_new = np.linalg.norm(new_point)
        if r_new > 1.0:
            # Project back to boundary or reject? 
            # In SA, we can allow it but energy will be huge. 
            # However, to maintain validity at the end, we should project or reject.
            # Let's project to boundary to keep it valid.
            new_point = new_point / r_new
        
        # Calculate Delta E
        dE = calc_delta_energy(pos, idx, new_point)
        
        # Accept?
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            pos[idx] = new_point
            # Update best
            # Calculating total energy every step is expensive O(N^2).
            # We can track current energy incrementally.
            # Let's just check every 1000 steps or use a cheaper metric.
            # For now, let's calculate total energy every 500 steps to update best.
            
        if i % 500 == 0:
            curr_energy = calc_total_energy(pos)
            if curr_energy < best_energy:
                best_energy = curr_energy
                best_pos = pos.copy()
                
        T *= cooling_rate
        if T < T_min:
            break
            
    # 3. Local Polish
    # Use scipy minimize on the best position found
    # Objective: Minimize max violation for a given radius? 
    # Or just minimize the energy further.
    # Let's try to maximize the minimum distance directly using a local search.
    
    def min_distance_metric(x):
        # x is flat array
        pos_flat = x.reshape(N, 2)
        # Check validity
        radii = np.linalg.norm(pos_flat, axis=1)
        if np.any(radii > 1.0):
            return 1e6 # Invalid
        
        diff = pos_flat[:, np.newaxis, :] - pos_flat[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        min_boundary_dist = np.min(1.0 - radii)
        
        # We want to maximize min(min_pair_dist/2, min_boundary_dist)
        # So we minimize the negative of this
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return -r

    # Initial guess for minimizer
    x0 = best_pos.flatten()
    
    # Use Powell method which is derivative-free and robust
    # Note: This might be slow, but N=300 is manageable for a few iterations.
    # We limit the number of iterations.
    try:
        res = minimize(min_distance_metric, x0, method='Powell', options={'maxiter': 2000, 'maxfev': 20000})
        if res.success or res.fun < -1e-10:
            best_pos = res.x.reshape(N, 2)
    except Exception:
        pass

    # Final check and projection
    radii = np.linalg.norm(best_pos, axis=1)
    outside_mask = radii > 1.0
    if np.any(outside_mask):
        best_pos[outside_mask] = best_pos[outside_mask] / radii[outside_mask, np.newaxis]

    return best_pos

# EVOLVE-BLOCK-END
