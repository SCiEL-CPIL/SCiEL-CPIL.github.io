import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # We want to maximize the minimum distance.
    # Let d_ij be the distance between sphere i and j.
    # Let d_iw be the distance from sphere i to the nearest wall.
    # We want to maximize min(d_ij, d_iw).
    # A smooth approximation is to maximize -sum(exp(-d_ij^2 / T)) - sum(exp(-d_iw^2 / T))
    # Or simpler: maximize the sum of d_ij^k for some large k? No, that's not quite right.
    # Let's use a "soft" objective: F = -sum_{i<j} exp(-d_ij^2 / sigma) - sum_i exp(-d_iw^2 / sigma)
    # As sigma -> 0, this approaches the min.
    
    def compute_score(positions):
        """Compute the actual radius (min distance / 2)."""
        # Pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Zero out diagonal
        dists[np.diag_indices(N)] = np.inf
        
        # Distances to walls
        # Cube is [-0.5, 0.5]^3
        # Distance to wall for axis k is 0.5 - |pos_ik|
        wall_dists = 0.5 - np.abs(positions)
        min_wall_dist = np.min(wall_dists, axis=1)
        
        min_pair_dist = np.min(dists)
        min_dist = np.min([min_pair_dist, np.min(min_wall_dist)])
        
        # Ensure validity: positions must be inside
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
            
        return min_dist / 2.0

    def objective_and_grad(positions, sigma):
        """
        Compute a smooth objective value and its gradient.
        Objective: -sum_{i<j} exp(-d_ij^2 / sigma) - sum_i exp(-d_iw^2 / sigma)
        We want to maximize this (minimize the negative).
        Actually, let's define energy E = sum_{i<j} exp(-d_ij^2 / sigma) + sum_i exp(-d_iw^2 / sigma)
        We want to minimize E.
        """
        # Pairwise
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists_sq[np.diag_indices(N)] = np.inf # Avoid self-interaction
        
        # exp(-d^2 / sigma)
        pair_terms = np.exp(-dists_sq / sigma)
        # Gradient of pair term w.r.t pos_i:
        # d/dx_i exp(-d_ij^2/sigma) = exp(...) * (-1/sigma) * d/dx_i (d_ij^2)
        # d_ij^2 = (x_i - x_j)^2 ...
        # d/dx_i (d_ij^2) = 2(x_i - x_j)
        # So grad_i = -2/sigma * (x_i - x_j) * exp(-d_ij^2/sigma)
        
        # Sum over j for each i
        # We can use matrix multiplication or broadcasting.
        # Let A_ij = exp(-d_ij^2/sigma). Note A is symmetric, A_ii = 0.
        # grad_i = -2/sigma * sum_j (x_i - x_j) * A_ij
        # sum_j (x_i - x_j) A_ij = x_i * sum_j A_ij - sum_j x_j A_ij
        
        sum_A = np.sum(pair_terms, axis=1)
        # Weighted sum of positions
        # sum_j x_j A_ij. Since A is symmetric, this is A @ positions.T? No, A is (N,N), pos is (N,3).
        # (A @ pos.T).T gives sum_j A_ij x_j for each i.
        weighted_pos = (pair_terms @ positions.T).T
        
        grad_pair = (-2.0 / sigma) * (positions * sum_A[:, np.newaxis] - weighted_pos)
        
        # Walls
        # d_iw_k = 0.5 - |x_ik|
        # Term: exp(-(0.5 - |x_ik|)^2 / sigma)
        # Let u_ik = 0.5 - |x_ik|
        # d/dx_ik exp(-u_ik^2/sigma) = exp(-u_ik^2/sigma) * (-2 u_ik / sigma) * d/dx_ik u_ik
        # d/dx_ik (0.5 - |x_ik|) = -sign(x_ik)
        # So grad = exp(-u_ik^2/sigma) * (-2 u_ik / sigma) * (-sign(x_ik))
        #         = 2/sigma * sign(x_ik) * u_ik * exp(-u_ik^2/sigma)
        
        abs_pos = np.abs(positions)
        u = 0.5 - abs_pos
        u_sq = u**2
        wall_terms = np.exp(-u_sq / sigma)
        
        # sign(x_ik) * u_ik
        # If x_ik > 0, sign=1, u=0.5-x. Term = (0.5-x)
        # If x_ik < 0, sign=-1, u=0.5+x. Term = -(0.5+x)
        # Note: u is always positive.
        # Let's just compute directly.
        signs = np.sign(positions)
        # Handle 0 carefully, sign(0)=0 is fine as gradient is 0 at 0 for |x|? 
        # d/dx |x| is undefined at 0, but subgradient. Here we have smooth exp.
        # At x=0, u=0.5. grad = 2/sigma * 0 * ... = 0. Correct.
        
        grad_wall = (2.0 / sigma) * signs * u * wall_terms
        
        total_grad = grad_pair + grad_wall
        
        # Objective value (Energy)
        energy = np.sum(pair_terms) + np.sum(wall_terms)
        
        return energy, total_grad

    # Initialization
    # Use a grid perturbed with noise as a good starting point
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_pos = grid[:N].copy()
    
    # Add small noise to break symmetry
    np.random.seed(42)
    initial_pos += np.random.normal(0, 0.01, initial_pos.shape)
    
    # Clip to box
    initial_pos = np.clip(initial_pos, -0.49, 0.49)
    
    best_pos = initial_pos.copy()
    best_score = compute_score(best_pos)
    
    # Simulated Annealing / Gradient Descent
    T = 0.05 # Initial temperature (sigma)
    lr = 0.005 # Learning rate
    alpha = 0.999 # Cooling rate
    
    # We will iterate until deadline
    # Each iteration: compute grad, update, accept if better or with probability exp(-delta_E/T)
    
    current_pos = best_pos.copy()
    current_energy, _ = objective_and_grad(current_pos, T)
    
    # Pre-allocate for speed
    # We'll use a simple loop.
    
    # To speed up, we can reduce the number of pairs considered? No, N=100 is small.
    # 100*100 = 10k operations per gradient. Very fast.
    
    while time.time() < deadline:
        # Compute gradient
        energy, grad = objective_and_grad(current_pos, T)
        
        # Gradient step
        new_pos = current_pos - lr * grad
        
        # Keep in box
        new_pos = np.clip(new_pos, -0.49, 0.49)
        
        # Compute new energy
        new_energy, _ = objective_and_grad(new_pos, T)
        
        # Acceptance criterion
        if new_energy < energy:
            current_pos = new_pos
            current_energy = new_energy
            
            # Update best
            score = compute_score(current_pos)
            if score > best_score:
                best_score = score
                best_pos = current_pos.copy()
        else:
            # Metropolis acceptance
            delta_e = new_energy - energy
            # If delta_e is small and T is high, accept
            if np.random.random() < np.exp(-delta_e / T):
                current_pos = new_pos
                current_energy = new_energy
        
        # Cool down
        T *= alpha
        if T < 1e-5:
            T = 1e-5 # Floor to allow some movement? Or break?
            # Let's keep it small but non-zero to allow fine tuning via gradient descent effectively
            # Actually, if T is very small, exp(-delta/T) is 0 for positive delta.
            # So it becomes pure gradient descent (steepest descent).
            # This is good for final polishing.
            
    return best_pos
# EVOLVE-BLOCK-END
