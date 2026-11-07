"""Shrink-Wrap Annealing Strategy for Sutton-Chen Clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _time_left(deadline, margin=5.0):
    return deadline - time.time() - margin > 0

def _is_valid(positions):
    """Check that positions are finite and no two atoms are closer than 0.35."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    # Vectorized distance check for speed
    # To avoid O(N^2) memory explosion for large N, we chunk it
    # But for N<=80, N^2 is 6400, which is fine.
    dists = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.35:
        return False
    return True

def _soft_core_energy(positions, sigma, alpha):
    """
    Compute a soft-core repulsive potential and its gradient.
    E_soft = sum_{i<j} (1 / (r_ij^2 + sigma^2)^(alpha/2))
    This is smooth and finite even at r=0, unlike 1/r^n.
    """
    n = len(positions)
    # Compute pairwise differences
    # diff[i, j, :] = positions[i] - positions[j]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    # Squared distances
    r2 = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero issues in gradient calculation, though sigma handles it
    denom = (r2 + sigma ** 2) ** (alpha / 2.0)
    # We only want i < j
    iu = np.triu_indices(n, k=1)
    r2_pairs = r2[iu]
    denom_pairs = denom[iu]
    
    # Energy
    E = np.sum(1.0 / denom_pairs)
    
    # Gradient
    # dE/dr2 = -alpha/2 * (r2 + sigma^2)^(-alpha/2 - 1)
    # dE/dx_i = sum_{j != i} dE/dr2 * (x_i - x_j)
    # Note: d(r^2)/d(x_i) = 2*(x_i - x_j). 
    # So dE/dx_i = sum_{j != i} (-alpha/2 * denom_inv * 2 * (x_i - x_j))
    #            = -alpha * sum_{j != i} (x_i - x_j) / (r2 + sigma^2)^(alpha/2 + 1)
    
    denom_grad = (r2 + sigma ** 2) ** (alpha / 2.0 + 1.0)
    # Force on i from j is proportional to (x_i - x_j) / denom_grad
    # We need the sum over j for each i.
    
    # Vectorized approach:
    # grad[i] = -alpha * sum_j (positions[i] - positions[j]) / denom_grad[i,j]
    # Note: denom_grad is symmetric.
    
    # Create a matrix of weights W[i,j] = 1 / denom_grad[i,j]
    # Set diagonal to 0 to avoid self-interaction
    W = 1.0 / denom_grad
    np.fill_diagonal(W, 0.0)
    
    # grad[i] = -alpha * ( sum_j W[i,j] * positions[i] - sum_j W[i,j] * positions[j] )
    # sum_j W[i,j] * positions[i] = positions[i] * sum_j W[i,j]
    # sum_j W[i,j] * positions[j] = W @ positions
    
    row_sums = np.sum(W, axis=1)
    W_pos = W @ positions
    
    grad = -alpha * (positions * row_sums[:, np.newaxis] - W_pos)
    
    return E, grad

def construct(instance):
    """
    Shrink-Wrap Annealing:
    1. Initialize atoms in a dense, overlapping sphere.
    2. Minimize E_total = E_SC + Lambda(t) * E_soft.
    3. Lambda(t) decays over time/iterations, allowing the soft core to guide
       atoms apart smoothly before the SC potential takes over.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    # Parameters for soft core
    # Sigma controls the width of the repulsion. 
    # For N=80, typical equilibrium distance is ~1.0-1.2.
    # We want repulsion to be significant when atoms are close (< 1.0).
    sigma = 0.5 
    alpha = 6.0 # Exponent, similar to m_exp for SC
    
    # Initial Lambda (strength of soft core)
    # Should be strong enough to overcome SC attraction at short range
    lambda_start = 50.0
    lambda_end = 0.0
    
    # Initial configuration: Random points in a small sphere
    # Radius should be small enough to cause overlaps
    # Volume of cluster ~ N * v_atom. v_atom ~ 1/10.
    # R ~ N^(1/3). Let's start with R ~ 0.5 * N^(1/3) to force overlaps.
    R_init = 0.6 * (size ** (1.0 / 3.0))
    
    rng = np.random.default_rng(42)
    
    # Generate initial positions
    # Rejection sampling to ensure they are within R_init
    positions = np.zeros((size, 3))
    for i in range(size):
        while True:
            p = rng.uniform(-R_init, R_init, 3)
            if np.linalg.norm(p) <= R_init:
                positions[i] = p
                break
    
    best_energy = np.inf
    best_positions = None
    
    # We will perform a series of L-BFGS-B runs with decreasing Lambda.
    # This mimics a slow annealing.
    # Number of steps depends on time budget.
    # Let's try 5-10 steps.
    n_steps = 8
    
    current_positions = positions.copy()
    
    for step in range(n_steps):
        if not _time_left(deadline, margin=10.0):
            break
            
        # Linearly decrease Lambda
        t = step / max(1, n_steps - 1)
        current_lambda = lambda_start * (1.0 - t)
        
        if current_lambda < 1e-6:
            current_lambda = 0.0
            
        def objective(x):
            pos = x.reshape(-1, 3)
            # Sutton-Chen Energy and Gradient
            E_sc, G_sc = sutton_chen_energy_gradient(pos, kind)
            
            if current_lambda > 0:
                E_soft, G_soft = _soft_core_energy(pos, sigma, alpha)
                E_total = E_sc + current_lambda * E_soft
                G_total = G_sc + current_lambda * G_soft
            else:
                E_total = E_sc
                G_total = G_sc
                
            return E_total, G_total.flatten()
        
        try:
            res = minimize(
                objective,
                current_positions.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            new_positions = res.x.reshape(-1, 3)
            
            if _is_valid(new_positions):
                # Check if this is a better solution (based on pure SC energy)
                E_final, _ = sutton_chen_energy_gradient(new_positions, kind)
                if E_final < best_energy:
                    best_energy = E_final
                    best_positions = new_positions
                    
                # Update current positions for the next step
                current_positions = new_positions
            else:
                # If invalid (e.g., atoms too close), try to push them apart more
                # This shouldn't happen with soft core, but just in case
                pass
                
        except Exception as e:
            # If minimization fails, break
            break

    # Final Polish: Pure L-BFGS-B on SC energy with high precision
    if best_positions is not None and _time_left(deadline, margin=15.0):
        try:
            def obj_sc(x):
                pos = x.reshape(-1, 3)
                E, G = sutton_chen_energy_gradient(pos, kind)
                return E, G.flatten()
            
            res = minimize(
                obj_sc,
                best_positions.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-15,
                    'gtol': 1e-10,
                    'maxls': 100
                }
            )
            
            polished = res.x.reshape(-1, 3)
            if _is_valid(polished):
                E_polished, _ = sutton_chen_energy_gradient(polished, kind)
                if E_polished < best_energy:
                    best_positions = polished
                    best_energy = E_polished
        except:
            pass

    # Fallback if no valid solution found (should not happen)
    if best_positions is None:
        # Return a valid random configuration
        R = 1.0 * (size ** (1.0 / 3.0))
        for i in range(size):
            while True:
                p = rng.uniform(-R, R, 3)
                if np.linalg.norm(p) <= R:
                    best_positions[i] = p
                    break
        # Relax it once
        try:
            E, pos = relax(best_positions, potential, kind)
            if _is_valid(pos):
                best_positions = pos
        except:
            pass

    return best_positions
# EVOLVE-BLOCK-END
