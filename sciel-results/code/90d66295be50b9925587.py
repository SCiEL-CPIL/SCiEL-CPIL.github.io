"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

def _energy_and_grad(P, r):
    """
    Compute the energy and gradient for a fixed radius r.
    Energy penalizes overlaps and boundary violations.
    """
    N = P.shape[0]
    energy = 0.0
    grad = np.zeros_like(P)
    
    # 1. Pairwise Overlaps
    tree = cKDTree(P)
    pairs = tree.query_pairs(r=2*r)
    
    for i, j in pairs:
        d_vec = P[i] - P[j]
        dist = np.linalg.norm(d_vec)
        if dist < 2*r:
            # Penalty: (2r - dist)^2
            # d/dPi (2r - dist)^2 = 2(2r - dist) * d/dPi (2r - dist)
            # d/dPi dist = d_vec / dist
            # So d/dPi (2r - dist)^2 = 2(2r - dist) * (-d_vec / dist)
            if dist > 1e-9:
                penalty = (2*r - dist)
                energy += penalty * penalty
                # Gradient contribution
                grad[i] += 2 * penalty * (-d_vec / dist)
                grad[j] += 2 * penalty * (d_vec / dist) # Note: d/dPj dist = -d_vec/dist, so -(-d_vec/dist) = d_vec/dist?
                # Let's re-verify.
                # f = (2r - ||Pi - Pj||)^2
                # df/dPi = 2(2r - ||Pi-Pj||) * (-1) * d/dPi ||Pi-Pj||
                # d/dPi ||Pi-Pj|| = (Pi - Pj) / ||Pi-Pj||
                # So df/dPi = -2(2r - dist) * (Pi - Pj) / dist
                # My code above: 2 * penalty * (-d_vec / dist) where d_vec = Pi - Pj.
                # This matches.
                # df/dPj = 2(2r - ||Pi-Pj||) * (-1) * d/dPj ||Pi-Pj||
                # d/dPj ||Pi-Pj|| = (Pj - Pi) / ||Pi-Pj|| = -d_vec / dist
                # So df/dPj = -2(2r - dist) * (-d_vec / dist) = 2(2r - dist) * (d_vec / dist)
                # My code above: 2 * penalty * (d_vec / dist). Matches.
            else:
                # If dist is 0, add small random gradient to break symmetry
                energy += (2*r)**2
                grad[i] += np.random.normal(0, 1, 2)
                grad[j] -= np.random.normal(0, 1, 2)

    # 2. Boundary Violations
    norms = np.linalg.norm(P, axis=1)
    # Constraint: ||Pi|| <= 1 - r
    # Penalty: max(0, ||Pi|| - (1-r))^2
    violation = norms - (1 - r)
    mask = violation > 0
    if np.any(mask):
        v = violation[mask]
        energy += np.sum(v * v)
        # d/dPi (||Pi|| - (1-r))^2 = 2(||Pi|| - (1-r)) * d/dPi ||Pi||
        # d/dPi ||Pi|| = Pi / ||Pi||
        # So grad = 2 * v * (Pi / ||Pi||)
        P_masked = P[mask]
        n_masked = norms[mask]
        # Avoid division by zero
        n_safe = np.where(n_masked > 1e-9, n_masked, 1e-9)
        grad[mask] += 2 * v[:, np.newaxis] * (P_masked / n_safe[:, np.newaxis])
        
    return energy, grad

def _optimize_for_radius(P_init, r):
    """
    Use L-BFGS-B to minimize the energy for a fixed radius r.
    Returns the best position and energy found.
    """
    res = minimize(
        fun=lambda x: _energy_and_grad(x, r)[0],
        x0=P_init,
        jac=lambda x: _energy_and_grad(x, r)[1],
        method='L-BFGS-B',
        bounds=[(-1.1, 1.1)] * (2 * N),
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9}
    )
    return res.x, res.fun

def _random_init():
    """
    Generate random points inside the unit circle.
    """
    # Sample uniformly in area
    r = np.sqrt(np.random.rand(N))
    theta = 2 * np.pi * np.random.rand(N)
    P = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return P

def _score(P):
    """
    Calculate the actual score: min(min_pair_dist/2, min_dist_to_boundary)
    """
    norms = np.linalg.norm(P, axis=1)
    min_boundary_dist = 1.0 - np.max(norms)
    
    tree = cKDTree(P)
    dists, _ = tree.query(P, k=2)
    min_pair_dist = np.min(dists[:, 1])
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def construct():
    best_P = None
    best_r = 0.0
    
    # Binary search for the maximum feasible radius
    # Known best is ~0.05315
    # We search in [0.05, 0.055]
    low_r = 0.050
    high_r = 0.055
    
    # First, find a feasible starting point for a lower radius to ensure we have a valid config
    # Then binary search upwards
    
    # To save time, we will do a few trials for each radius
    num_trials = 3
    
    # We will iterate from high to low, or binary search.
    # Binary search is more efficient if the feasibility is monotonic.
    
    # However, finding a feasible point is stochastic.
    # Let's try a sequence of radii descending from high to low, stopping when we find a feasible one?
    # Or binary search: if r is feasible, try higher. If not, try lower.
    
    # Since "feasible" depends on finding a local minimum, it's noisy.
    # Let's use a fixed set of radii to check, picking the best one that works.
    
    radii_to_try = [0.0535, 0.0534, 0.0533, 0.0532, 0.0531, 0.0530, 0.0525, 0.0520]
    
    for r in radii_to_try:
        found = False
        for trial in range(num_trials):
            P_init = _random_init()
            # Add small noise to break symmetry if needed, but random init is already noisy
            P_opt, energy = _optimize_for_radius(P_init, r)
            
            # Check if energy is effectively zero
            if energy < 1e-6:
                # Verify with exact score
                actual_score = _score(P_opt)
                if actual_score >= r - 1e-6: # Allow tiny numerical error
                    if best_P is None or r > best_r:
                        best_r = r
                        best_P = P_opt
                    found = True
                    break # Move to next (lower) radius if we found a valid one? 
                    # Actually, we want the HIGHEST r. So if we find one at 0.0535, we should keep it.
                    # The loop goes high to low. So if we find one, we can break out of trials for this r.
                    # We should NOT break out of the radius loop immediately, because maybe we can find a valid one at 0.0535, but we already tried it.
                    # Wait, the list is sorted high to low. If we find a valid one at 0.0535, that's our best so far.
                    # We can break the trials loop.
                    # We should continue checking lower radii? No, we want the highest.
                    # So if we find a valid one at the current highest untried radius, we can stop?
                    # Yes, because all subsequent radii are lower.
                    # So if found, we can return or break the outer loop.
                    # However, "found" might be a local minimum that is valid, but maybe a different random start at the SAME r could yield a better configuration? 
                    # The score is determined by r. If r is feasible, the score is at least r.
                    # So finding ANY valid config for r gives score >= r.
                    # So we want the highest r for which we can find a valid config.
                    # If we find one for 0.0535, we can stop searching for lower radii.
                    # We should break the outer loop.
                    pass 
        
        if found:
            break
            
    if best_P is None:
        # Fallback: use the last tried radius's best attempt
        # This shouldn't happen if low_r is feasible
        P_opt, _ = _optimize_for_radius(_random_init(), 0.050)
        best_P = P_opt
        best_r = _score(best_P)
        
    # Final polish: try to improve the score of the best_P found
    # We can run a quick local optimization on the actual score? 
    # Or just use the result from the binary search.
    
    return best_P
# EVOLVE-BLOCK-END
