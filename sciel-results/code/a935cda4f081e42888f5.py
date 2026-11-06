"""Morse cluster global minima: Differential Evolution approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import differential_evolution
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy_with_penalty(positions, rho, min_dist=0.5, penalty_factor=50.0):
    """
    Compute Morse energy with a penalty for minimum distance violations.
    positions: (n, 3) array
    rho: parameter
    """
    n = positions.shape[0]
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # We only need i < j
    iu = np.triu_indices(n, k=1)
    r = dists[iu]
    
    # Morse potential: E = sum exp(rho(1-r)) (exp(rho(1-r)) - 2)
    x = rho * (1.0 - r)
    exp_x = np.exp(x)
    e_pairs = exp_x * (exp_x - 2.0)
    energy = np.sum(e_pairs)
    
    # Penalty for min distance
    if min_dist > 0:
        violations = r < min_dist
        if np.any(violations):
            viol_dists = r[violations]
            shortfall = min_dist - viol_dists
            penalty = penalty_factor * np.sum(shortfall**2)
            energy += penalty
            
    return energy

def construct(instance):
    """Find low-energy Morse cluster via Differential Evolution."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    t_start = time.time()
    time_limit = 45.0 # Leave margin for 6 instances in 300s
    
    # Bounds for DE
    R = 2.0 * (size ** (1.0/3.0))
    bounds = [(-R, R)] * (3 * size)
    
    def objective(x):
        pos = x.reshape((size, 3))
        if not np.all(np.isfinite(pos)):
            return 1e10
        return _morse_energy_with_penalty(pos, rho, min_dist=0.5, penalty_factor=50.0)
    
    best_pos = None
    best_energy = np.inf
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=15, 
                tol=1e-5,
                seed=42,
                workers=4,
                init='sobol'
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_pos = res.x.reshape((size, 3))
                
    except Exception:
        pass
    
    # Polish with relax
    if best_pos is not None:
        try:
            e_relaxed, pos_relaxed = relax(best_pos.copy(), potential, rho)
            if e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = pos_relaxed
        except Exception:
            pass
    
    # Second round if time remains
    remaining_time = time_limit - (time.time() - t_start)
    if remaining_time > 5.0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res2 = differential_evolution(
                    objective, 
                    bounds, 
                    maxiter=30, 
                    popsize=20, 
                    tol=1e-5,
                    seed=12345,
                    workers=4,
                    init='latinhypercube'
                )
                if res2.fun < best_energy:
                    best_energy = res2.fun
                    best_pos = res2.x.reshape((size, 3))
                    e_relaxed, pos_relaxed = relax(best_pos.copy(), potential, rho)
                    if e_relaxed < best_energy:
                        best_energy = e_relaxed
                        best_pos = pos_relaxed
        except Exception:
            pass

    if best_pos is None:
        rng = np.random.default_rng(0)
        R = 1.5 * (size ** (1.0/3.0))
        best_pos = rng.uniform(-R, R, (size, 3))
        try:
            _, best_pos = relax(best_pos.copy(), potential, rho)
        except:
            pass

    return best_pos
# EVOLVE-BLOCK-END
