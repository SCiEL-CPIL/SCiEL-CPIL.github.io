"""
Broadband AR coating design using Global Multi-Start Nelder-Mead Simplex Optimization.
Strategy:
1. Enumerate potential layer counts from 1 to 20.
2. For each count, initialize the material sequence using a greedy insertion heuristic 
   that minimizes the mean reflectance at each step.
3. Use scipy.optimize.minimize with Nelder-Mead to refine the continuous thicknesses.
4. Select the global best solution across all layer counts and restarts.
"""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _reflectance_calc(mats, thicks, n0, ns, materials, lambdas):
    """
    Fast vectorized reflectance calculation using characteristic matrix method.
    mats: list of 'L'/'H'
    thicks: numpy array of thicknesses
    """
    N = len(thicks)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(lambdas, np.abs(r)**2)
        
    M11 = np.ones(len(lambdas), dtype=complex)
    M12 = np.zeros(len(lambdas), dtype=complex)
    M21 = np.zeros(len(lambdas), dtype=complex)
    M22 = np.ones(len(lambdas), dtype=complex)
    
    for i in range(N):
        n_i = materials[mats[i]]
        d_i = thicks[i]
        
        delta = 2.0 * np.pi * n_i * d_i / lambdas
        c = np.cos(delta)
        s = np.sin(delta)
        
        m11 = c
        m12 = 1j * s / n_i
        m21 = 1j * n_i * s
        m22 = c
        
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    Y_s = ns
    Y_in = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    r = (Y_in - n0) / (Y_in + n0)
    R = np.abs(r)**2
    return R

def design(problem):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    band = problem["band_nm"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    n_wav = len(lambdas)
    
    def evaluate(mats, thicks):
        if len(mats) == 0:
            return np.mean(_reflectance_calc(mats, np.array([]), n0, ns, materials, lambdas)) * 100.0
        return np.mean(_reflectance_calc(mats, thicks, n0, ns, materials, lambdas)) * 100.0

    def objective(x):
        # x is the vector of thicknesses
        # Check constraints
        if np.any(x < 0) or np.any(x > max_layer_nm):
            return 1e6
        if np.sum(x) > max_total_nm:
            return 1e6
        
        # x corresponds to the fixed mats for this call
        return evaluate(current_mats, x)

    best_score = np.inf
    best_mats = []
    best_thicks = np.array([])
    
    # Heuristic: Greedy Layer Insertion
    # Start with 0 layers, add layers one by one choosing the material and position
    # that minimizes the mean reflectance.
    # This provides a good initial guess for the material sequence.
    
    # We will try a few different "seeds" for the material sequence generation
    # to ensure we explore different local minima.
    
    seeds = [
        ['L'], # Start with L
        ['H'], # Start with H
        ['L', 'H'], # Start with LH
        ['H', 'L'], # Start with HL
    ]
    
    # Precompute quarter wave thicknesses for initialization
    center_lambda = 0.5 * (band[0] + band[1])
    qw_L = center_lambda / (4.0 * materials['L'])
    qw_H = center_lambda / (4.0 * materials['H'])
    
    # We will iterate over possible layer counts from 1 to max_layers
    # But to save time, we'll do a coarse pass first, then refine the best few.
    
    # Phase 1: Coarse search over layer counts with greedy initialization
    # For each N, generate a greedy sequence, then run a quick Nelder-Mead.
    
    candidates = [] # List of (score, mats, thicks)
    
    # We'll use a random seed for reproducibility in case we add randomness later, 
    # but for now we use deterministic seeds.
    
    for seed_mats in seeds:
        current_mats = seed_mats.copy()
        current_thicks = np.array([qw_L if m=='L' else qw_H for m in seed_mats])
        
        # Greedily add layers up to max_layers
        for target_n in range(len(current_mats) + 1, max_layers + 1):
            if time.time() > deadline - 10.0:
                break
            
            best_add_score = np.inf
            best_add_mats = None
            best_add_thicks = None
            
            # Try inserting L and H at every position
            for pos in range(len(current_mats) + 1):
                for new_mat in ['L', 'H']:
                    trial_mats = current_mats[:pos] + [new_mat] + current_mats[pos:]
                    # Initialize new layer thickness to quarter wave
                    trial_thicks = np.insert(current_thicks, pos, qw_L if new_mat=='L' else qw_H)
                    
                    # Check constraint
                    if np.sum(trial_thicks) > max_total_nm:
                        # Scale down if needed? Or skip?
                        # If we scale down, we might be far from optimal.
                        # Let's just skip if it violates, as greedy should stay within limits 
                        # if we start small. But 20 layers * 100nm = 2000nm > 1000nm.
                        # So we must be careful.
                        # Let's try to scale down to fit if possible.
                        scale = max_total_nm / np.sum(trial_thicks)
                        if scale > 0.1: # Don't scale down too much
                            trial_thicks *= scale
                        else:
                            continue
                    
                    score = evaluate(trial_mats, trial_thicks)
                    if score < best_add_score:
                        best_add_score = score
                        best_add_mats = trial_mats
                        best_add_thicks = trial_thicks
            
            if best_add_mats is None:
                break # Cannot add more layers
            
            current_mats = best_add_mats
            current_thicks = best_add_thicks
            
            # Evaluate the current greedy state
            score = evaluate(current_mats, current_thicks)
            if score < best_score:
                best_score = score
                best_mats = current_mats.copy()
                best_thicks = current_thicks.copy()
            
            # Store as a candidate for local refinement
            candidates.append((score, current_mats.copy(), current_thicks.copy()))
            
            # Check time
            if time.time() > deadline - 20.0:
                break
        
        # Check time
        if time.time() > deadline - 20.0:
            break

    # Phase 2: Local Refinement using Nelder-Mead
    # We will refine the top K candidates.
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0])
    
    # Limit the number of refinements to stay within time budget
    # Each Nelder-Mead call takes ~0.1-0.5s depending on dimension.
    # With 20 dims, it might be slower.
    # Let's refine the top 5 candidates.
    
    num_refinements = 5
    refined = 0
    
    for score, mats, thicks in candidates:
        if refined >= num_refinements:
            break
        if time.time() > deadline - 5.0:
            break
        
        # Set up the optimization problem
        # Bounds: [0, max_layer_nm] for each thickness
        # Also total thickness constraint.
        
        # Nelder-Mead does not handle bounds or equality constraints directly.
        # We use a penalty in the objective function.
        
        current_mats = mats
        x0 = thicks
        
        # To improve convergence, we can use a transformed space or just rely on the penalty.
        # Let's use a simple penalty.
        
        def obj(x):
            if np.any(x < 0) or np.any(x > max_layer_nm):
                return 1e6
            if np.sum(x) > max_total_nm:
                # Penalty for exceeding total thickness
                excess = np.sum(x) - max_total_nm
                return evaluate(mats, x) + 10.0 * excess
            return evaluate(mats, x)
        
        # Initial simplex
        # Nelder-Mead creates the simplex automatically, but we can guide it.
        # Default is fine.
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 0.1, 'fatol': 0.001})
            
            if res.success or res.fun < score:
                new_thicks = res.x
                # Clip to valid range
                new_thicks = np.clip(new_thicks, 0, max_layer_nm)
                
                # Check total thickness
                if np.sum(new_thicks) > max_total_nm:
                    # Scale down if slightly over, or reject
                    if np.sum(new_thicks) > max_total_nm * 1.01:
                        continue
                    new_thicks *= (max_total_nm / np.sum(new_thicks))
                
                new_score = evaluate(mats, new_thicks)
                if new_score < best_score:
                    best_score = new_score
                    best_mats = mats.copy()
                    best_thicks = new_thicks.copy()
                
                refined += 1
        except Exception:
            pass

    # If no layers were found (unlikely), return empty
    if len(best_mats) == 0:
        return [], []
        
    return best_mats, list(best_thicks)

# EVOLVE-BLOCK-END
