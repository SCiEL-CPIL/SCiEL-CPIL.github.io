"""Broadband antireflection coating design via Multi-Start Basin Hopping with Fixed Alternating Structure."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _vectorized_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance fraction for a single design across all wavelengths.
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return float(r * r)
    
    n_layers_arr = np.array([n_H if m == 'H' else n_L for m in materials])
    
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(n_layers):
        n = n_layers_arr[i]
        d = thicknesses[i]
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.real(r * r)
    
    return float(np.mean(R))

def _evaluate_design(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm):
    """
    Evaluate a design. Returns reflectance percentage if valid, inf otherwise.
    """
    if len(thicknesses) == 0:
        return float('inf')
    
    # Check constraints
    if any(t < -1e-6 or t > max_layer_nm + 1e-6 for t in thicknesses):
        return float('inf')
    if sum(thicknesses) > max_total_nm + 1e-6:
        return float('inf')
        
    t_calc = np.clip(thicknesses, 0, max_layer_nm)
    R_frac = _vectorized_reflectance(t_calc, materials, n_ambient, n_substrate, n_L, n_H, wls)
    return R_frac * 100.0

def design(problem):
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    wls = np.linspace(wl_min, wl_max, 401)
    wl_center = 0.5 * (wl_min + wl_max)
    
    # Fallback solution: Single quarter-wave L layer
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    best_score = float('inf')
    best_thicknesses = [qw_L]
    best_materials = ['L']
    
    # We will try several different alternating structures and optimize their thicknesses.
    # Strategy: Basin Hopping
    # 1. Define a set of candidate structures (alternating H/L with varying counts)
    # 2. For each structure, generate multiple initial guesses
    # 3. Run Nelder-Mead on each guess
    # 4. Keep the global best
    
    # Candidate structures: 
    # We know that optimal broadband AR coatings often have ~5-15 layers.
    # We will test a range of layer counts.
    # To keep it fast, we limit the number of structures and restarts.
    
    structures = []
    # Try different starting materials and counts
    # Max layers is 20. Let's try odd and even counts up to 15 to save time.
    for count in [3, 5, 7, 9, 11, 13, 15]:
        for start_mat in ['H', 'L']:
            mats = []
            for i in range(count):
                if i % 2 == 0:
                    mats.append(start_mat)
                else:
                    mats.append('L' if start_mat == 'H' else 'H')
            structures.append(mats)
    
    # Also add a few random alternating structures with higher counts if time permits
    # But let's stick to the systematic ones first.
    
    # Initial guesses for each structure:
    # 1. Quarter-wave at center wavelength
    # 2. Quarter-wave at min wavelength
    # 3. Quarter-wave at max wavelength
    # 4. Random
    # 5. Optimized for a specific "meta" wavelength (e.g., 550nm)
    
    def get_initial_guesses(mats, count):
        guesses = []
        # 1. Center QW
        g1 = np.array([ (wl_center / (4.0 * (n_H if m=='H' else n_L))) for m in mats ])
        guesses.append(g1)
        
        # 2. Min QW
        g2 = np.array([ (wl_min / (4.0 * (n_H if m=='H' else n_L))) for m in mats ])
        guesses.append(g2)
        
        # 3. Max QW
        g3 = np.array([ (wl_max / (4.0 * (n_H if m=='H' else n_L))) for m in mats ])
        guesses.append(g3)
        
        # 4. Random
        g4 = np.random.uniform(50, 200, count)
        guesses.append(g4)
        
        # 5. "Thick" QW (half wave)
        g5 = np.array([ (wl_center / (2.0 * (n_H if m=='H' else n_L))) for m in mats ])
        guesses.append(g5)
        
        return guesses

    # Objective function for Nelder-Mead
    def objective(thicknesses):
        # Ensure constraints are handled
        if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
            return 1e6
        if np.sum(thicknesses) > max_total_nm:
            return 1e6
        # Evaluate
        return _evaluate_design(thicknesses, current_mats, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)

    # We need to be careful with the closure for objective.
    # We will define a helper that takes mats as arg.
    
    def optimize_structure(mats, initial_guess):
        # Bounds for Nelder-Mead
        bounds = [(0, max_layer_nm)] * len(mats)
        
        # Nelder-Mead doesn't support bounds directly in older scipy, 
        # but we can use a penalty or transform. 
        # However, for 5-15 dimensions, NM is slow.
        # Let's use L-BFGS-B if available? No, NM is often better for non-smooth/noisy, 
        # but this is smooth. L-BFGS-B handles bounds.
        # Let's try L-BFGS-B first as it's faster for smooth bounded problems.
        
        def obj(x):
            if np.any(x < 0) or np.any(x > max_layer_nm) or np.sum(x) > max_total_nm:
                return 1e6
            return _evaluate_design(x, mats, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
        
        try:
            res = minimize(obj, initial_guess, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-5})
            if res.success:
                return res.x, res.fun
            else:
                # If it fails, try Nelder-Mead as fallback? Or just return current best.
                # Let's just return the best found during the process if available, 
                # but L-BFGS-B usually gives a decent result even if not converged.
                return res.x, res.fun
        except Exception:
            return initial_guess, obj(initial_guess)

    # Main loop: Iterate through structures and guesses
    # To manage time, we limit the number of optimizations.
    # Total structures: 7 counts * 2 starts = 14 structures.
    # Guesses per structure: 5.
    # Total optimizations: 70.
    # Each optimization takes ~0.1-0.5s. Total ~7-35s. Safe.
    
    # We will shuffle the structures to avoid biasing early on one type? 
    # Or just go in order. Order doesn't matter much for global best.
    
    # To improve diversity, we can also randomize the order of guesses.
    
    for mats in structures:
        if time.time() > deadline - 10.0: # Leave 10s for cleanup
            break
            
        count = len(mats)
        guesses = get_initial_guesses(mats, count)
        
        # Shuffle guesses to randomize
        np.random.shuffle(guesses)
        
        for guess in guesses:
            if time.time() > deadline - 5.0:
                break
                
            # Clip initial guess to be valid
            guess_clipped = np.clip(guess, 0, max_layer_nm)
            if np.sum(guess_clipped) > max_total_nm:
                # Scale down
                scale = max_total_nm / np.sum(guess_clipped)
                guess_clipped *= scale
                
            # Run optimizer
            opt_thick, opt_score = optimize_structure(mats, guess_clipped)
            
            # Final check and cleanup
            opt_thick = np.clip(opt_thick, 0, max_layer_nm)
            if np.sum(opt_thick) > max_total_nm:
                scale = max_total_nm / np.sum(opt_thick)
                opt_thick *= scale
            
            # Re-evaluate to be sure
            final_score = _evaluate_design(opt_thick, mats, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
            
            if final_score < best_score:
                best_score = final_score
                best_thicknesses = opt_thick.tolist()
                best_materials = mats[:]
                
    # Final validity check
    if not best_thicknesses or any(t < 0 or t > max_layer_nm for t in best_thicknesses) or sum(best_thicknesses) > max_total_nm:
        return ["L"], [qw_L]
        
    cleaned_thick = [max(0.0, t) for t in best_thicknesses]
    if sum(cleaned_thick) > max_total_nm:
        scale = max_total_nm / sum(cleaned_thick)
        cleaned_thick = [t * scale for t in cleaned_thick]
        
    return best_materials, cleaned_thick

# EVOLVE-BLOCK-END
