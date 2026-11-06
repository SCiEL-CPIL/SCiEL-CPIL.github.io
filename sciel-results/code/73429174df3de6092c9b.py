"""Multi-start Nelder-Mead over fixed layer structures for broadband AR coating."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _mean_reflectance_vec(materials, thicknesses, problem):
    """
    Compute mean reflectance (percent) over the band using vectorized characteristic matrix method.
    """
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    
    n_mat = np.array([n_L if m == 'L' else n_H for m in materials])
    d = np.array(thicknesses, dtype=float)
    
    # Precompute phases for all wavelengths and layers
    # shape: (n_layers, n_wls)
    phases = 2.0 * np.pi * n_mat[:, None] * d[:, None] / wls[None, :]
    
    cos_p = np.cos(phases)
    sin_p = np.sin(phases)
    
    # Initialize characteristic matrix product for each wavelength
    # M is shape (n_wls, 2, 2)
    M = np.zeros((len(wls), 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for i in range(len(materials)):
        n_i = n_mat[i]
        # B matrix for each wavelength
        B = np.zeros((len(wls), 2, 2), dtype=complex)
        B[:, 0, 0] = cos_p[i]
        B[:, 0, 1] = 1j * sin_p[i] / n_i
        B[:, 1, 0] = 1j * n_i * sin_p[i]
        B[:, 1, 1] = cos_p[i]
        
        # M = M @ B
        M = np.einsum('wij,wjk->wik', M, B)
        
    # Calculate Reflectance
    Y = M[:, 0, 1] / M[:, 1, 1]
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2
    return np.mean(R) * 100.0

def _is_valid(materials, thicknesses, problem):
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    max_layers = problem["max_layers"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def design(problem):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Fallback: single quarter-wave layer
    best_mats = ["L"]
    best_thick = [centre / (4.0 * n_L)]
    best_score = _mean_reflectance_vec(best_mats, best_thick, problem)
    
    # Generate candidate layer structures
    # We will test various numbers of layers and sequences
    # Focus on physically motivated sequences: alternating, symmetric, etc.
    
    def get_initial_guess(mats, n_layers):
        """
        Physics-informed initial guess.
        For broadband AR, we want to approximate an impedance gradient.
        A common heuristic is to distribute thicknesses such that the optical
        thickness is roughly uniform or follows a specific profile.
        Here we use a simple heuristic: start with uniform optical thickness
        scaled to fit the total budget, or use quarter-wave values.
        """
        # Total budget
        total_budget = max_total_nm
        # Distribute thicknesses
        # Heuristic: Use quarter-wave at center wavelength as a base, then scale
        # Or just uniform thickness
        # Let's try a mix: some quarter-wave, some uniform
        # For now, uniform thickness is a safe start for Nelder-Mead
        thicks = np.full(n_layers, total_budget / n_layers)
        
        # Alternative: Impedance matching gradient
        # Index profile from n_amb to n_sub
        # n_j = n_amb * (n_sub/n_amb)^(j/N)
        # But we are restricted to L and H.
        # So we just use the sequence provided.
        # Let's try initializing with quarter-wave values for each layer
        # d_j = lambda_c / (4 * n_j)
        # Sum d_j might exceed max_total_nm, so we scale
        d_qw = np.array([centre / (4.0 * (n_L if m == 'L' else n_H)) for m in mats])
        if np.sum(d_qw) > max_total_nm:
            d_qw = d_qw * (max_total_nm / np.sum(d_qw))
        
        # Clip to max_layer_nm
        d_qw = np.clip(d_qw, 0, max_layer_nm)
        
        return d_qw.tolist()

    # Define structures to test
    # We'll test N = 1 to 12 layers
    # For each N, we'll test a few sequences
    # Sequence types:
    # 1. Alternating starting with L
    # 2. Alternating starting with H
    # 3. All L
    # 4. All H
    # 5. Symmetric: L H L H ... H L
    # 6. Random (a few)
    
    structures = []
    
    # Add single layer
    structures.append(["L"])
    structures.append(["H"])
    
    # Add multi-layer structures
    for n in range(2, 13):
        # Alternating L
        alt_l = []
        for i in range(n):
            alt_l.append('L' if i % 2 == 0 else 'H')
        structures.append(alt_l)
        
        # Alternating H
        alt_h = []
        for i in range(n):
            alt_h.append('H' if i % 2 == 0 else 'L')
        structures.append(alt_h)
        
        # All L
        structures.append(['L'] * n)
        
        # All H
        structures.append(['H'] * n)
        
        # Symmetric
        sym = []
        for i in range(n):
            if i < n // 2:
                sym.append('L')
            elif i == n // 2 and n % 2 == 1:
                sym.append('H')
            else:
                sym.append('L')
        # This simple symmetric isn't great. Let's do a proper symmetric alternating
        # For even n: L H L H ... H L is not symmetric if L!=H.
        # Symmetric means mats[i] == mats[n-1-i]
        # Let's just generate a few random symmetric ones
        half = n // 2
        for _ in range(2):
            first_half = [random.choice(['L', 'H']) for _ in range(half)]
            if n % 2 == 0:
                full = first_half + first_half[::-1]
            else:
                full = first_half + ['H'] + first_half[::-1]
            structures.append(full)
            
    # Deduplicate structures
    unique_structures = []
    seen = set()
    for s in structures:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_structures.append(s)
    
    # Sort structures by complexity (length) to start simple
    unique_structures.sort(key=len)
    
    # Optimization function
    def optimize_structure(mats, x0):
        """
        Run Nelder-Mead for a fixed material sequence.
        Returns (score, thicknesses)
        """
        def obj(x):
            # Clip to valid range
            x = np.clip(x, 0, max_layer_nm)
            if np.sum(x) > max_total_nm:
                return 100.0 # Penalty
            return _mean_reflectance_vec(mats, list(x), problem)
        
        # Initial simplex
        # Use a scaled initial guess
        x0_arr = np.array(x0)
        
        # Set up Nelder-Mead
        # xatol and fatol for convergence
        # maxiter to prevent hanging
        # We want to be fast
        max_iter = 1000
        try:
            res = minimize(obj, x0_arr, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-3, 'fatol': 1e-6})
            if res.success or res.fun < 100.0: # If it didn't fail badly
                x_opt = np.clip(res.x, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm:
                    return float(res.fun), list(x_opt)
        except Exception:
            pass
        
        # Fallback to initial guess score if optimization fails
        x0_clipped = np.clip(x0_arr, 0, max_layer_nm)
        if np.sum(x0_clipped) <= max_total_nm:
            return obj(x0_clipped), list(x0_clipped)
        else:
            return 100.0, list(x0_clipped)

    # Iterate through structures
    # We can run multiple starts per structure if time permits
    # But let's start with one good start per structure
    # And maybe a random start for better ones
    
    for mats in unique_structures:
        if time.time() > deadline - 5.0:
            break
            
        # Initial guess
        x0 = get_initial_guess(mats, len(mats))
        
        # Run optimization
        score, thicks = optimize_structure(mats, x0)
        
        if score < best_score:
            best_score = score
            best_mats = mats[:]
            best_thick = thicks[:]
            
        # If this structure is promising (score < current best * 1.5), try a second start
        # with a perturbed initial guess
        if score < best_score * 1.5 and score < 10.0:
            if time.time() > deadline - 10.0:
                break
            
            # Perturb
            x1 = np.array(x0) + np.random.normal(0, 10, len(x0))
            x1 = np.clip(x1, 0, max_layer_nm)
            if np.sum(x1) > max_total_nm:
                x1 = x1 * (max_total_nm / np.sum(x1))
            
            score2, thicks2 = optimize_structure(mats, x1)
            if score2 < best_score:
                best_score = score2
                best_mats = mats[:]
                best_thick = thicks2[:]

    # Final local refinement on the best found structure
    if time.time() < deadline - 2 and len(best_mats) > 0:
        try:
            def obj(x):
                x = np.clip(x, 0, max_layer_nm)
                if np.sum(x) > max_total_nm:
                    return 100.0
                return _mean_reflectance_vec(best_mats, list(x), problem)
            
            x0 = np.array(best_thick, dtype=float)
            # Use tighter tolerance
            res = minimize(obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-7})
            
            if res.fun < best_score:
                x_opt = np.clip(res.x, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm:
                    best_score = float(res.fun)
                    best_thick = list(x_opt)
        except Exception:
            pass

    return best_mats, best_thick
# EVOLVE-BLOCK-END
