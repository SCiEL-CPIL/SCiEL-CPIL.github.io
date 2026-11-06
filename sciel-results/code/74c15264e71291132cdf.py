"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_reflectance(materials, thicknesses, n_air, n_sub, wavelengths):
    """
    Compute reflectance for a stack across all wavelengths.
    materials: list of 'L' or 'H'
    thicknesses: list of physical thicknesses in nm
    n_air: ambient index
    n_sub: substrate index
    wavelengths: array of wavelengths in nm
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_air - n_sub) / (n_air + n_sub)
        return np.full_like(wavelengths, abs(r)**2)
    
    M_total = np.ones((len(wavelengths), 2, 2), dtype=complex)
    n_vals = [problem_global["materials"][m] for m in materials]
    
    for i in range(n_layers):
        n = n_vals[i]
        d = thicknesses[i]
        delta = 2.0 * np.pi * n * d / wavelengths
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        M_i = np.ones((len(wavelengths), 2, 2), dtype=complex)
        M_i[:, 0, 1] = 1j * sin_d / n
        M_i[:, 1, 0] = 1j * n * sin_d
        M_i[:, 0, 0] = cos_d
        M_i[:, 1, 1] = cos_d
        
        M_total = np.einsum('nij,njk->nik', M_total, M_i)
        
    m11 = M_total[:, 0, 0]
    m12 = M_total[:, 0, 1]
    m21 = M_total[:, 1, 0]
    m22 = M_total[:, 1, 1]
    
    Y_in = (m21 + m22 * n_sub) / (m11 + m12 * n_sub)
    r = (n_air - Y_in) / (n_air + Y_in)
    R = np.abs(r)**2
    return R

problem_global = None

def design(problem):
    global problem_global
    problem_global = problem
    
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    def eval_stack(mats, thicks):
        R = _compute_reflectance(mats, thicks, n_air, n_sub, wavelengths)
        return np.mean(R) * 100.0
    
    # Start with a known good multi-layer AR stack
    # Quarter-wave stack: L H L H ... ending with L on substrate
    # For broadband, use 5-7 layers
    
    def make_quarter_wave_stack(n_layers):
        """Create a quarter-wave stack with alternating L/H, starting with L from air side."""
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append('L')
                # Quarter wave at center wavelength
                lam_c = 0.5 * (band_nm[0] + band_nm[1])
                thicks.append(lam_c / (4.0 * n_L))
            else:
                mats.append('H')
                lam_c = 0.5 * (band_nm[0] + band_nm[1])
                thicks.append(lam_c / (4.0 * n_H))
        return mats, thicks
    
    # Try several initial configurations
    candidates = []
    
    # 5-layer quarter-wave stack
    mats5, thicks5 = make_quarter_wave_stack(5)
    candidates.append((mats5, thicks5))
    
    # 7-layer quarter-wave stack
    mats7, thicks7 = make_quarter_wave_stack(7)
    candidates.append((mats7, thicks7))
    
    # 3-layer quarter-wave stack
    mats3, thicks3 = make_quarter_wave_stack(3)
    candidates.append((mats3, thicks3))
    
    # Single quarter-wave L layer
    lam_c = 0.5 * (band_nm[0] + band_nm[1])
    candidates.append((['L'], [lam_c / (4.0 * n_L)]))
    
    # Evaluate all candidates
    best_mats = None
    best_thicks = None
    best_score = float('inf')
    
    for mats, thicks in candidates:
        score = eval_stack(mats, thicks)
        if score < best_score:
            best_score = score
            best_mats = mats[:]
            best_thicks = thicks[:]
    
    # Now refine the best candidate using gradient-free local search
    start_time = time.time()
    time_budget = 50.0
    
    # Local search: perturb each thickness and material, keep if better
    current_mats = best_mats[:]
    current_thicks = np.array(best_thicks, dtype=float)
    current_score = best_score
    
    n_iter = 0
    while time.time() - start_time < time_budget:
        n_iter += 1
        improved = False
        
        # Try perturbing each thickness
        for i in range(len(current_mats)):
            if current_thicks[i] < 1e-9:
                continue
            original = current_thicks[i]
            
            # Try several perturbations
            for factor in [0.9, 0.8, 0.7, 1.1, 1.2, 1.3]:
                new_thicks = current_thicks.copy()
                new_thicks[i] = original * factor
                new_thicks[i] = np.clip(new_thicks[i], 0, max_layer_nm)
                
                if np.sum(new_thicks) > max_total_nm:
                    new_thicks = new_thicks * (max_total_nm / np.sum(new_thicks))
                
                score = eval_stack(current_mats, new_thicks)
                if score < current_score:
                    current_thicks = new_thicks
                    current_score = score
                    improved = True
                    break
        
        # Try swapping materials
        for i in range(len(current_mats)):
            if current_thicks[i] < 1e-9:
                continue
            original_mat = current_mats[i]
            new_mat = 'H' if original_mat == 'L' else 'L'
            current_mats[i] = new_mat
            
            score = eval_stack(current_mats, current_thicks)
            if score < current_score:
                current_score = score
                improved = True
            else:
                current_mats[i] = original_mat
        
        # Try adding a layer
        if len(current_mats) < max_layers and time.time() - start_time < time_budget - 5:
            for pos in range(len(current_mats) + 1):
                for new_mat in ['L', 'H']:
                    new_mats = current_mats[:]
                    new_thicks = current_thicks.copy()
                    new_mats.insert(pos, new_mat)
                    new_thicks = np.insert(new_thicks, pos, 50.0)
                    
                    if np.sum(new_thicks) > max_total_nm:
                        new_thicks = new_thicks * (max_total_nm / np.sum(new_thicks))
                    
                    score = eval_stack(new_mats, new_thicks)
                    if score < current_score:
                        current_mats = new_mats
                        current_thicks = new_thicks
                        current_score = score
                        improved = True
                        break
                if improved:
                    break
        
        # Try removing a layer if it's thin
        for i in range(len(current_mats)):
            if current_thicks[i] < 20.0:
                new_mats = current_mats[:i] + current_mats[i+1:]
                new_thicks = np.delete(current_thicks, i)
                
                if len(new_mats) == 0:
                    continue
                    
                score = eval_stack(new_mats, new_thicks)
                if score < current_score:
                    current_mats = new_mats
                    current_thicks = new_thicks
                    current_score = score
                    improved = True
                    break
        
        if not improved and n_iter > 20:
            break
    
    # Final cleanup: remove zero-thickness layers
    final_mats = []
    final_thicks = []
    for m, t in zip(current_mats, current_thicks):
        if t > 1e-6:
            final_mats.append(m)
            final_thicks.append(float(t))
    
    if len(final_mats) == 0:
        lam_c = 0.5 * (band_nm[0] + band_nm[1])
        final_mats = ["L"]
        final_thicks = [lam_c / (4.0 * n_L)]
    
    return final_mats, final_thicks
# EVOLVE-BLOCK-END
