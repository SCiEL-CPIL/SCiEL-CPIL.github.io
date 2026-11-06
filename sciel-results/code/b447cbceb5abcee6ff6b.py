"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """
    Compute reflectance at all wavelengths using the characteristic matrix method (vectorized).
    """
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    n_layers = len(materials)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wls, r*r)
        
    # Initialize total matrix for all wavelengths
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for mat, thick in zip(materials, thicknesses):
        n = materials_dict[mat]
        if thick < 1e-9:
            continue
            
        beta = 2.0 * np.pi * n * thick / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix
        L11 = cos_b
        L12 = 1.0j * sin_b / n
        L21 = n * 1.0j * sin_b
        L22 = cos_b
        
        # M_total = M_total @ L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    # Input admittance Y
    Y_s = substrate_index
    Y = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    
    # Reflectance
    Y_0 = ambient_index
    r = (Y_0 - Y) / (Y_0 + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance(materials, thicknesses, band_nm=(400, 800), n_points=401,
                       ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """Compute mean reflectance in percent over the band."""
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    wls = np.linspace(band_nm[0], band_nm[1], n_points)
    refls = _reflectance_vectorized(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < -1e-6 or t > max_layer_nm + 1e-6:
            return False
    if sum(thicknesses) > max_total_nm + 1e-6:
        return False
    return True

def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-6, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """Simple Nelder-Mead implementation with box constraints."""
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        step = 0.05 * (bounds[i][1] - bounds[i][0]) + 0.01
        x_new[i] += step
        x_new[i] = np.clip(x_new[i], bounds[i][0], bounds[i][1])
        simplex.append(x_new)
        
    simplex_f = [f(x) for x in simplex]
    
    for it in range(max_iter):
        order = np.argsort(simplex_f)
        simplex = [simplex[i] for i in order]
        simplex_f = [simplex_f[i] for i in order]
        
        if abs(simplex_f[-1] - simplex_f[0]) < tol:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = f(x_r)
        
        if f_r < simplex_f[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                simplex_f[-1] = f_e
            else:
                simplex[-1] = x_r
                simplex_f[-1] = f_r
        elif f_r < simplex_f[-2]:
            simplex[-1] = x_r
            simplex_f[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = f(x_c)
            if f_c < simplex_f[-1]:
                simplex[-1] = x_c
                simplex_f[-1] = f_c
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    simplex_f[i] = f(simplex[i])
                    
    best_idx = np.argmin(simplex_f)
    return simplex[best_idx], simplex_f[best_idx]

def design(problem):
    """
    Design using variable-length discrete search (Simulated Annealing) 
    over layer types with global scaling, followed by continuous polish.
    """
    start_time = time.time()
    deadline = 145.0
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem.get("max_total_nm", 1000)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Pre-compute QWOT thicknesses at center wavelength
    q_l = wl_center / (4.0 * nL)
    q_h = wl_center / (4.0 * nH)
    
    # Objective for a given sequence of types and scale factor
    # We optimize the scale factor 's' such that thicknesses = s * base_thick
    # To keep total thickness < max_total_nm, we can normalize or just bound s.
    # Better: optimize the scale 's' directly, but ensure individual layers don't exceed max_layer_nm.
    
    def objective_for_seq(seq, s):
        """
        seq: list of 'L' or 'H'
        s: scaling factor applied to QWOT thicknesses
        Returns mean reflectance.
        """
        thicknesses = []
        for m in seq:
            base = q_l if m == 'L' else q_h
            t = s * base
            if t > max_layer_nm:
                return 100.0 # Penalty
            thicknesses.append(t)
        
        if sum(thicknesses) > max_total_nm:
            return 100.0
            
        return _mean_reflectance(seq, thicknesses, band_nm, 401, n_amb, n_sub, materials_dict)

    def optimize_scale_for_seq(seq):
        """
        Find optimal scale 's' for a fixed sequence using Nelder-Mead.
        Bounds for s: [0.1, max_s] where max_s is limited by max_layer_nm / min_qwot
        """
        min_qwot = min(q_l, q_h)
        max_s = max_layer_nm / min_qwot
        # Also limited by total thickness: sum(s * q_i) <= max_total_nm -> s <= max_total_nm / sum(q_i)
        sum_q = sum(q_l if m == 'L' else q_h for m in seq)
        max_s_total = max_total_nm / sum_q
        max_s = min(max_s, max_s_total)
        
        if max_s < 0.1:
            return 0.1, 100.0
            
        bounds = [(0.1, max_s)]
        x0 = np.array([1.0])
        
        def f_s(s_arr):
            return objective_for_seq(seq, s_arr[0])
            
        try:
            x_opt, f_opt = _nelder_mead(f_s, x0, bounds, max_iter=100, tol=1e-5)
            return x_opt[0], f_opt
        except Exception:
            return 1.0, 100.0

    best_score = 100.0
    best_seq = ['L']
    best_scale = 1.0
    
    # Initialize with a few known good patterns
    initial_seqs = [
        ['L'],
        ['H'],
        ['L', 'H'],
        ['H', 'L'],
        ['L', 'H', 'L'],
        ['H', 'L', 'H'],
        ['L', 'H', 'L', 'H'],
        ['H', 'L', 'H', 'L'],
        ['L', 'H', 'L', 'H', 'L'],
        ['H', 'L', 'H', 'L', 'H'],
    ]
    
    # Simulated Annealing over sequence structure
    # State: current_seq
    # Moves: insert, delete, flip
    current_seq = initial_seqs[np.random.randint(len(initial_seqs))]
    current_scale, current_score = optimize_scale_for_seq(current_seq)
    
    if current_score < best_score:
        best_score = current_score
        best_seq = current_seq[:]
        best_scale = current_scale
    
    T = 1.0 # Temperature
    T_min = 0.01
    cooling_rate = 0.98
    max_sa_steps = 2000
    
    sa_step = 0
    while T > T_min and time.time() - start_time < deadline - 10:
        sa_step += 1
        if sa_step % 100 == 0:
            T *= cooling_rate
            
        # Generate neighbor
        new_seq = current_seq[:]
        move_type = np.random.choice(['insert', 'delete', 'flip'])
        
        if move_type == 'insert' and len(new_seq) < 20:
            pos = np.random.randint(0, len(new_seq) + 1)
            mat = np.random.choice(['L', 'H'])
            new_seq.insert(pos, mat)
        elif move_type == 'delete' and len(new_seq) > 1:
            pos = np.random.randint(0, len(new_seq))
            new_seq.pop(pos)
        elif move_type == 'flip' and len(new_seq) > 0:
            pos = np.random.randint(0, len(new_seq))
            new_seq[pos] = 'H' if new_seq[pos] == 'L' else 'L'
        else:
            continue
            
        # Evaluate neighbor
        new_scale, new_score = optimize_scale_for_seq(new_seq)
        
        # Acceptance criterion
        if new_score < current_score:
            current_seq = new_seq
            current_scale = new_scale
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_seq = current_seq[:]
                best_scale = current_scale
        else:
            # Metropolis acceptance
            if np.random.rand() < np.exp(-(new_score - current_score) / T):
                current_seq = new_seq
                current_scale = new_scale
                current_score = new_score

    # Final continuous optimization on the best discrete sequence found
    if time.time() - start_time < deadline - 5:
        # We can also try to optimize thicknesses individually for the best sequence
        # This is a higher dimensional problem but might find better local minima.
        # Let's do a multi-start Nelder-Mead on the thicknesses directly for best_seq.
        
        def objective_thicks(thicks):
            if any(t < -1e-6 or t > max_layer_nm + 1e-6 for t in thicks):
                return 100.0
            if sum(thicks) > max_total_nm + 1e-6:
                return 100.0
            return _mean_reflectance(best_seq, thicks, band_nm, 401, n_amb, n_sub, materials_dict)
            
        bounds = [(0.0, max_layer_nm) for _ in range(len(best_seq))]
        
        # Initial guess: scaled QWOTs
        init_thicks = [best_scale * (q_l if m == 'L' else q_h) for m in best_seq]
        
        # Multi-start
        starts = [init_thicks]
        # Add some random perturbations
        for _ in range(5):
            perturbed = [t * np.random.uniform(0.5, 1.5) for t in init_thicks]
            # Clip to bounds
            perturbed = [np.clip(t, 0.0, max_layer_nm) for t in perturbed]
            if sum(perturbed) <= max_total_nm:
                starts.append(perturbed)
                
        for start in starts:
            if time.time() - start_time > deadline - 2:
                break
            try:
                x_opt, f_opt = _nelder_mead(objective_thicks, np.array(start), bounds, max_iter=300, tol=1e-6)
                if f_opt < best_score:
                    best_score = f_opt
                    best_thicknesses = list(x_opt)
                    # Update best_seq? No, best_seq is fixed here.
                    # We need to store the final thicknesses.
                    # The return interface requires materials and thicknesses.
                    # So we just keep best_seq and update best_thicknesses.
                    # But wait, best_seq might not be the one that gives the best thicknesses if we changed it?
                    # No, we are optimizing thicknesses for the current best_seq.
                    # So we just need to store the thicknesses.
                    # We need a variable to hold the final thicknesses.
                    if not hasattr(design, '_best_thicks'):
                        design._best_thicks = best_thicknesses if 'best_thicknesses' in locals() else None
                    design._best_thicks = list(x_opt)
            except Exception:
                pass

    # Construct final result
    if hasattr(design, '_best_thicks') and design._best_thicks is not None:
        final_materials = best_seq
        final_thicknesses = design._best_thicks
    else:
        final_materials = best_seq
        final_thicknesses = [best_scale * (q_l if m == 'L' else q_h) for m in best_seq]
        
    # Clamp and validate
    final_thicknesses = [np.clip(t, 0.0, max_layer_nm) for t in final_thicknesses]
    if sum(final_thicknesses) > max_total_nm:
        # Scale down
        scale_down = max_total_nm / sum(final_thicknesses)
        final_thicknesses = [t * scale_down for t in final_thicknesses]
        
    if not _is_valid(final_materials, final_thicknesses, problem):
        # Fallback
        final_materials = ["L"]
        final_thicknesses = [q_l]
        
    return final_materials, final_thicknesses

# EVOLVE-BLOCK-END
