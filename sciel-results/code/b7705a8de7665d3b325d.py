"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _compute_mean_reflectance(materials, thicknesses, ambient_index, substrate_index,
                               band_nm, n_wavelengths=401):
    """
    Compute the mean normal-incidence reflectance (in percent) over the band
    using the characteristic-matrix method.
    """
    n_L = len(materials)
    if n_L == 0:
        # Bare substrate reflectance
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return 100.0 * r * r

    wls = np.linspace(band_nm[0], band_nm[1], n_wavelengths)
    
    # Precompute refractive indices
    n_amb = ambient_index
    n_sub = substrate_index
    n_layers = np.array([problem_materials_dict[m] for m in materials])
    t_layers = np.array(thicknesses, dtype=float)

    reflectances = np.zeros(n_wavelengths)

    for i, wl in enumerate(wls):
        ki = 2.0 * np.pi / wl
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for j in range(n_L):
            delta_j = ki * n_layers[j] * t_layers[j]
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            M_j = np.array([[cos_d, 1j * sin_d / n_layers[j]],
                            [1j * n_layers[j] * sin_d, cos_d]], dtype=complex)
            M = M_j @ M
        
        B = M[0, 0] + M[0, 1] / n_sub
        C = n_sub * M[1, 0] + M[1, 1]
        Y = B / C
        r = (n_amb - Y) / (n_amb + Y)
        reflectances[i] = np.abs(r) ** 2

    return 100.0 * np.mean(reflectances)


def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem.get('max_layers', 20)
    max_total_nm = problem.get('max_total_nm', 1000)
    max_layer_nm = problem.get('max_layer_nm', 400)
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return True
    
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    
    if sum(thicknesses) > max_total_nm:
        return False
    
    return True


# Global materials dict, set in design
problem_materials_dict = {}

def design(problem):
    """
    Design an antireflection coating using simulated annealing on 
    quarter-wave optical thickness sequences with a global scaling factor.
    """
    global problem_materials_dict
    problem_materials_dict = problem['materials']
    ambient_index = problem['ambient_index']
    substrate_index = problem['substrate_index']
    band_nm = problem['band_nm']
    max_layers = problem.get('max_layers', 20)
    max_total_nm = problem.get('max_total_nm', 1000)
    max_layer_nm = problem.get('max_layer_nm', 400)
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    n_L_idx = problem_materials_dict['L']
    n_H_idx = problem_materials_dict['H']
    
    # Quarter-wave thickness at 600nm for each material
    qw_L = 600.0 / (4.0 * n_L_idx)
    qw_H = 600.0 / (4.0 * n_H_idx)
    
    # Simulated annealing parameters
    n_layers = 16  # Number of layers in the coating
    initial_temp = 50.0
    cooling_rate = 0.999
    min_temp = 0.01
    max_iterations = 50000
    
    # Initialize with a good starting point: alternating L and H layers
    # with quarter-wave thicknesses at 600nm
    best_materials = []
    best_thicknesses = []
    
    # Start with a classic AR coating design: LH pair
    # L on top, H on bottom, both quarter-wave at 600nm
    init_mats = ['L', 'H']
    init_thicks = [qw_L, qw_H]
    
    # Try a few initial designs and pick the best
    candidates = [
        (['L'], [qw_L]),
        (['L', 'H'], [qw_L, qw_H]),
        (['H', 'L'], [qw_H, qw_L]),
        (['L', 'H', 'L'], [qw_L, qw_H, qw_L]),
        (['H', 'L', 'H'], [qw_H, qw_L, qw_H]),
        (['L', 'H', 'L', 'H'], [qw_L, qw_H, qw_L, qw_H]),
        (['H', 'L', 'H', 'L'], [qw_H, qw_L, qw_H, qw_L]),
    ]
    
    best_score = float('inf')
    best_mats = None
    best_thicks = None
    
    for mats, thicks in candidates:
        if _is_valid(mats, thicks, problem):
            score = _compute_mean_reflectance(mats, thicks, ambient_index, substrate_index,
                                               band_nm, problem_materials_dict)
            if score < best_score:
                best_score = score
                best_mats = mats[:]
                best_thicks = thicks[:]
    
    if best_mats is None:
        best_mats = ['L']
        best_thicks = [qw_L]
        best_score = _compute_mean_reflectance(best_mats, best_thicks, ambient_index, substrate_index,
                                                band_nm, problem_materials_dict)
    
    # Simulated annealing on the layer sequence
    # State: list of materials (L or H) and a global scaling factor for thicknesses
    # We'll fix the number of layers and optimize the sequence + scale factor
    
    # Start from the best initial design
    current_mats = best_mats[:]
    current_thicks = best_thicks[:]
    current_score = best_score
    
    # Extend to a fixed number of layers if needed, or keep as is
    # Let's use a variable number of layers approach with SA
    
    # Define temperature schedule
    T = initial_temp
    
    # Random number generator
    rng = np.random.default_rng(42)
    
    iteration = 0
    while T > min_temp and iteration < max_iterations:
        if time.time() > deadline - 5.0:
            break
        
        iteration += 1
        
        # Generate a neighbor
        r = rng.random()
        
        if r < 0.5 and len(current_mats) > 1:
            # Change a random layer's material
            idx = rng.integers(0, len(current_mats))
            new_mats = current_mats[:]
            new_mats[idx] = 'H' if new_mats[idx] == 'L' else 'L'
            new_thicks = current_thicks[:]
            # Update thickness to quarter-wave for the new material at 600nm
            new_thicks[idx] = qw_H if new_mats[idx] == 'H' else qw_L
            
        elif r < 0.7:
            # Add a layer
            if len(current_mats) < max_layers:
                pos = rng.integers(0, len(current_mats) + 1)
                new_mat = 'L' if rng.random() < 0.5 else 'H'
                new_mats = current_mats[:]
                new_mats.insert(pos, new_mat)
                new_thicks = current_thicks[:]
                new_thicks.insert(pos, qw_H if new_mat == 'H' else qw_L)
            else:
                new_mats = current_mats
                new_thicks = current_thicks
        elif r < 0.85:
            # Remove a layer
            if len(current_mats) > 1:
                pos = rng.integers(0, len(current_mats))
                new_mats = current_mats[:]
                new_thicks = current_thicks[:]
                del new_mats[pos]
                del new_thicks[pos]
            else:
                new_mats = current_mats
                new_thicks = current_thicks
        else:
            # Perturb a random thickness
            idx = rng.integers(0, len(current_thicks))
            delta = rng.normal(0, 20.0)
            new_mats = current_mats
            new_thicks = current_thicks[:]
            new_thicks[idx] += delta
        
        # Check validity
        if not _is_valid(new_mats, new_thicks, problem):
            # Try to repair: clip thicknesses
            if new_mats is not current_mats or new_thicks is not current_thicks:
                new_thicks = [max(0, min(max_layer_nm, t)) for t in new_thicks]
                if sum(new_thicks) > max_total_nm:
                    scale = max_total_nm / sum(new_thicks) if sum(new_thicks) > 0 else 1.0
                    new_thicks = [t * scale for t in new_thicks]
                    new_thicks = [max(0, min(max_layer_nm, t)) for t in new_thicks]
                if not _is_valid(new_mats, new_thicks, problem):
                    continue
        
        # Evaluate
        new_score = _compute_mean_reflectance(new_mats, new_thicks, ambient_index, substrate_index,
                                               band_nm, problem_materials_dict)
        
        # Metropolis criterion
        delta_E = new_score - current_score
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_mats = new_mats
            current_thicks = new_thicks
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_mats = current_mats[:]
                best_thicks = current_thicks[:]
        
        T *= cooling_rate
    
    # Local refinement using Nelder-Mead on the best solution
    if len(best_mats) > 0 and time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            
            def objective(thicks):
                # Ensure validity
                if any(t < 0 or t > max_layer_nm for t in thicks):
                    return 100.0
                if sum(thicks) > max_total_nm:
                    return 100.0
                return _compute_mean_reflectance(best_mats, thicks, ambient_index, 
                                                 substrate_index, band_nm, problem_materials_dict)
            
            x0 = np.array(best_thicks, dtype=float)
            bounds = [(0, max_layer_nm) for _ in range(len(best_mats))]
            
            # Use Nelder-Mead
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 200})
            
            if res.fun < best_score:
                new_thicks = np.clip(res.x, 0, max_layer_nm)
                if sum(new_thicks) <= max_total_nm and all(t <= max_layer_nm for t in new_thicks):
                    new_score = _compute_mean_reflectance(best_mats, new_thicks, ambient_index, 
                                                          substrate_index, band_nm, problem_materials_dict)
                    if new_score < best_score:
                        best_score = new_score
                        best_thicks = list(new_thicks)
                        
        except Exception:
            pass
        
        # Try a second refinement pass with smaller steps
        if len(best_mats) > 0 and time.time() < deadline - 5.0:
            try:
                from scipy.optimize import minimize
                
                def objective2(thicks):
                    if any(t < 0 or t > max_layer_nm for t in thicks):
                        return 100.0
                    if sum(thicks) > max_total_nm:
                        return 100.0
                    return _compute_mean_reflectance(best_mats, thicks, ambient_index, 
                                                     substrate_index, band_nm, problem_materials_dict)
                
                x0 = np.array(best_thicks, dtype=float)
                bounds = [(0, max_layer_nm) for _ in range(len(best_mats))]
                
                res = minimize(objective2, x0, method='Powell', 
                               options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100, 'maxfev': 500})
                
                if res.fun < best_score:
                    new_thicks = np.clip(res.x, 0, max_layer_nm)
                    if sum(new_thicks) <= max_total_nm and all(t <= max_layer_nm for t in new_thicks):
                        new_score = _compute_mean_reflectance(best_mats, new_thicks, ambient_index, 
                                                              substrate_index, band_nm, problem_materials_dict)
                        if new_score < best_score:
                            best_score = new_score
                            best_thicks = list(new_thicks)
                            
            except Exception:
                pass
    
    # Final validation
    if not _is_valid(best_mats, best_thicks, problem):
        # Fallback to a simple valid design
        best_mats = ['L']
        best_thicks = [qw_L]
    
    return best_mats, best_thicks

# EVOLVE-BLOCK-END

