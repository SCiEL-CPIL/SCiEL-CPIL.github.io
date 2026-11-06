"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _compute_mean_reflectance(materials, thicknesses, ambient_index, substrate_index,
                               band_nm, materials_dict, n_wavelengths=401):
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
    k = 2.0 * np.pi / wls  # 1/nm

    # Precompute refractive indices
    n_amb = ambient_index
    n_sub = substrate_index
    n_layers = np.array([materials_dict[m] for m in materials])
    t_layers = np.array(thicknesses, dtype=float)

    reflectances = np.zeros(n_wavelengths)

    for i, wl in enumerate(wls):
        ki = 2.0 * np.pi / wl
        # Build the characteristic matrix from the air side
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for j in range(n_L):
            delta_j = ki * n_layers[j] * t_layers[j]
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            M_j = np.array([[cos_d, 1j * sin_d / n_layers[j]],
                            [1j * n_layers[j] * sin_d, cos_d]], dtype=complex)
            M = M_j @ M
        
        # Compute B and C
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


def design(problem):
    """
    Design an antireflection coating using coordinate descent optimization.
    """
    materials_dict = problem['materials']
    ambient_index = problem['ambient_index']
    substrate_index = problem['substrate_index']
    band_nm = problem['band_nm']
    max_layers = problem.get('max_layers', 20)
    max_total_nm = problem.get('max_total_nm', 1000)
    max_layer_nm = problem.get('max_layer_nm', 400)
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Global best
    best_materials = ['L']
    best_thicknesses = [600.0 / (4.0 * materials_dict['L'])]
    best_score = _compute_mean_reflectance(best_materials, best_thicknesses,
                                           ambient_index, substrate_index,
                                           band_nm, materials_dict)
    
    # Multi-seed initialization
    np.random.seed(42)
    
    # Generate diverse initial designs
    initial_designs = []
    
    # Design 1: Single quarter-wave L layer
    initial_designs.append((['L'], [600.0 / (4.0 * materials_dict['L'])]))
    
    # Design 2: Single quarter-wave H layer
    initial_designs.append((['H'], [600.0 / (4.0 * materials_dict['H'])]))
    
    # Design 3: LH quarter-wave pair
    initial_designs.append((['L', 'H'], [600.0 / (4.0 * materials_dict['L']),
                                         600.0 / (4.0 * materials_dict['H'])]))
    
    # Design 4: HL quarter-wave pair
    initial_designs.append((['H', 'L'], [600.0 / (4.0 * materials_dict['H']),
                                         600.0 / (4.0 * materials_dict['L'])]))
    
    # Design 5: Random designs
    for _ in range(30):
        n_layers = np.random.randint(1, min(7, max_layers + 1))
        mats = list(np.random.choice(['L', 'H'], size=n_layers))
        # Quarter-wave thicknesses at 600nm
        thicks = [600.0 / (4.0 * materials_dict[m]) for m in mats]
        # Ensure total thickness constraint
        total = sum(thicks)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicks = [t * scale for t in thicks]
        initial_designs.append((mats, thicks))
    
    # Evaluate all initial designs
    candidates = []
    for mats, thicks in initial_designs:
        if _is_valid(mats, thicks, problem):
            score = _compute_mean_reflectance(mats, thicks, ambient_index,
                                              substrate_index, band_nm, materials_dict)
            candidates.append((mats, thicks, score))
            if score < best_score:
                best_score = score
                best_materials = mats[:]
                best_thicknesses = thicks[:]
    
    # Sort candidates by score
    candidates.sort(key=lambda x: x[2])
    
    # Coordinate descent optimization on top candidates
    for mats, thicks, score in candidates[:10]:
        if time.time() > deadline:
            break
        
        current_materials = mats[:]
        current_thicknesses = thicks[:]
        current_score = score
        
        # Try to improve by adjusting thicknesses and structure
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adjusting each thickness with multiple step sizes
            for i in range(len(current_thicknesses)):
                for delta in [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0,
                             10.0, -10.0, 20.0, -20.0, 50.0, -50.0]:
                    new_thicks = current_thicknesses[:]
                    new_thicks[i] = current_thicknesses[i] + delta
                    
                    if _is_valid(current_materials, new_thicks, problem):
                        new_score = _compute_mean_reflectance(current_materials, new_thicks,
                                                             ambient_index, substrate_index,
                                                             band_nm, materials_dict)
                        if new_score < current_score:
                            current_thicknesses = new_thicks
                            current_score = new_score
                            improved = True
                            break  # Move to next layer
                
                if time.time() > deadline:
                    break
            
            if not improved:
                # Try adding a layer
                if len(current_materials) < max_layers:
                    for new_mat in ['L', 'H']:
                        for pos in range(len(current_materials) + 1):
                            if time.time() > deadline:
                                break
                            
                            new_mats = current_materials[:]
                            new_mats.insert(pos, new_mat)
                            new_thicks = current_thicknesses[:]
                            new_thicks.insert(pos, 600.0 / (4.0 * materials_dict[new_mat]))
                            
                            if _is_valid(new_mats, new_thicks, problem):
                                new_score = _compute_mean_reflectance(new_mats, new_thicks,
                                                                     ambient_index, substrate_index,
                                                                     band_nm, materials_dict)
                                if new_score < current_score:
                                    current_materials = new_mats
                                    current_thicknesses = new_thicks
                                    current_score = new_score
                                    improved = True
                                    break
                        
                        if time.time() > deadline:
                            break
                        if improved:
                            break
                
                # Try removing a layer
                if not improved and len(current_materials) > 1:
                    for i in range(len(current_materials)):
                        if time.time() > deadline:
                            break
                        
                        new_mats = current_materials[:]
                        new_thicks = current_thicknesses[:]
                        del new_mats[i]
                        del new_thicks[i]
                        
                        if _is_valid(new_mats, new_thicks, problem):
                            new_score = _compute_mean_reflectance(new_mats, new_thicks,
                                                                 ambient_index, substrate_index,
                                                                 band_nm, materials_dict)
                            if new_score < current_score:
                                current_materials = new_mats
                                current_thicknesses = new_thicks
                                current_score = new_score
                                improved = True
                                break
                
                if not improved:
                    break
        
        # Update global best
        if current_score < best_score:
            best_score = current_score
            best_materials = current_materials[:]
            best_thicknesses = current_thicknesses[:]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
