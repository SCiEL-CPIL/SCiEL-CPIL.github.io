"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """
    Compute reflectance (0-1) for a stack using the characteristic matrix method.
    
    wavelengths: array of wavelengths in nm
    n_ambient: refractive index of ambient
    n_substrate: refractive index of substrate
    n_layers: array of refractive indices of layers
    d_layers: array of physical thicknesses in nm
    
    Returns: array of reflectance values (0-1)
    """
    n = len(n_layers)
    if n == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.abs(r) ** 2
    
    # Initialize identity matrix for each wavelength
    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)
    
    for i in range(n):
        ni = n_layers[i]
        di = d_layers[i]
        delta = 2.0 * np.pi * ni * di / wavelengths
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        L11 = cos_d
        L12 = 1j * sin_d / ni
        L21 = 1j * ni * sin_d
        L22 = cos_d
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return R


def _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """Compute mean reflectance in percent."""
    R = _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers)
    return np.mean(R) * 100.0


def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return True
    
    total = sum(thicknesses)
    if total > max_total_nm:
        return False
    
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    
    for m in materials:
        if m not in problem["materials"]:
            return False
    
    return True


def _evaluate_design(materials, thicknesses, problem, wavelengths):
    """Evaluate a design and return mean reflectance in percent."""
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_layers = np.array([problem["materials"][m] for m in materials], dtype=float)
    d_layers = np.array(thicknesses, dtype=float)
    
    return _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers)


def _optimize_thicknesses(mats, thicks, problem, wavelengths, deadline, max_iter=3000):
    """Optimize thicknesses for a given layer sequence using adaptive step sizes."""
    current_thicks = list(thicks)
    current_score = _evaluate_design(mats, current_thicks, problem, wavelengths)
    
    n = len(mats)
    if n == 0:
        return mats, current_thicks, current_score
    
    # Initialize step sizes
    step_sizes = [15.0] * n
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Try increasing
            new_thicks = list(current_thicks)
            new_thicks[i] = min(current_thicks[i] + step_sizes[i], 400.0)
            if _is_valid(mats, new_thicks, problem):
                new_score = _evaluate_design(mats, new_thicks, problem, wavelengths)
                if new_score < current_score:
                    current_thicks = new_thicks
                    current_score = new_score
                    improved = True
                    continue
            
            # Try decreasing
            new_thicks = list(current_thicks)
            new_thicks[i] = max(current_thicks[i] - step_sizes[i], 0.0)
            if _is_valid(mats, new_thicks, problem):
                new_score = _evaluate_design(mats, new_thicks, problem, wavelengths)
                if new_score < current_score:
                    current_thicks = new_thicks
                    current_score = new_score
                    improved = True
                    continue
        
        if not improved:
            step_sizes = [s * 0.6 for s in step_sizes]
            if min(step_sizes) < 0.5:
                break
        else:
            step_sizes = [min(s * 1.1, 30.0) for s in step_sizes]
    
    return mats, current_thicks, current_score


def _polish_thicknesses(mats, thicks, problem, wavelengths, deadline, max_iter=5000):
    """High-precision polish using finer step sizes and line search."""
    current_thicks = list(thicks)
    current_score = _evaluate_design(mats, current_thicks, problem, wavelengths)
    
    n = len(mats)
    if n == 0:
        return mats, current_thicks, current_score
    
    # Start with smaller step sizes for polish
    step_sizes = [3.0] * n
    min_step = 0.01
    max_step = 10.0
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Try several step sizes around current value
            best_local_score = current_score
            best_local_thick = current_thicks[i]
            
            for factor in [1.0, 0.5, 0.25, 0.1]:
                step = step_sizes[i] * factor
                
                # Try increasing
                new_thick = min(current_thicks[i] + step, 400.0)
                new_thicks = list(current_thicks)
                new_thicks[i] = new_thick
                if _is_valid(mats, new_thicks, problem):
                    new_score = _evaluate_design(mats, new_thicks, problem, wavelengths)
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local_thick = new_thick
                
                # Try decreasing
                new_thick = max(current_thicks[i] - step, 0.0)
                new_thicks = list(current_thicks)
                new_thicks[i] = new_thick
                if _is_valid(mats, new_thicks, problem):
                    new_score = _evaluate_design(mats, new_thicks, problem, wavelengths)
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local_thick = new_thick
            
            if best_local_score < current_score:
                current_thicks[i] = best_local_thick
                current_score = best_local_score
                improved = True
        
        if not improved:
            step_sizes = [s * 0.7 for s in step_sizes]
            if min(step_sizes) < min_step:
                break
        else:
            step_sizes = [min(s * 1.2, max_step) for s in step_sizes]
    
    return mats, current_thicks, current_score


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    # Generate diverse initial designs
    initial_designs = []
    
    # Single layer designs
    initial_designs.append((["L"], [centre / (4.0 * n_L)]))
    initial_designs.append((["H"], [centre / (4.0 * n_H)]))
    
    # Two-layer designs
    initial_designs.append((["L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Three-layer designs
    initial_designs.append((["H", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Four-layer designs
    initial_designs.append((["H", "L", "H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    initial_designs.append((["L", "H", "L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # Five-layer designs
    initial_designs.append((["H", "L", "H", "L", "H"], 
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), 
                            centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["L", "H", "L", "H", "L"], 
                           [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), 
                            centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Six-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Seven-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H)]))
    
    # Eight-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Nine-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # Ten-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L)]))
    
    # Non-standard patterns
    # HLLH pattern
    initial_designs.append((["H", "L", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["H", "L", "L", "H", "L", "L", "H"], 
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), 
                            centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # LHL pattern
    initial_designs.append((["L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Evaluate all initial designs and keep the best
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    for mats, thicks in initial_designs:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _evaluate_design(mats, thicks, problem, wavelengths)
        if score < best_score:
            best_score = score
            best_materials = list(mats)
            best_thicknesses = list(thicks)
    
    # Fallback if no valid initial design
    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)
    
    # Optimize the best initial design
    best_materials, best_thicknesses, best_score = _optimize_thicknesses(
        best_materials, best_thicknesses, problem, wavelengths, deadline
    )
    
    # Try other promising sequences with shorter optimization
    # Generate additional candidate sequences
    candidate_sequences = []
    
    # Alternating sequences of various lengths
    for n_layers in range(1, min(15, max_layers + 1)):
        # Starting with H
        seq = []
        for i in range(n_layers):
            seq.append("H" if i % 2 == 0 else "L")
        candidate_sequences.append(seq)
        
        # Starting with L
        seq = []
        for i in range(n_layers):
            seq.append("L" if i % 2 == 0 else "H")
        candidate_sequences.append(seq)
    
    # Some non-alternating patterns
    for n_layers in [4, 6, 8, 10, 12]:
        # HLLH repeated
        if n_layers % 4 == 0:
            seq = []
            for i in range(n_layers):
                pos = i % 4
                if pos == 0 or pos == 3:
                    seq.append("H")
                else:
                    seq.append("L")
            candidate_sequences.append(seq)
        
        # LHHL repeated
        if n_layers % 4 == 0:
            seq = []
            for i in range(n_layers):
                pos = i % 4
                if pos == 0 or pos == 1:
                    seq.append("L" if pos == 0 else "H")
                else:
                    seq.append("H" if pos == 2 else "L")
            # Simplified: just add a few more patterns
            pass
    
    # Add some specific patterns
    candidate_sequences.append(["H", "L", "L", "H", "L", "L", "H"])
    candidate_sequences.append(["L", "H", "H", "L", "H", "H", "L"])
    candidate_sequences.append(["H", "L", "H", "L", "L", "H", "L", "H"])
    candidate_sequences.append(["L", "H", "L", "H", "H", "L", "H", "L"])
    
    # Deduplicate and filter
    seen = set()
    unique_candidates = []
    for seq in candidate_sequences:
        if len(seq) > max_layers:
            continue
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(seq)
    
    # Evaluate and optimize each candidate
    for seq in unique_candidates:
        if time.time() > deadline:
            break
        
        if seq == best_materials:
            continue
        
        thicks = [centre / (4.0 * materials_dict[m]) for m in seq]
        
        if not _is_valid(seq, thicks, problem):
            continue
        
        score = _evaluate_design(seq, thicks, problem, wavelengths)
        
        # Only optimize if promising
        if score < best_score + 3.0:
            mats, thicks, score = _optimize_thicknesses(
                seq, thicks, problem, wavelengths, deadline, max_iter=1500
            )
            if score < best_score:
                best_score = score
                best_materials = list(seq)
                best_thicknesses = list(thicks)
    
    # Final refinement on the best design
    if time.time() < deadline - 5.0:
        best_materials, best_thicknesses, best_score = _optimize_thicknesses(
            best_materials, best_thicknesses, problem, wavelengths, deadline, max_iter=2000
        )
    
    # High-precision polish pass
    if time.time() < deadline - 2.0:
        best_materials, best_thicknesses, best_score = _polish_thicknesses(
            best_materials, best_thicknesses, problem, wavelengths, deadline, max_iter=5000
        )
    
    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
