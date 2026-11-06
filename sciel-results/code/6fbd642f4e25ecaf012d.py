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
    # Convert to radians for phase
    # delta_i = 2*pi*n_i*d_i / lambda
    # For a stack, the characteristic matrix is product of individual layer matrices
    # M_i = [[cos(delta_i), j*sin(delta_i)/n_i], [j*n_i*sin(delta_i), cos(delta_i)]]
    
    # Stack all layers
    # The total characteristic matrix M = M_1 * M_2 * ... * M_N
    # Then Y = (M11 + M12 * n_sub) / (M21 + M22 * n_sub)
    # R = |(n_amb - Y) / (n_amb + Y)|^2
    
    n = len(n_layers)
    if n == 0:
        # Bare substrate
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.abs(r) ** 2
    
    # Initialize identity matrix for each wavelength
    # We'll process all wavelengths in parallel using vectorized operations
    # M is a 2x2 matrix for each wavelength
    
    # Start with identity
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
        
        # Layer matrix:
        # [[cos(delta), j*sin(delta)/ni], [j*ni*sin(delta), cos(delta)]]
        L11 = cos_d
        L12 = 1j * sin_d / ni
        L21 = 1j * ni * sin_d
        L22 = cos_d
        
        # M = M * L (matrix multiplication)
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    # Admittance Y
    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    
    # Reflectance
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
        if t < -1e-9:  # allow tiny numerical error
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


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # 140s budget, leaving 10s margin
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # 401 equally spaced wavelengths from 400 to 800 nm
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    
    # Initial design: quarter-wave stack
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    # Start with a reasonable initial design
    # Try a few initial designs
    initial_designs = []
    
    # Single quarter-wave L layer
    initial_designs.append((["L"], [centre / (4.0 * n_L)]))
    
    # Single quarter-wave H layer
    initial_designs.append((["H"], [centre / (4.0 * n_H)]))
    
    # Two-layer quarter-wave stack (L then H)
    initial_designs.append((["L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # Three-layer quarter-wave stack (H, L, H)
    initial_designs.append((["H", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # Five-layer quarter-wave stack (H, L, H, L, H)
    initial_designs.append((["H", "L", "H", "L", "H"], 
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), 
                            centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # Seven-layer quarter-wave stack
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H)]))
    
    # Evaluate all initial designs
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
    
    # If no valid initial design, use the simplest one
    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)
    
    # Now perform local optimization using a simple gradient-free approach
    # We'll use a coordinate descent / hill climbing approach
    
    num_layers = len(best_materials)
    
    # Optimize thicknesses with fixed layer sequence
    def optimize_thicknesses(mats, thicks, problem, wavelengths, deadline, max_iter=5000):
        """Optimize thicknesses for a given layer sequence."""
        current_thicks = list(thicks)
        current_score = _evaluate_design(mats, current_thicks, problem, wavelengths)
        
        n = len(mats)
        if n == 0:
            return mats, current_thicks, current_score
        
        # Step size for each thickness
        step_sizes = [20.0] * n  # start with 20nm steps
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(n):
                # Try increasing
                new_thicks = list(current_thicks)
                new_thicks[i] = min(current_thicks[i] + step_sizes[i], max_layer_nm)
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
                # Reduce step sizes
                step_sizes = [s * 0.7 for s in step_sizes]
                if min(step_sizes) < 0.1:
                    break
            else:
                # Slightly increase step sizes when improving
                step_sizes = [min(s * 1.05, 40.0) for s in step_sizes]
        
        return mats, current_thicks, current_score
    
    # Optimize the best initial design
    best_materials, best_thicknesses, best_score = optimize_thicknesses(
        best_materials, best_thicknesses, problem, wavelengths, deadline
    )
    
    # Try different layer sequences
    # Generate candidate sequences
    candidate_sequences = []
    
    # Quarter-wave style sequences
    for n_layers in range(1, min(11, max_layers + 1)):
        # Alternating starting with H
        seq = []
        for i in range(n_layers):
            seq.append("H" if i % 2 == 0 else "L")
        candidate_sequences.append(seq)
        
        # Alternating starting with L
        seq = []
        for i in range(n_layers):
            seq.append("L" if i % 2 == 0 else "H")
        candidate_sequences.append(seq)
    
    # Also try some non-alternating patterns
    for n_layers in range(3, min(9, max_layers + 1)):
        # HLLH pattern
        if n_layers % 2 == 1:
            seq = []
            for i in range(n_layers):
                if i % 4 == 0:
                    seq.append("H")
                else:
                    seq.append("L")
            candidate_sequences.append(seq)
    
    # Evaluate and optimize each candidate sequence
    for seq in candidate_sequences:
        if time.time() > deadline:
            break
        
        if len(seq) > max_layers:
            continue
        
        # Check if this sequence is already in our best
        if seq == best_materials:
            continue
        
        # Initialize with quarter-wave thicknesses
        thicks = [centre / (4.0 * materials_dict[m]) for m in seq]
        
        # Check validity
        if not _is_valid(seq, thicks, problem):
            continue
        
        # Evaluate
        score = _evaluate_design(seq, thicks, problem, wavelengths)
        
        # Only optimize if it looks promising
        if score < best_score + 2.0:  # within 2% of best
            mats, thicks, score = optimize_thicknesses(
                seq, thicks, problem, wavelengths, deadline, max_iter=2000
            )
            if score < best_score:
                best_score = score
                best_materials = list(seq)
                best_thicknesses = list(thicks)
    
    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to simplest valid design
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
