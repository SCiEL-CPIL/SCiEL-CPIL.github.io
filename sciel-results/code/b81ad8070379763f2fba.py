"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _reflectance_spectrum(materials, thicknesses, problem, wavelengths):
    """Compute reflectance spectrum using the characteristic matrix method."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_layers = len(materials)
    
    # Precompute refractive indices for each layer
    n_vals = np.array([problem["materials"][m] for m in materials])
    d_vals = np.array(thicknesses)
    
    # Wavelengths in meters for phase calculations, but we work in nm
    # Phase delta = 2*pi*n*d/lambda
    # For each wavelength, compute the characteristic matrix
    
    n_waves = len(wavelengths)
    R = np.zeros(n_waves)
    
    for i, lam in enumerate(wavelengths):
        # Build the characteristic matrix
        # M = product of layer matrices
        # For normal incidence, each layer matrix is:
        # [[cos(delta), (i/sin(delta))/n_j], [i*n_j*sin(delta), cos(delta)]]
        # where delta_j = 2*pi*n_j*d_j/lambda
        
        M = np.array([[1.0, 0.0]], dtype=complex)  # Start with identity
        
        for j in range(n_layers):
            delta = 2.0 * np.pi * n_vals[j] * d_vals[j] / lam
            cd = np.cos(delta)
            sd = np.sin(delta)
            
            if abs(sd) < 1e-12:
                # Avoid division by zero
                inv_n_sd = 0.0
            else:
                inv_n_sd = 1.0 / (n_vals[j] * sd)
            
            B = n_vals[j] * sd
            
            layer_M = np.array([[cd, 1j * inv_n_sd], [1j * B, cd]], dtype=complex)
            M = M @ layer_M
        
        # The total matrix M = [[M11, M12], [M21, M22]]
        # Reflectance: r = (n_amb * M11 - M12) / (n_amb * M11 + M12) ... wait
        # Actually, for the characteristic matrix method:
        # The admittance Y = M21/M22 (or similar)
        # Let me use the standard formula:
        # r = (Y_s - Y_0) / (Y_s + Y_0) where Y_s is the input admittance
        # Y_s = M21 / M22 ... no
        
        # Standard: If M = M_1 M_2 ... M_N, then
        # The field vector at the input is related to the substrate by:
        # [E_in]   = M [E_sub]
        # [H_in]       [H_sub]
        # 
        # At the substrate, H_sub = n_sub * E_sub (for normal incidence, non-magnetic)
        # So [E_in, H_in]^T = M [1, n_sub]^T * E_sub
        # 
        # The input admittance Y_in = H_in / E_in = (M12 + M22 * n_sub) / (M11 + M21 * n_sub)
        # Wait, let me be more careful.
        
        # E_in = M11 * E_sub + M12 * H_sub = M11 * E_sub + M12 * n_sub * E_sub = (M11 + M12 * n_sub) * E_sub
        # H_in = M21 * E_sub + M22 * H_sub = M21 * E_sub + M22 * n_sub * E_sub = (M21 + M22 * n_sub) * E_sub
        # 
        # Y_in = H_in / E_in = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
        
        M11, M12 = M[0, 0], M[0, 1]
        M21, M22 = M[1, 0], M[1, 1]
        
        Y_in = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
        
        # Reflection coefficient at air-coating interface
        r = (n_amb - Y_in) / (n_amb + Y_in)
        
        R[i] = np.abs(r) ** 2 * 100.0  # Reflectance in percent
    
    return R

def _nelder_mead(objective, x0, bounds, maxiter=500, tol=1e-8):
    """Simple Nelder-Mead implementation with bounds."""
    n = len(x0)
    
    # Simplex initialization
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        step = 0.05 * abs(point[i]) + 0.00025
        point[i] += step
        simplex.append(point)
    
    # Evaluate initial simplex
    values = [objective(p) for p in simplex]
    
    for it in range(maxiter):
        # Sort simplex
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[0] - values[-1]) < tol:
            break
        
        # Centroid (excluding worst point)
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < values[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            # Accept reflection
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = objective(simplex[i])
    
    # Return best point
    best_idx = np.argmin(values)
    return np.array(simplex[best_idx]), values[best_idx]

def design(problem):
    """Design an antireflection coating using optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    
    best_score = 1e6
    # Initialize with a valid single layer to ensure consistency
    best_materials = ["L"]
    best_thicknesses = [band[0] / (4.0 * n_L)]
    
    # Precompute wavelengths for evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def _objective_constrained(params):
        """Objective with constraint handling."""
        thicknesses = np.array(params)
        
        # Check constraints
        if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
            return 1e6
        if np.sum(thicknesses) > max_total_nm:
            return 1e6
        
        R = _reflectance_spectrum(best_materials, thicknesses, problem, wavelengths)
        return np.mean(R)
    
    # Try various layer counts
    for n_layers in range(2, min(max_layers + 1, 11)):  # Try up to 10 layers first
        if time.time() > deadline:
            break
        
        # Generate a few candidate material patterns
        candidate_patterns = []
        
        # Alternating starting with L
        pattern1 = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        candidate_patterns.append(pattern1)
        
        # Alternating starting with H
        pattern2 = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        candidate_patterns.append(pattern2)
        
        # For larger stacks, try symmetric patterns
        if n_layers >= 4:
            base = ["L", "H", "H", "L"]
            pattern3 = []
            for i in range(n_layers):
                pattern3.append(base[i % 4])
            candidate_patterns.append(pattern3)
        
        if n_layers >= 4:
            base = ["L", "L", "H", "H"]
            pattern4 = []
            for i in range(n_layers):
                pattern4.append(base[i % 4])
            candidate_patterns.append(pattern4)
        
        # Unique patterns
        seen = set()
        unique_patterns = []
        for p in candidate_patterns:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                unique_patterns.append(p)
        
        for materials in unique_patterns:
            if time.time() > deadline:
                break
            
            # Initial guess: quarter-wave thicknesses at center wavelength
            centre = 0.5 * (band[0] + band[1])
            x0 = []
            for m in materials:
                n_m = problem["materials"][m]
                d0 = centre / (4.0 * n_m)
                x0.append(d0)
            
            # Ensure initial guess satisfies constraints
            if sum(x0) > max_total_nm:
                scale = max_total_nm / sum(x0)
                x0 = [d * scale for d in x0]
            
            # Use Nelder-Mead for local optimization
            try:
                result_x, result_f = _nelder_mead(
                    _objective_constrained,
                    x0,
                    bounds=[(0.0, max_layer_nm)] * n_layers,
                    maxiter=300
                )
                
                # Verify constraints
                if all(0 <= t <= max_layer_nm for t in result_x) and sum(result_x) <= max_total_nm:
                    if result_f < best_score:
                        best_score = result_f
                        best_materials = materials
                        best_thicknesses = list(result_x)
            except Exception:
                pass
        
        # If we have time, try a few more layers
        if time.time() > deadline - 10:
            break
    
    # Ensure final result is valid and consistent
    if len(best_materials) != len(best_thicknesses):
        # Fallback to the initial valid single layer if mismatch occurs
        best_materials = ["L"]
        best_thicknesses = [band[0] / (4.0 * n_L)]
        
    if not all(0 <= t <= max_layer_nm for t in best_thicknesses):
        best_thicknesses = [min(max(t, 0), max_layer_nm) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
