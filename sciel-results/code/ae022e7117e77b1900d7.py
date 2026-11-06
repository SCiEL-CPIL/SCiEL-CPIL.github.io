"""Broadband antireflection coating design via multi-start Powell optimization."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _characteristic_matrix(n, d, lam):
    """Compute 2x2 characteristic matrix for a single layer at wavelength lam (nm)."""
    beta = 2.0 * np.pi * n * d / lam
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    # For normal incidence: B11 = cos(beta), B12 = i*sin(beta)/n
    # We work with real and imaginary parts separately or use complex
    return np.array([
        [np.cos(beta), 1j * np.sin(beta) / n],
        [1j * n * np.sin(beta), np.cos(beta)]
    ], dtype=complex)


def _reflectance_stack(materials, thicknesses, n_ambient, n_substrate, wavelengths):
    """Compute reflectance spectrum using characteristic matrix method.
    
    Args:
        materials: list of refractive indices
        thicknesses: list of physical thicknesses in nm
        n_ambient: ambient index
        n_substrate: substrate index
        wavelengths: array of wavelengths in nm
    
    Returns:
        Array of reflectance values (0 to 1) for each wavelength
    """
    n_layers = len(materials)
    n_wl = len(wavelengths)
    R = np.zeros(n_wl)
    
    for i, lam in enumerate(wavelengths):
        # Start with identity
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        # Multiply by each layer's characteristic matrix
        for j in range(n_layers):
            n_j = materials[j]
            d_j = thicknesses[j]
            beta = 2.0 * np.pi * n_j * d_j / lam
            cos_b = np.cos(beta)
            sin_b = np.sin(beta)
            
            layer_M = np.array([
                [cos_b, 1j * sin_b / n_j],
                [1j * n_j * sin_b, cos_b]
            ], dtype=complex)
            
            M = M @ layer_M
        
        # Compute admittance at the air side
        # Y = (M11 * n_sub + M12 * n_ambient * n_sub) / (M21 * n_sub + M22 * n_ambient)
        # Actually, the standard formula:
        # The characteristic matrix gives us the relation between fields
        # Y = (n_sub * M11 + M12 * n_ambient * n_sub) / (n_sub * M21 + M22 * n_ambient)
        # Let me use the standard approach:
        
        # The admittance Y seen from the ambient side:
        # Y = (M11 * n_sub + M12) / (M21 * n_sub + M22)
        # where M is the total characteristic matrix
        
        Y = (M[0, 0] * n_substrate + M[0, 1]) / (M[1, 0] * n_substrate + M[1, 1])
        
        # Reflectance
        r = (Y - n_ambient) / (Y + n_ambient)
        R[i] = np.abs(r) ** 2
    
    return R


def _objective(x, problem, wavelengths):
    """Compute mean reflectance in percent for a given set of layer parameters.
    
    x encodes: [n_layers, t1, t2, ..., tn] where t_i are optical thicknesses (n*d)
    The layer sequence alternates L, H, L, H, ... starting with L.
    """
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    
    n_layers = int(round(x[0]))
    n_layers = max(1, min(n_layers, max_layers))
    
    if n_layers == 0:
        return 100.0  # No coating, high reflectance
    
    optical_thicknesses = x[1:1 + n_layers]
    
    # Convert optical thickness to physical thickness
    # Layer sequence: L, H, L, H, ...
    materials = []
    thicknesses = []
    total_nm = 0.0
    
    for i in range(n_layers):
        if i % 2 == 0:
            n_mat = n_L
        else:
            n_mat = n_H
        
        d = optical_thicknesses[i] / n_mat
        d = max(0.0, min(d, max_layer_nm))
        
        materials.append(n_mat)
        thicknesses.append(d)
        total_nm += d
    
    # Check total thickness constraint
    if total_nm > max_total_nm + 1e-9:
        return 100.0  # Penalty for violating constraint
    
    # Compute reflectance
    R = _reflectance_stack(materials, thicknesses, n_ambient, n_substrate, wavelengths)
    
    return np.mean(R) * 100.0


def _powell_optimize(x0, problem, wavelengths, max_iter=50, tol=1e-8):
    """Powell's method for optimization."""
    from scipy.optimize import minimize
    
    # Bounds: n_layers in [1, 20], optical thicknesses in [0, 400*n_max]
    # Max optical thickness per layer: 400 nm * 2.35 = 940 nm
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    max_opt_thick = max_layer_nm * max(problem["materials"].values())
    
    # We'll optimize with a fixed number of layers initially, then try different counts
    n0 = int(round(x0[0]))
    n0 = max(1, min(n0, max_layers))
    
    # Optimize optical thicknesses for this fixed layer count
    def obj_fixed_layers(t):
        # t is array of optical thicknesses
        x = np.concatenate([[float(n0)], t])
        return _objective(x, problem, wavelengths)
    
    bounds = [(0.0, max_opt_thick)] * n0
    x0_fixed = x0[1:1 + n0]
    
    try:
        result = minimize(obj_fixed_layers, x0_fixed, method='Powell', 
                         bounds=bounds, options={'maxiter': max_iter, 'ftol': tol})
        return result.x, result.fun
    except Exception:
        return x0_fixed, obj_fixed_layers(x0_fixed)


def design(problem):
    """Design a broadband antireflection coating using multi-start Powell optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points from 400 to 800 nm)
    wavelengths = np.linspace(band[0], band[1], 401)
    
    best_score = float('inf')
    best_x = None
    
    # Multi-start: try different initial layer counts and thicknesses
    # Layer counts to try: 1, 2, 3, 4, 5, 6, 7, 8
    layer_counts = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # For each layer count, try a few random initializations
    rng = np.random.RandomState(42)
    
    max_opt_thick = max_layer_nm * max(n_L, n_H)
    
    for n_layers in layer_counts:
        if time.time() > deadline:
            break
        
        # Try 3 different random starts for each layer count
        for start_idx in range(3):
            if time.time() > deadline:
                break
            
            # Initial optical thicknesses: quarter-wave at center wavelength
            center_wl = 0.5 * (band[0] + band[1])
            if start_idx == 0:
                # Quarter-wave design
                init_t = np.zeros(n_layers)
                for i in range(n_layers):
                    n_mat = n_L if i % 2 == 0 else n_H
                    init_t[i] = n_mat * center_wl / 4.0
            elif start_idx == 1:
                # Random initialization
                init_t = rng.uniform(0.1 * max_opt_thick, 0.9 * max_opt_thick, size=n_layers)
            else:
                # Half-quarter wave variation
                init_t = np.zeros(n_layers)
                for i in range(n_layers):
                    n_mat = n_L if i % 2 == 0 else n_H
                    init_t[i] = n_mat * center_wl / 4.0 * (0.8 + 0.4 * rng.random())
            
            x0 = np.concatenate([[float(n_layers)], init_t])
            
            # Run Powell optimization
            try:
                t_opt, score = _powell_optimize(x0, problem, wavelengths, max_iter=30, tol=1e-7)
                
                if score < best_score:
                    best_score = score
                    best_x = np.concatenate([[float(n_layers)], t_opt])
            except Exception:
                pass
        
        # Try a few more starts with different layer counts if time permits
        if time.time() > deadline:
            break
    
    # If no valid solution found, use a simple quarter-wave coating
    if best_x is None:
        center_wl = 0.5 * (band[0] + band[1])
        d_L = center_wl / (4.0 * n_L)
        return ["L"], [d_L]
    
    # Convert best solution to materials and thicknesses
    n_layers = int(round(best_x[0]))
    n_layers = max(1, min(n_layers, max_layers))
    
    materials = []
    thicknesses = []
    total_nm = 0.0
    
    for i in range(n_layers):
        if i % 2 == 0:
            n_mat = n_L
            label = 'L'
        else:
            n_mat = n_H
            label = 'H'
        
        opt_thick = best_x[1 + i]
        d = opt_thick / n_mat
        d = max(0.0, min(d, max_layer_nm))
        
        materials.append(label)
        thicknesses.append(d)
        total_nm += d
    
    # If total thickness exceeds limit, scale down
    if total_nm > max_total_nm and total_nm > 0:
        scale = max_total_nm / total_nm
        thicknesses = [d * scale for d in thicknesses]
    
    return materials, thicknesses

# EVOLVE-BLOCK-END
