"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using characteristic matrix method."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_dict = problem["materials"]
    lam0, lam1 = problem["band_nm"]
    n_waves = 401
    lambdas = np.linspace(lam0, lam1, n_waves)
    
    total_r = 0.0
    for lam in lambdas:
        # Build total characteristic matrix
        A = 1.0 + 0j
        B = 0.0 + 0j
        C = 0.0 + 0j
        D = 1.0 + 0j
        
        for i, mat in enumerate(materials):
            n_i = n_dict[mat]
            d_i = thicknesses[i]
            # Phase thickness delta = 2*pi*n*d/lambda
            delta = 2.0 * np.pi * n_i * d_i / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Layer matrix:
            # [cos_d, i*sin_d/n_i]
            # [i*n_i*sin_d, cos_d]
            M11 = cos_d
            M12 = 1j * sin_d / n_i
            M21 = 1j * n_i * sin_d
            M22 = cos_d
            
            # Multiply current total matrix M by layer matrix M_i
            # New M = M * M_i
            nA = A * M11 + B * M21
            nB = A * M12 + B * M22
            nC = C * M11 + D * M21
            nD = C * M12 + D * M22
            
            A, B, C, D = nA, nB, nC, nD
        
        # Input admittance Y_in = (C + D*ns) / (A + B*ns)
        Y_in = (C + D * ns) / (A + B * ns)
        
        # Reflectance R = | (Y0 - Y_in) / (Y0 + Y_in) |^2
        r = (n0 - Y_in) / (n0 + Y_in)
        R = np.abs(r) ** 2
        total_r += R
        
    return 100.0 * total_r / n_waves

def _evaluate_candidate(materials, thicknesses, problem):
    """Check constraints and return reflectance or infinity if invalid."""
    if not materials:
        # No layers, just bare glass
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
        
    if len(materials) > problem["max_layers"]:
        return float('inf')
    if len(materials) != len(thicknesses):
        return float('inf')
    
    total_thick = 0.0
    for i, mat in enumerate(materials):
        if mat not in problem["materials"]:
            return float('inf')
        d = thicknesses[i]
        if d < 0 or d > problem["max_layer_nm"]:
            return float('inf')
        total_thick += d
        
    if total_thick > problem["max_total_nm"]:
        return float('inf')
        
    return _mean_reflectance(materials, thicknesses, problem)

def _optimize_thicknesses(materials, problem):
    """Use differential evolution to optimize thicknesses for a given material sequence."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_dict = problem["materials"]
    lam0, lam1 = problem["band_nm"]
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    n_layers = len(materials)
    
    # Define bounds for each thickness
    bounds = [(0, max_layer)] * n_layers
    
    def objective(thicknesses):
        # Check total thickness constraint
        total = sum(thicknesses)
        if total > max_total + 1e-6:
            return 100.0  # Penalty for violating total thickness
        
        # Compute mean reflectance
        total_r = 0.0
        n_waves = 401
        lambdas = np.linspace(lam0, lam1, n_waves)
        
        for lam in lambdas:
            A = 1.0 + 0j
            B = 0.0 + 0j
            C = 0.0 + 0j
            D = 1.0 + 0j
            
            for i, mat in enumerate(materials):
                n_i = n_dict[mat]
                d_i = thicknesses[i]
                delta = 2.0 * np.pi * n_i * d_i / lam
                cos_d = np.cos(delta)
                sin_d = np.sin(delta)
                
                M11 = cos_d
                M12 = 1j * sin_d / n_i
                M21 = 1j * n_i * sin_d
                M22 = cos_d
                
                nA = A * M11 + B * M21
                nB = A * M12 + B * M22
                nC = C * M11 + D * M21
                nD = C * M12 + D * M22
                
                A, B, C, D = nA, nB, nC, nD
            
            Y_in = (C + D * ns) / (A + B * ns)
            r = (n0 - Y_in) / (n0 + Y_in)
            R = np.abs(r) ** 2
            total_r += R
            
        return 100.0 * total_r / n_waves
    
    # Run differential evolution with limited iterations
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=100,
        popsize=15,
        tol=1e-4,
        seed=42,
        polish=True
    )
    
    return list(result.x), result.fun

def design(problem):
    """Design a stack by optimizing thicknesses for several material sequences."""
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    lam_center = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Quarter wave thicknesses at center wavelength for initial guesses
    tqL = lam_center / (4.0 * nL)
    tqH = lam_center / (4.0 * nH)
    
    best_materials = None
    best_thicknesses = None
    best_r = float('inf')
    
    # Try various material sequences with differential evolution optimization
    sequences_to_try = [
        ["L"],
        ["H", "L"],
        ["L", "H"],
        ["L", "H", "L"],
        ["H", "L", "H"],
        ["L", "H", "L", "H"],
        ["H", "L", "H", "L"],
        ["L", "H", "L", "H", "L"],
        ["H", "L", "H", "L", "H"],
        ["L", "H", "L", "H", "L", "H"],
        ["H", "L", "H", "L", "H", "L"],
        ["L", "H", "L", "H", "L", "H", "L"],
        ["H", "L", "H", "L", "H", "L", "H"],
    ]
    
    for mats in sequences_to_try:
        try:
            # Optimize thicknesses for this sequence
            opt_thick, opt_r = _optimize_thicknesses(mats, problem)
            
            # Verify the result
            if opt_r < best_r:
                # Check if the optimized solution is valid
                if all(0 <= d <= problem["max_layer_nm"] for d in opt_thick):
                    if sum(opt_thick) <= problem["max_total_nm"] + 1e-6:
                        best_r = opt_r
                        best_materials = mats
                        best_thicknesses = opt_thick
        except Exception:
            # If optimization fails, try quarter-wave values as fallback
            try:
                if len(mats) == 1:
                    if mats[0] == "L":
                        thicks = [tqL]
                    else:
                        thicks = [tqH]
                else:
                    thicks = []
                    for m in mats:
                        if m == "L":
                            thicks.append(tqL)
                        else:
                            thicks.append(tqH)
                
                r = _evaluate_candidate(mats, thicks, problem)
                if r < best_r:
                    best_r = r
                    best_materials = mats
                    best_thicknesses = thicks
            except Exception:
                pass
    
    # If all failed, fallback to single L layer
    if best_materials is None:
        return ["L"], [tqL]
        
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
