import numpy as np
from scipy.optimize import minimize

def design(problem):
    """
    Design a broadband antireflection coating using local optimization.
    """
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    materials = problem["materials"]
    n_L = materials["L"]
    n_H = materials["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    def calc_mean_reflectance(materials_seq, thicknesses):
        """
        Calculate mean reflectance for a given stack.
        """
        N = len(thicknesses)
        if N == 0:
            # Bare substrate
            r = (n_air - n_sub) / (n_air + n_sub)
            return (abs(r)**2) * 100.0

        total_reflectance = 0.0
        for wl in wavelengths:
            # Characteristic matrix for the stack
            # M_total = M_1 * M_2 * ... * M_N
            M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
            
            for i in range(N):
                n_i = n_L if materials_seq[i] == 'L' else n_H
                d_i = thicknesses[i]
                # Optical phase delta = 2 * pi * n * d / wl
                delta = 2.0 * np.pi * n_i * d_i / wl
                cos_d = np.cos(delta)
                sin_d = np.sin(delta)
                
                # Matrix for layer i
                # [ cos(delta), i * sin(delta) / n_i ]
                # [ i * n_i * sin(delta), cos(delta) ]
                M_i = np.array([
                    [cos_d, 1j * sin_d / n_i],
                    [1j * n_i * sin_d, cos_d]
                ], dtype=complex)
                
                M_total = M_total @ M_i
            
            # Admittance at the front surface
            # Y_in = (M_21 + M_22 * n_sub) / (M_11 + M_12 * n_sub)
            # Note: The matrix M_total maps fields from substrate to air.
            # E_0 = M_11 E_s + M_12 H_s
            # H_0 = M_21 E_s + M_22 H_s
            # In substrate, H_s = n_sub * E_s (assuming normalized to air impedance)
            # So Y_in = H_0/E_0 = (M_21 + M_22*n_sub) / (M_11 + M_12*n_sub)
            
            denom = M_total[0, 0] + M_total[0, 1] * n_sub
            numer = M_total[1, 0] + M_total[1, 1] * n_sub
            
            if abs(denom) < 1e-12:
                Y_in = 0
            else:
                Y_in = numer / denom
            
            # Reflectance R = |(Y_in - n_air) / (Y_in + n_air)|^2
            # n_air is 1.0
            r = (Y_in - n_air) / (Y_in + n_air)
            R = (abs(r)**2)
            total_reflectance += R
        
        return (total_reflectance / len(wavelengths)) * 100.0

    def objective(x):
        """
        Objective function for the optimizer.
        x: array of thicknesses
        """
        # x corresponds to the fixed materials_seq
        # We need to handle the penalty for total thickness
        total_thick = np.sum(x)
        penalty = 0.0
        if total_thick > max_total_nm:
            penalty = 100.0 * (total_thick - max_total_nm)
        
        # Also ensure individual thicknesses are within bounds
        # scipy minimize with bounds handles individual bounds
        # But we should check for negative values just in case
        if np.any(x < 0):
            return 1000.0 + penalty
            
        R_mean = calc_mean_reflectance(materials_seq, x)
        return R_mean + penalty

    # Define a few initial guesses to increase robustness
    # Guess 1: Alternating L/H quarter-wave at 600nm
    N1 = 10
    mat_seq1 = ['L' if i % 2 == 0 else 'H' for i in range(N1)]
    d1 = np.array([(600.0 / (4.0 * (n_L if m == 'L' else n_H))) for m in mat_seq1])
    
    # Guess 2: All L, quarter-wave at 600nm
    N2 = 10
    mat_seq2 = ['L'] * N2
    d2 = np.array([(600.0 / (4.0 * n_L)) for _ in range(N2)])
    
    # Guess 3: Random valid stack
    N3 = 12
    mat_seq3 = np.random.choice(['L', 'H'], size=N3)
    d3 = np.random.uniform(50, 200, size=N3)
    
    best_result = None
    best_score = float('inf')
    
    # Bounds for thicknesses
    bounds = [(0, max_layer_nm) for _ in range(N1)]
    
    # Run optimization for each guess
    # Use Nelder-Mead for derivative-free optimization
    # maxiter and maxfev to control time
    # 60s budget is generous, but let's be safe.
    
    for mat_seq, d_init, N in [(mat_seq1, d1, N1), (mat_seq2, d2, N2), (mat_seq3, d3, N3)]:
        # Ensure initial guess is within bounds
        d_init = np.clip(d_init, 0, max_layer_nm)
        
        # If initial guess violates total thickness, scale it down
        if np.sum(d_init) > max_total_nm:
            d_init = d_init * (max_total_nm / np.sum(d_init)) * 0.9 # Leave some margin
        
        res = minimize(objective, d_init, args=(mat_seq,), method='Nelder-Mead', 
                       bounds=bounds, 
                       options={'maxiter': 2000, 'maxfev': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.success:
            final_thick = res.x
            # Check constraints again
            if np.all(final_thick >= 0) and np.all(final_thick <= max_layer_nm) and np.sum(final_thick) <= max_total_nm:
                score = calc_mean_reflectance(mat_seq, final_thick)
                if score < best_score:
                    best_score = score
                    best_result = (mat_seq, final_thick)
        else:
            # If optimization fails, check the best point found so far
            final_thick = res.x
            if np.all(final_thick >= 0) and np.all(final_thick <= max_layer_nm) and np.sum(final_thick) <= max_total_nm:
                score = calc_mean_reflectance(mat_seq, final_thick)
                if score < best_score:
                    best_score = score
                    best_result = (mat_seq, final_thick)

    # If no valid result found, fallback to a simple valid design
    if best_result is None:
        # Fallback: Single L layer
        d_fallback = 600.0 / (4.0 * n_L)
        if d_fallback > max_layer_nm:
            d_fallback = max_layer_nm
        return ['L'], [d_fallback]
    
    mat_seq, thicknesses = best_result
    return mat_seq, list(thicknesses)
