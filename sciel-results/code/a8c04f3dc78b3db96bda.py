import numpy as np
from scipy.optimize import minimize
import time

def design(problem):
    """
    Designs a broadband antireflection coating using a genetic-like optimization
    of layer thicknesses for fixed alternating sequences.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s budget
    
    n_0 = problem.get("ambient_index", 1.0)
    n_s = problem.get("substrate_index", 1.52)
    materials = problem["materials"]
    n_L = materials["L"]
    n_H = materials["H"]
    band_nm = problem["band_nm"]
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # 401 equally spaced wavelengths
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    def compute_reflectance(thicknesses, layer_indices):
        """
        Computes mean reflectance for given thicknesses and layer indices.
        """
        N = len(thicknesses)
        if N == 0:
            # Bare substrate
            r = (n_0 - n_s) / (n_0 + n_s)
            return abs(r)**2 * 100.0
            
        # Precompute delta for all wavelengths and layers
        # delta[i, j] = 2 * pi * n_i * d_i / lambda_j
        # n_layers x N_wavelengths
        deltas = np.zeros((N, len(wavelengths)))
        for i in range(N):
            n_i = layer_indices[i]
            d_i = thicknesses[i]
            deltas[i, :] = 2.0 * np.pi * n_i * d_i / wavelengths
        
        # Initialize total matrix M = I
        # M is a 2x2 matrix for each wavelength
        # We can store A, B, C, D as arrays of shape (N_wavelengths,)
        A = np.ones(len(wavelengths)) + 0j
        B = np.zeros(len(wavelengths)) + 0j
        C = np.zeros(len(wavelengths)) + 0j
        D = np.ones(len(wavelengths)) + 0j
        
        for i in range(N):
            n_i = layer_indices[i]
            delta = deltas[i, :]
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Matrix M_i
            # A_i = cos_d
            # B_i = i * n_i * sin_d
            # C_i = i * sin_d / n_i
            # D_i = cos_d
            
            # M_total = M_total @ M_i
            # [A B]   [A_i B_i]   [A*A_i + B*C_i, A*B_i + B*D_i]
            # [C D] @ [C_i D_i] = [C*A_i + D*C_i, C*B_i + D*D_i]
            
            new_A = A * cos_d + B * (1j * sin_d / n_i)
            new_B = A * (1j * n_i * sin_d) + B * cos_d
            new_C = C * cos_d + D * (1j * sin_d / n_i)
            new_D = C * (1j * n_i * sin_d) + D * cos_d
            
            A = new_A
            B = new_B
            C = new_C
            D = new_D
            
        # Admittance Y
        # Y = (A*n_s + B) / (C*n_s + D)
        Y = (A * n_s + B) / (C * n_s + D)
        
        # Reflectance
        r = (n_0 - Y) / (n_0 + Y)
        R = np.abs(r)**2
        return np.mean(R) * 100.0

    # Candidate sequences
    # L is 1.38, H is 2.35
    # Start with L (close to air), end with L (close to substrate)
    sequences = [
        ["L", "H", "L", "H", "L"],
        ["L", "H", "L", "H", "L", "H", "L"],
        ["L", "H", "L", "H", "L", "H", "L", "H", "L"]
    ]
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    for seq in sequences:
        if time.time() > deadline:
            break
            
        n_layers = len(seq)
        indices = [materials[mat] for mat in seq]
        
        # Initial guess: quarter wave at 600 nm
        x0 = np.zeros(n_layers)
        for i, n_i in enumerate(indices):
            x0[i] = 600.0 / (4.0 * n_i)
            
        def objective(x):
            if np.any(x < 0) or np.any(x > max_layer_nm):
                return 1e6
            if np.sum(x) > max_total_nm:
                return 1e6 + (np.sum(x) - max_total_nm)
            return compute_reflectance(x, indices)
        
        # Optimize
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        except Exception:
            continue
            
        if res.fun < best_score:
            x_opt = res.x
            if np.all(x_opt >= 0) and np.all(x_opt <= max_layer_nm) and np.sum(x_opt) <= max_total_nm:
                best_score = res.fun
                best_materials = seq
                best_thicknesses = x_opt
                
    if best_materials is None:
        # Fallback to seed
        centre = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [centre / (4.0 * n_L)]
        
    return best_materials, list(best_thicknesses)

