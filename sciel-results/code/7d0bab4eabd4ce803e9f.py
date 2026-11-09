import numpy as np
from scipy.optimize import minimize
import time

def design(problem):
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    # 401 points between 400 and 800
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Function to calculate mean reflectance
    def mean_reflectance(thicknesses):
        # thicknesses is a list of floats
        # Check bounds
        for t in thicknesses:
            if t < 0 or t > max_layer_nm:
                return 100.0 # Penalty
        if sum(thicknesses) > max_total_nm:
            return 100.0 # Penalty
        
        # Characteristic Matrix Method
        # M = product of individual layer matrices
        # R = |(n0*Y - 1)/(n0*Y + 1)|^2 where Y is the admittance
        
        # Convert to angular frequency or just use phase
        # Phase delta = 2 * pi * n * d / lambda
        
        # We compute the complex admittance Y
        # Y_sub = n_sub (real)
        # Y_amb = n_amb (real)
        
        # Iterate layers from substrate to air? Or air to substrate?
        # Standard: M_total = M_1 * M_2 * ... * M_N
        # Y_total = (m11*n_sub + m12) / (m21*n_sub + m22) ? 
        # Let's use the standard formula:
        # For a stack of layers, the characteristic matrix M is:
        # M = M_1 M_2 ... M_N
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
        # The admittance Y of the stack is:
        # Y = (m11*n_s + m12) / (m21*n_s + m22)
        # Reflectance R = |(n_0*Y - 1)/(n_0*Y + 1)|^2
        
        m11 = 1.0 + 0.0j
        m12 = 0.0 + 0.0j
        m21 = 0.0 + 0.0j
        m22 = 1.0 + 0.0j
        
        # Note: Order of multiplication matters.
        # If we go from air to substrate, the first layer is closest to air.
        # The matrix product M = M_1 M_2 ... M_N corresponds to light traveling from air to substrate.
        # The admittance at the air interface is Y = (M_11*n_s + M_12) / (M_21*n_s + M_22)
        
        for i, t in enumerate(thicknesses):
            n_layer = n_L if i % 2 == 0 else n_H # Assuming L starts
            # Wait, the sequence is fixed by the caller, so I need to pass indices too.
            # I will refactor to pass indices as well.
            pass
        # This helper needs indices. I'll make a closure or pass them.
        
        # Let's rewrite mean_reflectance to take indices as well.
        
        R = 0.0
        for lam in wavelengths:
            m11 = 1.0 + 0.0j
            m12 = 0.0 + 0.0j
            m21 = 0.0 + 0.0j
            m22 = 1.0 + 0.0j
            
            for i, t in enumerate(thicknesses):
                n_layer = indices[i]
                delta = 2.0 * np.pi * n_layer * t / lam
                cos_d = np.cos(delta)
                sin_d = np.sin(delta)
                
                # Layer matrix
                m11_l = cos_d
                m12_l = 1.0j * sin_d / n_layer
                m21_l = 1.0j * n_layer * sin_d
                m22_l = cos_d
                
                # Multiply M = M * M_l
                new_m11 = m11 * m11_l + m12 * m21_l
                new_m12 = m11 * m12_l + m12 * m22_l
                new_m21 = m21 * m11_l + m22 * m21_l
                new_m22 = m21 * m12_l + m22 * m22_l
                
                m11, m12, m21, m22 = new_m11, new_m12, new_m21, new_m22
            
            # Calculate Admittance Y
            # Y = (m11*n_s + m12) / (m21*n_s + m22)
            Y = (m11 * n_substrate + m12) / (m21 * n_substrate + m22)
            
            # Reflectance
            r = (n_ambient * Y - 1.0) / (n_ambient * Y + 1.0)
            R_val = np.abs(r) ** 2
            R += R_val
            
        return R / len(wavelengths) * 100.0 # Return in percent

    def optimize_sequence(indices):
        n_layers = len(indices)
        # Initial guess: quarter wave at 600nm
        lam_center = 600.0
        x0 = []
        for n in indices:
            x0.append(lam_center / (4.0 * n))
        
        def objective(x):
            return mean_reflectance_with_indices(x, indices)
            
        # Use L-BFGS-B for bounds
        bounds = [(0, max_layer_nm)] * n_layers
        
        # Check time
        if time.time() - start_time > time_limit:
            return None, 100.0
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10})
            return res.x, res.fun
        except Exception as e:
            return None, 100.0

    # We need a helper that uses the outer scope's mean_reflectance logic but with specific indices
    def mean_reflectance_with_indices(thicknesses, indices):
        for t in thicknesses:
            if t < 0 or t > max_layer_nm:
                return 100.0
        if sum(thicknesses) > max_total_nm:
            return 100.0
            
        R = 0.0
        for lam in wavelengths:
            m11 = 1.0 + 0.0j
            m12 = 0.0 + 0.0j
            m21 = 0.0 + 0.0j
            m22 = 1.0 + 0.0j
            
            for i, t in enumerate(thicknesses):
                n_layer = indices[i]
                delta = 2.0 * np.pi * n_layer * t / lam
                cos_d = np.cos(delta)
                sin_d = np.sin(delta)
                
                m11_l = cos_d
                m12_l = 1.0j * sin_d / n_layer
                m21_l = 1.0j * n_layer * sin_d
                m22_l = cos_d
                
                new_m11 = m11 * m11_l + m12 * m21_l
                new_m12 = m11 * m12_l + m12 * m22_l
                new_m21 = m21 * m11_l + m22 * m21_l
                new_m22 = m21 * m12_l + m22 * m22_l
                
                m11, m12, m21, m22 = new_m11, new_m12, new_m21, new_m22
            
            Y = (m11 * n_substrate + m12) / (m21 * n_substrate + m22)
            r = (n_ambient * Y - 1.0) / (n_ambient * Y + 1.0)
            R_val = np.abs(r) ** 2
            R += R_val
            
        return R / len(wavelengths) * 100.0

    best_score = 100.0
    best_thick = None
    best_labels = None
    
    # Candidate sequences
    # Start with L (low index) to match air better
    sequences = [
        ["L", "H", "L"],
        ["L", "H", "L", "H", "L"],
        ["L", "H", "L", "H", "L", "H", "L"],
        ["L", "H", "L", "H", "L", "H", "L", "H", "L"],
        # Also try starting with H? Less likely but possible
        ["H", "L", "H"],
        ["H", "L", "H", "L", "H"],
    ]
    
    for seq in sequences:
        if time.time() - start_time > time_limit:
            break
            
        indices = []
        for label in seq:
            if label == "L":
                indices.append(n_L)
            else:
                indices.append(n_H)
                
        x_opt, score = optimize_sequence(indices)
        
        if x_opt is not None and score < best_score:
            # Verify constraints one last time
            if all(0 <= t <= max_layer_nm for t in x_opt) and sum(x_opt) <= max_total_nm:
                best_score = score
                best_thick = list(x_opt)
                best_labels = list(seq)
                
    # If no optimization succeeded, return the seed or a simple guess
    if best_thick is None:
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        return ["L"], [centre / (4.0 * n_L)]
        
    return best_labels, best_thick

