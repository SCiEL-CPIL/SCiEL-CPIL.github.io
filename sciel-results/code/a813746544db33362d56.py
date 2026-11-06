import numpy as np
import time
from scipy.optimize import minimize

def _calculate_reflectance(materials, thicknesses, wavelengths, n0, ns, nL, nH):
    """
    Calculate mean reflectance percentage for a given stack.
    materials: list of 'L' or 'H'
    thicknesses: list of floats (nm)
    wavelengths: array of floats (nm)
    """
    N = len(materials)
    if N == 0:
        # Bare substrate
        r = (n0 - ns) / (n0 + ns)
        R = np.abs(r)**2
        return np.mean(R) * 100.0

    # Pre-compute indices
    indices = np.array([nL if m == 'L' else nH for m in materials])
    thicknesses = np.array(thicknesses)
    
    # Vectorize over wavelengths
    # k = 2 * pi / lambda
    k = 2 * np.pi / wavelengths
    
    # delta = k * n * d
    # shape: (num_wavelengths, num_layers)
    delta = k[:, np.newaxis] * indices[np.newaxis, :] * thicknesses[np.newaxis, :]
    
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    
    # Characteristic Matrix for each layer
    # M_i = [[cos(d), i sin(d)/n], [i n sin(d), cos(d)]]
    # We need to multiply all matrices.
    # M_total = M_1 @ M_2 @ ... @ M_N
    
    # Initialize M_total as Identity for each wavelength
    M_total = np.eye(2, dtype=complex)
    M_total = np.broadcast_to(M_total, (len(wavelengths), 2, 2)).copy()
    
    for i in range(N):
        n_i = indices[i]
        # M_i for all wavelengths
        # M_i[0,0] = cos(delta_i)
        # M_i[0,1] = 1j * sin(delta_i) / n_i
        # M_i[1,0] = 1j * n_i * sin(delta_i)
        # M_i[1,1] = cos(delta_i)
        
        M_i = np.zeros((len(wavelengths), 2, 2), dtype=complex)
        M_i[:, 0, 0] = cos_d[:, i]
        M_i[:, 0, 1] = 1j * sin_d[:, i] / n_i
        M_i[:, 1, 0] = 1j * n_i * sin_d[:, i]
        M_i[:, 1, 1] = cos_d[:, i]
        
        # M_total = M_total @ M_i
        M_total = np.matmul(M_total, M_i)
        
    M11 = M_total[:, 0, 0]
    M12 = M_total[:, 0, 1]
    M21 = M_total[:, 1, 0]
    M22 = M_total[:, 1, 1]
    
    # Admittance Y
    # Y = (M11 * Ys + M12) / (M21 * Ys + M22)
    # Ys = ns (since normal incidence, admittance is n)
    Ys = ns
    Y = (M11 * Ys + M12) / (M21 * Ys + M22)
    
    # Reflectance
    # R = | (Y0 - Y) / (Y0 + Y) |^2
    Y0 = n0
    r = (Y0 - Y) / (Y0 + Y)
    R = np.abs(r)**2
    
    return np.mean(R) * 100.0

def design(problem):
    n0 = problem.get("ambient_index", 1.0)
    ns = problem.get("substrate_index", 1.52)
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # 401 equally spaced wavelengths
    wavelengths = np.linspace(band[0], band[1], 401)
    
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    best_score = float('inf')
    best_materials = ["L"]
    best_thicknesses = [100.0]
    
    # Candidate patterns
    # We will try a few fixed material sequences and optimize thicknesses
    # N=10 and N=20
    patterns = []
    
    # Pattern 1: All L
    patterns.append(["L"] * 10)
    patterns.append(["L"] * 20)
    
    # Pattern 2: All H
    patterns.append(["H"] * 10)
    patterns.append(["H"] * 20)
    
    # Pattern 3: Alternating L, H
    patterns.append(["L" if i % 2 == 0 else "H" for i in range(10)])
    patterns.append(["L" if i % 2 == 0 else "H" for i in range(20)])
    
    # Pattern 4: Alternating H, L
    patterns.append(["H" if i % 2 == 0 else "L" for i in range(10)])
    patterns.append(["H" if i % 2 == 0 else "L" for i in range(20)])
    
    # Pattern 5: Block L then H
    patterns.append(["L"] * 5 + ["H"] * 5)
    patterns.append(["L"] * 10 + ["H"] * 10)
    
    # Pattern 6: Block H then L
    patterns.append(["H"] * 5 + ["L"] * 5)
    patterns.append(["H"] * 10 + ["L"] * 10)
    
    # Pattern 7: Gradient L->H (more Ls first)
    patterns.append(["L"] * 8 + ["H"] * 2)
    patterns.append(["L"] * 16 + ["H"] * 4)
    
    # Pattern 8: Gradient H->L
    patterns.append(["H"] * 8 + ["L"] * 2)
    patterns.append(["H"] * 16 + ["L"] * 4)
    
    # Random patterns
    np.random.seed(42)
    for _ in range(5):
        N = np.random.choice([10, 15, 20])
        mat = np.random.choice(["L", "H"], size=N)
        patterns.append(list(mat))
    
    for mat in patterns:
        if time.time() > deadline:
            break
            
        N = len(mat)
        
        # Initial guess: Quarter wave at center wavelength
        center_lambda = 0.5 * (band[0] + band[1])
        init_thicknesses = []
        for m in mat:
            n = nL if m == 'L' else nH
            t = center_lambda / (4.0 * n)
            # Scale down to respect total thickness constraint roughly
            # If N=20, sum t ~ 20 * 100 = 2000. We need <= 1000.
            # So divide by 2.
            t = t * (max_total_nm / (N * center_lambda / (4.0 * 1.8))) # Rough scaling
            # Better: just set to a small value like 20nm to start
            init_thicknesses.append(20.0)
            
        x0 = np.array(init_thicknesses)
        
        def objective(x):
            # Check constraints
            if np.any(x < 0) or np.any(x > max_layer_nm):
                return 100.0 + 1000.0 * np.sum(np.abs(x - np.clip(x, 0, max_layer_nm)))
            total = np.sum(x)
            penalty = 0.0
            if total > max_total_nm:
                penalty = 10.0 * (total - max_total_nm)
            
            r = _calculate_reflectance(mat, x, wavelengths, n0, ns, nL, nH)
            return r + penalty
        
        bounds = [(0, max_layer_nm) for _ in range(N)]
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.success or res.fun < best_score:
                x_opt = res.x
                # Ensure constraints are strictly met for the final result
                if np.sum(x_opt) > max_total_nm:
                    # Scale down
                    x_opt = x_opt * (max_total_nm / np.sum(x_opt))
                
                # Re-evaluate true score
                score = _calculate_reflectance(mat, x_opt, wavelengths, n0, ns, nL, nH)
                
                if score < best_score:
                    best_score = score
                    best_materials = mat
                    best_thicknesses = list(x_opt)
                    
        except Exception as e:
            pass

    # Fallback if no optimization succeeded or all were bad
    if best_score == float('inf'):
        return ["L"], [100.0]
        
    return best_materials, best_thicknesses
