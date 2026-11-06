"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def _char_matrix(n, d, lam):
    """Return the characteristic matrix for a single layer (vectorized over lam)."""
    beta = 2.0 * np.pi * n * d / lam
    c = np.cos(beta)
    s = np.sin(beta)
    # For normal incidence, A = 1, B = n
    # M = [[A*cos(beta), i*B*sin(beta)], [i*(1/B)*sin(beta), A*cos(beta)]]
    M11 = c
    M12 = 1j * n * s
    M21 = 1j * s / n
    M22 = c
    return np.stack([np.stack([M11, M12], axis=-1),
                     np.stack([M21, M22], axis=-1)], axis=-2)

def _reflectance(materials, thicknesses, wavelengths, n_ambient, n_substrate):
    """Compute reflectance (fraction) at normal incidence for given layers."""
    # Build total characteristic matrix
    # M_total = M1 * M2 * ... * M_N
    total = np.zeros((2, 2, len(wavelengths)), dtype=complex)
    total[0, 0] = 1.0
    total[1, 1] = 1.0
    
    for mat, th in zip(materials, thicknesses):
        n = mat
        if th < 1e-6:
            continue
        M = _char_matrix(n, th, wavelengths)
        # total = total @ M (element-wise matrix multiplication over last axis)
        new_total = np.zeros_like(total)
        new_total[0, 0] = total[0, 0] * M[0, 0] + total[0, 1] * M[1, 0]
        new_total[0, 1] = total[0, 0] * M[0, 1] + total[0, 1] * M[1, 1]
        new_total[1, 0] = total[1, 0] * M[0, 0] + total[1, 1] * M[1, 0]
        new_total[1, 1] = total[1, 0] * M[0, 1] + total[1, 1] * M[1, 1]
        total = new_total
    
    # Admittance at the air side
    Y = (total[1, 0] + total[1, 1] * n_substrate) / (total[0, 0] + total[0, 1] * n_substrate)
    
    # Reflectance: R = |(Y - n_ambient) / (Y + n_ambient)|^2
    r = (Y - n_ambient) / (Y + n_ambient)
    R = np.abs(r)**2
    return R

def design(problem):
    """Design an antireflection coating using Differential Evolution."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    lam0, lam1 = band
    
    wavelengths = np.linspace(lam0, lam1, 401)
    
    best_result = None
    best_score = float('inf')
    
    def objective(x):
        """
        x: [m1, m2, ..., mN, t1, t2, ..., tN]
        mi in [0, 1] -> material index
        ti in [0, max_layer_nm] -> thickness
        """
        N = len(x) // 2
        m_params = x[:N]
        t_params = x[N:]
        
        # Map material params to indices
        # < 0.5 -> L, >= 0.5 -> H
        indices = np.where(m_params < 0.5, n_L, n_H)
        
        # Clamp thicknesses to [0, max_layer_nm]
        t_clamped = np.clip(t_params, 0, max_layer_nm)
        
        # Check total thickness constraint
        total_th = np.sum(t_clamped)
        if total_th > max_total_nm:
            # Penalty
            return 100.0 + (total_th - max_total_nm)
        
        # If all thicknesses are zero, return high value
        if np.all(t_clamped < 1e-6):
            return 100.0
            
        # Compute reflectance
        try:
            R = _reflectance(indices, t_clamped, wavelengths, n_ambient, n_substrate)
            mean_R = np.mean(R) * 100.0
        except Exception:
            return 100.0
            
        return mean_R

    # Try different numbers of layers
    # We prioritize lower layer counts for speed, but higher counts for performance.
    # 6 and 8 are good trade-offs.
    for N_layers in [6, 8, 10]:
        if time.time() > deadline - 30:  # Leave 30s for safety and finalization
            break
            
        bounds = [(0, 1)] * N_layers + [(0, max_layer_nm)] * N_layers
        
        try:
            # Use differential evolution
            # maxiter and popsize control the search depth
            # 100 iterations * 20 popsize = 2000 evals.
            # Each eval is fast.
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                seed=42,
                tol=1e-5,
                init='sobol'
            )
            
            if res.fun < best_score:
                x_opt = res.x
                m_params = x_opt[:N_layers]
                t_params = x_opt[N_layers:]
                
                indices = np.where(m_params < 0.5, n_L, n_H)
                t_clamped = np.clip(t_params, 0, max_layer_nm)
                
                # Filter out zero-thickness layers
                valid_indices = []
                valid_thicknesses = []
                for i in range(N_layers):
                    if t_clamped[i] > 1e-6:
                        # Determine label
                        if indices[i] == n_L:
                            valid_indices.append('L')
                        else:
                            valid_indices.append('H')
                        valid_thicknesses.append(float(t_clamped[i]))
                
                if len(valid_indices) > 0:
                    # Final check on constraints
                    if sum(valid_thicknesses) <= max_total_nm and all(0 <= t <= max_layer_nm for t in valid_thicknesses):
                        score = objective(x_opt)
                        if score < best_score:
                            best_score = score
                            best_result = (valid_indices, valid_thicknesses)
                            
        except Exception as e:
            pass

    if best_result is None:
        # Fallback: single quarter-wave layer
        centre = 0.5 * (lam0 + lam1)
        return ["L"], [centre / (4.0 * n_L)]
        
    return best_result

# EVOLVE-BLOCK-END
