"""
Broadband AR coating design using continuous admittance profile optimization via L-BFGS + quantization.
Strategy:
1. Represent the coating as a continuous optical admittance profile using a smooth basis (cosine series).
2. Optimize the continuous shape via L-BFGS-B to minimize mean reflectance.
3. Quantize the optimal continuous profile into discrete 'L'/'H' layers by mapping local admittance to nearest material.
4. Refine the discrete layers using coordinate descent.
"""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _reflectance_calc(mats, thicks, n0, ns, materials, lambdas):
    """
    Fast vectorized reflectance calculation.
    mats: list of 'L'/'H'
    thicks: numpy array of thicknesses
    """
    N = len(thicks)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(lambdas, np.abs(r)**2)
        
    M11 = np.ones(len(lambdas), dtype=complex)
    M12 = np.zeros(len(lambdas), dtype=complex)
    M21 = np.zeros(len(lambdas), dtype=complex)
    M22 = np.ones(len(lambdas), dtype=complex)
    
    for i in range(N):
        n_i = materials[mats[i]]
        d_i = thicks[i]
        
        delta = 2.0 * np.pi * n_i * d_i / lambdas
        c = np.cos(delta)
        s = np.sin(delta)
        
        m11 = c
        m12 = 1j * s / n_i
        m21 = 1j * n_i * s
        m22 = c
        
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    Y_s = ns
    Y_in = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    r = (Y_in - n0) / (Y_in + n0)
    R = np.abs(r)**2
    return R

def _continuous_reflectance(params, n0, ns, materials, lambdas, num_basis):
    """
    Compute reflectance for a continuous admittance profile represented by cosine series coefficients.
    params: array of length 2*num_basis, first num_basis are even coefficients, next num_basis are odd.
    The profile is: Y(z) = sum_{k=0}^{K-1} a_k * cos(k*pi*z) + sum_{k=0}^{K-1} b_k * sin(k*pi*z)
    where z in [0, 1] is the normalized position from air to substrate.
    """
    K = num_basis
    a = params[:K]
    b = params[K:]
    
    # Discretize the profile into many thin layers
    num_slices = 200
    z = np.linspace(0, 1, num_slices + 1)
    dz = 1.0 / num_slices
    
    # Compute admittance at each slice boundary
    Y = np.zeros(num_slices + 1)
    for i in range(num_slices + 1):
        Y[i] = 0.0
        for k in range(K):
            Y[i] += a[k] * np.cos(k * np.pi * z[i])
        for k in range(K):
            Y[i] += b[k] * np.sin(k * np.pi * z[i])
    
    # Ensure Y is positive (admittance must be positive)
    Y = np.maximum(Y, 0.1)
    
    # Use the average admittance in each slice
    Y_slice = 0.5 * (Y[:-1] + Y[1:])
    
    # Map each slice's admittance to either L or H material
    # For the continuous calculation, we use the actual admittance value
    # but we need to compute reflectance. Let's use the actual Y values.
    
    # Compute characteristic matrix for each slice
    M11 = np.ones(len(lambdas), dtype=complex)
    M12 = np.zeros(len(lambdas), dtype=complex)
    M21 = np.zeros(len(lambdas), dtype=complex)
    M22 = np.ones(len(lambdas), dtype=complex)
    
    # Physical thickness of each slice
    # Total coating thickness is not fixed in continuous formulation
    # We need to map the normalized z to physical thickness
    # Let's assume the total coating thickness is T, and we optimize T as well
    # Actually, let's fix the total thickness and optimize the shape
    # For simplicity, let's use a fixed total thickness of 500 nm
    T_total = 500.0
    d_slice = T_total / num_slices
    
    for i in range(num_slices):
        n_i = Y_slice[i]  # Use admittance directly as refractive index (approximation)
        # Actually, admittance Y = n for non-magnetic materials, so n = Y
        d_i = d_slice
        
        delta = 2.0 * np.pi * n_i * d_i / lambdas
        c = np.cos(delta)
        s = np.sin(delta)
        
        m11 = c
        m12 = 1j * s / n_i
        m21 = 1j * n_i * s
        m22 = c
        
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    Y_s = ns
    Y_in = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    r = (Y_in - n0) / (Y_in + n0)
    R = np.abs(r)**2
    return np.mean(R) * 100.0

def _quantize_profile(Y_profile, n_L, n_H, max_layers, max_total_nm, max_layer_nm):
    """
    Quantize a continuous admittance profile into discrete L/H layers.
    Y_profile: array of admittance values
    Returns: (mats, thicks)
    """
    n0 = 1.0
    # Map each point to nearest material
    # Threshold: geometric mean of n_L and n_H
    threshold = np.sqrt(n_L * n_H)
    
    # Determine material for each point
    mat_array = np.where(Y_profile >= threshold, 'H', 'L')
    
    # Merge consecutive same-material points
    mats = []
    thicks = []
    current_mat = mat_array[0]
    current_count = 1
    
    for i in range(1, len(mat_array)):
        if mat_array[i] == current_mat:
            current_count += 1
        else:
            mats.append(current_mat)
            thicks.append(current_count)
            current_mat = mat_array[i]
            current_count = 1
    mats.append(current_mat)
    thicks.append(current_count)
    
    # Convert counts to physical thicknesses
    # Assume total thickness is 500 nm
    T_total = 500.0
    total_count = sum(thicks)
    if total_count == 0:
        return [], []
    thicks = np.array(thicks, dtype=float) * (T_total / total_count)
    
    # Clip to max_layer_nm
    thicks = np.clip(thicks, 0, max_layer_nm)
    
    # If total exceeds max_total_nm, scale down
    if np.sum(thicks) > max_total_nm:
        thicks *= (max_total_nm / np.sum(thicks))
    
    # If too many layers, merge smallest adjacent layers
    while len(mats) > max_layers:
        # Find the pair of adjacent layers with smallest combined thickness
        min_idx = 0
        min_val = np.inf
        for i in range(len(mats) - 1):
            combined = thicks[i] + thicks[i+1]
            if combined < min_val:
                min_val = combined
                min_idx = i
        # Merge
        mats[min_idx] = mats[min_idx]  # Keep first material
        mats = mats[:min_idx] + mats[min_idx+1:]
        thicks = np.concatenate([thicks[:min_idx], thicks[min_idx+1:]])
        thicks[min_idx] = thicks[min_idx] + thicks[min_idx+1] if min_idx < len(thicks) else thicks[-1]
        # Actually, need to be careful with indices
        # Let's redo this more carefully
        break  # Simplified for now
    
    # Re-merge properly
    while len(mats) > max_layers:
        min_idx = 0
        min_val = np.inf
        for i in range(len(mats) - 1):
            combined = thicks[i] + thicks[i+1]
            if combined < min_val:
                min_val = combined
                min_idx = i
        mats = mats[:min_idx] + mats[min_idx+1:]
        thicks = np.delete(thicks, min_idx+1)
        thicks[min_idx] = min_val
    
    # Final check
    if len(mats) > max_layers:
        mats = mats[:max_layers]
        thicks = thicks[:max_layers]
    
    if np.sum(thicks) > max_total_nm:
        thicks *= (max_total_nm / np.sum(thicks))
    
    return mats, list(thicks)

def design(problem):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    band = problem["band_nm"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[800-400+1-1] if False else band[1], 401)
    
    n_L = materials['L']
    n_H = materials['H']
    
    def evaluate(mats, thicks):
        if len(mats) == 0:
            return np.mean(_reflectance_calc(mats, np.array([]), n0, ns, materials, lambdas)) * 100.0
        return np.mean(_reflectance_calc(mats, np.array(thicks), n0, ns, materials, lambdas)) * 100.0
    
    best_score = np.inf
    best_mats = []
    best_thicks = np.array([])
    
    # Step 1: Optimize continuous profile
    num_basis = 8
    # Initial guess: linear taper from n0 to ns
    # Y(z) = n0 + (ns - n0) * z
    # In cosine series: a_0 = (n0 + ns)/2, b_0 = (ns - n0) * (2/pi) * ... 
    # Let's just start with a_0 = sqrt(n0*ns), others zero
    params0 = np.zeros(2 * num_basis)
    params0[0] = np.sqrt(n0 * ns)  # Geometric mean
    
    # Bounds: keep admittance in reasonable range [0.5, 3.0]
    bounds = [(0.5, 3.0)] * (2 * num_basis)
    
    try:
        if time.time() < deadline - 30.0:
            result = minimize(
                _continuous_reflectance,
                params0,
                args=(n0, ns, materials, lambdas, num_basis),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'maxfun': 1000}
            )
            if result.success or result.fun < best_score:
                best_score = result.fun
                params_opt = result.x
            else:
                params_opt = params0
        else:
            params_opt = params0
    except Exception as e:
        params_opt = params0
    
    # Step 2: Quantize the optimal profile
    K = num_basis
    a = params_opt[:K]
    b = params_opt[K:]
    
    num_slices = 100
    z = np.linspace(0, 1, num_slices)
    Y_profile = np.zeros(num_slices)
    for i in range(num_slices):
        Y_profile[i] = 0.0
        for k in range(K):
            Y_profile[i] += a[k] * np.cos(k * np.pi * z[i])
        for k in range(K):
            Y_profile[i] += b[k] * np.sin(k * np.pi * z[i])
    Y_profile = np.maximum(Y_profile, 0.1)
    
    mats, thicks = _quantize_profile(Y_profile, n_L, n_H, max_layers, max_total_nm, max_layer_nm)
    
    if len(mats) > 0:
        score = evaluate(mats, np.array(thicks))
        if score < best_score:
            best_score = score
            best_mats = mats
            best_thicks = np.array(thicks)
    
    # Step 3: Coordinate descent refinement
    if len(best_mats) > 0 and time.time() < deadline - 10.0:
        current_mats = best_mats.copy()
        current_thicks = best_thicks.copy()
        current_score = best_score
        
        step_sizes = [20, 10, 5, 2, 1, 0.5, 0.2, 0.1]
        
        for step in step_sizes:
            if time.time() > deadline - 5.0:
                break
            
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                N = len(current_thicks)
                for i in range(N):
                    for delta in [-step, step]:
                        trial_thicks = current_thicks.copy()
                        trial_thicks[i] = np.clip(current_thicks[i] + delta, 0, max_layer_nm)
                        
                        if np.sum(trial_thicks) > max_total_nm:
                            continue
                        
                        trial_score = evaluate(current_mats, trial_thicks)
                        if trial_score < current_score - 1e-9:
                            current_thicks = trial_thicks
                            current_score = trial_score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_mats = current_mats.copy()
                                best_thicks = current_thicks.copy()
                            break
                
                # Material flips
                for i in range(len(current_mats)):
                    trial_mats = current_mats.copy()
                    trial_mats[i] = 'H' if trial_mats[i] == 'L' else 'L'
                    trial_score = evaluate(trial_mats, current_thicks)
                    if trial_score < current_score - 1e-9:
                        current_mats = trial_mats
                        current_score = trial_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_mats = current_mats.copy()
                            best_thicks = current_thicks.copy()
    
    return best_mats, list(best_thicks)

# EVOLVE-BLOCK-END
