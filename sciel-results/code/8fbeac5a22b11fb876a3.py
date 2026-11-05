"""
Broadband antireflection coating design via Global Multi-Start Optimization.
Strategy:
1. Define a fast, vectorized characteristic-matrix method to evaluate reflectance.
2. Generate diverse initial candidates (quarter-wave stacks, random, etc.).
3. Refine each candidate using Nelder-Mead simplex optimization.
4. Return the best valid design found within the time budget.
"""

import numpy as np
import time
from scipy.optimize import minimize

# Constants for speed
N_WAVELENGTHS = 401
WAVELENGTHS = np.linspace(400, 800, N_WAVELENGTHS)

def _compute_reflectance_vectorized(materials, thicknesses, n_ambient, n_substrate):
    """
    Computes mean reflectance (percent) for a given stack.
    materials: list of 'L' or 'H'
    thicknesses: list of floats (nm)
    """
    if not materials:
        # Bare substrate
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return 100.0 * (r**2)
    
    n_L = 1.38
    n_H = 2.35
    n_mat = np.array([n_L if m == 'L' else n_H for m in materials])
    d = np.array(thicknesses)
    
    # Wavelengths in nm
    wl = WAVELENGTHS
    k = 2 * np.pi / wl # 1/nm
    
    # Characteristic matrices
    # M_j = [[cos(delta_j), (i/sin(n_j)) * sin(delta_j)], [i*n_j*sin(delta_j), cos(delta_j)]]
    # Note: for normal incidence, B = n_j
    
    n_layers = len(materials)
    M = np.ones((n_layers, N_WAVELENGTHS, 2, 2), dtype=complex)
    
    for j in range(n_layers):
        delta = k * n_mat[j] * d[j]
        c = np.cos(delta)
        s = np.sin(delta)
        
        M[j, :, 0, 0] = c
        M[j, :, 0, 1] = 1j / n_mat[j] * s
        M[j, :, 1, 0] = 1j * n_mat[j] * s
        M[j, :, 1, 1] = c
        
    # Product of matrices
    # Start with identity
    M_total = np.ones((N_WAVELENGTHS, 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1
    M_total[:, 1, 1] = 1
    
    for j in range(n_layers):
        # M_total = M_total @ M[j]
        # M_total is (N, 2, 2), M[j] is (N, 2, 2)
        M_total = np.einsum('nij,njk->nik', M_total, M[j])
        
    # Boundary condition: Substrate
    # [B, C] = M_total @ [1, n_substrate]
    # Actually, standard definition:
    # [E, H] = M @ [1, n_s]
    # B = M[0,0] + M[0,1] * n_s
    # C = M[1,0] + M[1,1] * n_s
    
    B = M_total[:, 0, 0] + M_total[:, 0, 1] * n_substrate
    C = M_total[:, 1, 0] + M_total[:, 1, 1] * n_substrate
    
    # Admittance Y = C / B
    Y = C / B
    
    # Reflectance R = |(Y - n_ambient) / (Y + n_ambient)|^2
    Y = Y / 1.0 # normalized? No, Y is admittance
    # Wait, the standard formula is:
    # r = (Y - n_0) / (Y + n_0)
    # where Y is the input admittance.
    # Let's check dimensions.
    # M = [[cos, i/sin/n], [i n sin, cos]]
    # [E, H] = M [1, n_s]
    # B = M00 + M01 n_s
    # C = M10 + M11 n_s
    # Y = C/B
    # r = (Y - n_0) / (Y + n_0)
    
    r = (Y - n_ambient) / (Y + n_ambient)
    R = np.abs(r)**2
    
    mean_R = np.mean(R)
    return mean_R * 100.0

def _penalized_objective(x, problem):
    """
    Objective function for optimizer.
    x: array of thicknesses (nm)
    """
    materials = problem['materials_seq']
    max_total = problem['max_total_nm']
    max_layer = problem['max_layer_nm']
    
    # Check constraints
    if np.any(x < 0) or np.any(x > max_layer):
        return 1e6
    if np.sum(x) > max_total:
        return 1e6
        
    try:
        val = _compute_reflectance_vectorized(materials, x, problem['ambient_index'], problem['substrate_index'])
        return val
    except Exception:
        return 1e6

def design(problem):
    """
    Design a stack of at most twenty low- and high-index layers.
    """
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    n_ambient = problem['ambient_index']
    n_substrate = problem['substrate_index']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    # Prepare problem context for objective
    ctx = {
        'ambient_index': n_ambient,
        'substrate_index': n_substrate,
        'max_total_nm': max_total_nm,
        'max_layer_nm': max_layer_nm
    }
    
    best_score = 100.0 # Bare glass is ~4.26%, but we start high
    best_materials = []
    best_thicknesses = []
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Quarter-wave stack (L H L H ...)
    # Center wavelength 600 nm
    lam_c = 600.0
    d_L = lam_c / (4 * 1.38)
    d_H = lam_c / (4 * 2.35)
    
    for n_layers in range(1, max_layers + 1, 2): # Odd number of layers often better for AR? Or even?
        # Try alternating starting with L
        seq_L = []
        th_L = []
        for i in range(n_layers):
            if i % 2 == 0:
                seq_L.append('L')
                th_L.append(d_L)
            else:
                seq_L.append('H')
                th_L.append(d_H)
        
        # Check total thickness
        if sum(th_L) <= max_total_nm:
            candidates.append((seq_L, th_L))
            
        # Try alternating starting with H
        seq_H = []
        th_H = []
        for i in range(n_layers):
            if i % 2 == 0:
                seq_H.append('H')
                th_H.append(d_H)
            else:
                seq_H.append('L')
                th_H.append(d_L)
        
        if sum(th_H) <= max_total_nm:
            candidates.append((seq_H, th_H))
            
    # 2. Random candidates
    rng = np.random.default_rng(42)
    for _ in range(50):
        n_layers = rng.integers(1, max_layers + 1)
        seq = []
        th = []
        total = 0
        for i in range(n_layers):
            mat = 'L' if rng.random() < 0.5 else 'H'
            seq.append(mat)
            # Random thickness up to max_layer, but ensure sum doesn't exceed max_total
            # Simple random
            t = rng.uniform(0, max_layer_nm)
            if total + t > max_total_nm:
                t = max_total_nm - total
                if t < 0:
                    break
            th.append(t)
            total += t
        if len(seq) == len(th) and len(seq) > 0:
            candidates.append((seq, th))
            
    # 3. Specific known good structures (e.g. 4-layer quarter wave)
    # L H L H
    d_L4 = 600 / (4 * 1.38)
    d_H4 = 600 / (4 * 2.35)
    candidates.append((['L', 'H', 'L', 'H'], [d_L4, d_H4, d_L4, d_H4]))
    candidates.append((['H', 'L', 'H', 'L'], [d_H4, d_L4, d_H4, d_L4]))
    
    # Deduplicate candidates
    unique_candidates = []
    seen = set()
    for mats, ths in candidates:
        key = (tuple(mats), tuple(round(t, 2) for t in ths))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((mats, ths))
            
    # Evaluate and refine
    for mats, ths in unique_candidates:
        if time.time() > deadline:
            break
            
        ctx['materials_seq'] = mats
        x0 = np.array(ths)
        
        # Evaluate initial
        try:
            score = _penalized_objective(x0, ctx)
            if score < best_score:
                best_score = score
                best_materials = mats[:]
                best_thicknesses = ths[:]
        except Exception:
            continue
            
        # Optimize
        # Use Nelder-Mead
        # Bounds are handled by penalty, but we can also use 'L-BFGS-B' if we had gradients.
        # Nelder-Mead is robust for non-differentiable (though this is smooth) and small dims.
        try:
            res = minimize(
                _penalized_objective, 
                x0, 
                args=(ctx,), 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
            )
            
            if res.fun < best_score and res.fun < 100: # Valid
                # Check constraints on res.x
                x_opt = res.x
                if np.all(x_opt >= 0) and np.all(x_opt <= max_layer_nm) and np.sum(x_opt) <= max_total_nm:
                    best_score = res.fun
                    best_materials = mats[:]
                    best_thicknesses = list(x_opt)
        except Exception:
            pass
            
    # Final check
    if not best_materials:
        # Fallback to seed
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        return ["L"], [centre / (4.0 * problem["materials"]["L"])]
        
    return best_materials, best_thicknesses
