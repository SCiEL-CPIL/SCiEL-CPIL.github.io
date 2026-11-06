"""Antireflection coating design via characteristic-matrix optimization."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _char_matrix(n, d, lam, n0):
    """Compute the 2x2 characteristic matrix for a single layer."""
    beta = 2.0 * np.pi * n * d / lam
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    # For normal incidence, Y = n (admittance)
    # M = [[cos(b), i*sin(b)/Y], [i*Y*sin(b), cos(b)]]
    # We work with real/imag parts or use complex directly
    Y = n  # normal incidence admittance
    M = np.array([
        [cos_b, 1j * sin_b / Y],
        [1j * Y * sin_b, cos_b]
    ], dtype=complex)
    return M

def _reflectance_stack(materials, thicknesses, lambdas, n0, ns):
    """
    Compute reflectance for a stack of layers.
    materials: list of 'L' or 'H'
    thicknesses: list of nm
    lambdas: array of wavelengths
    n0: ambient index
    ns: substrate index
    Returns array of reflectance (0 to 1).
    """
    nL = materials['L']
    nH = materials['H']
    n_layers = len(materials)
    
    refl = np.zeros(len(lambdas))
    
    for i, lam in enumerate(lambdas):
        # Substrate admittance
        Ys = ns
        # Start with substrate matrix (identity for admittance propagation)
        # Actually, we build the total matrix from substrate up
        # Total matrix M_total = M1 * M2 * ... * Mn
        # Then Y_in = Ys * M22 + Ys * M12 * Ys ... 
        # Let's use the standard formulation:
        # [B, C]^T = M_total * [1, Ys]^T  where M_total is product of layer matrices
        # Y_in = B/C
        # R = |(Y_in - n0)/(Y_in + n0)|^2
        
        # Build total matrix
        M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_layer = nL if materials[j] == 'L' else nH
            d_layer = thicknesses[j]
            M_layer = _char_matrix(n_layer, d_layer, lam, n0)
            M_total = M_total @ M_layer
        
        # Propagate through substrate
        # The field vector at the air interface is:
        # [E, H]^T = M_total * [E_s, H_s]^T
        # At substrate: H_s = Y_s * E_s (for forward wave)
        # So [E_s, H_s]^T = [1, Y_s]^T * E_s
        # Thus [E, H]^T = M_total * [1, Y_s]^T
        vec = M_total @ np.array([1.0, Ys], dtype=complex)
        B = vec[0]
        C = vec[1]
        Y_in = B / C
        refl[i] = np.abs((Y_in - n0) / (Y_in + n0)) ** 2
    
    return refl

def _objective(params, problem, n_eval_wavelengths=50):
    """
    Objective function for optimization.
    params: [n_layers, t1, t2, ..., t20, m1, m2, ..., m20]
    where m_i are relaxed binary (0 for L, 1 for H)
    """
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    n0 = problem['ambient_index']
    ns = problem['substrate_index']
    band = problem['band_nm']
    
    # Extract n_layers as integer (rounded)
    n_layers = int(np.round(params[0]))
    n_layers = np.clip(n_layers, 1, max_layers)
    
    # Extract thicknesses and materials
    thicknesses = params[1:1+max_layers]
    mat_rels = params[1+max_layers:1+2*max_layers]
    
    # Only use first n_layers
    thicknesses = thicknesses[:n_layers]
    mat_rels = mat_rels[:n_layers]
    
    # Convert relaxed materials to binary
    # m < 0.5 -> L, m >= 0.5 -> H
    materials = ['L' if m < 0.5 else 'H' for m in mat_rels]
    
    # Check constraints
    if np.sum(thicknesses) > max_total_nm + 1e-6:
        return 1e10
    if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
        return 1e10
    
    # Clip thicknesses to valid range
    thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
    
    # Evaluate reflectance at design wavelengths
    lambdas = np.linspace(band[0], band[1], n_eval_wavelengths)
    refl = _reflectance_stack(materials, thicknesses, lambdas, n0, ns)
    
    # Mean reflectance in percent
    return np.mean(refl) * 100.0

def _evaluate_full(materials, thicknesses, problem):
    """Evaluate with full 401 wavelength grid."""
    n0 = problem['ambient_index']
    ns = problem['substrate_index']
    band = problem['band_nm']
    lambdas = np.linspace(band[0], band[1], 401)
    refl = _reflectance_stack(materials, thicknesses, lambdas, n0, ns)
    return np.mean(refl) * 100.0

def design(problem):
    """
    Design an antireflection coating stack.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    n0 = problem['ambient_index']
    ns = problem['substrate_index']
    band = problem['band_nm']
    nL = problem['materials']['L']
    nH = problem['materials']['H']
    
    # Best solution found so far
    best_score = 1e10
    best_materials = ['L']
    best_thicknesses = [band[0] / (4.0 * nL)]
    
    # Try several initial guesses and local optimizations
    # Strategy: try different layer counts and initial patterns
    
    n_eval_wavelengths = 30  # Use fewer wavelengths for speed during optimization
    
    # Initial guesses: (n_layers, initial_thicknesses, initial_materials)
    initial_guesses = []
    
    # 1. Quarter-wave stack (HLHL... or LHLH...)
    for n_layers in [2, 4, 6, 8, 10, 12]:
        if n_layers > max_layers:
            continue
        t_q = (band[0] + band[1]) / 8.0  # approximate quarter wave
        # Alternate H and L starting with H
        mats = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        thicks = [t_q / nH if m == 'H' else t_q / nL for m in mats]
        # Scale to fit within max_total_nm
        total = sum(thicks)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicks = [t * scale for t in thicks]
        initial_guesses.append((n_layers, thicks, mats))
        
        # Alternate starting with L
        mats2 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        thicks2 = [t_q / nL if m == 'L' else t_q / nH for m in mats2]
        total2 = sum(thicks2)
        if total2 > max_total_nm:
            scale2 = max_total_nm / total2
            thicks2 = [t * scale2 for t in thicks2]
        initial_guesses.append((n_layers, thicks2, mats2))
    
    # 2. Random initial guesses
    np.random.seed(42)
    for _ in range(10):
        n_layers = np.random.randint(2, max_layers + 1)
        mats = [np.random.choice(['L', 'H']) for _ in range(n_layers)]
        thicks = np.random.uniform(0, max_layer_nm, n_layers)
        # Ensure total doesn't exceed max
        total = sum(thicks)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicks = thicks * scale
        initial_guesses.append((n_layers, list(thicks), mats))
    
    # 3. Thin-layer design (many thin layers)
    for n_layers in [15, 20]:
        if n_layers > max_layers:
            continue
        mats = [np.random.choice(['L', 'H']) for _ in range(n_layers)]
        thicks = np.random.uniform(5, 50, n_layers)
        total = sum(thicks)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicks = thicks * scale
        initial_guesses.append((n_layers, list(thicks), mats))
    
    # Optimize each initial guess
    for idx, (n_layers, thicks, mats) in enumerate(initial_guesses):
        if time.time() > deadline - 10.0:
            break
        
        # Build parameter vector
        params = np.zeros(1 + 2 * max_layers)
        params[0] = float(n_layers)
        for i in range(n_layers):
            params[1 + i] = thicks[i]
            params[1 + max_layers + i] = 0.0 if mats[i] == 'L' else 1.0
        
        # Bounds
        bounds = [(0.5, float(max_layers))]  # n_layers
        for i in range(max_layers):
            bounds.append((0.0, max_layer_nm))  # thickness
        for i in range(max_layers):
            bounds.append((0.0, 1.0))  # material relaxation
        
        # Optimize
        try:
            result = minimize(
                _objective,
                params,
                args=(problem, n_eval_wavelengths),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < best_score:
                opt_params = result.x
                opt_n_layers = int(np.round(opt_params[0]))
                opt_n_layers = np.clip(opt_n_layers, 1, max_layers)
                opt_thicks = np.clip(opt_params[1:1+opt_n_layers], 0.0, max_layer_nm)
                opt_mats_rel = opt_params[1+max_layers:1+max_layers+opt_n_layers]
                opt_mats = ['L' if m < 0.5 else 'H' for m in opt_mats_rel]
                
                # Check total thickness
                if np.sum(opt_thicks) <= max_total_nm + 1e-6:
                    score = _evaluate_full(opt_mats, list(opt_thicks), problem)
                    if score < best_score:
                        best_score = score
                        best_materials = opt_mats
                        best_thicknesses = list(opt_thicks)
                        
        except Exception:
            pass
    
    # Also try a simple gradient-free approach with differential evolution on a few cases
    if time.time() < deadline - 20.0:
        try:
            from scipy.optimize import differential_evolution
            
            def _de_objective(params, problem):
                n_layers = int(np.round(params[0]))
                n_layers = np.clip(n_layers, 2, max_layers)
                thicknesses = np.clip(params[1:1+n_layers], 0.0, max_layer_nm)
                mat_rels = params[1+max_layers:1+max_layers+n_layers]
                materials = ['L' if m < 0.5 else 'H' for m in mat_rels]
                
                if np.sum(thicknesses) > max_total_nm + 1e-6:
                    return 1e10
                
                lambdas = np.linspace(band[0], band[1], 20)
                refl = _reflectance_stack(materials, thicknesses, lambdas, n0, ns)
                return np.mean(refl) * 100.0
            
            bounds_de = [(2.0, float(max_layers))]
            for i in range(max_layers):
                bounds_de.append((0.0, max_layer_nm))
            for i in range(max_layers):
                bounds_de.append((0.0, 1.0))
            
            x0 = np.random.uniform([b[0] for b in bounds_de], [b[1] for b in bounds_de])
            
            result_de = differential_evolution(
                _de_objective,
                bounds_de,
                args=(problem,),
                seed=42,
                maxiter=50,
                tol=1e-4,
                popsize=10,
                polish=True
            )
            
            if result_de.fun < best_score:
                opt_params = result_de.x
                opt_n_layers = int(np.round(opt_params[0]))
                opt_n_layers = np.clip(opt_n_layers, 1, max_layers)
                opt_thicks = np.clip(opt_params[1:1+opt_n_layers], 0.0, max_layer_nm)
                opt_mats_rel = opt_params[1+max_layers:1+max_layers+opt_n_layers]
                opt_mats = ['L' if m < 0.5 else 'H' for m in opt_mats_rel]
                
                if np.sum(opt_thicks) <= max_total_nm + 1e-6:
                    score = _evaluate_full(opt_mats, list(opt_thicks), problem)
                    if score < best_score:
                        best_score = score
                        best_materials = opt_mats
                        best_thicknesses = list(opt_thicks)
                        
        except Exception:
            pass
    
    # Final refinement with more wavelengths if time permits
    if time.time() < deadline - 5.0 and best_materials:
        try:
            n_layers = len(best_materials)
            params = np.zeros(1 + 2 * max_layers)
            params[0] = float(n_layers)
            for i in range(n_layers):
                params[1 + i] = best_thicknesses[i]
                params[1 + max_layers + i] = 0.0 if best_materials[i] == 'L' else 1.0
            
            bounds = [(float(n_layers), float(n_layers))]  # Fix n_layers
            for i in range(max_layers):
                bounds.append((0.0, max_layer_nm))
            for i in range(max_layers):
                bounds.append((0.0, 1.0))
            
            result = minimize(
                _objective,
                params,
                args=(problem, 60),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                opt_params = result.x
                opt_n_layers = int(np.round(opt_params[0]))
                opt_n_layers = np.clip(opt_n_layers, 1, max_layers)
                opt_thicks = np.clip(opt_params[1:1+opt_n_layers], 0.0, max_layer_nm)
                opt_mats_rel = opt_params[1+max_layers:1+max_layers+opt_n_layers]
                opt_mats = ['L' if m < 0.5 else 'H' for m in opt_mats_rel]
                
                if np.sum(opt_thicks) <= max_total_nm + 1e-6:
                    score = _evaluate_full(opt_mats, list(opt_thicks), problem)
                    if score < best_score:
                        best_score = score
                        best_materials = opt_mats
                        best_thicknesses = list(opt_thicks)
                        
        except Exception:
            pass
    
    # Ensure valid output
    # Trim trailing zero-thickness layers
    while best_materials and best_thicknesses and best_thicknesses[-1] < 1e-6:
        best_materials.pop()
        best_thicknesses.pop()
    
    if not best_materials:
        best_materials = ['L']
        best_thicknesses = [band[0] / (4.0 * nL)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
