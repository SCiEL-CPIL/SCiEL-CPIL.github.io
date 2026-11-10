"""Broadband AR coating design using Differential Evolution with heuristic initialization."""
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START

def _characteristic_matrix_reflectance(n0, ns, materials, thicknesses, lambdas):
    """
    Compute reflectance using the characteristic matrix method.
    n0: ambient index, ns: substrate index
    materials: dict of layer indices (same order as thicknesses)
    thicknesses: list of physical thicknesses in nm
    lambdas: array of wavelengths in nm
    Returns: array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    if N == 0:
        # Bare substrate
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(lambdas, r * r, dtype=float)
    
    # Pre-compute for all wavelengths at once
    # M_total starts as identity
    M11 = np.ones((N + 1, len(lambdas)), dtype=complex)
    M12 = np.zeros((N + 1, len(lambdas)), dtype=complex)
    M21 = np.zeros((N + 1, len(lambdas)), dtype=complex)
    M22 = np.ones((N + 1, len(lambdas)), dtype=complex)
    
    # Build the total characteristic matrix from substrate side to air side
    # Actually, we build from air side to substrate side
    # The total matrix M = M1 * M2 * ... * MN
    # where Mi is the characteristic matrix of layer i
    
    # Start with identity (no layers)
    # For each layer, multiply
    
    # Layer indices in order from air to substrate
    # thicknesses[i] corresponds to materials[labels[i]]
    
    # We need the indices for each layer
    # Let's assume thicknesses is paired with labels externally
    # Actually, let's pass the indices directly for speed
    
    # For now, this function expects materials to be a list of indices
    # Let's refactor: pass indices as a list
    
    # Actually, let's just compute inline in the main function for speed
    # This is a helper, let's make it take indices directly
    
    # Wait, the current signature takes materials as dict. Let me change approach.
    # I'll write a faster inline version in the design function.
    # For now, let's keep this but make it efficient.
    
    # Let's assume materials is a list of indices
    # If it's a dict, we need labels. Let's change the interface internally.
    
    # Let me rewrite this to take a list of indices
    pass

def _reflectance_from_indices(n0, ns, indices, thicknesses, lambdas):
    """
    Compute reflectance given a list of layer indices and thicknesses.
    indices: list of indices (floats)
    thicknesses: list of physical thicknesses in nm
    lambdas: array of wavelengths in nm
    Returns: array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(lambdas, r * r, dtype=float)
    
    # Phase factors: delta_i = 2*pi*n_i*d_i / lambda
    # For all wavelengths at once
    # indices: (N,), thicknesses: (N,), lambdas: (W,)
    
    # delta: (N, W)
    # n_i * d_i: (N, 1)
    # lambda: (1, W)
    # delta = 2*pi*n*d/lambda
    
    nd = np.array(indices) * np.array(thicknesses)  # (N,)
    delta = 2.0 * np.pi * nd[:, np.newaxis] / lambdas[np.newaxis, :]  # (N, W)
    
    # cos and sin
    cos_d = np.cos(delta)  # (N, W)
    sin_d = np.sin(delta)  # (N, W)
    
    # Characteristic matrix for each layer:
    # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
    # For normal incidence
    
    # Total matrix M = M_1 * M_2 * ... * M_N
    # Start with identity
    M11 = np.ones((len(lambdas),), dtype=complex)
    M12 = np.zeros((len(lambdas),), dtype=complex)
    M21 = np.zeros((len(lambdas),), dtype=complex)
    M22 = np.ones((len(lambdas),), dtype=complex)
    
    for i in range(N):
        n_i = indices[i]
        c = cos_d[i]  # (W,)
        s = sin_d[i]  # (W,)
        
        # M_i = [[c, i*s/n_i], [i*n_i*s, c]]
        # M_new = M_old * M_i
        # M11_new = M11*c + M12*(i*s/n_i)
        # M12_new = M11*(i*n_i*s) + M12*c
        # M21_new = M21*c + M22*(i*s/n_i)
        # M22_new = M21*(i*n_i*s) + M22*c
        
        M11_new = M11 * c + M12 * (1j * s / n_i)
        M12_new = M11 * (1j * n_i * s) + M12 * c
        M21_new = M21 * c + M22 * (1j * s / n_i)
        M22_new = M21 * (1j * n_i * s) + M22 * c
        
        M11, M12, M21, M22 = M11_new, M12_new, M21_new, M22_new
    
    # Reflectance:
    # Y = (M11 + M12*ns) / (M21 + M22*ns)  # admittance
    # r = (n0*Y - 1) / (n0*Y + 1)  # wait, let me check the formula
    
    # Actually, the standard formula:
    # The characteristic matrix relates (E, H) at air side to (E, H) at substrate side
    # M * [E_s, H_s]^T = [E_0, H_0]^T where E_0 is at air side
    # At substrate, H_s = E_s / ns (for normal incidence, impedance = 1/n)
    # So [E_0, H_0] = M * [E_s, E_s/ns]^T
    # E_0 = M11*E_s + M12*E_s/ns
    # H_0 = M21*E_s + M22*E_s/ns
    # r = (E_0 - H_0) / (E_0 + H_0)  (since E_0 = E_inc + E_ref, H_0 = E_inc - E_ref, and n0=1)
    # Wait, if n0 != 1, we need to be careful.
    
    # Let's use the standard formula:
    # B = M11 + M12/ns
    # C = M21 + M22/ns
    # r = (n0*C - B) / (n0*C + B)
    
    B = M11 + M12 / ns
    C = M21 + M22 / ns
    r = (n0 * C - B) / (n0 * C + B)
    
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(n0, ns, materials_dict, labels, thicknesses, lambdas):
    """Compute mean reflectance in percent."""
    indices = [materials_dict[l] for l in labels]
    R = _reflectance_from_indices(n0, ns, indices, thicknesses, lambdas)
    return np.mean(R) * 100.0

def _is_valid(labels, thicknesses, max_layers, max_total_nm, max_layer_nm):
    """Check if the design is valid."""
    if len(labels) > max_layers:
        return False
    if len(labels) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def design(problem):
    """Design a broadband AR coating using Differential Evolution."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    lambdas = np.linspace(band[0], band[1], 401)
    
    # Material indices
    nL = materials["L"]
    nH = materials["H"]
    
    # Objective function: encode labels and thicknesses into a single vector
    # We'll use a fixed number of layers (max_layers) and allow thickness to be 0
    # to effectively remove a layer. But we also need to choose which material.
    # Alternative: encode each layer as (material_choice, thickness) where material_choice
    # is 0 or 1 (L or H), but DE works on continuous variables.
    # We can use a continuous variable in [0, 1] for material choice and round.
    # Or we can fix the layer structure and optimize thicknesses.
    
    # Let's try: optimize thicknesses for a fixed layer count, and also try different counts.
    # Better: use a continuous optimization where each "layer slot" has a thickness in [0, max_layer_nm]
    # and a material variable in [0, 1] (0 -> L, 1 -> H). Round material to nearest.
    
    # Dimension: 2 * max_layers
    # First max_layers: thicknesses
    # Next max_layers: material choices (continuous, will be rounded)
    
    def objective(x):
        # x: array of length 2*max_layers
        thicknesses = x[:max_layers]
        mat_vars = x[max_layers:]
        
        # Round material variables to 0 or 1
        mat_rounded = np.round(mat_vars)
        
        # Build labels
        labels = []
        thick = []
        for i in range(max_layers):
            if thicknesses[i] > 1e-6:  # non-zero thickness
                if mat_rounded[i] < 0.5:
                    labels.append("L")
                else:
                    labels.append("H")
                thick.append(thicknesses[i])
        
        if not labels:
            # No layers, bare substrate
            return _mean_reflectance_percent(n0, ns, materials, [], [], lambdas)
        
        # Check constraints
        if len(labels) > max_layers:
            return 1e6  # penalty
        if sum(thick) > max_total_nm:
            # Scale down or penalize
            return 1e6
        for t in thick:
            if t > max_layer_nm:
                return 1e6
        
        return _mean_reflectance_percent(n0, ns, materials, labels, thick, lambdas)
    
    # Heuristic initialization: quarter-wave stack
    # For a quarter-wave stack, thickness = lambda / (4*n)
    # Center wavelength
    center = 0.5 * (band[0] + band[1])
    
    # Start with a few quarter-wave layers
    # Let's initialize with alternating L and H layers
    x0 = np.zeros(2 * max_layers)
    
    # Initialize with a reasonable starting point
    # Use a 5-layer quarter-wave stack as initial guess
    n_init = 5
    for i in range(n_init):
        if i % 2 == 0:
            # L layer
            x0[i] = center / (4.0 * nL)
            x0[i + max_layers] = 0.0  # L
        else:
            # H layer
            x0[i] = center / (4.0 * nH)
            x0[i + max_layers] = 1.0  # H
    
    # Use scipy's differential_evolution
    from scipy.optimize import differential_evolution
    
    # Bounds: thickness in [0, max_layer_nm], material in [0, 1]
    bounds = [(0.0, max_layer_nm)] * max_layers + [(0.0, 1.0)] * max_layers
    
    # Run DE with time limit
    # We need to be careful about the time limit
    # DE doesn't have a direct time limit, so we'll use a callback
    
    best_result = None
    best_value = 1e10
    
    # We'll run DE with a maxiter that we adjust based on time
    # Or we can use a callback to stop early
    
    def callback(xk, convergence):
        nonlocal best_result, best_value
        # xk is the current best
        val = objective(xk)
        if val < best_value:
            best_value = val
            best_result = xk.copy()
        # Check time
        if time.time() > deadline:
            return True  # stop
        return False
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            x0=x0,
            maxiter=100,
            tol=1e-6,
            seed=42,
            polish=True,
            callback=callback,
            workers=-1
        )
        
        if result.success or result.x is not None:
            if best_result is not None:
                x_best = best_result
            else:
                x_best = result.x
            
            # Decode
            thicknesses = x_best[:max_layers]
            mat_vars = x_best[max_layers:]
            mat_rounded = np.round(mat_vars)
            
            labels = []
            thick = []
            for i in range(max_layers):
                if thicknesses[i] > 1e-6:
                    if mat_rounded[i] < 0.5:
                        labels.append("L")
                    else:
                        labels.append("H")
                    thick.append(thicknesses[i])
            
            # Check validity
            if not _is_valid(labels, thick, max_layers, max_total_nm, max_layer_nm):
                # Try to fix by scaling
                if thick and sum(thick) > max_total_nm:
                    scale = max_total_nm / sum(thick)
                    thick = [t * scale for t in thick]
                if not _is_valid(labels, thick, max_layers, max_total_nm, max_layer_nm):
                    # Fallback to simple design
                    center = 0.5 * (band[0] + band[1])
                    return ["L"], [center / (4.0 * nL)]
            
            return labels, thick
    
    except Exception as e:
        # Fallback
        center = 0.5 * (band[0] + band[1])
        return ["L"], [center / (4.0 * nL)]
    
    # Fallback
    center = 0.5 * (band[0] + band[1])
    return ["L"], [center / (4.0 * nL)]

# EVOLVE-BLOCK-END
