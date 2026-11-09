"""Design a broadband antireflection coating using differential evolution on a periodic L/H stack."""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.constants import c  # not needed, but available

def _reflectance_stack(thicknesses, indices, ambient_index, substrate_index, wavelengths_nm):
    """
    Compute normal-incidence reflectance (as a fraction, 0-1) for a stack of layers
    using the characteristic-matrix method.
    
    thicknesses: list of physical thicknesses in nm (air to substrate order)
    indices: list of refractive indices for each layer
    ambient_index: index of ambient (air)
    substrate_index: index of substrate (glass)
    wavelengths_nm: array of wavelengths in nm
    """
    n_amb = ambient_index
    n_sub = substrate_index
    n_layers = len(thicknesses)
    
    # Convert wavelengths to angular frequency / wavenumber
    # k = 2*pi*n / lambda
    # For each wavelength, compute the characteristic matrix
    
    # M = product of individual layer matrices
    # For layer j: M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # where delta_j = 2*pi*n_j*d_j / lambda
    
    # For a stack, M_total = M_1 * M_2 * ... * M_N
    # Then:
    # Y = (M_11 + M_12 * n_sub) / (M_21 + M_22 * n_sub)
    # R = | (n_amb - Y) / (n_amb + Y) |^2
    
    reflectances = np.zeros(len(wavelengths_nm))
    
    for i, lam in enumerate(wavelengths_nm):
        # Build total matrix
        M_total = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
        
        for j in range(n_layers):
            n_j = indices[j]
            d_j = thicknesses[j]
            delta_j = 2.0 * np.pi * n_j * d_j / lam
            c_j = np.cos(delta_j)
            s_j = np.sin(delta_j)
            
            M_j = np.array([
                [c_j, 1j * s_j / n_j],
                [1j * n_j * s_j, c_j]
            ])
            
            M_total = M_total @ M_j
        
        M11, M12 = M_total[0, 0], M_total[0, 1]
        M21, M22 = M_total[1, 0], M_total[1, 1]
        
        Y = (M11 + M12 * n_sub) / (M21 + M22 * n_sub)
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2
        reflectances[i] = R
    
    return reflectances

def design(problem):
    """
    Design a broadband antireflection coating using differential evolution
    on a periodic L/H stack with a base thickness and gradient factor.
    """
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials["L"]
    n_H = materials["H"]
    
    # Wavelengths for evaluation: 401 equally spaced points
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # We'll use a periodic structure: alternating L and H layers.
    # Parameterize by:
    # - base_thickness: base thickness for all layers
    # - gradient: linear gradient factor (0 to 1) that modulates thickness
    #   to create a smooth transition from low to high optical density.
    #
    # Strategy: Use N pairs of (L, H) layers, so total layers = 2*N.
    # N can be up to max_layers // 2 = 10.
    #
    # For layer index k (0-indexed from air side):
    #   If k is even: material is L
    #   If k is odd: material is H
    #
    # Thickness for layer k:
    #   t_k = base_thickness * (1.0 + gradient * (k / (N_total - 1) - 0.5) * 2.0)
    #   where N_total is total number of layers.
    #   This creates a linear gradient from (1 - gradient) to (1 + gradient) times base.
    #
    # We'll optimize base_thickness and gradient, and also choose N (number of pairs).
    # To keep the search space manageable, we'll try a few values of N and pick the best.
    
    def objective(params):
        base_thickness, gradient, N_pairs = params
        N_pairs = int(round(N_pairs))
        N_pairs = max(1, min(N_pairs, max_layers // 2))
        N_total = 2 * N_pairs
        
        # Compute thicknesses
        thicknesses = []
        indices = []
        for k in range(N_total):
            if k % 2 == 0:
                indices.append(n_L)
            else:
                indices.append(n_H)
            
            if N_total > 1:
                frac = k / (N_total - 1)
            else:
                frac = 0.5
            # Gradient factor: from (1 - gradient) to (1 + gradient)
            scale = 1.0 + gradient * (frac - 0.5) * 2.0
            t = base_thickness * scale
            # Clamp to valid range
            t = max(0.0, min(t, max_layer_nm))
            thicknesses.append(t)
        
        # Check total thickness constraint
        total_t = sum(thicknesses)
        if total_t > max_total_nm:
            # Penalize heavily
            return 1.0 + (total_t - max_total_nm) / max_total_nm
        
        # Compute mean reflectance
        R = _reflectance_stack(thicknesses, indices, ambient_index, substrate_index, wavelengths)
        return np.mean(R)
    
    # Define bounds for differential evolution
    # base_thickness: [10, 400] nm
    # gradient: [0, 1]
    # N_pairs: [1, max_layers // 2]
    bounds = [(10.0, 400.0), (0.0, 1.0), (1.0, float(max_layers // 2))]
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds,
        maxiter=50,
        popsize=15,
        tol=1e-6,
        seed=42,
        polish=False,
        init='latinhypercube'
    )
    
    best_base, best_grad, best_N_pairs = result.x
    best_N_pairs = int(round(best_N_pairs))
    best_N_pairs = max(1, min(best_N_pairs, max_layers // 2))
    N_total = 2 * best_N_pairs
    
    # Build the optimal stack
    thicknesses = []
    labels = []
    for k in range(N_total):
        if k % 2 == 0:
            labels.append("L")
        else:
            labels.append("H")
        
        if N_total > 1:
            frac = k / (N_total - 1)
        else:
            frac = 0.5
        scale = 1.0 + best_grad * (frac - 0.5) * 2.0
        t = best_base * scale
        t = max(0.0, min(t, max_layer_nm))
        thicknesses.append(t)
    
    # Final check on total thickness
    total_t = sum(thicknesses)
    if total_t > max_total_nm:
        # Scale down all thicknesses
        scale_factor = max_total_nm / total_t
        thicknesses = [t * scale_factor for t in thicknesses]
    
    return labels, thicknesses
