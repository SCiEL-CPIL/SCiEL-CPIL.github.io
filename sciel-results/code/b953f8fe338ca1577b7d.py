"""Rewritten program for broadband antireflection coating design."""
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START
def _reflectance_spectrum(thicknesses, materials, ambient_index, substrate_index,
                          band_nm, n_wavelengths=401):
    """
    Compute normal-incidence reflectance spectrum using characteristic matrix method.
    
    Parameters:
    -----------
    thicknesses : list of float
        Physical thicknesses in nm, ordered from air to substrate.
    materials : dict
        Mapping from layer label to refractive index.
    ambient_index : float
        Refractive index of ambient (air).
    substrate_index : float
        Refractive index of substrate (glass).
    band_nm : list/tuple
        (lambda_min, lambda_max) in nm.
    n_wavelengths : int
        Number of wavelengths to sample.
    
    Returns:
    --------
    reflectance : ndarray
        Reflectance in percent at each wavelength.
    """
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, n_wavelengths)
    
    n_layers = len(thicknesses)
    if n_layers == 0:
        # Bare substrate
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return (np.abs(r0) ** 2) * 100.0
    
    # We need to know the indices of each layer. Since thicknesses alone don't tell us
    # the index, we need to pass indices too. Let's refactor.
    # Actually, the design function returns (materials_labels, thicknesses).
    # So we need to pass the labels as well for the reflectance calculation.
    # Let's create a helper that takes labels and thicknesses.
    pass


def _reflectance_spectrum_full(labels, thicknesses, materials, ambient_index, substrate_index,
                                band_nm, n_wavelengths=401):
    """
    Compute normal-incidence reflectance spectrum using characteristic matrix method.
    
    Parameters:
    -----------
    labels : list of str
        Layer labels ('L' or 'H'), ordered from air to substrate.
    thicknesses : list of float
        Physical thicknesses in nm.
    materials : dict
        Mapping from layer label to refractive index.
    ambient_index : float
        Refractive index of ambient (air).
    substrate_index : float
        Refractive index of substrate (glass).
    band_nm : list/tuple
        (lambda_min, lambda_max) in nm.
    n_wavelengths : int
        Number of wavelengths to sample.
    
    Returns:
    --------
    reflectance : ndarray
        Reflectance in percent at each wavelength.
    """
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, n_wavelengths)
    
    n_layers = len(labels)
    if n_layers == 0:
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return (np.abs(r0) ** 2) * 100.0
    
    # Precompute indices and optical thicknesses
    indices = np.array([materials[label] for label in labels], dtype=complex)
    thicknesses_arr = np.array(thicknesses, dtype=float)
    
    # For each wavelength, compute the characteristic matrix
    # M_total = product of M_j for j=1..N
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # where delta_j = 2*pi*n_j*d_j/lambda
    
    reflectance = np.zeros(len(wavelengths))
    
    for i, lam in enumerate(wavelengths):
        # Start with identity matrix
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = indices[j]
            d_j = thicknesses_arr[j]
            delta_j = 2.0 * np.pi * n_j * d_j / lam
            
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            
            # Characteristic matrix for layer j
            M_j = np.array([[cos_d, 1j * sin_d / n_j],
                            [1j * n_j * sin_d, cos_d]], dtype=complex)
            
            M = M @ M_j
        
        # Boundary conditions:
        # At substrate: Y_s = n_s (for normal incidence)
        # Total admittance: Y = (M[1,0] * n_s + M[1,1]) / (M[0,0] * n_s + M[0,1])
        # Wait, let me be careful with the characteristic matrix method.
        
        # The characteristic matrix M relates the fields at the top of the stack to the substrate:
        # [E_top]   = M @ [E_sub]
        # [H_top]        [H_sub]
        
        # At the substrate, H_sub = Y_sub * E_sub, where Y_sub = n_s (for normal incidence, TE/TM same)
        # So [E_top] = M @ [E_sub, Y_sub * E_sub]^T = M @ E_sub * [1, Y_sub]^T
        # E_top = M[0,0] * E_sub + M[0,1] * Y_sub * E_sub = (M[0,0] + M[0,1] * Y_sub) * E_sub
        # H_top = M[1,0] * E_sub + M[1,1] * Y_sub * E_sub = (M[1,0] + M[1,1] * Y_sub) * E_sub
        
        # The admittance at the top of the stack is Y_top = H_top / E_top = (M[1,0] + M[1,1] * Y_sub) / (M[0,0] + M[0,1] * Y_sub)
        
        Y_sub = substrate_index  # real number
        Y_top = (M[1, 0] + M[1, 1] * Y_sub) / (M[0, 0] + M[0, 1] * Y_sub)
        
        # Reflection coefficient at the air-coating interface:
        # r = (n_ambient - Y_top) / (n_ambient + Y_top)
        r = (ambient_index - Y_top) / (ambient_index + Y_top)
        reflectance[i] = np.abs(r) ** 2 * 100.0
    
    return reflectance


def design(problem):
    """
    Design a broadband antireflection coating using global optimization.
    
    We optimize both the number of layers (up to max_layers), the layer types (L/H),
    and the thicknesses to minimize the mean reflectance over the band.
    """
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # We'll try different numbers of layers and find the best design.
    # For efficiency, we'll use differential evolution for each fixed number of layers.
    
    best_labels = []
    best_thicknesses = []
    best_mean_ref = np.inf
    
    # Try from 1 up to max_layers, but limit to reasonable numbers for speed.
    # Given 60s CPU budget, we should be careful.
    # Let's try a few promising configurations.
    
    # Strategy: Use a fixed number of layers (say, 8-12) and optimize thicknesses
    # with a smart initialization based on quarter-wave stacks.
    
    # First, let's try a simple approach: optimize thicknesses for a fixed layer sequence.
    # Common effective sequences for broadband AR:
    # - Alternating L and H layers
    # - Starting with L (low index close to air) and ending with L (low index close to glass? no, glass is 1.52, L is 1.38, H is 2.35)
    # Actually, for AR on glass (n=1.52), we want to match air (n=1.0) to glass (n=1.52).
    # The geometric mean is sqrt(1.0 * 1.52) ≈ 1.233. L=1.38 is closer to this than H=2.35.
    # So starting with L makes sense.
    
    # Let's try optimizing with a fixed sequence of alternating L and H layers.
    # We'll try different numbers of layers.
    
    # To stay within time limits, we'll use a coarse optimization first, then refine.
    
    # Let's define the objective function
    def objective(params):
        """
        params: array of length 2*N, where first N are discrete (0 for L, 1 for H)
                and last N are continuous thicknesses.
        But differential evolution works better with continuous variables.
        Let's use a different approach: fix the layer types, optimize thicknesses.
        """
        pass
    
    # Better approach: For each candidate layer sequence, optimize thicknesses using
    # a continuous optimizer.
    
    # Let's generate candidate sequences.
    # A good starting point is alternating L and H, starting with L.
    
    # We'll try sequences of length 4, 6, 8, 10, 12.
    
    for n_layers in [4, 6, 8, 10, 12]:
        # Generate a few candidate sequences
        candidates = []
        
        # Alternating starting with L
        seq1 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        candidates.append(seq1)
        
        # Alternating starting with H
        seq2 = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        candidates.append(seq2)
        
        # All L
        seq3 = ['L'] * n_layers
        candidates.append(seq3)
        
        # All H
        seq4 = ['H'] * n_layers
        candidates.append(seq4)
        
        # L...L H...H (half and half)
        half = n_layers // 2
        seq5 = ['L'] * half + ['H'] * (n_layers - half)
        candidates.append(seq5)
        
        seq6 = ['H'] * half + ['L'] * (n_layers - half)
        candidates.append(seq6)
        
        for labels in candidates:
            # Optimize thicknesses for this fixed sequence
            # Initial guess: quarter-wave at center wavelength
            centre = 0.5 * (band_nm[0] + band_nm[1])
            init_thicknesses = []
            for label in labels:
                n_mat = materials[label]
                d = centre / (4.0 * n_mat)
                # Clip to valid range
                d = min(max(d, 0.0), max_layer_nm)
                init_thicknesses.append(d)
            
            # Check if initial total thickness is within limit
            if sum(init_thicknesses) > max_total_nm:
                # Scale down
                scale = max_total_nm / sum(init_thicknesses)
                init_thicknesses = [d * scale for d in init_thicknesses]
            
            # Define the objective function for thickness optimization
            def obj(thicknesses):
                # Ensure thicknesses are within bounds
                if any(t < 0 or t > max_layer_nm for t in thicknesses):
                    return 1e6
                if sum(thicknesses) > max_total_nm:
                    return 1e6
                
                refl = _reflectance_spectrum_full(labels, thicknesses, materials,
                                                   ambient_index, substrate_index, band_nm)
                return np.mean(refl)
            
            # Bounds for each thickness
            bounds = [(0.0, max_layer_nm)] * n_layers
            
            # Use differential evolution for robustness
            try:
                result = differential_evolution(obj, bounds, maxiter=100, popsize=15,
                                                 tol=1e-6, seed=42, polish=True)
                if result.success or result.fun < best_mean_ref:
                    best_mean_ref = result.fun
                    best_labels = labels[:]
                    best_thicknesses = result.x.tolist()
            except Exception:
                pass
    
    # If we haven't found anything good, fall back to a simple design
    if not best_labels:
        centre = 0.5 * (band_nm[0] + band_nm[1])
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * materials["L"])]
    
    # Final validation and clipping
    best_thicknesses = [min(max(t, 0.0), max_layer_nm) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses
# EVOLVE-BLOCK-END
