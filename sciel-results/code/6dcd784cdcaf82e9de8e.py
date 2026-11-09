"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def _reflectance_spectrum(thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """
    Compute normal-incidence reflectance using the characteristic-matrix method.
    
    Parameters:
    -----------
    thicknesses : list or array of floats (nm)
    materials : dict mapping 'L' -> n_L, 'H' -> n_H
    ambient_index : float
    substrate_index : float
    wavelengths : array of floats (nm)
    
    Returns:
    --------
    reflectance : array of floats (fraction, not percent)
    """
    n_L = materials['L']
    n_H = materials['H']
    
    # Determine layer indices from thicknesses? No, we need labels too.
    # Actually, the function signature in the task is design(problem) -> (materials, thicknesses)
    # where materials is a list of 'L'/'H'. So the reflectance function needs labels.
    # Let's restructure. The helper should take labels and thicknesses.
    pass

def _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """
    Compute mean reflectance in percent.
    """
    n_L = materials['L']
    n_H = materials['H']
    
    n_layers = len(labels)
    if n_layers == 0:
        # Bare substrate
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return (r * r) * 100.0
    
    # Build the characteristic matrix
    # M = M1 * M2 * ... * Mn
    # Each Mi is a 2x2 matrix
    # For a layer with index ni and thickness di at wavelength lambda:
    # delta_i = 2 * pi * ni * di / lambda
    # Mi = [[cos(delta_i), i * sin(delta_i) / ni],
    #       [i * ni * sin(delta_i), cos(delta_i)]]
    
    # We want the mean reflectance over the wavelengths
    # To be efficient, we can vectorize over wavelengths.
    # However, for simplicity and robustness within the time limit, let's compute
    # for each wavelength. But 401 wavelengths * up to 20 layers is fine.
    
    # Let's vectorize over wavelengths for speed.
    # wavelengths: shape (W,)
    # thicknesses: shape (N,)
    # We need delta for each layer and each wavelength: shape (N, W)
    
    deltas = 2.0 * np.pi * np.array([[materials[lbl] for lbl in labels]] * len(thicknesses)) * \
             np.array(thicknesses)[:, None] / wavelengths[None, :]
    
    # Initialize M as identity for each wavelength
    # M shape: (W, 2, 2)
    M = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for i, lbl in enumerate(labels):
        ni = materials[lbl]
        delta = deltas[i, :]  # shape (W,)
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Mi for each wavelength
        # Mi = [[cos_d, 1j * sin_d / ni],
        #       [1j * ni * sin_d, cos_d]]
        
        # M = M * Mi
        # M[:, 0, 0] = M00*Mi00 + M01*Mi10
        # M[:, 0, 1] = M00*Mi01 + M01*Mi11
        # M[:, 1, 0] = M10*Mi00 + M11*Mi10
        # M[:, 1, 1] = M10*Mi01 + M11*Mi11
        
        Mi00 = cos_d
        Mi01 = 1j * sin_d / ni
        Mi10 = 1j * ni * sin_d
        Mi11 = cos_d
        
        new_M00 = M[:, 0, 0] * Mi00 + M[:, 0, 1] * Mi10
        new_M01 = M[:, 0, 0] * Mi01 + M[:, 0, 1] * Mi11
        new_M10 = M[:, 1, 0] * Mi00 + M[:, 1, 1] * Mi10
        new_M11 = M[:, 1, 0] * Mi01 + M[:, 1, 1] * Mi11
        
        M[:, 0, 0] = new_M00
        M[:, 0, 1] = new_M01
        M[:, 1, 0] = new_M10
        M[:, 1, 1] = new_M11
    
    # Now compute reflectance
    # The field vector in the incident medium (air) is:
    # [E_inc + E_ref] = M00 * E_sub
    # [E_inc - E_ref] * (n_ambient / n_substrate) = M10 * E_sub  (Wait, the standard formula is slightly different)
    
    # Standard formula for normal incidence:
    # Let the substrate have index n_s.
    # The field at the substrate interface (after the last layer) is:
    # [E_s] = M * [E_s]  (where the second component is related to H)
    # Actually, the characteristic matrix relates the fields at the first interface to the substrate.
    # Let's use the standard result:
    # r = (n_ambient * M11 + n_ambient * M12 * n_substrate - M11 * n_substrate - M12 * n_ambient^2) / ... 
    # No, let's derive it properly.
    
    # Let the fields at the air-coating interface be (E, H) and at the coating-substrate interface be (E', H').
    # The characteristic matrix M relates them:
    # [E]   = [M11 M12] [E']
    # [H]     [M21 M22] [H']
    
    # In the substrate, there is only a transmitted wave:
    # E' = E_t
    # H' = E_t / n_s  (assuming normalized impedance, or similar)
    # Actually, for normal incidence, H = E / n (in normalized units where Z0=1)
    # So H' = E_t / n_s
    
    # In the air, we have incident and reflected waves:
    # E = E_i + E_r
    # H = (E_i - E_r) / n_0
    
    # So:
    # E_i + E_r = M11 * E_t + M12 * (E_t / n_s) = E_t * (M11 + M12 / n_s)
    # (E_i - E_r) / n_0 = M21 * E_t + M22 * (E_t / n_s) = E_t * (M21 + M22 / n_s)
    
    # Let A = M11 + M12 / n_s
    # Let B = M21 + M22 / n_s
    # Then:
    # E_i + E_r = A * E_t
    # E_i - E_r = n_0 * B * E_t
    
    # Adding: 2 E_i = (A + n_0 B) E_t  => E_i = (A + n_0 B) / 2 * E_t
    # Subtracting: 2 E_r = (A - n_0 B) E_t => E_r = (A - n_0 B) / 2 * E_t
    
    # r = E_r / E_i = (A - n_0 B) / (A + n_0 B)
    
    n_0 = ambient_index
    n_s = substrate_index
    
    A = M[:, 0, 0] + M[:, 0, 1] / n_s
    B = M[:, 1, 0] + M[:, 1, 1] / n_s
    
    r = (A - n_0 * B) / (A + n_0 * B)
    
    reflectance = np.abs(r) ** 2
    mean_reflectance_percent = np.mean(reflectance) * 100.0
    
    return mean_reflectance_percent

def design(problem):
    """
    Design a broadband antireflection coating using a multi-layer stack optimized
    via a global optimization algorithm.
    """
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # We will search over a fixed number of layers, say 10, to keep the search space manageable.
    # But the problem allows up to 20. Let's start with a smaller number and increase if needed.
    # Actually, let's use a variable number of layers by encoding the problem as a fixed-length
    # vector where some layers can have zero thickness.
    
    # Let's fix the number of layers to 12 (even number, good for symmetric designs)
    # and allow thicknesses to be 0.
    N_layers = 12
    
    # Define the objective function
    def objective(params):
        # params: array of length 2 * N_layers
        # First N_layers are thicknesses, next N_layers are labels (0 for L, 1 for H)
        thicknesses = params[:N_layers]
        labels_float = params[N_layers:]
        
        # Clip thicknesses to valid range
        thicknesses = np.clip(thicknesses, 0, max_layer_nm)
        
        # Convert labels to 'L' or 'H'
        # Threshold at 0.5
        labels = ['H' if l > 0.5 else 'L' for l in labels_float]
        
        # Check total thickness constraint
        if np.sum(thicknesses) > max_total_nm:
            return 100.0  # Large penalty
        
        # Compute reflectance
        return _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, wavelengths)
    
    # Initial guess: a simple quarter-wave stack
    centre_wavelength = 0.5 * (band_nm[0] + band_nm[1])
    initial_thick_L = centre_wavelength / (4.0 * materials['L'])
    initial_thick_H = centre_wavelength / (4.0 * materials['H'])
    
    # Create an initial guess with alternating L and H layers
    initial_params = np.zeros(2 * N_layers)
    for i in range(N_layers):
        if i % 2 == 0:
            initial_params[i] = initial_thick_L
            initial_params[i + N_layers] = 0.0  # L
        else:
            initial_params[i] = initial_thick_H
            initial_params[i + N_layers] = 1.0  # H
    
    # Bounds for the optimization
    # Thicknesses: [0, max_layer_nm]
    # Labels: [0, 1] (discrete, but we'll treat them as continuous for the optimizer and round)
    bounds = [(0, max_layer_nm)] * N_layers + [(0, 1)] * N_layers
    
    # Use differential evolution for global optimization
    # This is robust for multimodal problems
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-6,
        polish=True,
        mutation=(0.3, 1.0),
        recombination=0.7,
        popsize=15
    )
    
    # Extract the best solution
    best_params = result.x
    best_thicknesses = np.clip(best_params[:N_layers], 0, max_layer_nm)
    best_labels_float = best_params[N_layers:]
    
    # Convert labels
    best_labels = ['H' if l > 0.5 else 'L' for l in best_labels_float]
    
    # Remove layers with zero thickness
    final_labels = []
    final_thicknesses = []
    for lbl, th in zip(best_labels, best_thicknesses):
        if th > 1e-6:  # Small threshold to avoid numerical noise
            final_labels.append(lbl)
            final_thicknesses.append(th)
    
    # If no layers are left, return a single layer
    if len(final_labels) == 0:
        final_labels = ['L']
        final_thicknesses = [initial_thick_L]
    
    # Check if the total thickness exceeds the limit
    if np.sum(final_thicknesses) > max_total_nm:
        # Scale down the thicknesses
        scale = max_total_nm / np.sum(final_thicknesses)
        final_thicknesses = [th * scale for th in final_thicknesses]
    
    return final_labels, final_thicknesses
# EVOLVE-BLOCK-END
