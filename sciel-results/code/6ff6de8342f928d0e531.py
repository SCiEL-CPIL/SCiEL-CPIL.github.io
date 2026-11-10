"""Antireflection coating design using differential evolution global optimization."""
import numpy as np
import time
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a broadband AR coating using a fixed alternating stack structure
    and global optimization (differential evolution) for thicknesses.
    """
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    # Pre-compute wavelengths for evaluation (401 points from 400 to 800 nm)
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Define a fixed stack structure to reduce search space dimensionality
    # Alternating L and H layers starting with L (low index near air)
    # 14 layers is a good balance between complexity and optimization time
    num_layers = 14
    labels = []
    for i in range(num_layers):
        if i % 2 == 0:
            labels.append('L')
        else:
            labels.append('H')
    
    # Pre-compute refractive indices for the fixed structure
    n_stack = np.array([nL if l == 'L' else nH for l in labels])
    
    def objective(thicknesses):
        """
        Compute mean reflectance for a given set of thicknesses.
        Uses the characteristic matrix method.
        """
        # Check constraints quickly (DE handles bounds, but sum constraint needs checking)
        # If sum exceeds max_total_nm, penalize heavily
        if np.sum(thicknesses) > max_total_nm:
            return 100.0  # High penalty
        
        # Optical thicknesses (n*d)
        od = n_stack * thicknesses
        k = 2.0 * np.pi / wavelengths  # Wavenumber
        
        # Phase delta = k * d * n
        # Note: The characteristic matrix uses delta = 2*pi*n*d/lambda
        # So delta = k * n * d = k * od
        delta = k[:, None] * od[None, :]  # Shape: (401, num_layers)
        
        # Initialize characteristic matrices
        # M = product of individual layer matrices
        # Individual matrix for layer j:
        # [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
        
        # We can compute this vectorized for all wavelengths
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Matrix elements
        # M_11 = cos_d
        # M_12 = 1j * sin_d / n_stack
        # M_21 = 1j * n_stack * sin_d
        # M_22 = cos_d
        
        # To multiply matrices for all wavelengths simultaneously:
        # We can't simply use np.matmul on the full 3D array easily without loops or einsum.
        # Let's use a loop over layers for clarity and correctness, as num_layers is small (14).
        
        # Initialize identity matrices for all wavelengths
        # Shape: (401, 2, 2)
        M = np.zeros((len(wavelengths), 2, 2), dtype=complex)
        M[:, 0, 0] = 1.0
        M[:, 1, 1] = 1.0
        
        for j in range(num_layers):
            n_j = n_stack[j]
            # Current layer matrix for all wavelengths
            # M_j[:, 0, 0] = cos_d[:, j]
            # M_j[:, 0, 1] = 1j * sin_d[:, j] / n_j
            # M_j[:, 1, 0] = 1j * n_j * sin_d[:, j]
            # M_j[:, 1, 1] = cos_d[:, j]
            
            M_j = np.zeros((len(wavelengths), 2, 2), dtype=complex)
            M_j[:, 0, 0] = cos_d[:, j]
            M_j[:, 0, 1] = 1j * sin_d[:, j] / n_j
            M_j[:, 1, 0] = 1j * n_j * sin_d[:, j]
            M_j[:, 1, 1] = cos_d[:, j]
            
            # M = M @ M_j
            # Using einsum for batched matrix multiplication
            M = np.einsum('ijk,klk->il' if False else 'ijk,jlk->il', M, M_j) 
            # Wait, einsum indices: M is (N, 2, 2), M_j is (N, 2, 2).
            # We want M_new[i, k, l] = sum_j M[i, k, j] * M_j[i, j, l]
            # 'ikj,ijl->ikl'
            M = np.einsum('ikj,ijl->ikl', M, M_j)
        
        # Calculate Reflectance
        # Y = M[1,0] / M[0,0] (admittance)
        # R = |(Y - n0) / (Y + n0)|^2
        # But Y is relative to substrate?
        # Standard formula:
        # The characteristic matrix M relates the field at the entrance to the field at the substrate.
        # M = [[M11, M12], [M21, M22]]
        # The effective admittance Y is given by:
        # Y = (M21 + M22 * ns) / (M11 + M12 * ns)
        # Where ns is the substrate index.
        
        M11 = M[:, 0, 0]
        M12 = M[:, 0, 1]
        M21 = M[:, 1, 0]
        M22 = M[:, 1, 1]
        
        Y = (M21 + M22 * ns) / (M11 + M12 * ns)
        
        R = np.abs((Y - n0) / (Y + n0)) ** 2
        mean_R = np.mean(R)
        return mean_R * 100.0  # Return in percent as per task description? 
        # Task says "mean normal-incidence reflectance in percent". 
        # Score is the raw value. Lower is better.
        # The evaluator computes it exactly. I should return the value in the same units.
        # "Score (minimize): mean normal-incidence reflectance in percent"
        # So returning percent is correct.

    # Define bounds for differential evolution
    # Each thickness must be in [0, max_layer_nm]
    bounds = [(0, max_layer_nm) for _ in range(num_layers)]
    
    # Run Differential Evolution
    # Maxiter and popsize need to be tuned to fit in time.
    # 14 dimensions.
    # Let's estimate time. One evaluation is ~14 matrix mults of 401x2x2. Very fast.
    # DE with popsize=100, maxiter=200 might be too slow.
    # Let's try popsize=50, maxiter=100.
    
    # Check time before starting
    if time.time() - start_time > DEADLINE:
        # Fallback to a simple analytical guess if we are already out of time
        centre = 0.5 * (band[0] + band[1])
        return labels, [centre / (4.0 * nL) for _ in range(num_layers)]
        
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=50, 
            tol=1e-4, 
            seed=42,
            polish=True,  # Use L-BFGS-B for final polish
            update_interval=1  # Not used in standard DE, but good for debugging if we had it
        )
    except Exception as e:
        # Fallback if DE fails
        centre = 0.5 * (band[0] + band[1])
        return labels, [centre / (4.0 * nL) for _ in range(num_layers)]
        
    best_thicknesses = result.x
    
    # Verify constraints on the best result
    if np.sum(best_thicknesses) > max_total_nm:
        # Scale down if necessary
        scale = max_total_nm / np.sum(best_thicknesses)
        best_thicknesses = best_thicknesses * scale
        
    # Final check for time
    if time.time() - start_time > DEADLINE:
        # Return whatever we have, even if not fully polished
        pass

    return labels, list(best_thicknesses)
# EVOLVE-BLOCK-END
