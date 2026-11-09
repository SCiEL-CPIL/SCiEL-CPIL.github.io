import time
import numpy as np

def design(problem):
    """
    Design a 6-layer quarter-wave antireflection coating.
    Layers: L, H, L, H, L, H
    Center wavelength: 600 nm (midpoint of 400-800 nm band)
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Problem parameters
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    center_lambda = 0.5 * (band[0] + band[1])
    
    # 6-layer quarter-wave stack
    # Order from air to substrate: L, H, L, H, L, H
    materials = ['L', 'H', 'L', 'H', 'L', 'H']
    indices = [n_L, n_H, n_L, n_H, n_L, n_H]
    
    # Calculate physical thicknesses for quarter-wave at center_lambda
    thicknesses = [center_lambda / (4.0 * n) for n in indices]
    
    # Check constraints
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    total_thickness = sum(thicknesses)
    if total_thickness > max_total_nm:
        # Scale down if necessary (should not happen for 6 layers at 600nm)
        scale_factor = max_total_nm / total_thickness
        thicknesses = [t * scale_factor for t in thicknesses]
        
    # Ensure individual layers are within limits
    for i in range(len(thicknesses)):
        if thicknesses[i] > max_layer_nm:
            thicknesses[i] = max_layer_nm
        if thicknesses[i] < 0:
            thicknesses[i] = 0.0

    # Optional: Simple local optimization if time allows
    # For now, return the fixed quarter-wave design which is a strong baseline improvement
    # A more complex optimizer could refine this, but the structural change from 1 to 6 layers is key.
    
    return materials, thicknesses
