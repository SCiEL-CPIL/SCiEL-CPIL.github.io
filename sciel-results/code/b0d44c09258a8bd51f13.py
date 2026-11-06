"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """
    Design a 32x32 unit cell using a Fourier-harmonic superposition approach.
    This method explicitly engineers the reciprocal lattice vector components
    required for strong Bragg coupling between the 4th and 5th TM bands.
    """
    start_time = time.time()
    deadline = 200  # Safety margin before 230s kill
    
    size = problem["pixels"]
    N = size
    cell = [[0 for _ in range(N)] for _ in range(N)]
    
    # Create coordinate grids
    x = np.arange(N) - N/2.0 + 0.5
    y = np.arange(N) - N/2.0 + 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Normalize coordinates to [-1, 1]
    X_norm = X / (N/2.0)
    Y_norm = Y / (N/2.0)
    
    # Generate field as superposition of Fourier harmonics
    # These harmonics correspond to specific reciprocal lattice vectors
    # that are important for Bragg scattering in a square lattice
    
    # Harmonic (2,0): x-direction coupling
    amp_20 = 0.8
    phase_20 = 0.0
    f_20 = amp_20 * np.cos(2 * np.pi * X_norm + phase_20)
    
    # Harmonic (0,2): y-direction coupling
    amp_02 = 0.8
    phase_02 = 0.0
    f_02 = amp_02 * np.cos(2 * np.pi * Y_norm + phase_02)
    
    # Harmonic (2,2): diagonal coupling
    amp_22 = 0.5
    phase_22 = np.pi/4
    f_22 = amp_22 * np.cos(2 * np.pi * (X_norm + Y_norm) + phase_22)
    
    # Harmonic (4,0): higher-order x modulation
    amp_40 = 0.3
    phase_40 = np.pi/2
    f_40 = amp_40 * np.cos(4 * np.pi * X_norm + phase_40)
    
    # Harmonic (0,4): higher-order y modulation
    amp_04 = 0.3
    phase_04 = np.pi/2
    f_04 = amp_04 * np.cos(4 * np.pi * Y_norm + phase_04)
    
    # Sum of all harmonics
    field = f_20 + f_02 + f_22 + f_40 + f_04
    
    # Threshold to binary (0/1)
    # Use the median as threshold for ~50% fill fraction
    threshold = np.median(field)
    binary = (field > threshold).astype(np.uint8)
    
    # Apply D4 symmetry: average with all 8 symmetry operations
    sym = binary.astype(float).copy()
    sym += np.rot90(binary, 1).astype(float)
    sym += np.rot90(binary, 2).astype(float)
    sym += np.rot90(binary, 3).astype(float)
    sym += np.fliplr(binary).astype(float)
    sym += np.flipud(binary).astype(float)
    sym += np.fliplr(np.flipud(binary)).astype(float)
    sym += binary.T.astype(float)
    
    # Average and threshold
    sym_avg = sym / 8.0
    sym_threshold = 0.5
    sym_binary = (sym_avg > sym_threshold).astype(np.uint8)
    
    # Convert to nested list
    for i in range(N):
        for j in range(N):
            cell[i][j] = int(sym_binary[i, j])
    
    # Check time and return
    if time.time() - start_time > deadline:
        return cell
    
    return cell
# EVOLVE-BLOCK-END
