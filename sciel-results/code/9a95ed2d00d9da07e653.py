"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell for a 2D square-lattice photonic crystal
    that maximizes the gap between TM bands 4 and 5.
    
    Strategy: Fourier Synthesis.
    Instead of defining geometric shapes, we define the dielectric permittivity
    distribution as a sum of sinusoidal components in the real space. This
    allows us to directly control the strength of specific Fourier coefficients
    (scattering vectors) that are responsible for opening the band gap.
    
    We focus on enhancing the scattering at the M point (diagonal) and X point (edge)
    by combining cosine terms. The pattern is then thresholded to create a binary
    air/dielectric map.
    
    The cell must be symmetric under:
    - 180-degree rotation
    - Transpose symmetry
    """
    size = problem["pixels"]
    
    # Create coordinate grids
    # Use pixel centers for better symmetry
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    # Normalize coordinates to [0, 1)
    x_norm = X / size
    y_norm = Y / size
    
    # Define the potential function V(x,y)
    # We use a combination of modes to target both X and M points.
    # Mode (1,1): cos(2*pi*x)*cos(2*pi*y) -> targets M point
    # Mode (1,0): cos(2*pi*x) -> targets X point (vertical edges)
    # Mode (0,1): cos(2*pi*y) -> targets X point (horizontal edges)
    
    # Amplitude tuning:
    # A strong (1,1) component creates a checkerboard-like pattern.
    # Adding (1,0) and (0,1) creates a more complex, often higher-performing structure
    # resembling a "plus" or "cross" hybrid with diagonal features.
    
    A_diag = 1.0
    A_edge = 0.5
    
    # Calculate the superposition
    V = A_diag * np.cos(2 * np.pi * x_norm) * np.cos(2 * np.pi * y_norm)
    V += A_edge * (np.cos(2 * np.pi * x_norm) + np.cos(2 * np.pi * y_norm))
    
    # Normalize V to have zero mean and unit standard deviation for thresholding
    V = V - np.mean(V)
    std = np.std(V)
    if std > 0:
        V = V / std
    
    # Thresholding:
    # We want a filling fraction close to 50%.
    # To ensure exactly 50% filling (or close to it), we can use the median as threshold.
    # However, for symmetry and robustness, we pick a threshold that ensures
    # the pattern is centered around 0.
    # Let's aim for a filling fraction of 0.5.
    # We can sort the values and pick the median.
    threshold = np.median(V)
    
    # Create binary cell
    # 1 where V > threshold (dielectric), 0 otherwise
    cell_np = (V > threshold).astype(int)
    
    # Ensure strict symmetry by enforcing it explicitly if floating point errors occurred
    # (Though cosine sums should be symmetric, let's be safe)
    # The construction is inherently symmetric under x->1-x, y->1-y and x<->y.
    # cos(2*pi*(1-x)) = cos(2*pi - 2*pi*x) = cos(2*pi*x).
    # So the mathematical function is symmetric. The discretization on integer grid
    # 0..31 is symmetric around 15.5?
    # x=0 and x=31: cos(0)=1, cos(2*pi*31/32)=cos(2*pi - pi/16) != 1.
    # Wait, the grid is 0..31.
    # x_norm = i/32.
    # i=0 -> 0. i=31 -> 31/32.
    # cos(2*pi*0) = 1.
    # cos(2*pi*31/32) = cos(2*pi - pi/16) = cos(pi/16).
    # These are not equal. The pattern is NOT symmetric under i <-> 31-i on a discrete grid
    # unless we shift the coordinates or use a different grid definition.
    
    # To ensure symmetry cell[i][j] == cell[31-i][31-j], we should define coordinates
    # relative to the center.
    # Let's redefine X, Y to be centered.
    # Center is 15.5.
    # x_centered = x - 15.5
    # x_norm_centered = x_centered / size
    
    x_c = (X - (size - 1) / 2.0) / size
    y_c = (Y - (size - 1) / 2.0) / size
    
    V2 = A_diag * np.cos(2 * np.pi * x_c) * np.cos(2 * np.pi * y_c)
    V2 += A_edge * (np.cos(2 * np.pi * x_c) + np.cos(2 * np.pi * y_c))
    
    V2 = V2 - np.mean(V2)
    std2 = np.std(V2)
    if std2 > 0:
        V2 = V2 / std2
        
    threshold2 = np.median(V2)
    cell_np = (V2 > threshold2).astype(int)
    
    # Convert to list of lists
    cell = cell_np.tolist()
    
    return cell
# EVOLVE-BLOCK-END
