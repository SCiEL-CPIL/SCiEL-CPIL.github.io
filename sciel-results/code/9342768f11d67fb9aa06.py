"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    Uses spectral clustering on a compatibility graph to derive item coordinates,
    then combines with product and inverse waste squared to guide bin packing.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin before 330s kill
    
    # Convert to numpy array for vectorized operations
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
        
    # Calculate sums to check if items fit together
    sums = d[:, None] + d[None, :]
    
    # Calculate products to measure "compatibility" (favoring large-large pairs)
    products = d[:, None] * d[None, :]
    
    # Calculate waste (how much space is left in the bin after placing both items)
    waste = capacity - sums
    
    # Condition: items must fit together
    fits = sums <= capacity
    
    # Use a small epsilon to avoid division by zero when waste is exactly 0
    epsilon = 1e-6
    waste_safe = np.where(fits, waste + epsilon, 1.0)
    
    # Base heuristic: product * (1 / waste_safe^4)
    # Stronger inverse-gap penalty: heavily favors pairs that fill the bin tightly,
    # reducing fragmentation and bin count (measured improvement direction).
    H_fitting = products / (waste_safe ** 4)
    H_base = np.where(fits, H_fitting, 1e-9)
    
    # Spectral clustering approach:
    # Build a graph where edge weights are inversely proportional to residual space
    # for pairs that fit together. Use Fiedler vector to get 1D compatibility coordinates.
    
    # Edge weights: for fitting pairs, weight = 1 / (waste + 1.0)^2 to avoid extreme values
    # Stronger emphasis on tight pairs in the spectral graph so the Fiedler vector
    # captures the tight-fill structure
    # For non-fitting pairs, weight = 0
    W = np.zeros((n, n), dtype=np.float64)
    W[fits] = 1.0 / (waste[fits] + 1.0) ** 2
    
    # Compute degree matrix
    degrees = W.sum(axis=1)
    
    # Build Laplacian L = D - W
    L = np.diag(degrees) - W
    
    # Solve for Fiedler vector (eigenvector corresponding to second smallest eigenvalue)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector for the second smallest eigenvalue
        # eigh returns eigenvalues in ascending order
        fiedler = eigenvectors[:, 1]
        
        # Normalize the Fiedler vector to have mean 0 and unit variance
        fiedler = fiedler - fiedler.mean()
        std = fiedler.std()
        if std > 1e-10:
            fiedler = fiedler / std
        else:
            fiedler = np.zeros(n)
        
        # Use the Fiedler coordinates to modulate the heuristic
        # Items with similar coordinates (close together) are more compatible
        # Create a compatibility factor based on how close items are in Fiedler space
        # coord_diff = |fiedler[i] - fiedler[j]|
        # compatibility_factor = exp(-coord_diff^2) - higher when items are close
        
        coord_diff = np.abs(fiedler[:, None] - fiedler[None, :])
        # Scale the Gaussian kernel to have meaningful spread
        # Use a narrower bandwidth so the spectral signal has more discriminative power
        # without zeroing out most entries
        bandwidth = max(np.max(coord_diff) * 0.35, 1e-6)
        compat_factor = np.exp(-(coord_diff / bandwidth) ** 2)
        
        # Combine: H = H_base * (0.3 + 0.7 * compat_factor)
        # Let the spectral compatibility modulate more strongly while keeping a
        # floor so the tight-fill signal dominates
        H_spectral = H_base * (0.3 + 0.7 * compat_factor)
        
    except Exception:
        # If spectral approach fails, fall back to base heuristic
        H_spectral = H_base
    
    # Ensure non-fitting pairs remain at 1e-9
    H = np.where(fits, H_spectral, 1e-9)
    
    # Ensure shape is correct
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
        
    # Check for non-finite values and sanitize if needed
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    # Ensure all values are positive
    H = np.where(H > 0, H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
