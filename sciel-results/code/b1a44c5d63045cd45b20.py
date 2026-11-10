import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Strategy: Spectral clustering-based heuristic.
    1. Construct a graph where nodes are items and edge weights are inversely 
       proportional to the residual space (waste) if two items are packed together.
    2. Use the Fiedler vector (second eigenvector of the graph Laplacian) to assign
       a one-dimensional "compatibility coordinate" to each item.
    3. Set the heuristic matrix entries to the product of these coordinates, 
       scaled to ensure positive values and appropriate magnitude.
    4. For non-fitting pairs, set the heuristic to 1e-9.
    """
    start_time = time.time()
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
        
    # Calculate sums for all pairs
    sums = d[:, None] + d[None, :]
    
    # Determine which pairs fit in the bin
    fits = sums <= capacity
    
    # Calculate waste for fitting pairs
    waste = capacity - sums
    
    # Construct edge weights: inversely proportional to waste
    # For fitting pairs, waste >= 0. We use 1/(waste + 1) to avoid division by zero
    # and to ensure larger weights for tighter fits.
    # For non-fitting pairs, weight is 0.
    
    # Safe waste for calculation
    safe_waste = np.where(fits, waste + 1.0, 1.0)
    weights = np.where(fits, 1.0 / safe_waste, 0.0)
    
    # Ensure symmetry (it should already be symmetric, but just in case)
    weights = 0.5 * (weights + weights.T)
    
    # Construct the graph Laplacian
    # L = D - W, where D is the degree matrix (diagonal with row sums of W)
    degrees = np.sum(weights, axis=1)
    D = np.diag(degrees)
    L = D - weights
    
    # Find the Fiedler vector (second eigenvector of L)
    # Since n=120, we can use dense eigenvalue decomposition
    # We need the eigenvalues and eigenvectors of L
    # The smallest eigenvalue is 0 (with eigenvector [1,1,...,1])
    # The second smallest eigenvalue's eigenvector is the Fiedler vector
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Sort eigenvalues and eigenvectors (eigh returns them sorted in ascending order)
        # The first eigenvector corresponds to eigenvalue 0
        # The second eigenvector is the Fiedler vector
        
        if n < 2:
            fiedler = np.ones(n)
        else:
            fiedler = eigenvectors[:, 1]
            
            # The sign of the Fiedler vector is arbitrary. 
            # We want to ensure that the heuristic is positive.
            # We'll center the Fiedler vector and use its absolute value or a shifted version.
            # Actually, the directive says "product of these coordinates".
            # If we use the raw Fiedler vector, some values will be negative.
            # The product of two negatives is positive, but product of one negative and one positive is negative.
            # The heuristic must be positive (non-positive becomes 1e-9).
            
            # Let's shift the Fiedler vector to make all values positive.
            # We add the absolute value of the minimum to make the minimum 0, 
            # and then add a small epsilon to avoid zeros.
            fiedler_min = np.min(fiedler)
            if fiedler_min < 0:
                fiedler = fiedler - fiedler_min + 1e-6
            else:
                fiedler = fiedler + 1e-6
            
            # Alternatively, we could use the absolute value, but shifting preserves the relative ordering better.
            
    except Exception:
        # Fallback to a simple heuristic if spectral decomposition fails
        fiedler = d.copy()
    
    # Calculate the heuristic matrix as the outer product of the Fiedler coordinates
    H = np.outer(fiedler, fiedler)
    
    # For non-fitting pairs, set to 1e-9
    H = np.where(fits, H, 1e-9)
    
    # Ensure no non-finite values
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)
        
    # Ensure all values are positive (non-positive becomes 1e-9)
    H = np.where(H > 0, H, 1e-9)
    
    # Ensure shape
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
        
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Return immediately if we are close to the limit
        return H

    return H
