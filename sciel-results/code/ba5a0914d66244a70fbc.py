import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    demands = np.array(demands)
    
    # 1. Construct Weight Matrix based on fit quality
    # We want to strongly connect items that fit well together (low residual)
    # W_ij = exp(-alpha * residual) if fits, else 0
    sums = demands[:, None] + demands[None, :]
    residuals = capacity - sums
    fit_mask = residuals >= 0
    
    # Alpha controls how sensitive the weight is to the residual space
    # Larger alpha -> stronger preference for tight fits
    alpha = 1.0 / capacity  # Normalize to be scale-invariant
    
    # Calculate weights
    # For fitting pairs, weight is exp(-alpha * residual)
    # For non-fitting, weight is 0
    # Use safe residuals to avoid exp of positive (if residual < 0, we mask it anyway)
    W = np.zeros((n, n))
    # Only compute exp for fitting pairs to save time and avoid overflow/underflow issues
    # We can use np.where but exp of large positive is bad, so we mask first
    # residuals can be up to capacity (150). alpha * 150 = 150/150 = 1. exp(-1) is fine.
    # If alpha was 1, exp(-150) is tiny. 
    # Let's use a fixed alpha to ensure variation.
    # If residual is 0 (perfect fit), W=1. If residual is 100, W=exp(-100/150) ~ 0.5.
    
    # To avoid numerical issues with exp on the whole matrix, we apply it only where fit_mask is True
    # But vectorized is easier.
    # We need to be careful: exp(-alpha * residual) for residual < 0 would be > 1.
    # But we only care about fit_mask.
    
    safe_residuals = np.where(fit_mask, residuals, 0)
    W = np.where(fit_mask, np.exp(-alpha * safe_residuals), 0.0)
    
    # Symmetric, diagonal 0
    np.fill_diagonal(W, 0.0)
    
    # 2. Compute Normalized Laplacian
    # L_sym = I - D^(-1/2) W D^(-1/2)
    degrees = W.sum(axis=1)
    
    # Handle isolated nodes (degree 0) to avoid division by zero
    # If a node has degree 0, it's isolated. Its Laplacian row/col should be identity-like?
    # In normalized Laplacian, if degree is 0, D^(-1/2) is undefined.
    # Standard approach: treat isolated nodes as having eigenvalue 0 or 1?
    # For clustering, isolated nodes are their own cluster.
    # We can set D_inv_sqrt to 0 for zero degrees, which makes the corresponding row/col of the 
    # similarity matrix 0, so L = I - 0 = I for those nodes.
    # This means eigenvalue 1. The Fiedler vector (2nd smallest) will likely be 0 for isolated nodes 
    # if there are enough connected nodes, or they will be part of the null space if there are many.
    # Given 120 items and capacity 150, min demand 20, max 99.
    # Min sum = 40, Max sum = 198.
    # Most items will have many connections. Isolated nodes are unlikely unless all other items are > 150 - d_i.
    # Since max demand is 99, min demand 20.
    # If d_i = 99, need d_j <= 51.
    # If d_i = 20, need d_j <= 130 (all fit).
    # So small items connect to everyone. Large items connect to small/medium.
    # Isolated nodes are extremely unlikely. But we handle it just in case.
    
    D_inv_sqrt = np.zeros_like(degrees)
    non_zero = degrees > 0
    D_inv_sqrt[non_zero] = 1.0 / np.sqrt(degrees[non_zero])
    
    # Similarity matrix S = D^(-1/2) W D^(-1/2)
    S = W * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]
    
    # L_sym = I - S
    L_sym = np.eye(n) - S
    
    # 3. Eigendecomposition
    # We only need the 2nd smallest eigenvalue's eigenvector.
    # np.linalg.eigh returns sorted ascending.
    # For n=120, full eigh is fast enough.
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    
    # Fiedler vector is index 1
    fiedler = eigenvectors[:, 1]
    
    # 4. Heuristic Matrix
    # We want H_ij to be high if i and j are close in the Fiedler space.
    # H_ij = exp(-beta * (fiedler_i - fiedler_j)^2)
    # This creates a Gaussian kernel on the spectral embedding.
    
    # Beta controls the width of the attraction.
    # If beta is too small, everything is attracted to everything.
    # If beta is too large, only identical coordinates attract.
    # Let's normalize the Fiedler vector to have variance 1 or range [0,1] to pick a stable beta.
    
    # Center and scale fiedler
    f_mean = np.mean(fiedler)
    f_std = np.std(fiedler)
    if f_std > 1e-9:
        f_norm = (fiedler - f_mean) / f_std
    else:
        f_norm = np.zeros(n)
        
    # Calculate squared distances in 1D
    diffs = f_norm[:, None] - f_norm[None, :]
    sq_diffs = diffs * diffs
    
    # Beta selection:
    # If sq_diffs is ~1 (unit variance), exp(-1) is 0.37.
    # We want a reasonable spread.
    beta = 2.0
    
    H = np.exp(-beta * sq_diffs)
    
    # Apply fitting mask
    # If they don't fit, heuristic should be very low (1e-9)
    H = np.where(fit_mask, H, 1e-9)
    
    # Ensure diagonal is 0 or low? Ants don't pick the same item twice usually, 
    # but if they do, it shouldn't be attractive. 
    # The problem says "placing item j right after item i". j != i usually.
    # But the matrix is n x n.
    # Let's set diagonal to 1e-9 to be safe.
    np.fill_diagonal(H, 1e-9)
    
    # Ensure no non-finite values
    H = np.nan_to_num(H, nan=1e-9, posinf=1.0, neginf=1e-9)
    
    # Ensure positive
    H = np.where(H > 0, H, 1e-9)
    
    return H
