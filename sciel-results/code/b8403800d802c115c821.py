import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 115.0  # 5s margin
    
    n = 24
    # Evaluation points on unit circle
    N = 512
    theta = 2 * np.pi * np.arange(N) / N
    # A[j, k] = exp(i * theta[j] * k) for k=0..23
    # P(z_j) = sum_k c_k * z_j^k
    # = c @ [1, z_j, z_j^2, ..., z_j^23]
    
    # Precompute the Vandermonde-like matrix
    # V[j, k] = exp(i * theta[j] * k)
    k_range = np.arange(n)
    V = np.exp(1j * np.outer(theta, k_range))  # N x n
    
    # Generate all 2^24 sign patterns in batches
    # Use a Gray code or simple enumeration
    # 2^24 = 16777216
    
    best_val = np.inf
    best_poly = None
    
    batch_size = 20000
    total = 1 << n
    
    # Precompute V.T for matrix multiply: C (batch x n) @ V.T (n x N) -> batch x N
    VT = V.T  # n x N
    
    for start in range(0, total, batch_size):
        if time.time() > deadline:
            break
        
        end = min(start + batch_size, total)
        # Generate sign vectors for indices start to end
        # Convert index to binary, map 0->-1, 1->+1
        idx = np.arange(start, end)
        # Get bits
        bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
        # Map to {-1, 1}
        C = 2 * bits - 1  # batch_size x n
        
        # Evaluate: C @ VT -> batch_size x N
        vals = C @ VT  # complex
        max_vals = np.max(np.abs(vals), axis=1)
        
        # Track best
        min_idx = np.argmin(max_vals)
        if max_vals[min_idx] < best_val:
            best_val = max_vals[min_idx]
            best_poly = C[min_idx].astype(int)
    
    # If we didn't find anything (shouldn't happen), use default
    if best_poly is None:
        # Golay-Rudin-Shapiro fallback
        p, q = [1], [1]
        while len(p) < 24:
            p, q = p + q, p + [-c for c in q]
        return p[:24]
    
    return best_poly.tolist()
