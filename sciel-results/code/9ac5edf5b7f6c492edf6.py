import numpy as np

N = 100

def compute_overlaps(h):
    """Compute C_k for all k."""
    C = np.zeros(N)
    for k in range(1, N):
        # h[0:N-k] and h[k:N]
        C[k] = np.sum(h[:N-k] * (1 - h[k:]))
    return C

def project(h):
    """Project h to sum=N/2 and 0<=h<=1."""
    # Simple projection: adjust to sum, then clip
    # This is not a true projection but works for small steps
    diff = N/2 - np.sum(h)
    h += diff / N
    h = np.clip(h, 0, 1)
    # Re-adjust sum after clipping
    diff = N/2 - np.sum(h)
    # Add diff to elements not at bounds
    mask = (h > 0) & (h < 1)
    if np.sum(mask) > 0:
        h[mask] += diff / np.sum(mask)
    # If still not summing correctly due to clipping, do a final clip and accept error or force
    h = np.clip(h, 0, 1)
    return h

def construct():
    h = np.full(N, 0.5)
    step_size = 0.01
    best_C = np.max(compute_overlaps(h))
    
    for _ in range(1000):
        C = compute_overlaps(h)
        k_max = np.argmax(C)
        
        # Gradient of C_k_max
        # C_k = sum_{i=0}^{N-k-1} h_i (1-h_{i+k})
        # dC/dh_j = (1-h_{j+k}) if j+k < N else 0
        #           - h_{j-k} if j-k >= 0 else 0
        
        grad = np.zeros(N)
        k = k_max
        # Term 1: h_j is the first factor
        # h_j * (1-h_{j+k}) -> deriv wrt h_j is (1-h_{j+k})
        # valid for j in 0..N-k-1
        grad[:N-k] += (1 - h[k:])
        
        # Term 2: - h_j * h_{j-k} (from h_{i} where i=j-k, so i+k=j)
        # Actually, let's just use numerical gradient or careful indexing.
        # C_k = sum_{i} h_i - h_i h_{i+k}
        # d/dh_j (sum h_i) = 1
        # d/dh_j (-h_i h_{i+k})
        # if j is the first index: -h_{j+k}
        # if j is the second index: -h_{j-k}
        
        # Let's restart gradient calc properly
        grad = np.zeros(N)
        # Derivative of sum(h[:N-k]) is 1 for indices 0..N-k-1
        grad[:N-k] += 1
        
        # Derivative of -sum(h[:N-k] * h[k:])
        # For j in 0..N-k-1: term is -h_j h_{j+k}. Deriv wrt h_j is -h_{j+k}.
        grad[:N-k] -= h[k:]
        
        # For j in k..N-1: term is -h_{j-k} h_j. Deriv wrt h_j is -h_{j-k}.
        grad[k:] -= h[:N-k]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            break
        grad = grad / grad_norm
        
        # Step
        h_new = h - step_size * grad
        h_new = project(h_new)
        
        C_new = compute_overlaps(h_new)
        max_C_new = np.max(C_new)
        
        if max_C_new < best_C:
            h = h_new
            best_C = max_C_new
        else:
            # Reduce step size or break
            step_size *= 0.9
            if step_size < 1e-5:
                break

    return h
