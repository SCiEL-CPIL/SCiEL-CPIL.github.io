import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def _objective_and_grad(h):
    """Compute objective value and gradient w.r.t. h (assuming h >= 0, sum(h)=1).
    We work on the simplex, so gradient is projected.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0, np.zeros_like(h)
    
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    
    if conv_inf < 1e-15:
        return 0.0, np.zeros_like(h)
    
    value = h2_sq / conv_inf
    
    # Gradient of value w.r.t. h
    # value = (sum(h^2)/l1^2) / max(conv)
    # Let f = sum(h^2)/l1^2, g = max(conv)
    # dvalue = (df*g - f*dg) / g^2
    # df/dh_i = (2*h_i * l1 - sum(h^2)*2*h_i/l1) / l1^2 = 2*h_i*(l1^2 - sum(h^2))/(l1^3)
    #          = 2*h_i * (l1^2 - l1^2*sum(h_norm^2)) / l1^3 = 2*h_norm_i * (1 - h2_sq) / l1
    # Wait, let me redo this more carefully.
    
    # f = sum(h^2) / l1^2
    # df/dh_i = 2*h_i/l1^2 - sum(h^2)*2*h_i/l1^3 = 2*h_i*(l1 - sum(h^2)/l1)/l1^2 = 2*h_i*(l1 - f*l1)/l1^2
    # Actually: df/dh_i = 2*h_i/l1^2 - 2*h_i*sum(h^2)/l1^3 = 2*h_i*(l1 - sum(h^2)/l1)/l1^2
    # Let me just use: f = sum(h_norm^2), and h_norm = h/l1
    # df/dh_i = 2*h_norm_i / l1 (since h2_sq = sum(h_norm^2), d(h2_sq)/d(h_norm_i) = 2*h_norm_i, and d(h_norm_i)/d(h_i) = 1/l1)
    
    # For g = max(conv), the gradient is sparse: only at the argmax of conv.
    # conv[k] = sum_j h_norm[j] * h_norm[k-j] (with appropriate indexing)
    # d(conv[k])/d(h_norm[i]) = h_norm[k-i] + h_norm[i] (if indices are valid)
    # But since we take max, we only care about the argmax index.
    
    # Find argmax of conv
    k_max = np.argmax(conv)
    
    # Gradient of conv[k_max] w.r.t. h_norm
    grad_conv_hnorm = np.zeros(N)
    # conv[k_max] = sum_{j} h_norm[j] * h_norm[k_max - j] where h_norm is defined on [0, N-1]
    # So d(conv[k_max])/d(h_norm[i]) = h_norm[k_max - i] (if 0 <= k_max-i < N) + h_norm[i] (if 0 <= i < N, which is always true)
    # Wait, conv is full mode: conv[m] = sum_{j} a[j] * b[m-j]
    # For autocorrelation: conv[m] = sum_{j} h_norm[j] * h_norm[m-j]
    # d(conv[m])/d(h_norm[i]) = h_norm[m-i] (if 0<=m-i<N) + h_norm[i] (if 0<=i<N, but we need m-i to be valid for the first term, and i to be valid for the second)
    # Actually: conv[m] = sum_{j=0}^{N-1} h_norm[j] * h_norm[m-j] where h_norm[k]=0 if k<0 or k>=N
    # d(conv[m])/d(h_norm[i]) = h_norm[m-i] (from j=i term) + h_norm[i] (from j=m-i term, if m-i != i)
    # If m-i == i, i.e., m=2i, then it's 2*h_norm[i]
    
    # Let's compute directly
    for i in range(N):
        j1 = m_idx = k_max - i
        if 0 <= j1 < N:
            grad_conv_hnorm[i] += h_norm[j1]
        # The second term: when j = k_max - i, we have h_norm[k_max-i] * h_norm[i], so derivative w.r.t. h_norm[i] is h_norm[k_max-i]
        # But this is the same as above! Wait no.
        # conv[m] = sum_j h[j] h[m-j]
        # d/dh[i] = h[m-i] (from j=i) + h[i] (from j=m-i, if m-i != i)
        # If m-i == i, then it's 2*h[i]
        # So:
        j2 = k_max - i
        if 0 <= j2 < N and j2 != i:
            grad_conv_hnorm[i] += h_norm[i]
        # If j2 == i, we already counted h_norm[j1] = h_norm[i] once, need to add another h_norm[i]
        # Actually my loop above adds h_norm[j1] where j1=k_max-i. That's the j=i term.
        # The j=m-i term gives h_norm[i] if m-i is a valid index AND m-i != i.
        # If m-i == i, then j=i and j=m-i are the same, so we only have one term, and derivative is 2*h[i]? No.
        # Let me just compute it numerically-safe:
    
    # Actually, let me just do it cleanly:
    grad_conv_hnorm = np.zeros(N)
    for i in range(N):
        # Term 1: j = i, contributes h_norm[k_max - i] if valid
        j1 = k_max - i
        if 0 <= j1 < N:
            grad_conv_hnorm[i] += h_norm[j1]
        # Term 2: j = k_max - i, contributes h_norm[i] if valid and j != i
        j2 = k_max - i
        if 0 <= j2 < N and j2 != i:
            grad_conv_hnorm[i] += h_norm[i]
        # If j2 == i, the two terms are the same, so we should have 2*h_norm[i]
        # But in my loop, if j1==j2==i, I added h_norm[j1] = h_norm[i] once.
        # I need to add h_norm[i] again.
        if j2 == i and 0 <= j2 < N:
            grad_conv_hnorm[i] += h_norm[i]
    
    # Now convert gradient from h_norm to h: d(conv)/d(h_i) = d(conv)/d(h_norm_i) * d(h_norm_i)/d(h_i) = grad_conv_hnorm[i] / l1
    grad_g_h = grad_conv_hnorm / l1
    
    # Gradient of f = h2_sq = sum(h_norm^2) w.r.t. h
    grad_f_h = 2.0 * h_norm / l1
    
    # Gradient of value = f/g
    grad_value_h = (grad_f_h * conv_inf - f * grad_g_h) / (conv_inf ** 2)
    
    # Project to simplex: we want to maximize, so we do gradient ascent
    # But we need to stay on the simplex (sum=1, h>=0)
    # For simplicity, return the gradient and let the caller handle projection
    
    return value, grad_value_h

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Initialize with a smooth profile
    # Try a few different initializations and pick the best
    best_h = None
    best_score = 0.0
    
    # Initialization 1: Gaussian-like
    x = np.linspace(-1, 1, N)
    h1 = np.exp(-x**2 * 5)
    h1 = np.maximum(h1, 0)
    h1 = h1 / np.sum(h1)
    
    # Initialization 2: Uniform
    h2 = np.ones(N) / N
    
    # Initialization 3: Triangular
    h3 = np.maximum(0, 1 - np.abs(x) * 2)
    h3 = h3 / np.sum(h3)
    
    # Initialization 4: Sparse - few peaks
    h4 = np.zeros(N)
    h4[::5] = 1.0  # Every 5th element
    h4 = h4 / np.sum(h4)
    
    # Initialization 5: Sidon-like - use a quadratic residue pattern
    h5 = np.zeros(N)
    # Use positions that are quadratic residues mod p, mapped to [0, N)
    p = 53  # prime > N
    residues = set()
    for r in range(p):
        residues.add((r*r) % p)
    positions = sorted([r for r in residues if r < N])
    if len(positions) > 0:
        h5[positions] = 1.0
        h5 = h5 / np.sum(h5)
    
    candidates = [h1, h2, h3, h4, h5]
    
    for h_init in candidates:
        if time.time() > deadline - 5:
            break
        score = compute_ratio(h_init)
        if score > best_score:
            best_score = score
            best_h = h_init.copy()
    
    if best_h is None:
        return list(np.ones(N) / N)
    
    # Gradient ascent with projection to simplex
    h = best_h.copy()
    lr = 0.1
    for step in range(2000):
        if time.time() > deadline - 5:
            break
        
        value, grad = _objective_and_grad(h)
        
        # Project gradient to simplex: subtract mean to ensure sum of update is 0
        grad_centered = grad - np.mean(grad)
        
        # Update
        h_new = h + lr * grad_centered
        h_new = np.maximum(h_new, 0)
        
        # Renormalize to sum=1
        s = np.sum(h_new)
        if s < 1e-15:
            break
        h_new = h_new / s
        
        new_score = compute_ratio(h_new)
        
        if new_score > value:
            h = h_new
            # Decrease learning rate slightly
            lr *= 0.999
        else:
            # Try smaller step
            lr *= 0.5
            if lr < 1e-6:
                break
    
    final_score = compute_ratio(h)
    
    # If we improved, return h; otherwise return best_h
    if final_score > best_score:
        return list(h)
    else:
        return list(best_h)

