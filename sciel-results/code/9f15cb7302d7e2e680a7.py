import numpy as np

N = 400

def construct():
    # Initialize with random noise
    rng = np.random.default_rng(123)
    h = rng.normal(size=N)
    # Normalize sum to 1
    h = h / np.sum(h)
    
    # Parameters
    T = 0.01
    alpha = 0.01
    steps = 2000
    
    for _ in range(steps):
        # Compute autocorrelation
        C = np.convolve(h, h)
        
        # Compute softmax probabilities
        # Subtract max for numerical stability
        C_max = np.max(C)
        exp_C = np.exp((C - C_max) / T)
        sum_exp = np.sum(exp_C)
        p = exp_C / sum_exp
        
        # Compute gradient of LogSumExp w.r.t h
        # grad_m = sum_k p_k * dC_k/dh_m
        # dC_k/dh_m = 2 * h_{k-m} (with boundary checks)
        # This is 2 * corr(p, h)
        
        # np.correlate(p, h, mode='full') gives sum p_i h_{i+k}?
        # Let's use convolution to be safe.
        # We want sum_k p_k h_{k-m}.
        # Let p_rev = p[::-1].
        # conv(p_rev, h)_m = sum_j p_rev_j h_{m-j} = sum_j p_{len(p)-1-j} h_{m-j}.
        # This is getting messy with indices.
        
        # Let's use the property: grad = 2 * (p * h) where * is cross-correlation?
        # Actually, for linear convolution C = h * h, the gradient is 2 * (h * x) where x is the "output" signal?
        # If we view C_k = sum_i h_i h_{k-i}, then C is the convolution of h with h.
        # The gradient of sum_k p_k C_k w.r.t h is the convolution of p with the gradient of C w.r.t h?
        # No.
        # Let's just compute it explicitly for small N or use a known identity.
        # Identity: grad_h (p^T (h*h)) = 2 (p * h) ? No, p is length 2N-1, h is N.
        
        # Let's use the fact that C = h * h.
        # L = sum p_k C_k.
        # dL/dh_m = sum_k p_k (h_{k-m} + h_{m-k})?
        # Since h is real, h_{k-m} = h_{-(m-k)}.
        # This is 2 * sum_k p_k h_{k-m}.
        
        # We can compute this by padding p and h.
        # p is length 2N-1. h is length N.
        # We want q_m = sum_k p_k h_{k-m} for m in 0..N-1.
        # This is the cross-correlation of p and h.
        # np.correlate(p, h, mode='valid' or 'full')?
        # np.correlate(p, h, mode='full')[N-1:N-1+N] ?
        
        # Let's verify:
        # corr(p, h)[k] = sum_i p[i] h[i+k] ? No, np.correlate(p, h) computes sum p[i] conj(h[i+k])?
        # Docs: np.correlate(a, v) = sum a[i] conj(v[i+k])? No.
        # "Correlation of a and v".
        # a[i] * v[i+k]?
        # Let's just use convolution with reversed p.
        # q_m = sum_k p_k h_{k-m}.
        # Let p_rev = p[::-1].
        # conv(p_rev, h)_m = sum_j p_rev[j] h[m-j] = sum_j p[2N-2-j] h[m-j].
        # Let k = 2N-2-j => j = 2N-2-k.
        # sum_k p[k] h[m - (2N-2-k)] = sum_k p[k] h[m - 2N + 2 + k].
        # This shifts the index.
        
        # Simpler:
        # Pad p to length 2N-1 (it already is).
        # Pad h to length 2N-1?
        # We want the central part of the correlation.
        
        # Let's try:
        # grad = 2 * np.convolve(p, h, mode='same')? No, dimensions don't match well.
        
        # Let's use the explicit sum for the gradient vector.
        # grad[m] = 2 * sum_{k} p[k] * h[k-m] where 0 <= k-m < N.
        
        # We can compute this using FFT?
        # Or just loop? N=400, 2N=800. 800*400 = 320k ops. 2000 steps = 640M ops. Might be slow in Python loops.
        # Use NumPy broadcasting?
        # k is 0..2N-2. m is 0..N-1.
        # We want h[k-m].
        # Create matrix M where M[k,m] = h[k-m] if valid else 0.
        # Then grad = 2 * (p[:, None] * M).sum(axis=0).
        
        # This creates a 800x400 matrix. 320k floats. 2.5 MB.
        # 2000 iterations * 2.5 MB = 5 GB memory traffic. Should be okay.
        
        k_idx = np.arange(2*N - 1)
        m_idx = np.arange(N)
        # k_idx[:, None] - m_idx[None, :]
        diffs = k_idx[:, None] - m_idx[None, :]
        
        # Mask for valid indices
        mask = (diffs >= 0) & (diffs < N)
        
        # Create h_matrix where h_matrix[k,m] = h[diffs[k,m]] if valid else 0
        # We can't directly index h with diffs if out of bounds.
        # Use a padded h.
        h_pad = np.zeros(2*N - 1)
        h_pad[N-1:N-1+N] = h # Center h? No, h is 0-indexed.
        # Let's align indices properly.
        # C_k = sum_{i=0}^{N-1} h_i h_{k-i}.
        # Valid i: 0 <= i < N, 0 <= k-i < N => k-N+1 <= i <= k.
        
        # Let's just use the property:
        # grad = 2 * np.convolve(p, h, mode='full') ?
        # Let's test with small N.
        
        # Alternative: Use scipy.signal.correlate
        from scipy.signal import correlate
        # correlate(p, h, mode='full')
        # We want the part corresponding to m.
        
        # Let's try a different, more robust gradient calculation using FFT.
        # C = ifft(fft(h, 2N)**2)
        # grad = 2 * ifft(fft(p, 2N) * fft(h, 2N)) ?
        # This computes the convolution of p and h.
        # conv(p, h)_m = sum_k p_k h_{m-k}.
        # We want sum_k p_k h_{k-m}.
        # This is conv(p, h) with h reversed?
        # conv(p, h_rev)_m = sum_k p_k h_rev_{m-k} = sum_k p_k h_{N-1-(m-k)} = sum_k p_k h_{N-1-m+k}.
        # Still messy.
        
        # Let's stick to the matrix multiplication method, it's safer.
        # To speed it up, we can precompute the mask.
        
        if _ == 0:
            # Precompute mask and indices
            # diffs is (2N-1) x N
            # We need h[diffs] where valid.
            # We can create a 2D array of indices, clip them, and mask out invalids.
            h_indices = np.clip(diffs, 0, N-1)
            # But we need to zero out invalid entries.
            # h_vals = np.take(h, h_indices)
            # h_vals[~mask] = 0
            # This works.
            
        h_indices = np.clip(diffs, 0, N-1)
        h_vals = np.take(h, h_indices)
        h_vals[~mask] = 0.0
        
        grad = 2.0 * np.sum(p[:, None] * h_vals, axis=0)
        
        # Update h
        # Project gradient onto subspace sum(h)=1?
        # Or just subtract mean gradient.
        grad_mean = np.mean(grad)
        h = h - alpha * (grad - grad_mean)
        
        # Normalize
        s = np.sum(h)
        if abs(s) < 1e-10:
            h = rng.normal(size=N)
            s = np.sum(h)
        h = h / s
        
    return h
