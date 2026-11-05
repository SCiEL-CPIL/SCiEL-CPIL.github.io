<<<<<<< SEARCH
def _smooth_grad(x, beta):
    """Gradient of the smooth objective using FFT-vectorized computation."""
    s = float(np.sum(x))
    if abs(s) < 1e-10:
        return np.zeros_like(x)
    conv = fftconvolve(x, x, mode='full')
    m = float(np.max(conv))
    w = np.exp(beta * (conv - m))
    w_sum = np.sum(w)
    w /= w_sum
    # Gradient of log-sum-exp w.r.t. x[i]:
    # d/dx[i] log(sum_k exp(beta*conv[k])) = beta * sum_k w[k] * d conv[k]/dx[i]
    # conv[k] = sum_j x[j]*x[k-j], so d conv[k]/dx[i] = x[k-i] + x[i-k] (with bounds)
    # Using FFT: the gradient can be computed as beta * (w * x) convolved appropriately
    # More directly: grad_lse[i] = 2 * sum_j w[i+j] * x[j] for the relevant range
    # Let's use the full correlation: grad[i] = 2 * (w ⊛ x)[i] where ⊛ is cross-correlation
    # Actually, d conv[k]/dx[i] = x[k-i] + x[i-k] when both indices are valid
    # So grad_lse = 2 * (w * x) in the appropriate sense. Let me compute it directly:
    # grad_lse[i] = 2 * sum_{j: 0<=j<N, 0<=i+j<2N-1} w[i+j] * x[j]
    # This is a cross-correlation of w and x, but w has length 2N-1.
    # Use FFT for speed:
    W = np.fft.rfft(w, n=_FFT_NP)
    X = np.fft.rfft(x, n=_FFT_NP)
    # cross-correlation: ifft(conj(W) * X) gives sum_j w[j+i] * x[j] (circular)
    # We need linear, so pad appropriately. w has len 2N-1, x has len N.
    # Let's just do it with numpy correlate for correctness, it's fast enough for N=400
    # grad_lse[i] = 2 * sum_j w[i+j] * x[j] for j in [0, N-1], i+j in [0, 2N-2]
    # This is equivalent to: for each i, take w[i:i+N] dot x, where w is zero-padded
    w_padded = np.zeros(2 * N - 1)
    w_padded[:len(w)] = w
    # Use stride tricks or just loop (N=400 is small)
    grad_lse = np.zeros(N)
    for i in range(N):
        grad_lse[i] = 2.0 * float(np.dot(w_padded[i:i + N], x))
    lse = m + np.log(w_sum) / beta
    f = 2.0 * N * lse / (s * s)
    df = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
    return df
=======
def _smooth_grad(x, beta):
    """Gradient of the smooth objective using FFT-vectorized computation."""
    s = float(np.sum(x))
    if abs(s) < 1e-10:
        return np.zeros_like(x)
    
    conv = fftconvolve(x, x, mode='full')
    m = float(np.max(conv))
    w = np.exp(beta * (conv - m))
    w_sum = np.sum(w)
    w /= w_sum
    
    # Compute grad_lse[i] = 2 * sum_j w[i+j] * x[j]
    # This is the cross-correlation of w and x, but w is length 2N-1 and x is length N.
    # Specifically, we want (w * x)_i = sum_j w[i+j] x[j].
    # This is equivalent to the cross-correlation of w and x, but with w "reversed" in a specific way?
    # Let's use FFT.
    # We want C_i = sum_j w_{i+j} x_j.
    # Let W = FFT(w, n=_FFT_NP) and X = FFT(x, n=_FFT_NP).
    # The circular cross-correlation is IFFT(conj(W) * X).
    # Let y = IFFT(conj(W) * X).
    # y_k = sum_j w_j x_{(k-j) mod M} ... this is standard correlation.
    # We need linear cross-correlation.
    # Since w is length 2N-1 and x is length N, the result length is 3N-2? No.
    # w indices: 0..2N-2. x indices: 0..N-1.
    # i ranges 0..N-1.
    # j ranges 0..N-1.
    # i+j ranges 0..2N-2. This matches w's range perfectly.
    # So we are computing a specific slice of the cross-correlation.
    
    # Efficient way:
    # Pad x to length _FFT_NP.
    # Pad w to length _FFT_NP.
    # We want C_i = sum_j w[i+j] x[j].
    # This is the same as the coefficient of z^{-(i+j)} in W(z) X(z^{-1})?
    # Actually, let's just use np.correlate if it's fast enough, but FFT is better.
    
    # Let's try:
    # C = np.correlate(w, x, mode='valid')? 
    # np.correlate(a, v, mode='valid') returns the convolution of a and v, 
    # but only the parts that do not involve the zero-padding.
    # For a of length M and v of length N (M >= N), 'valid' returns M-N+1 elements.
    # Here w is 2N-1, x is N. M = 2N-1, N=N. Result size = N.
    # The k-th element of 'valid' correlation is sum_{j=0}^{N-1} a[k+j] v[j].
    # This is exactly what we want: C_k = sum_j w[k+j] x[j].
    
    # np.correlate is O(N^2) in worst case but highly optimized in C. 
    # However, FFT is O(N log N).
    # Let's use FFT to be safe for larger N or better constants.
    
    # To compute C_i = sum_j w[i+j] x[j] via FFT:
    # This is the cross-correlation of w and x.
    # Let W = rfft(w, n=_FFT_NP)
    # Let X = rfft(x, n=_FFT_NP)
    # The cross-correlation is irfft(conj(W) * X).
    # Let corr = irfft(conj(W) * X).
    # corr[k] = sum_j w[j] x[(k-j) mod M].
    # We want sum_j w[i+j] x[j].
    # Let k = i. Then we want sum_j w[j+i] x[j].
    # In the circular correlation, if we shift indices:
    # Let w' be w shifted?
    
    # Alternative: 
    # C_i = sum_j w[i+j] x[j]
    # This is the i-th element of the cross-correlation of w and x, 
    # where the correlation is defined as R_wx[i] = sum_j w[i+j] x[j].
    # Standard np.correlate(w, x, 'valid') does exactly this if len(w) >= len(x).
    # Let's use np.correlate for simplicity and correctness, it's likely fast enough for N=400.
    # If not, I can implement FFT version.
    
    # Actually, scipy.signal.correlate is available.
    # But let's stick to numpy.
    # For N=400, np.correlate is ~160k ops. Very fast.
    
    # Wait, the previous code used a loop. That was slow.
    # Let's use np.correlate.
    
    # Note: np.correlate(a, v, mode='valid') 
    # a: w (length 2N-1)
    # v: x (length N)
    # Returns array of length (2N-1) - N + 1 = N.
    # Element i is sum_{j=0}^{N-1} w[i+j] x[j].
    # This is exactly grad_lse[i] / 2.
    
    grad_lse_half = np.correlate(w, x, mode='valid')
    grad_lse = 2.0 * grad_lse_half
    
    lse = m + np.log(w_sum) / beta
    f = 2.0 * N * lse / (s * s)
    df = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
    return df
>>>>>>> REPLACE

