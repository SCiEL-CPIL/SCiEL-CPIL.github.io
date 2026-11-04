import numpy as np
import time

N = 400
DEADLINE = 140.0  # safety margin below 150s

def _autoconv_max_abs(x):
    """Compute the maximum absolute value of the linear autoconvolution of x."""
    n = len(x)
    size = 2 * n
    f = np.fft.rfft(x, size)
    conv = np.fft.irfft(f * np.conj(f), size)[:n]
    return np.max(np.abs(conv))

def _metric(x):
    """
    C3' metric: max |autoconv| / (sum(x))^2
    For a signed step function with sum S, we minimize max|autoconv| / S^2.
    Note: The seed program uses ones(N), where sum=N and max_autoconv=N.
    Value = N / N^2 = 1/N.
    Score = seed_value / value = (1/N) / (max|autoconv| / S^2) = S^2 / (N * max|autoconv|).
    """
    s = np.sum(x)
    if abs(s) < 1e-15:
        return np.inf
    return _autoconv_max_abs(x) / (s ** 2)

def _spectral_flatten(x0, max_iters=2000, lr=0.05):
    """
    Optimize the spectral magnitude to flatten the autoconvolution.
    We parameterize x via its FFT coefficients. To keep it simple and fast,
    we iterate:
    1. Compute autoconvolution c.
    2. Compute a 'target' spectrum that would produce a flat autoconvolution (all ones).
       The autoconvolution c is the IFFT of |X|^2. We want |X|^2 to be flat.
       So we want |X|^2 = constant. This implies |X| = sqrt(constant).
    3. Update the phase of X to match x0, and set magnitude to a flattened version.
       Actually, a simpler iterative approach:
       Let X = FFT(x).
       We want to minimize max|IFFT(|X|^2)|.
       This is equivalent to making |X|^2 as flat as possible.
       We can use an iterative algorithm similar to Gerchberg-Saxton or just gradient descent on the magnitude.
       
       Alternative: Directly optimize x in the time domain using the fact that
       the autoconvolution is the IFFT of |X|^2.
       Let's try a simple iterative spectral shaping:
       x_new = IFFT( X * (target_mag / |X|) )
       where target_mag is a flattened version of |X|.
    """
    x = x0.copy()
    n = len(x)
    size = 2 * n
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
            
        X = np.fft.rfft(x, size)
        mag = np.abs(X)
        
        # We want to flatten mag.
        # A common technique is to replace mag with a constant or a smoothed version.
        # But we also need to preserve the sum (DC component) to some extent, or just let it go.
        # The metric depends on sum(x)^2. If sum(x) is small, the denominator is small.
        # We want large sum(x) and small max_autoconv.
        
        # Let's try to make mag as flat as possible, but keep the DC term high.
        # Actually, if we make mag flat, the autoconvolution becomes a delta function (ideally).
        # But the DC term (k=0) of |X|^2 corresponds to the sum of squares, not the sum of x.
        
        # Let's use a simpler heuristic:
        # 1. Compute current autoconv c.
        # 2. Compute the peaks of c.
        # 3. Adjust x to reduce the peaks.
        
        # Better approach: Gradient descent on the spectral magnitude.
        # Let m = |X|. We want to minimize max|IFFT(m^2)|.
        # Let's just use a simple iterative replacement:
        # m_new = (m + constant) / 2 ? No.
        
        # Let's try the "flattening" approach:
        # m_target = mean(m) * ones_like(m)
        # But this might reduce the DC term too much.
        
        # Let's stick to the time-domain gradient descent but with a better initialization.
        # The previous program's gradient descent was slow.
        
        # New Idea: Use the fact that for random ±1 sequences, the autoconvolution is roughly a triangle with small ripples.
        # We want to minimize the ripples.
        # This is equivalent to making the sequence "more random" or "less correlated".
        
        # Let's try a different strategy: 
        # 1. Generate a random ±1 sequence.
        # 2. Compute its autoconvolution.
        # 3. Find the lags with high absolute values.
        # 4. Flip signs at positions that reduce these peaks.
        
        # This is essentially what the previous program did, but maybe the previous implementation was flawed.
        
        # Let's try a spectral approach again, but more carefully.
        # We want to minimize max |IFFT(|X|^2)|.
        # Let Y = |X|^2. We want to minimize max |IFFT(Y)|.
        # We can update Y to be flatter.
        # Let Y_new = Y - lr * grad_Y.
        # grad_Y = FFT(sign(IFFT(Y))).
        
        Y = X * np.conj(X)
        c = np.fft.irfft(Y, size)[:n]
        
        # Soft max approximation for max |c|
        tau = 1.0
        weights = np.exp(np.abs(c) / tau)
        weights /= np.sum(weights)
        
        grad_c = weights * np.sign(c)
        
        # grad_Y = fft(grad_c_padded)
        grad_c_padded = np.zeros(size, dtype=complex)
        grad_c_padded[:n] = grad_c
        grad_Y = np.fft.rfft(grad_c_padded)
        
        # Update Y
        Y -= lr * grad_Y
        
        # Ensure Y is real and non-negative (it's a power spectrum)
        Y = np.real(Y)
        Y = np.maximum(Y, 0)
        
        # Recover X from Y, keeping the phase of the original X
        phase = np.angle(X)
        mag = np.sqrt(Y)
        X_new = mag * np.exp(1j * phase)
        
        # Recover x
        x_new = np.fft.irfft(X_new, size)[:n]
        
        # Normalize x to have sum = N (or some target sum) to keep the denominator large
        # Actually, the metric is max|autoconv| / sum(x)^2.
        # If we scale x by alpha, max|autoconv| scales by alpha^2 and sum(x)^2 scales by alpha^2.
        # So the metric is scale invariant!
        # Therefore, we can normalize x arbitrarily.
        # Let's normalize to have unit L2 norm or unit sum.
        # Let's normalize to have sum = N.
        s = np.sum(x_new)
        if abs(s) > 1e-10:
            x_new = x_new / s * N
            
        x = x_new
        
    return x

def _greedy_sign_alignment(x):
    """
    Given a continuous vector x, find a signed step function (±1) that minimizes the metric.
    We use a greedy approach: start with sign(x), then try flipping individual signs to improve the metric.
    """
    x_bin = np.sign(x)
    x_bin[x_bin == 0] = 1
    
    current_val = _metric(x_bin)
    n = len(x_bin)
    
    # Sort positions by magnitude of x, largest first
    # The signs of large magnitude elements are more "confident"
    # But maybe we should try flipping the ones that contribute most to the peaks?
    
    # Simple local search: try flipping each element, keep the best flip.
    # To speed up, we can use the autoconvolution to identify which elements to flip.
    
    # For now, let's do a simple iterative improvement.
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        best_val = current_val
        best_x = x_bin.copy()
        
        # Try flipping each element
        for i in range(n):
            x_bin[i] = -x_bin[i]
            val = _metric(x_bin)
            if val < best_val:
                best_val = val
                best_x = x_bin.copy()
            x_bin[i] = -x_bin[i]  # restore
            
        if best_val < current_val:
            x_bin = best_x
            current_val = best_val
            improved = True
            
    return x_bin

def construct():
    start_time = time.time()
    n = N
    
    best_x = np.ones(n)
    best_val = _metric(best_x)
    
    np.random.seed(42)
    
    # Try several random initializations
    for trial in range(10):
        if time.time() > DEADLINE - 20:
            break
            
        # Random ±1 sequence
        x0 = np.random.choice([-1, 1], size=n).astype(float)
        
        # Spectral flattening optimization
        x_opt = _spectral_flatten(x0, max_iters=500, lr=0.01)
        
        # Greedy sign alignment
        x_bin = _greedy_sign_alignment(x_opt)
        
        val_bin = _metric(x_bin)
        
        if val_bin < best_val:
            best_val = val_bin
            best_x = x_bin.copy()
            
    return best_x
