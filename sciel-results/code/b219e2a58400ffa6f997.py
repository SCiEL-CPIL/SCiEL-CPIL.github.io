import numpy as np
from scipy.optimize import minimize
import time

N = 400
_DEADLINE = 150.0
_START_TIME = None

def _fft_correlation(w, h):
    """
    Compute g_i = sum_k w_k * h[k-i] + sum_k w_k * h[i-k]
    which is 2 * cross_correlation(w, h) at lag i, roughly.
    Specifically, term1_i = sum_k w_k h[k-i].
    term2_i = sum_k w_k h[i-k].
    Note term2 is the reverse of term1 if w and h are symmetric? No.
    term1_i = (w * h_rev)[i] ?
    Let's just use np.correlate or FFT.
    """
    # We need g[i] = sum_{k} w[k] * (h[k-i] if 0<=k-i<N else 0 + h[i-k] if 0<=i-k<N else 0)
    # Let's compute c1[i] = sum_{k} w[k] h[k-i]
    # This is the cross correlation of w and h.
    # np.correlate(w, h, mode='full')[i + N - 1] ?
    # Let's use FFT for speed.
    n_fft = 1 << (2 * N - 1).bit_length()
    Fw = np.fft.rfft(w, n=n_fft)
    Fh = np.fft.rfft(h, n=n_fft)
    
    # c1[i] = sum_k w[k] h[k-i]
    # This is conv(w, h_rev) ?
    # h_rev[j] = h[N-1-j]
    # Let's stick to the definition.
    # c1[i] is the correlation.
    # Using FFT:
    # Correlation C[i] = sum_k w[k] h[k-i]
    # C = ifft(fft(w) * conj(fft(h))) ? No, that's circular.
    # For linear correlation, we pad.
    
    # Let's use scipy.signal.correlate which is optimized.
    from scipy.signal import correlate
    # c1[i] = sum_k w[k] h[k-i]
    # This is correlate(w, h, mode='full')[i + N - 1] ?
    # Let's check indices.
    # correlate(w, h, 'full')[k] = sum_i w[i] h[i - k + N - 1]
    # We want sum_k w[k] h[k-i]. Let j=k. sum_j w[j] h[j-i].
    # This matches correlate(w, h, 'full')[i + N - 1] if we shift correctly?
    # Actually, let's just use the FFT method for the gradient of the soft max.
    
    # Gradient of sum_k w_k L_k w.r.t h_i
    # L_k = sum_j h_j h_{k-j}
    # dL_k/dh_i = h_{k-i} + h_{i-k}
    # grad_i = sum_k w_k (h_{k-i} + h_{i-k})
    
    # Part 1: sum_k w_k h_{k-i}
    # Part 2: sum_k w_k h_{i-k}
    
    # Part 1 is cross-correlation of w and h.
    # Part 2 is cross-correlation of w and h reversed?
    # Let h_rev[j] = h[N-1-j].
    # sum_k w_k h[i-k]. Let j = i-k => k = i-j.
    # sum_j w[i-j] h[j].
    # This is the correlation of w and h at lag i?
    
    # Let's compute both parts using FFT.
    # P1_i = sum_k w_k h_{k-i}
    # P2_i = sum_k w_k h_{i-k}
    
    # P1 is corr(w, h)
    # P2 is corr(h, w) ? No.
    
    # Let's use scipy.signal.correlate for reliability.
    c1 = correlate(w, h, mode='full')
    c2 = correlate(h, w, mode='full')
    
    # We need to extract the linear part.
    # c1[k] = sum_j w[j] h[j - k + N - 1]
    # We want P1[i] = sum_k w[k] h[k-i]
    # Let j=k in c1 formula: c1[i] = sum_j w[j] h[j - i + N - 1]
    # This is not quite P1[i].
    
    # Let's just do it with the FFT trick for linear convolution/correlation.
    # P1_i = sum_{k=i}^{i+N-1} w[k] h[k-i]  (assuming h is 0 outside [0, N-1])
    # This is the convolution of w and h? No, correlation.
    
    # Efficient way:
    # grad = correlate(w, h, mode='same') + correlate(h, w, mode='same') ?
    # Let's verify.
    # If w and h are length N.
    # correlate(w, h, 'full') has length 2N-1.
    # The central part corresponds to the linear correlation.
    
    # Let's use the FFT implementation from the inspiration or derive it.
    # grad_i = sum_k w_k (h_{k-i} + h_{i-k})
    
    # Let's compute using FFT.
    # Let W = FFT(w), H = FFT(h).
    # The term sum_k w_k h_{k-i} is the cross-correlation.
    # In frequency domain, cross-correlation is W * conj(H) (for circular).
    # For linear, we need zero padding.
    
    n = 2 * N - 1
    W = np.fft.rfft(w, n=2 * n)
    H = np.fft.rfft(h, n=2 * n)
    
    # Circular cross-correlation of w and h
    # c_circ = ifft(W * conj(H))
    # This gives sum_k w[k] h[(k-i) mod 2n]
    # Since w and h are zero-padded to 2n, and 2n > 2N-1, the linear correlation is contained in the circular one without wrap-around interference for the first 2N-1 terms?
    # Actually, linear correlation of two length N sequences has length 2N-1.
    # If we pad to 2N-1 or more, the circular correlation equals the linear correlation (plus zeros) if the size is >= 2N-1.
    
    c1_circ = np.fft.irfft(W * np.conj(H), n=2 * n)
    # c1_circ[i] = sum_k w[k] h[(k-i) mod 2n]
    # For i in [0, N-1], we want sum_{k=i}^{i+N-1} w[k] h[k-i].
    # This corresponds to indices k-i in [0, N-1].
    # So c1_circ[i] should be the desired P1_i.
    
    # What about P2_i = sum_k w_k h[i-k]?
    # Let j = i-k => k = i-j.
    # sum_j w[i-j] h[j].
    # This is the correlation of w and h at lag i, but with w reversed?
    # It is the same as c1_circ[N-1-i]? No.
    # Let's check c1_circ[N-1-i].
    # c1_circ[N-1-i] = sum_k w[k] h[(k - (N-1-i)) mod 2n]
    # Let m = k - (N-1-i).
    # This is getting confusing.
    
    # Let's just use the property that grad is symmetric if w is symmetric? No.
    
    # Let's use scipy.signal.correlate which is robust.
    # c_full = correlate(w, h, mode='full')
    # c_full[k] = sum_j w[j] h[j - k + N - 1]
    # We want P1_i = sum_k w[k] h[k-i]
    # Let j=k in c_full: c_full[i] = sum_j w[j] h[j - i + N - 1]
    # This is not P1_i.
    
    # P1_i = sum_{k=i}^{i+N-1} w[k] h[k-i]
    # Let u = k-i. P1_i = sum_{u=0}^{N-1} w[u+i] h[u]
    # This is the inner product of w[i:i+N] and h[0:N].
    
    # P2_i = sum_{k=i-N+1}^{i} w[k] h[i-k]  (where valid)
    # Let v = i-k. P2_i = sum_{v=0}^{N-1} w[i-v] h[v]
    # This is the inner product of w[i-N+1:i+1] (reversed) and h[0:N].
    
    # We can compute these using FFT or just loop if N is small? N=400 is small.
    # But we call this in every gradient step.
    # Let's use the FFT method for P1 and P2.
    
    # P1_i = <w[i:], h>
    # P2_i = <w[:i+1][::-1], h>
    
    # Let's compute P1 and P2 using FFT of the full vectors?
    # Actually, `correlate(w, h, mode='full')` gives:
    # c[k] = sum_j w[j] h[j - k + N - 1]
    # For k = 0: sum_j w[j] h[j + N - 1] -> mostly 0
    # For k = N - 1: sum_j w[j] h[j] -> <w, h>
    # For k = N - 1 + i: sum_j w[j] h[j - i]
    # Let j' = j - i. sum_{j'} w[j'+i] h[j'] = P1_i.
    # So P1_i = c_full[N - 1 + i].
    
    # For P2_i:
    # P2_i = sum_{v=0}^{N-1} w[i-v] h[v]
    # Let j = i-v. v = i-j.
    # sum_j w[j] h[i-j]
    # This is c_full[N - 1 - i]?
    # c_full[N - 1 - i] = sum_j w[j] h[j - (N - 1 - i) + N - 1] = sum_j w[j] h[j + i]
    # This is not P2_i.
    
    # P2_i is the correlation of w and h at lag i, but defined as sum w[i-v]h[v].
    # This is the same as P1_i if w and h are symmetric? No.
    # Note that P2_i = P1_i if we swap w and h?
    # P1_i(w,h) = sum w[i+u] h[u]
    # P1_i(h,w) = sum h[i+u] w[u] = sum w[u] h[i+u]
    # Let v = i+u. u = v-i.
    # sum w[v-i] h[v].
    # This is P2_i(w,h) if we reverse the index?
    # P2_i(w,h) = sum w[i-v] h[v].
    # Let u = i-v. v = i-u.
    # sum w[u] h[i-u].
    # This is P1_i(h, w_rev)?
    
    # Let's just use the c_full array.
    # P1_i = c_full[N - 1 + i]
    # P2_i = c_full[N - 1 - i] ?
    # Let's check i=0.
    # P2_0 = sum_{v=0}^{N-1} w[0-v] h[v] = w[0]h[0] + w[-1]h[1]...
    # Only w[0]h[0] is non-zero if we assume w is 0 outside.
    # So P2_0 = w[0]h[0].
    # c_full[N-1] = <w, h> = sum w[j]h[j].
    # c_full[N-1] is not P2_0.
    
    # Okay, let's just compute the gradient using the loop for N=400.
    # 400 iterations of dot product of length 400. 160k ops.
    # This is fast enough in Python if vectorized?
    # No, a Python loop of 400 is slow.
    # But we can vectorize the gradient calculation.
    
    # grad[i] = sum_k w[k] h[k-i] + sum_k w[k] h[i-k]
    
    # Let's construct the gradient using FFT of the weighted convolution.
    # Actually, the inspiration 1 used `_fft_grad_conv`.
    # Let's copy that logic.
    
    n = 2 * N - 1
    Fh = np.fft.rfft(h, n=2 * n)
    Fsw = np.fft.rfft(sw, n=2 * n)
    circ = np.fft.irfft(Fh * Fsw, n=2 * n)[:2 * n]
    # lin[i] = circ[i] + circ[i + n]
    lin = circ[:2 * n - 1] + circ[n:2 * n]
    # lin is the linear convolution of sw and h.
    # (sw * h)[k] = sum_i sw[i] h[k-i]
    # We want sum_k sw[k] h[k-i].
    # This is the cross-correlation.
    # corr(i) = conv(sw, h_rev)(i + N - 1)
    
    # Let's just use the loop for the gradient, it's safer.
    # But we can make it faster.
    
    return None # Placeholder

def _objective_hard(x):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    return abs(2 * N * m / (s * s))

def _soft_max_obj(x, temp=0.05):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    exp_terms = np.exp((conv - m) / temp)
    logsumexp = m / temp + np.log(np.sum(exp_terms))
    smax = temp * logsumexp
    return smax / (s * s)

def _soft_max_grad(x, temp=0.05):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(x)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    exp_terms = np.exp((conv - m) / temp)
    softmax_weights = exp_terms / np.sum(exp_terms)
    sw = softmax_weights
    
    # Compute gradient efficiently
    # g[i] = sum_k sw[k] (h[k-i] + h[i-k])
    # Use FFT for linear correlation
    n = 2 * N - 1
    # We need corr1[i] = sum_k sw[k] h[k-i]
    # and corr2[i] = sum_k sw[k] h[i-k]
    
    # corr1 is cross-correlation of sw and h.
    # corr2 is cross-correlation of h and sw.
    
    # Using scipy.signal.correlate
    from scipy.signal import correlate
    c1 = correlate(sw, h, mode='full')
    c2 = correlate(h, sw, mode='full')
    
    # Extract linear parts
    # c1[k] = sum_j sw[j] h[j - k + N - 1]
    # We want corr1[i] = sum_k sw[k] h[k-i]
    # Let j=k. c1[i] = sum_j sw[j] h[j - i + N - 1]
    # This is not corr1[i].
    
    # Let's use the FFT method from the diff in the prompt.
    # The diff had:
    # circ = np.fft.irfft(Fh * Fsw, n=2 * n)[:2 * n]
    # lin = circ[:2 * n - 1] + circ[n:2 * n]
    # return lin[:N]
    # This computes the linear convolution of h and sw.
    # (h * sw)[k] = sum_i h[i] sw[k-i]
    # We want sum_k sw[k] h[k-i].
    # Let j = k-i. k = j+i.
    # sum_j sw[j+i] h[j]
    # This is the correlation.
    
    # Let's just use the loop for N=400. It's only 400*400 = 160k ops.
    # In Python, this might take ~1ms. 1000 iterations = 1s. Acceptable.
    
    grad = np.zeros(N)
    for i in range(N):
        # First sum: sum_{k=i}^{i+N-1} sw[k] h[k-i]
        # sw slice: sw[i : i+N]
        # h slice: h[0 : N]
        # But sw has length 2N-1.
        # Valid k for h[k-i]: 0 <= k-i < N => i <= k < i+N
        # Also 0 <= k < 2N-1.
        k_start = i
        k_end = min(i + N, 2 * N - 1)
        if k_start < k_end:
            grad[i] += np.dot(sw[k_start:k_end], h[0:k_end - k_start])
        
        # Second sum: sum_{k=max(0, i-N+1)}^{i} sw[k] h[i-k]
        # Valid k for h[i-k]: 0 <= i-k < N => i-N < k <= i
        k_start2 = max(0, i - N + 1)
        k_end2 = i + 1
        if k_start2 < k_end2:
            # sw slice: sw[k_start2 : k_end2]
            # h slice: h[i-k]. If k=k_start2, h[i-k_start2].
            # Let j = i-k. As k goes k_start2 to i, j goes i-k_start2 down to 0.
            # So we need h[i-k_start2 : -1 : -1] ?
            # Or just reverse the sw slice?
            # sum sw[k] h[i-k]
            # Let u = i-k.
            # sum sw[i-u] h[u]
            # This is dot(sw[i-k_start2 : -1 : -1], h[0 : i-k_start2+1])
            # sw indices: i, i-1, ..., i-(i-k_start2) = k_start2
            # sw_slice = sw[k_start2 : i+1][::-1]
            sw_slice = sw[k_start2 : i + 1][::-1]
            h_slice = h[0 : i - k_start2 + 1]
            grad[i] += np.dot(sw_slice, h_slice)
    
    smax = temp * (m / temp + np.log(np.sum(exp_terms)))
    d_obj = grad / (s * s) - 2 * smax / (s * s * s)
    
    return d_obj

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_val = 2.0
    best_h = np.ones(N)
    
    deadline = _START_TIME + _DEADLINE - 5.0
    
    # Diverse initializations
    inits = []
    
    # Random
    for i in range(10):
        r = np.random.RandomState(1000 + i * 137)
        inits.append(r.randn(N))
        
    # DCT-based random spectral profiles
    # Generate random amplitudes in frequency domain, inverse DCT
    for i in range(10):
        r = np.random.RandomState(2000 + i * 137)
        coeffs = r.randn(N)
        # Scale to keep sum non-zero
        h = np.real(np.fft.irfft(coeffs, n=N))
        inits.append(h)
        
    # Low frequency dominance
    for i in range(5):
        r = np.random.RandomState(3000 + i * 137)
        coeffs = r.randn(N)
        # Zero out high frequencies
        coeffs[N//2:] = 0
        h = np.real(np.fft.irfft(coeffs, n=N))
        inits.append(h)
        
    # Linear ramps
    inits.append(np.linspace(-1, 1, N))
    inits.append(np.linspace(1, -1, N))
    
    # Step functions
    for frac in [0.25, 0.5, 0.75]:
        h = np.zeros(N)
        h[:int(frac * N)] = 1.0
        h[int(frac * N):] = -0.3
        inits.append(h - np.mean(h))

    for init_h in inits:
        if time.time() > deadline - 3:
            break
        
        h = init_h.astype(float).copy()
        s = np.sum(h)
        if abs(s) < 1e-12:
            h += 0.01 * np.random.randn(N)
            s = np.sum(h)
            if abs(s) < 1e-12:
                continue
        
        # Multi-temperature annealing
        temps = [0.1, 0.05, 0.02, 0.01]
        current_h = h.copy()
        
        for temp in temps:
            if time.time() > deadline - 3:
                break
            
            try:
                res = minimize(
                    lambda x, t=temp: _soft_max_obj(x, t),
                    current_h,
                    jac=lambda x, t=temp: _soft_max_grad(x, t),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 500}
                )
                
                new_val = _objective_hard(res.x)
                if new_val < best_val:
                    best_val = new_val
                    best_h = res.x.copy()
                
                if res.fun < _soft_max_obj(current_h, temp):
                    current_h = res.x.copy()
            except Exception:
                pass
    
    # Final polish
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _soft_max_obj(x, 0.001),
                best_h,
                jac=lambda x: _soft_max_grad(x, 0.001),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
            )
            val = _objective_hard(res.x)
            if val < best_val:
                best_val = val
                best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h
