"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Efficient Gradient Descent with Projection.
We minimize the peak of the autoconvolution $M = \max(f * f)$ subject to
$f \ge 0$ and $\sum f = 1$.
We use an auxiliary variable $M$ and minimize a loss function:
$L(f, M) = M + \lambda \sum_k \max(0, (f*f)_k - M)^2$
The gradient with respect to $f$ is computed efficiently using FFT.
We project $f$ onto the simplex $\{f \ge 0, \sum f = 1\}$ at each step.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def project_simplex(v):
    """Project vector v onto the probability simplex {x >= 0, sum(x) = 1}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def autoconvolve_fft(f):
    """Compute linear autoconvolution using FFT for O(N log N) speed."""
    # Need size at least 2*N - 1 for linear convolution
    size = 2 * N
    F = np.fft.rfft(f, size)
    conv = np.fft.irfft(F * F, size)
    return conv[:2 * N - 1]

def gradient_step(f, M, lambda_penalty=10.0):
    """
    Compute gradient of the penalty term P = sum(max(0, conv - M)^2).
    dP/df = 2 * sum_{j} max(0, conv_j - M) * (d conv_j / d f_i)
    conv = f * f.
    d conv_j / d f_i = f_{j-i} + f_{j-i} (if j-i != i, else 2*f_i? No, derivative of sum f_k f_{j-k} wrt f_i is f_{j-i} + f_{j-i} if i != j-i, else 2 f_i.
    Actually, conv_j = sum_{k=0}^j f_k f_{j-k}.
    d conv_j / d f_i = f_{j-i} (if j-i in [0, N-1]) + f_{j-i} (if j-i in [0, N-1])?
    Let's use the correlation form.
    Convolution is commutative.
    Gradient of f*f wrt f is 2 * (f * f) reversed? No.
    Let g = f * f.
    dg_i = sum_j (df_j * f_{i-j} + f_j * df_{i-j})
    This is equivalent to 2 * (f * f) in the context of linear algebra if we consider the quadratic form?
    Actually, for linear convolution, the gradient of 0.5 * ||f*f||^2 is not simply f*f.
    But we are looking at the gradient of sum max(0, conv_k - M)^2.
    Let h_k = max(0, conv_k - M).
    d L / d f_i = sum_k 2 h_k * (d conv_k / d f_i).
    d conv_k / d f_i = f_{k-i} + f_{k-i} ? No.
    conv_k = sum_{m=0}^{N-1} f_m f_{k-m}.
    d conv_k / d f_i = f_{k-i} (if 0 <= k-i < N) + f_{k-i} (if 0 <= k-i < N)?
    Wait, if m=i, term is f_i f_{k-i}. Derivative wrt f_i is f_{k-i}.
    If k-m=i => m=k-i, term is f_{k-i} f_i. Derivative wrt f_i is f_{k-i}.
    So if i != k-i, derivative is 2 f_{k-i}.
    If i == k-i (i.e., k=2i), derivative is 2 f_i.
    So d conv_k / d f_i = f_{k-i} + f_{k-i} = 2 f_{k-i} (with appropriate bounds).
    Basically, the gradient vector G_i = 2 * sum_k h_k f_{k-i}.
    This is 2 * (h * f) where * is linear convolution, but we need to be careful with indices.
    Let H be the vector h.
    G = 2 * (H * f) ? No, it's the cross-correlation or convolution?
    sum_k h_k f_{k-i}. Let j = k-i => k = j+i.
    sum_j h_{j+i} f_j.
    This is the cross-correlation of h and f.
    Cross-correlation can be computed via FFT.
    """
    # h_k = max(0, conv_k - M)
    conv = autoconvolve_fft(f)
    h = np.maximum(conv - M, 0.0)
    
    # We need G_i = 2 * sum_k h_k * f_{k-i}
    # This is equivalent to 2 * (h * f) ? 
    # Convolution (h * f)_i = sum_j h_j f_{i-j}.
    # Our sum is sum_k h_k f_{k-i}. Let j = k-i => k = j+i.
    # sum_j h_{j+i} f_j.
    # This is the cross-correlation C_{hf}(i) = sum_j h(j+i) f(j).
    # Cross-correlation can be computed as ifft(fft(h) * conj(fft(f))) ?
    # Let's just use np.correlate for simplicity if N is small enough? N=600, 2N=1200.
    # np.correlate(h, f, mode='full') gives sum h_j f_{j+i} ?
    # Documentation: c_k = sum_n a_n * conj(b_{n-k})
    # We want sum_k h_k f_{k-i}.
    # This is c_{-i} of correlate(f, h)?
    # Let's just compute it using FFT to be safe and fast.
    # G = 2 * ifft( conj(fft(h)) * fft(f) ) ?
    # Let's test:
    # ifft( H_freq * F_freq )_i = sum_k h_k f_{(i-k) mod N}
    # We want linear correlation.
    # Simpler: Just use np.convolve(h, f[::-1], 'full')?
    # Let's stick to the explicit formula for gradient of quadratic form if possible, but here it's max(0, .)^2.
    
    # Efficient way:
    # G = 2 * np.correlate(h, f, mode='same') ? 
    # Let's verify dimensions. h has size 2N-1. f has size N.
    # We need G of size N.
    # G_i = 2 * sum_{k=0}^{2N-2} h_k f_{k-i} (where f_j=0 if j not in [0, N-1])
    
    # Using FFT:
    size = 2 * N
    H = np.fft.rfft(h, size)
    F = np.fft.rfft(f, size)
    # We want the cross-correlation sum_k h_k f_{k-i}
    # This is ifft( conj(H) * F ) ?
    # Let's check: ifft( conj(H) * F )_i = sum_k h_k f_{i-k} ? No.
    # Standard correlation: xcorr(x, y) = ifft( X * conj(Y) )
    # We want sum_k h_k f_{k-i}.
    # Let y = f. We want sum_k h_k y_{k-i}.
    # This is the correlation of h and y at lag i?
    # Correlation R_{hy}(i) = sum_k h(k) y(k-i).
    # So G_i = 2 * R_{hy}(i).
    # R_{hy} can be computed as ifft( H * conj(Y) )?
    # Yes, if we use circular correlation and pad sufficiently.
    
    R_hy = np.fft.irfft(np.conj(H) * F, size)
    # R_hy[i] corresponds to sum_k h[k] f[(i-k) mod size]
    # We want sum_k h[k] f[k-i].
    # Note that R_hy[i] = sum_k h[k] f[i-k].
    # We want C[i] = sum_k h[k] f[k-i].
    # C[i] = R_hy[-i] ?
    # R_hy[-i] = sum_k h[k] f[-i-k] = sum_k h[k] f[-(i+k)].
    # Due to symmetry of real signals and linear convolution bounds, we can just take the central part.
    # Actually, for linear convolution/correlation with zero padding,
    # The output of ifft(conj(H)*F) has the correlation centered.
    # Let's just compute the gradient numerically? No, too slow.
    
    # Let's use the property: Gradient of sum_k max(0, (f*f)_k - M)^2 wrt f
    # is 2 * (h * f) where * is linear convolution?
    # Let's check d/d f_i sum_k h_k (f*f)_k
    # = sum_k h_k (f_{k-i} + f_{k-i}) = 2 sum_k h_k f_{k-i}
    # This is 2 * (h * f)_{?}
    # Convolution (h * f)_i = sum_j h_j f_{i-j}.
    # Our sum is sum_k h_k f_{k-i}.
    # Let j = k-i => k = j+i.
    # sum_j h_{j+i} f_j.
    # This is (h * f)_{-i}? No.
    # (h * f)_{-i} = sum_j h_j f_{-i-j}.
    # Let m = -i-j => j = -i-m.
    # sum_m h_{-i-m} f_m.
    # If h is symmetric (which it is, since h = max(0, conv-M) and conv is symmetric), h_{-i-m} = h_{i+m}.
    # Then sum_m h_{i+m} f_m.
    # This matches our sum sum_j h_{j+i} f_j.
    # So the gradient is 2 * (h * f)_{-i} ?
    # Since h is symmetric, (h * f) is not necessarily symmetric if f is not.
    # But we can just compute the convolution h * f and take the appropriate slice.
    # The vector G_i = 2 * sum_k h_k f_{k-i}.
    # Let conv_hf = np.convolve(h, f, 'full').
    # conv_hf[k] = sum_j h_j f_{k-j}.
    # We want G_i = 2 * sum_j h_{j+i} f_j.
    # Let k = j+i => j = k-i.
    # G_i = 2 * sum_k h_k f_{k-i} ? No, the sum is over j.
    # G_i = 2 * sum_j h_{j+i} f_j.
    # This is the cross-correlation of h and f.
    # np.correlate(h, f, 'full') gives c_k = sum_j h_j f_{j-k}.
    # We want sum_j h_{j+i} f_j. Let m = j+i => j = m-i.
    # sum_m h_m f_{m-i}.
    # This is c_{-i} of correlate(h, f)?
    # c_{-i} = sum_j h_j f_{j-(-i)} = sum_j h_j f_{j+i}.
    # Yes! So G_i = 2 * c_{-i}.
    # The output of np.correlate(h, f, 'full') has length len(h)+len(f)-1.
    # The index 0 corresponds to c_0.
    # c_k is at index k + (len(f)-1) ? No, 'full' output indexing is tricky.
    # For 'full', output[k] = sum_n h[n] f[n-k] ? No.
    # numpy doc: c_k = sum_n a[n] * conj(b[n-k])
    # So output[k] = c_k.
    # We want G_i = 2 * c_{-i}.
    # c_{-i} is at index -i.
    # In 'full' mode, the index 0 is at position len(f)-1.
    # So c_0 is at index N-1.
    # c_{-i} is at index N-1 - i.
    
    # So we compute c = np.correlate(h, f, 'full')
    # Then G = 2 * c[N-1 : N-1 - N : -1] ?
    # Indices for i=0..N-1:
    # i=0 -> idx N-1
    # i=1 -> idx N-2
    # ...
    # i=N-1 -> idx 0
    # So G = 2 * c[N-1::-1]
    
    c = np.correlate(h, f, mode='full')
    G = 2.0 * c[N-1::-1]
    
    return G, M

def optimize(f_init, max_time=10.0):
    """Run gradient descent on f and M."""
    start_time = time.time()
    f = f_init.copy()
    s = np.sum(f)
    if s > 0:
        f = f / s
    M = np.max(autoconvolve_fft(f))
    
    lr_f = 1.0 / (N * 0.1) # Initial learning rate
    lr_M = 1.0 / (N * 0.1)
    lambda_penalty = 10.0
    
    best_score = compute_score(f)
    best_f = f.copy()
    
    while time.time() - start_time < max_time:
        # Compute gradient
        G_f, _ = gradient_step(f, M, lambda_penalty)
        
        # Update M: M should be >= max(conv). 
        # Gradient of M term is 1. Gradient of penalty is 0 if M > max(conv).
        # So if M > max(conv), decrease M. Else increase M.
        # Simple subgradient:
        max_conv = np.max(autoconvolve_fft(f))
        if M > max_conv:
            M -= lr_M
        else:
            M += lr_M * 0.5 # Slow down increase
        
        # Update f
        f_new = f - lr_f * G_f
        
        # Project to simplex
        f_new = project_simplex(f_new)
        
        # Ensure non-negative and normalized
        f_new = np.maximum(f_new, 0)
        s_new = np.sum(f_new)
        if s_new > 0:
            f_new = f_new / s_new
        else:
            break
            
        f = f_new
        
        # Check improvement
        current_score = compute_score(f)
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
            
        # Adaptive learning rate
        # If score increases, reduce lr
        if current_score > best_score * 1.01:
            lr_f *= 0.5
            lr_M *= 0.5
        else:
            lr_f *= 1.01
            lr_M *= 1.01
            
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Box (centered)
    for width in [0.2, 0.4, 0.6, 0.8]:
        box = np.zeros(N)
        start = int(N * (1 - width) / 2)
        end = int(N * (1 + width) / 2)
        box[start:end] = 1.0
        inits.append(box)
        
    # 3. Gaussian-like
    x = np.linspace(-1, 1, N)
    for sigma in [0.2, 0.4, 0.6]:
        gauss = np.exp(-x**2 / (2 * sigma**2))
        inits.append(gauss)
        
    # 4. Random positive
    for _ in range(5):
        rand = np.random.rand(N)
        inits.append(rand)
        
    # 5. Random Box
    for _ in range(5):
        box = np.zeros(N)
        w = np.random.uniform(0.2, 0.8)
        start = np.random.uniform(0, 1-w)
        s_idx = int(N * start)
        e_idx = int(N * (start + w))
        box[s_idx:e_idx] = 1.0
        inits.append(box)

    # Optimize each initialization
    time_per_init = (TIME_LIMIT - 5) / len(inits)
    
    for init in inits:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        f_opt, score_opt = optimize(init, max_time=time_per_init)
        
        # Final polish: ensure exact non-negativity and normalization
        f_opt = np.maximum(f_opt, 0.0)
        s = np.sum(f_opt)
        if s > 0:
            f_opt = f_opt / s
            
        actual_score = compute_score(f_opt)
        
        if actual_score < best_score:
            best_score = actual_score
            best_f = f_opt

    # Final check
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
