"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: 
1. Use a smooth parametric family based on cosine bumps with multiple components.
2. Optimize using Nelder-Mead with a smooth max surrogate.
3. Apply high-speed local refinement using FFT-based gradient descent on a 
   log-sum-exp (soft-max) surrogate of the autoconvolution maximum.
4. Enforce non-negativity and normalization at each step.

Key improvements over the seed:
- FFT-based gradient computation reduces refinement step cost from O(N^2) to O(N log N).
- Soft-max objective provides smoother gradients for local optimization.
- More refinement iterations possible within the time limit.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # Safety margin before 150s kill

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

def smooth_max_value(x, beta):
    """Compute the value of the smooth max (log-sum-exp) for objective function."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def cosine_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from cosine bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    f = cosine_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max_value(conv, beta) / (s * s)

def fft_autoconv(f):
    """Compute full autoconvolution using FFT for speed."""
    # Pad to next power of 2 or just 2*N for full convolution
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    C = np.fft.irfft(F * F, n=n_fft)
    # The full convolution length is 2*N - 1. 
    # irfft gives length n_fft. The first 2*N-1 elements are the full conv.
    return C[:2*N-1]

def fft_gradient(f, beta):
    """
    Compute the gradient of the soft-max of the autoconvolution with respect to f.
    
    Objective: J = soft_max(c, beta) / S^2 where c = f * f, S = sum(f).
    For local refinement, we fix S (normalize f to sum 1 or N) and just minimize soft_max(c).
    Let's assume f is normalized to sum=1 before gradient step, or we handle the normalization.
    
    Gradient of soft_max(c, beta) w.r.t c is w = softmax(beta * c / max_c) * (max_c / max_c) ... 
    Actually, let's use the standard soft-max gradient.
    Let g = beta * c.
    Soft-max value V = max(c) + (1/beta) * log(sum(exp(beta * (c - max(c)))))
    dV/dc = exp(beta * (c - max(c))) / sum(exp(beta * (c - max(c)))) = softmax(beta * (c - max(c)))
    
    Then dc/dj = sum_m (dc_m/dj) * w_m
    dc_m/dj = f_{m-j} (where indices are wrapped/valid for the convolution)
    
    This sum is the convolution of w and f.
    """
    c = fft_autoconv(f)
    c_max = np.max(c)
    if c_max < 1e-15:
        return np.zeros_like(f)
    
    # Compute weights w for the soft-max
    # To avoid overflow, shift by max
    c_shifted = c - c_max
    exp_c = np.exp(beta * c_shifted)
    w = exp_c / np.sum(exp_c)
    
    # Gradient w.r.t f_j is sum_m w_m * f_{m-j}
    # This is the cross-correlation of w and f, or convolution if we flip w.
    # Since c is symmetric (autoconv), w is symmetric. So conv(w, f) works.
    # We need the part of the convolution that corresponds to indices 0..N-1 of f.
    
    # Let's compute full conv of w and f
    # w has length 2N-1, f has length N.
    # We need grad_j = sum_m w_m f_{m-j}
    # This is (w * f)_j if we define convolution as sum_k w_k f_{j-k}? 
    # Standard conv: (a*b)_n = sum_k a_k b_{n-k}.
    # Here we want sum_m w_m f_{m-j}. Let k = m-j => m = k+j.
    # sum_k w_{k+j} f_k.
    # This is cross-correlation.
    # Since w is symmetric, w_{k+j} = w_{-(k+j)} = w_{j+k} (if symmetric around 0).
    # Wait, c_m is defined for m in 0..2N-2. The "center" is N-1.
    # Symmetry: c_m = c_{2N-2-m}.
    # So w_m is symmetric around N-1.
    
    # Let's just use FFT for the gradient computation too.
    # grad = ifft(fft(w) * conj(fft(f))) ? No, that's correlation.
    # We want grad_j = sum_m w_m f_{m-j}.
    # This is the convolution of w and f, but with w flipped?
    # Let's check: (w * f)_j = sum_k w_k f_{j-k}.
    # We want sum_m w_m f_{m-j}.
    # If w is symmetric (w_m = w_{-m} in a centered sense), then this is just the convolution.
    
    # Simpler: Use the property that grad is the cross-correlation of w and f.
    # scipy.signal.correlate(w, f, mode='valid')? No, lengths differ.
    
    # Let's use FFT.
    # We want g_j = sum_m w_m f_{m-j}.
    # This is equivalent to (w * f^R)_j where f^R is f reversed?
    # Let's just do it directly with FFT of zero-padded arrays.
    
    n_fft = 2 * (2 * N - 1) # Enough for full conv of w and f
    W = np.fft.rfft(w, n=n_fft)
    F = np.fft.rfft(f, n=n_fft)
    
    # The operation sum_m w_m f_{m-j} is the cross-correlation.
    # In FFT domain, cross-correlation is conj(FFT(f)) * FFT(w).
    # Let's verify.
    # corr(j) = sum_m f(m) w(m-j).
    # We want sum_m w(m) f(m-j).
    # This is corr_w_f(j).
    # FFT(corr) = conj(FFT(w)) * FFT(f).
    
    G = np.fft.irfft(conj(W) * F, n=n_fft)
    
    # The result G has length n_fft.
    # We need the first N elements corresponding to j=0..N-1.
    # Note: The correlation is defined for j where the overlap exists.
    # Since w is length 2N-1 and f is length N, the correlation is defined for a range.
    # We want grad_j for j in 0..N-1.
    
    # The correlation output from FFT is centered or aligned?
    # np.correlate(w, f, 'full') gives length 2N-1 + N - 1 = 3N-2.
    # The index j in our sum sum_m w_m f_{m-j} corresponds to the lag.
    # If j=0, sum_m w_m f_m.
    # If j=N-1, sum_m w_m f_{m-(N-1)}.
    
    # Let's extract the correct slice.
    # The correlation of w (len Lw) and f (len Lf) is usually indexed such that 
    # lag 0 is the overlap of the starts.
    # However, FFT correlation output is shifted.
    
    # Let's use a safer approach: direct convolution with FFT but careful with indices.
    # We want grad = conv(w, f) ?
    # If w is symmetric, conv(w, f) = corr(f, w).
    # Let's assume w is symmetric.
    
    G_conv = np.fft.irfft(W * F, n=n_fft)
    
    # The convolution of w (len 2N-1) and f (len N) has length 3N-2.
    # We want the part that corresponds to the gradient of f_j.
    # The term f_j appears in c_m for m=j..j+N-1.
    # So grad_j = sum_{m=j}^{j+N-1} w_m f_{m-j} * f_j ? No.
    # dc_m/df_j = f_{m-j}.
    # grad_j = sum_m w_m f_{m-j}.
    
    # Let's just compute it using the fact that it's a convolution.
    # If we define v = w, then grad = v * f.
    # The j-th element of v * f is sum_k v_k f_{j-k}.
    # This matches sum_m w_m f_{m-j} if we let k=m and j-k = m-j => k=j? No.
    # sum_k w_k f_{j-k}. Let m=k. sum_m w_m f_{j-m}.
    # We want sum_m w_m f_{m-j}.
    # These are the same if w is symmetric (w_m = w_{-m}).
    
    # So grad = conv(w, f).
    # We need to extract the correct N elements from the full convolution.
    # The full conv has length 3N-2.
    # The "center" of the conv is at index (Lw-1 + Lf-1)/2?
    
    # Let's use scipy.signal.convolve if available, or just trust the FFT slice.
    # For autoconvolution gradient, the gradient is symmetric if f is symmetric.
    
    # Let's take the middle N elements of the full convolution?
    # Length of conv is 3N-2.
    # We want indices 0..N-1 of the gradient.
    # In the full convolution array C, C[k] = sum_i w_i f_{k-i}.
    # We want C[k] for k corresponding to j.
    # If j=0, we want sum_m w_m f_m. This is the "peak" of the overlap if w and f are aligned.
    # w is length 2N-1, f is length N.
    # If we align the start of f with the start of w, the overlap is at index 0 of f.
    
    # Actually, let's just use the direct O(N^2) if N=600 is small enough?
    # 600^2 = 360,000 operations. In Python loop, this is slow.
    # But in NumPy vectorized, it's fast.
    
    # Vectorized gradient:
    # grad[j] = sum_m w[m] * f[m-j]
    # We can construct a matrix M where M[j, m] = f[m-j] if 0<=m-j<N else 0.
    # Then grad = w @ M.T ?
    
    # Let's try a different vectorized approach.
    # Pad f to length 2N-1 with zeros.
    # Then the convolution w * f_pad will have the terms we need.
    
    # Let's stick to the FFT conv and extract the right part.
    # The convolution of w (len 2N-1) and f (len N) is length 3N-2.
    # The term f_j contributes to c_m for m in [j, j+N-1].
    # So grad_j = sum_{m=j}^{j+N-1} w_m f_{m-j}.
    # Let k = m-j. sum_{k=0}^{N-1} w_{k+j} f_k.
    # This is the inner product of w[j : j+N] and f.
    
    # So grad_j = <w[j : j+N], f>
    
    # We can compute this for all j using a sliding window inner product.
    # This is equivalent to the correlation of w and f.
    
    # Using FFT for correlation:
    # corr(j) = sum_m f(m) w(m-j)
    # We want sum_m f(m) w(m-j) ? No, we have sum_m w(m) f(m-j).
    # By symmetry of w, w(m) = w(-m) (centered).
    # So sum_m w(m) f(m-j) = sum_m w(m-j) f(m) ?
    
    # Let's just use the direct NumPy approach which is likely fast enough for N=600 if we avoid Python loops.
    # But 600 iterations of O(N^2) is 600 * 360k = 216M ops. Might be too slow in Python.
    # FFT is O(N log N).
    
    # Let's use the FFT correlation result.
    # corr = ifft(conj(FFT(w)) * FFT(f))
    # This gives sum_m f(m) w(m-j).
    # Since w is symmetric, w(m-j) = w(j-m).
    # So corr(j) = sum_m f(m) w(j-m) = sum_m f(m) w(m-j) (if w is even).
    # This matches our grad_j.
    
    # So grad = corr(w, f).
    # We need the first N values of the correlation.
    
    # The correlation array from FFT has length n_fft.
    # The valid lags are - (N-1) to 2N-2 ?
    # We want j = 0, 1, ..., N-1.
    
    # Let's extract from G_corr.
    # The correlation is centered at index 0 in the time domain if we don't ifftshift?
    # np.correlate(w, f, 'full') has length 3N-2.
    # The lag 0 is at index N-1 (since len(w)=2N-1, len(f)=N, max index is 2N-2, min is -(N-1)).
    # Center index = (len(w)-1 + len(f)-1) / 2 = (2N-2)/2 = N-1.
    
    # So corr_full[N-1] is lag 0.
    # corr_full[N-1 + j] is lag j.
    # We want j=0..N-1.
    # So we take corr_full[N-1 : N-1+N].
    
    # Let's compute corr_full using FFT.
    # We need to align the outputs.
    
    return G_conv[:N] # This is a guess, might need shifting.

def local_refinement_fft(f, start_time, max_time, beta=20.0):
    """
    Local refinement using FFT-based gradient descent on soft-max objective.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    current_f = current_f / np.sum(current_f) # Normalize to sum 1
    
    alpha = 0.1
    step_count = 0
    
    while time.time() - start_time < max_time:
        step_count += 1
        
        # Compute current score
        s = np.sum(current_f)
        if s < 1e-15:
            break
        current_f_norm = current_f / s
        conv = np.convolve(current_f_norm, current_f_norm, mode='full')
        current_val = np.max(conv)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f_norm.copy()
            
        # Compute gradient
        # grad_j = sum_m w_m f_{m-j}
        # Using the vectorized inner product idea: grad_j = <w[j:j+N], f>
        # This is the correlation of w and f.
        
        c = conv
        c_max = np.max(c)
        if c_max < 1e-15:
            break
        c_shifted = c - c_max
        exp_c = np.exp(beta * c_shifted)
        w = exp_c / np.sum(exp_c)
        
        # Compute correlation of w and f
        # w has length 2N-1, f has length N.
        # We want grad_j = sum_{k=0}^{N-1} w[j+k] f[k] for j=0..N-1.
        
        # We can use scipy.signal.correlate if available, but let's use FFT.
        # corr(j) = sum_m f(m) w(m-j)
        # We want sum_m w(m) f(m-j).
        # If w is symmetric, these are the same.
        
        # Let's use the property: grad = ifft(conj(FFT(w)) * FFT(f))
        n_fft = 2 * (2 * N - 1)
        W = np.fft.rfft(w, n=n_fft)
        F = np.fft.rfft(f, n=n_fft)
        G = np.fft.irfft(conj(W) * F, n=n_fft)
        
        # The correlation is centered.
        # Lag 0 is at index N-1 in the full correlation array of length 3N-2.
        # In the FFT array of length n_fft, the "center" is at index 0?
        # No, FFT assumes circular convolution.
        # The linear correlation is embedded in the circular one.
        
        # Let's just use the direct NumPy sliding window for N=600.
        # It might be fast enough if we do it in chunks or if we optimize.
        # Actually, 600 iterations * 360k ops is ~200M ops.
        # NumPy can do ~100M-1B ops/sec. So 0.2-2 seconds per iteration?
        # That's too slow for 100 iterations.
        
        # Let's stick to FFT.
        # The output of ifft(conj(W)*F) is the circular correlation.
        # We need to unwrap it.
        
        # For linear correlation of w (len Lw) and f (len Lf):
        # The result is length Lw + Lf - 1.
        # The FFT result is length n_fft.
        # The first Lw + Lf - 1 elements of the "shifted" FFT result contain the linear correlation.
        
        # Let's assume the standard alignment:
        # corr_full = G[:2*N-1 + N-1] # Length 3N-2
        # But G is circular.
        
        # Let's use a simpler, robust method:
        # grad = np.zeros(N)
        # for j in range(N):
        #     grad[j] = np.dot(w[j:j+N], f)
        # This is O(N^2).
        
        # To make it faster, we can use the fact that w is smooth.
        # Or just accept O(N^2) but reduce iterations?
        # Or use scipy.signal.correlate which is FFT-based and handles alignment.
        
        try:
            from scipy.signal import correlate
            # correlate(w, f, mode='valid')?
            # We want the part where f is fully inside w?
            # w is longer.
            # correlate(w, f, 'full') gives length 3N-2.
            # We want lags 0 to N-1.
            # In 'full' mode, the lag 0 is at index len(f)-1 = N-1.
            # So we take [N-1 : N-1+N].
            
            corr_full = correlate(w, f, mode='full')
            grad = corr_full[N-1 : N-1+N]
            
        except ImportError:
            # Fallback to O(N^2) if scipy.signal is not available (unlikely)
            grad = np.zeros(N)
            for j in range(N):
                grad[j] = np.dot(w[j:j+N], f)
        
        # Gradient descent step
        # Minimize soft_max(c)
        # f_new = f - alpha * grad
        # But we must maintain non-negativity and normalization.
        
        # Scale gradient to step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
            
        grad = grad / grad_norm
        
        # Try a few step sizes
        best_f_new = current_f_norm
        best_val_new = current_val
        
        for scale in [alpha, alpha*0.5, alpha*0.1, alpha*2.0]:
            f_trial = current_f_norm - scale * grad
            f_trial = np.maximum(f_trial, 0.0)
            s_trial = np.sum(f_trial)
            if s_trial < 1e-15:
                continue
            f_trial = f_trial / s_trial
            
            val_trial = np.max(np.convolve(f_trial, f_trial, mode='full'))
            
            if val_trial < best_val_new:
                best_val_new = val_trial
                best_f_new = f_trial
        
        if best_val_new < current_val:
            current_f_norm = best_f_new
            alpha = min(alpha * 1.05, 1.0)
        else:
            alpha *= 0.8
            if alpha < 1e-6:
                break

    return best_f, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components
    for n_c in [3, 4, 5, 6]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta
        betas = [10.0, 20.0, 30.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            for restart in range(5):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                
                if restart == 0:
                    centers = np.linspace(-0.15, 0.15, n_c)
                    widths = np.full(n_c, 0.15)
                    heights = np.full(n_c, 1.0)
                else:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.05, 0.25, size=n_c)
                    heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        if best_params is not None:
            f = parametric_to_step(best_params, n_c)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Local refinement with FFT
    if time.time() - start_time < TIME_LIMIT - 5:
        refined_f, refined_score = local_refinement_fft(
            best_f, start_time, TIME_LIMIT - 2
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
