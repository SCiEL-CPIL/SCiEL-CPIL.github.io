"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max(h):
    """Compute max of full autoconvolution of h."""
    # Use FFT for efficiency
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(H * H, n=size)[:2*n-1]
    return np.max(conv)

def _objective(h):
    """Compute the task value: |2*N*max(autoconv)/sum(h)^2|"""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv_max = _autoconv_max(h)
    return abs(2.0 * N * conv_max / (s * s))

def _gradient_descent_step(h, lr, beta):
    """
    Perform gradient descent on the smooth surrogate:
    f(h) = 2*N * softmax(beta * conv) / sum(h)^2
    
    where softmax uses log-sum-exp for numerical stability.
    """
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return h
    
    # Compute autoconvolution via FFT
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(H * H, n=size)[:2*n-1]
    
    # Soft-max of conv with parameter beta
    max_c = np.max(conv)
    exp_conv = np.exp(beta * (conv - max_c))
    sum_exp = np.sum(exp_conv)
    
    # The objective is |2*N*max(conv)/s^2|. For smooth surrogate:
    # g(h) = 2*N * (1/beta) * log(sum(exp(beta*conv))) / s^2
    # But we want to minimize the max, so let's use the soft-max value directly.
    
    # Gradient of soft_max w.r.t. h_i:
    # d/dh_i [max_j conv_j] ≈ (1/sum_exp) * sum_j exp(beta*conv_j) * d(conv_j)/dh_i
    # conv_j = sum_k h_k * h_{j-k} (with appropriate bounds)
    # d(conv_j)/dh_i = h_{j-i} + h_{j+i} (for valid indices, considering symmetry)
    
    # More precisely, conv[j] = sum_{k=0}^{n-1} h[k] * h[j-k] where h[x]=0 for x outside [0,n-1]
    # d(conv[j])/dh[i] = h[j-i] + (1 if j==2i else 0)*... 
    # Actually conv[j] = sum_k h[k]*h[j-k], so d(conv[j])/dh[i] = h[j-i] (when j-i in [0,n-1]) + h[j-i]... 
    # Wait: conv[j] = sum_{k=max(0,j-(n-1))}^{min(j,n-1)} h[k]*h[j-k]
    # d(conv[j])/dh[i] = h[j-i] if 0 <= j-i <= n-1, else 0. Plus if i == j-k for some k, but that's the same term.
    # Actually it's just: for each j, d(conv[j])/dh[i] = h[j-i] if 0<=j-i<=n-1 else 0
    # But by symmetry of convolution, also h[i-j]... no, conv is not symmetric in that sense for the gradient.
    # Let me think again. conv[j] = sum_k h[k] h[j-k]. 
    # d(conv[j])/dh[i]: the term h[i] appears when k=i (giving h[j-i]) and when j-k=i i.e. k=j-i (giving h[i]*h[j-i]).
    # So d(conv[j])/dh[i] = h[j-i] + h[j-i] = 2*h[j-i] if i != j-i, and h[j-i] if i==j-i (i.e., j=2i).
    # Wait no: conv[j] = sum_k h[k]*h[j-k]. The partial derivative w.r.t. h[i]:
    # - When k=i: contributes h[j-i] (if j-i is valid index)
    # - When j-k=i, i.e., k=j-i: contributes h[k] = h[j-i]... wait, h[k]*h[j-k], derivative w.r.t h[i]:
    #   If k=i: d/dh[i](h[i]*h[j-i]) = h[j-i]
    #   If j-k=i (k=j-i): d/dh[i](h[j-i]*h[i]) = h[j-i]
    #   If i==j-i (j=2i): both terms are the same, so it's h[i], and d/dh[i](h[i]^2) = 2*h[i]
    # So d(conv[j])/dh[i] = h[j-i] (if 0<=j-i<=n-1) + h[j-i] (if 0<=j-i<=n-1) but counted once if j=2i.
    # More carefully: d(conv[j])/dh[i] = [0<=j-i<=n-1] * h[j-i] + [0<=j-i<=n-1] * h[j-i]
    #   but if j-i == i (i.e., j=2i), these are the same term h[i]*h[i], so derivative is 2*h[i] = h[i]+h[i].
    # So actually: d(conv[j])/dh[i] = 2*h[j-i] if 0<=j-i<=n-1, EXCEPT when j-i==i where it's still 2*h[i].
    # Hmm, let me just use: d(conv[j])/dh[i] = h[j-i] if 0<=j-i<=n-1 else 0, PLUS h[j-i] if 0<=j-i<=n-1 else 0,
    # but these are the same value, so it's 2*h[j-i] when valid.
    
    # Gradient of g = 2N * soft_max / s^2 w.r.t. h[i]:
    # d(soft_max)/dh[i] = sum_j (exp(beta*conv[j])/sum_exp) * d(conv[j])/dh[i]
    # = sum_j (exp(beta*conv[j])/sum_exp) * 2*h[j-i] (where valid)
    
    # Let w[j] = exp(beta*conv[j])/sum_exp
    # Then d(soft_max)/dh[i] = 2 * sum_j w[j] * h[j-i] (for valid j-i)
    # This is essentially 2 * (w * h)[i] in a cross-correlation sense.
    
    w = exp_conv / sum_exp
    
    # Compute sum_j w[j] * h[j-i] for each i
    # This is the cross-correlation of w and h, evaluated at lag i
    # corr[i] = sum_j w[j] * h[j-i] where j-i must be in [0,n-1]
    # Let m = j-i, so j = m+i. corr[i] = sum_m w[m+i] * h[m] for valid m, m+i
    # This is like a convolution/correlation.
    
    # Use FFT: corr = ifft(fft(w) * conj(fft(h)))... let me think.
    # Actually, let's define it via FFT.
    # We want for each i: sum_{m=0}^{n-1} w[m+i] * h[m] where m+i < len(w)
    # w has length 2n-1, h has length n.
    
    # Let's use direct computation for simplicity since n=400 is small.
    grad_softmax = np.zeros(n)
    for i in range(n):
        # sum_j w[j] * h[j-i] where 0 <= j-i < n, i.e., i <= j <= i+n-1
        j_start = max(0, i)
        j_end = min(2*n-1, i+n)
        if j_start < j_end:
            m_range = np.arange(j_start - i, j_end - i)  # m = j - i, from 0 to n-1
            grad_softmax[i] = 2.0 * np.dot(w[j_start:j_end], h[m_range])
    
    # Now g = 2N * soft_max / s^2
    # dg/dh[i] = 2N * (d(soft_max)/dh[i] * s^2 - soft_max * 2*s * 1) / s^4
    # = 2N * (d(soft_max)/dh[i] / s^2 - 2*soft_max/(s^3))
    # = 2N/s^2 * d(soft_max)/dh[i] - 4N*soft_max/s^3
    
    soft_max_val = max_c + np.log(sum_exp) / beta
    
    grad_g = (2.0 * N / (s * s)) * grad_softmax - (4.0 * N * soft_max_val / (s ** 3))
    
    # Also need to account for the absolute value, but since we're minimizing and the sign is handled
    # by the structure, let's just use the gradient as is.
    
    h_new = h - lr * grad_g
    
    # Project to maintain sum(h) = 1 (reparameterization)
    h_new = h_new / np.sum(h_new)
    
    return h_new

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = _objective(best_h)
    
    # Try multiple initializations
    np.random.seed(42)
    
    initializations = []
    
    # 1. All ones (baseline)
    initializations.append(np.ones(N))
    
    # 2. DC-removed random
    for trial in range(5):
        r = np.random.randn(N)
        r = r - np.mean(r)  # Remove DC component
        r = r / np.sum(r)  # Normalize sum to 1
        initializations.append(r.copy())
    
    # 3. Fourier-domain nulls: start with random, zero out some low frequencies
    for trial in range(5):
        r = np.random.randn(N)
        R = np.fft.rfft(r)
        # Zero out the first few and last few frequencies (keep DC for sum constraint)
        R[1:10] = 0
        R[-10:-1] = 0
        r = np.fft.irfft(R, n=N)
        r = r - np.mean(r)
        if abs(np.sum(r)) > 1e-10:
            r = r / np.sum(r)
        initializations.append(r.copy())
    
    # 4. Alternating signs
    alt = np.ones(N)
    alt[::2] = 1.0
    alt[1::2] = -1.0
    alt = alt / np.sum(alt) if abs(np.sum(alt)) > 1e-10 else alt
    initializations.append(alt.copy())
    
    # 5. Smooth random (low-pass filtered)
    for trial in range(3):
        r = np.random.randn(N)
        # Apply a simple low-pass via repeated averaging
        for _ in range(3):
            r = 0.5 * r + 0.25 * np.roll(r, 1) + 0.25 * np.roll(r, -1)
        r = r - np.mean(r)
        if abs(np.sum(r)) > 1e-10:
            r = r / np.sum(r)
        initializations.append(r.copy())
    
    beta = 150.0
    base_lr = 0.01
    max_iters = 500
    
    for init in initializations:
        if time.time() > deadline - 5.0:
            break
        
        h = init.copy()
        s = np.sum(h)
        if abs(s) < 1e-10:
            continue
        h = h / s  # Ensure sum = 1
        
        current_val = _objective(h)
        
        lr = base_lr
        patience = 0
        best_local_val = current_val
        best_local_h = h.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            h = _gradient_descent_step(h, lr, beta)
            
            # Ensure finite
            if not np.all(np.isfinite(h)):
                break
            
            val = _objective(h)
            
            if val < best_local_val:
                best_local_val = val
                best_local_h = h.copy()
                patience = 0
            else:
                patience += 1
            
            # Adaptive learning rate
            if patience > 20:
                lr *= 0.5
                patience = 0
                if lr < 1e-8:
                    break
            
            # Check if we've improved the global best
            if best_local_val < best_val:
                best_val = best_local_val
                best_h = best_local_h.copy()
        
        # After GD, try a few random restarts from the best local point
        if time.time() < deadline - 5.0:
            for restart in range(3):
                if time.time() > deadline - 3.0:
                    break
                h2 = best_local_h + 0.01 * np.random.randn(N)
                h2 = h2 / np.sum(h2)
                lr2 = 0.005
                for it2 in range(200):
                    if time.time() > deadline - 2.0:
                        break
                    h2 = _gradient_descent_step(h2, lr2, beta)
                    if not np.all(np.isfinite(h2)):
                        break
                    val2 = _objective(h2)
                    if val2 < best_val:
                        best_val = val2
                        best_h = h2.copy()
    
    # Final polish: try a few more iterations with very small learning rate
    if time.time() < deadline - 1.0:
        h = best_h.copy()
        for it in range(100):
            if time.time() > deadline - 0.5:
                break
            h = _gradient_descent_step(h, 0.001, beta)
            if not np.all(np.isfinite(h)):
                break
            val = _objective(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
