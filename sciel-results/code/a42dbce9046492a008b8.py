"""
Frequency-domain approach for the first autocorrelation inequality task.

Strategy:
- Parameterize the step function heights h (length N=600) as the real part of the inverse FFT
  of a complex spectral vector S, where S is constructed from non-negative amplitudes A >= 0
  with symmetric phases to ensure h is real.
- Apply ReLU activation in the time domain to enforce h >= 0.
- The autoconvolution's Fourier transform is |FFT(h)|^2, so minimizing the peak of the
  time-domain autoconvolution is equivalent to making the spectral energy as flat as possible.
- Use smooth gradient descent on the spectral amplitudes A to minimize the normalized peak
  autoconvolution, with the non-negativity constraint handled by ReLU in the time domain.

Key insight:
- If we let H = FFT(h), then the autoconvolution c has FFT = H * conj(H) = |H|^2.
- The peak of c is bounded by the L1 norm of |H|^2, which is sum |H|^2 = N * sum h_i^2 (Parseval).
- To minimize the peak, we want |H|^2 to be as flat as possible (equal energy at all frequencies).
- With the constraint h >= 0, we can't make H perfectly flat, but we can get close by
  choosing amplitudes that decay slowly and using the ReLU to clip negatives.

Implementation:
1. Initialize spectral amplitudes A with a slow decay (e.g., 1/sqrt(1+k^2)).
2. At each iteration:
   a. Construct H = A * exp(i*theta) with symmetric phases (theta_k = -theta_{N-k}).
   b. Compute h_raw = IFFT(H).
   c. Apply ReLU: h = max(h_raw, 0).
   d. Compute autoconvolution c and its peak.
   e. Compute score = peak / sum(h^2).
   f. Compute gradient of score w.r.t. A via chain rule.
   g. Update A with Adam optimizer.
3. Return the best h found.
"""

import numpy as np

N = 600

def construct():
    # Number of FFT bins for rfft
    n_bins = N // 2 + 1
    
    # Initialize spectral amplitudes with slow decay
    k = np.arange(n_bins)
    A = 1.0 / np.sqrt(1.0 + 0.1 * k**2)
    
    # Initialize phases to zero (symmetric => real h)
    theta = np.zeros(n_bins)
    
    # Adam optimizer state
    m_A = np.zeros(n_bins)
    v_A = np.zeros(n_bins)
    m_theta = np.zeros(n_bins)
    v_theta = np.zeros(n_bins)
    
    lr_A = 0.01
    lr_theta = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    best_h = None
    best_score = np.inf
    
    n_iters = 800
    
    for it in range(n_iters):
        # Construct H = A * exp(i*theta) with symmetry for real h
        H = A * np.exp(1j * theta)
        
        # Inverse FFT to get time domain
        h_raw = np.fft.irfft(H, n=N)
        
        # Apply ReLU to enforce non-negativity
        h = np.maximum(h_raw, 0.0)
        
        # Compute autoconvolution
        size = 2 * N
        H_full = np.fft.rfft(h, n=size)
        c = np.fft.irfft(H_full * H_full.conj(), n=size)
        c = c[:2*N-1]
        
        peak = np.max(c)
        norm = np.sum(h**2)
        
        if norm < 1e-10:
            continue
        
        score = peak / norm
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Compute gradient of score w.r.t. h
        # score = max(c) / sum(h^2)
        # Let M = max(c), S = sum(h^2)
        # d(score)/d(h_j) = (dM/dh_j * S - M * dS/dh_j) / S^2
        # dS/dh_j = 2*h_j
        # dM/dh_j = gradient of c_{k*} w.r.t. h_j where k* = argmax(c)
        
        k_star = int(np.argmax(c))
        
        # d(c_k)/d(h_j) for linear autoconvolution:
        # c_k = sum_i h_i * h_{i-k} (with h_i = 0 for i < 0 or i >= N)
        # d(c_k)/d(h_j) = h_{j-k} + h_{j+k} (with out-of-bounds = 0)
        
        # Efficient computation using shifts
        grad_c = np.zeros(N)
        if k_star == 0:
            grad_c = 2.0 * h
        else:
            # h shifted by k_star
            shift_pos = np.zeros(N)
            if k_star < N:
                shift_pos[k_star:] = h[:N-k_star]
            shift_neg = np.zeros(N)
            if k_star < N:
                shift_neg[:N-k_star] = h[k_star:]
            grad_c = shift_pos + shift_neg
        
        dS = 2.0 * h
        dM = grad_c
        
        # Gradient of score w.r.t. h
        grad_score_h = (dM * norm - peak * dS) / (norm * norm)
        
        # Gradient w.r.t. h_raw (before ReLU)
        # ReLU: h = max(h_raw, 0), so dh/dh_raw = 1 where h_raw > 0, else 0
        mask = (h_raw > 0).astype(float)
        grad_score_hraw = grad_score_h * mask
        
        # Gradient w.r.t. H (complex)
        # h_raw = IFFT(H), so dh_raw/dH = conj(FFT_basis)
        # For real h_raw from irfft, the relationship is:
        # h_raw = irfft(H) => H = rfft(h_raw)
        # Gradient w.r.t. H: grad_H = rfft(grad_score_hraw)
        grad_H = np.fft.rfft(grad_score_hraw, n=N)
        
        # H = A * exp(i*theta)
        # dH/dA = exp(i*theta)
        # dH/dtheta = i * A * exp(i*theta)
        
        # Gradient w.r.t. A: Re(grad_H * conj(dH/dA)) = Re(grad_H * exp(-i*theta))
        grad_A = np.real(grad_H * np.exp(-1j * theta))
        
        # Gradient w.r.t. theta: Re(grad_H * conj(dH/dtheta)) = Re(grad_H * (-i) * A * exp(-i*theta))
        grad_theta = np.real(grad_H * (-1j) * A * np.exp(-1j * theta))
        
        # Adam update for A
        t = it + 1
        m_A = beta1 * m_A + (1 - beta1) * grad_A
        v_A = beta2 * v_A + (1 - beta2) * grad_A**2
        m_A_hat = m_A / (1 - beta1**t)
        v_A_hat = v_A / (1 - beta2**t)
        A = A - lr_A * m_A_hat / (np.sqrt(v_A_hat) + eps)
        A = np.maximum(A, 0.0)  # Keep amplitudes non-negative
        
        # Adam update for theta (only for bins 1 to n_bins-1; bin 0 and bin N/2 have fixed phase 0 for real h)
        if n_bins > 2:
            m_theta = beta1 * m_theta + (1 - beta1) * grad_theta
            v_theta = beta2 * v_theta + (1 - beta2) * grad_theta**2
            m_theta_hat = m_theta / (1 - beta1**t)
            v_theta_hat = v_theta / (1 - beta2**t)
            theta_update = lr_theta * m_theta_hat / (np.sqrt(v_theta_hat) + eps)
            theta[1:-1] = theta[1:-1] - theta_update[1:-1]
        
        # Enforce symmetry: for real h, we need H[k] = conj(H[N-k])
        # This means A[k] = A[N-k] and theta[k] = -theta[N-k]
        # For rfft, we only have bins 0 to N//2, so we just need:
        # - Bin 0: phase 0 (DC)
        # - Bin N//2 (if N even): phase 0 (Nyquist)
        # The rfft automatically handles the symmetry when we do irfft
        
        theta[0] = 0.0
        if N % 2 == 0:
            theta[-1] = 0.0
    
    if best_h is None:
        # Fallback: uniform function
        best_h = np.ones(N) / np.sqrt(N)
    
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
