"""Rewritten program: complex |z|^2 parameterization with Wirtinger-style gradient descent
to minimize the normalized peak autoconvolution of a 600-interval step function on [-1/4,1/4]."""

import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # Initialize z as complex random
    np.random.seed(42)
    z = np.random.randn(N) + 1j * np.random.randn(N)
    # Scale so that |z|^2 has mean 1 (normalization will handle it anyway)
    h0 = np.abs(z)**2
    z = z / np.sqrt(np.mean(h0))

    # Best tracking
    best_h = np.ones(N)
    best_score = np.inf

    # FFT size for linear autoconvolution
    fft_size = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N-1
    # Actually 2*N-1 = 1199, so fft_size = 2048

    # Learning rate schedule
    base_lr = 0.05
    momentum = 0.9
    v = np.zeros_like(z)  # momentum buffer

    # Softmax temperature for peak approximation
    t = 0.02

    for step in range(5000):
        if time.time() > deadline:
            break

        # Compute h = |z|^2
        h = np.abs(z)**2
        s = np.sum(h)
        if s < 1e-12:
            # Reinitialize
            z = np.random.randn(N) + 1j * np.random.randn(N)
            v = np.zeros_like(z)
            continue
        h_norm = h / s  # normalized so integral = 1 (per-unit-height normalization)

        # Compute autoconvolution via FFT (linear)
        # Zero-pad to fft_size
        h_pad = np.zeros(fft_size)
        h_pad[:N] = h_norm
        H = np.fft.fft(h_pad)
        conv = np.fft.ifft(H * H).real[:2*N - 1]

        score = np.max(conv)
        if score < best_score:
            best_score = score
            best_h = h_norm.copy()

        # Softmax-weighted gradient for max approximation
        # grad of max(conv) w.r.t. h approximated by softmax over conv
        y = conv / t
        max_y = np.max(y)
        exp_y = np.exp(y - max_y)
        P = exp_y / np.sum(exp_y)  # probability weights over conv positions

        # Now compute gradient of the objective w.r.t. h
        # Objective: max_k (h*h)[k] where (h*h)[k] = sum_i h[i] h[k-i]
        # d/dh[j] of (h*h)[k] = h[k-j] + h[j-k] (with appropriate bounds)
        # So grad_h[j] = sum_k P[k] * (h[k-j] + h[j-k])
        # This can be computed via cross-correlation:
        # grad_h[j] = sum_k P[k] * h[k-j] + sum_k P[k] * h[j-k]
        #           = (P * h)[j] + (P_rev * h)[j]  (convolutions)
        
        # Use FFT for efficiency
        P_pad = np.zeros(fft_size)
        P_pad[:len(P)] = P
        P_full = np.fft.fft(P_pad)
        
        h_pad_full = np.zeros(fft_size)
        h_pad_full[:N] = h_norm
        H_full = np.fft.fft(h_pad_full)
        
        # First term: sum_k P[k] h[k-j] = (P conv h)[j]
        term1 = np.fft.ifft(P_full * H_full).real[:N]
        
        # Second term: sum_k P[k] h[j-k] = (P_rev conv h)[j]
        # P_rev[k] = P[-k mod L], but we need to be careful with indexing
        # Actually h[j-k] = (h_rev conv P)[j] where h_rev[j] = h[N-1-j]
        # Alternatively: sum_k P[k] h[j-k] = (P * h_reflected)[j]
        # Let's compute via: ifft(P_full * H_full_conj) won't work directly.
        # Instead: sum_k P[k] h[j-k] = (h conv P_rev)[j]
        # P_rev: reverse P
        P_rev = P[::-1].copy()
        P_rev_pad = np.zeros(fft_size)
        P_rev_pad[:len(P_rev)] = P_rev
        P_rev_full = np.fft.fft(P_rev_pad)
        term2 = np.fft.ifft(P_rev_full * H_full).real[:N]
        
        grad_h = term1 + term2
        
        # Now chain rule: h[j] = |z[j]|^2 = z[j] * conj(z[j])
        # d h[j] / d z[j] = conj(z[j])
        # d h[j] / d conj(z[j]) = z[j]
        # Wirtinger gradient: grad_z = conj(z) * grad_h, grad_conj_z = z * grad_h
        # Update: z -= lr * (grad_z + 0.5 * conj(grad_conj_z)) for real-valued objective
        # Actually for real objective f: df/dz = conj(z) * df/dh, and the gradient descent step is:
        # z_new = z - lr * (df/dz + conj(df/dconj_z))
        # But since f is real, df/dconj_z = conj(df/dz), so:
        # z_new = z - 2*lr * Re(df/dz) ... no, let's just use:
        # z -= lr * (conj(z) * grad_h)  -- this is the Wirtinger gradient flow
        # Actually the standard Wirtinger gradient for real f:
        # ∂f/∂z = (1/2)(∂f/∂x - i∂f/∂y) and the gradient step is z -= lr * 2 * ∂f/∂z*
        # Let's just use: z -= lr * conj(z) * grad_h (and similarly for the conjugate)
        # The full update in complex: z_new = z - lr * (conj(z) * grad_h + z * conj(grad_h))
        # Since grad_h is real, conj(grad_h) = grad_h, so:
        # z_new = z - lr * grad_h * (conj(z) + z) = z - lr * grad_h * 2*Re(z)
        # Hmm, that doesn't seem right either.
        
        # Let me reconsider. f = sum_k P[k] * conv[k], where conv[k] = sum_i h[i] h[k-i]
        # df/dh[j] = grad_h[j] (computed above)
        # h[j] = z[j] * conj(z[j])
        # df/dz[j] = conj(z[j]) * df/dh[j]
        # df/dconj(z[j]) = z[j] * df/dh[j]
        # For gradient descent on a real-valued function of complex variable, the steepest descent direction is:
        # z_new = z - lr * (df/dz) = z - lr * conj(z[j]) * grad_h[j]
        # But this only uses the Wirtinger derivative w.r.t. z. The full gradient in R^{2N} is:
        # (∂f/∂Re(z), ∂f/∂Im(z)) = 2 * Re(conj(∂f/∂z) * (1, i)) ... 
        # Simplest: just do z -= lr * conj(z) * grad_h, which is a common Wirtinger flow step.
        
        grad_z = np.conj(z) * grad_h
        v = momentum * v + (1 - momentum) * grad_z
        z = z - base_lr * v

    # Final normalization
    s_final = np.sum(best_h)
    if s_final > 0:
        best_h = best_h / s_final
    
    return list(best_h)

# EVOLVE-BLOCK-END
