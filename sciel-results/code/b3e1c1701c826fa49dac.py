"""
Strategy: Spectral Flatness Optimization via Direct Gradient Descent on Magnitudes/Phases.

The autoconvolution of a signal x is related to its power spectral density (PSD).
Specifically, the discrete autoconvolution c[k] is the inverse FFT of |X[k]|^2.
Minimizing the maximum absolute value of c[k] is equivalent to making the PSD
as "flat" as possible (white noise spectrum), subject to the constraint that
the integral (sum) of x is fixed.

We parameterize x by its FFT components: x = IFFT(M * exp(i*P)).
We optimize M (magnitudes) and P (phases) using gradient descent on a smooth
approximation of the max absolute autoconvolution.
This reduces the effective search space dimensionality and leverages the
spectral structure to find flatter spectra than time-domain methods.
"""

import numpy as np
import time

N = 400

# Precompute FFT size for efficient convolution
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _compute_conv_from_x(x):
    """Compute linear autoconvolution of x using FFT."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv_full = np.fft.irfft(fx * fx, _FFT_SIZE)
    # Extract lags 0 to N-1 and -(N-1) to -1
    # Lag k (0..N-1) is at index k
    # Lag -k (1..N-1) is at index _FFT_SIZE - k
    conv_pos = conv_full[:N]
    conv_neg = conv_full[_FFT_SIZE - (N - 1):_FFT_SIZE]
    # Note: conv_neg[0] corresponds to lag -(N-1), conv_neg[-1] to lag -1
    # We want to combine them. The problem asks for max |conv[k]| over all lags.
    # Due to symmetry, |conv[k]| = |conv[-k]|. So we just need max over 0..N-1.
    # Let's verify: c[k] = sum x[i]x[i+k]. c[-k] = sum x[i]x[i-k] = sum x[j+k]x[j] = c[k].
    # So we only need to consider lags 0..N-1.
    return conv_pos

def _soft_max_abs(conv, sigma=0.05):
    """
    Smooth approximation of max(|conv[k]|).
    Uses log-sum-exp of squared values to handle absolute value and max.
    f(x) = sigma * log(sum(exp((x^2 - max_x^2)/sigma))) + max_x^2
    This approximates max(x^2), and sqrt of that is max|x|.
    Actually, let's minimize max(conv^2). Then sqrt(result) is the metric.
    """
    sq = conv * conv
    max_sq = np.max(sq)
    if max_sq < 1e-12:
        return 0.0, np.zeros_like(conv)
    
    # Avoid overflow in exp
    shifted = (sq - max_sq) / sigma
    # Clip to prevent numerical issues, though shifted <= 0
    shifted = np.clip(shifted, -100, 0)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    val = max_sq + sigma * np.log(sum_exp)
    
    # Gradient w.r.t. conv
    # d/dconv (sigma * log(sum(exp((conv^2 - max_sq)/sigma)))) 
    # Note: max_sq is treated as constant for gradient step in this approximation 
    # (common in soft-max implementations to stabilize)
    # Actually, the true gradient of soft-max involves all terms.
    # Let u_i = conv[i]^2. f = sigma * log(sum(exp(u_i/sigma)))
    # df/du_i = exp(u_i/sigma) / sum(exp(u_j/sigma))
    # du_i/dconv[i] = 2*conv[i]
    # So d f / d conv[i] = 2 * conv[i] * weight[i]
    
    weights = exp_shifted / sum_exp
    grad_conv = 2.0 * conv * weights
    
    return val, grad_conv

def _metric(x):
    """Calculate the raw metric value."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    conv = _compute_conv_from_x(x)
    max_abs_conv = np.max(np.abs(conv))
    return max_abs_conv / (s * s)

def construct():
    deadline = time.time() + 145.0
    
    # We work in the frequency domain.
    # x = ifft(X). X = M * exp(1j*P).
    # Constraint: sum(x) = 1. sum(x) = X[0]. So X[0] must be 1 (real).
    # We optimize M[1:] and P[1:]. M[0] is fixed to 1, P[0] is fixed to 0.
    
    # Initial guess: White noise (flat spectrum)
    rng = np.random.default_rng(42)
    
    # Start with a flat magnitude spectrum
    M = np.ones(_FFT_SIZE // 2 + 1)
    P = np.zeros(_FFT_SIZE // 2 + 1)
    
    # Add some random phases to break symmetry
    P[1:] = rng.uniform(-np.pi, np.pi, _FFT_SIZE // 2)
    
    # Ensure X[0] = 1
    M[0] = 1.0
    P[0] = 0.0
    
    def get_x(M, P):
        X = M * np.exp(1j * P)
        # Force X[0] = 1 to maintain sum(x)=1
        X[0] = 1.0
        x = np.fft.irfft(X, _FFT_SIZE)
        return x[:N]
    
    def objective(M, P):
        x = get_x(M, P)
        conv = _compute_conv_from_x(x)
        val, grad_conv = _soft_max_abs(conv)
        return val, grad_conv
        
    def gradients(M, P):
        x = get_x(M, P)
        conv = _compute_conv_from_x(x)
        val, grad_conv = _soft_max_abs(conv)
        
        # We need d val / d M and d val / d P
        # x = ifft(X). X = M * e^{iP}
        # dx/dM_k = ifft(dX/dM_k) = ifft(e^{iP_k} * delta_{k})
        # dx/dP_k = ifft(dX/dP_k) = ifft(i * M_k * e^{iP_k} * delta_{k})
        
        # This is expensive if done naively for each k.
        # However, grad_val = sum_k grad_conv[k] * dx_k
        # dx_k = sum_j (d x_k / d X_j) X_j ... no, x_k = 1/M sum_j X_j e^{i 2 pi j k / M}
        # d x_k / d X_j = 1/M e^{i 2 pi j k / M}
        
        # Let's use the chain rule via FFT.
        # val = f(conv(x))
        # grad_x = sum_k grad_conv[k] * d conv[k] / d x
        # d conv[k] / d x is the gradient of autoconvolution.
        # conv[k] = sum_i x[i] x[i+k]
        # d conv[k] / d x[j] = x[j+k] + x[j-k]
        
        # Let's compute grad_x directly.
        grad_x = np.zeros(N)
        for k in range(N):
            # d conv[k] / d x[j]
            # Term 1: x[j+k] for j=0..N-1-k
            # Term 2: x[j-k] for j=k..N-1
            g_k = np.zeros(N)
            g_k[:N-k] += x[k:]
            g_k[k:] += x[:N-k]
            # Note: conv[k] is real.
            grad_x += grad_conv[k] * g_k
            
        # Now map grad_x to grad_M and grad_P
        # x = ifft(X)
        # X = fft(x) ... wait, x is the first N points of ifft(X, _FFT_SIZE).
        # This truncation makes the gradient mapping tricky.
        # The relationship x = ifft(X) is linear.
        # Let F be the IFFT matrix. x = F X.
        # X = F^H x ... no, X is the full FFT vector.
        # If x is the first N samples of the IFFT of X (length M),
        # then x = T F X, where T is the truncation matrix.
        # grad_X = F^H T^H grad_x
        # T^H grad_x is just grad_x padded with zeros to length M.
        
        grad_x_padded = np.zeros(_FFT_SIZE)
        grad_x_padded[:N] = grad_x
        
        # F^H is the FFT (conjugate transpose of IFFT is FFT)
        # Note: numpy ifft has 1/M factor. fft does not.
        # x = ifft(X) => X = fft(x) * M ? No.
        # numpy: ifft(fft(x)) = x.
        # So F = ifft_matrix. F^H = fft_matrix.
        # grad_X = fft(grad_x_padded)
        
        grad_X = np.fft.fft(grad_x_padded)
        
        # X = M * exp(iP)
        # dX/dM_k = exp(iP_k)
        # dX/dP_k = i * M_k * exp(iP_k)
        
        # grad_M_k = Re( grad_X_k * conj(dX/dM_k) ) = Re( grad_X_k * exp(-iP_k) )
        # grad_P_k = Re( grad_X_k * conj(dX/dP_k) ) = Re( grad_X_k * (-i) * M_k * exp(-iP_k) )
        #          = M_k * Re( -i * grad_X_k * exp(-iP_k) )
        
        phase_factor = np.exp(-1j * P)
        
        # Handle X[0] constraint: M[0]=1, P[0]=0. Gradients for these are 0.
        grad_M = np.real(grad_X * phase_factor)
        grad_P = np.real(-1j * grad_X * phase_factor) * M
        
        grad_M[0] = 0.0
        grad_P[0] = 0.0
        
        return val, grad_M, grad_P

    # Gradient Descent with momentum
    best_x = get_x(M, P)
    best_val = _metric(best_x)
    
    M_curr = M.copy()
    P_curr = P.copy()
    
    v_M = np.zeros_like(M_curr)
    v_P = np.zeros_like(P_curr)
    
    lr = 0.01
    momentum = 0.9
    
    # Adaptive learning rate
    for i in range(5000):
        if time.time() > deadline - 10:
            break
        
        val, g_M, g_P = gradients(M_curr, P_curr)
        
        # Update
        v_M = momentum * v_M + (1 - momentum) * g_M
        v_P = momentum * v_P + (1 - momentum) * g_P
        
        M_curr -= lr * v_M
        P_curr -= lr * v_P
        
        # Enforce constraints
        M_curr[0] = 1.0
        P_curr[0] = 0.0
        # Magnitudes must be non-negative
        M_curr = np.maximum(M_curr, 0)
        
        # Check metric
        x_curr = get_x(M_curr, P_curr)
        curr_metric = _metric(x_curr)
        
        if curr_metric < best_val:
            best_val = curr_metric
            best_x = x_curr.copy()
            
        # Simple restart if stuck
        if i % 100 == 0 and i > 0:
            # Slightly perturb phases
            P_curr[1:] += rng.normal(0, 0.1, _FFT_SIZE // 2)
            M_curr[1:] *= np.exp(rng.normal(0, 0.1, _FFT_SIZE // 2))
            M_curr[0] = 1.0
            P_curr[0] = 0.0
            M_curr = np.maximum(M_curr, 0)

    # Final local search in time domain for polish
    if time.time() < deadline - 2:
        x = best_x.copy()
        # Normalize to sum=1
        s = np.sum(x)
        if abs(s) > 1e-12:
            x = x / s
        else:
            x = np.ones(N) / N
            
        best_val = _metric(x)
        
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            improved = False
            for _ in range(N):
                idx = rng.integers(0, N)
                orig = x[idx]
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    x[idx] = orig + delta
                    val = _metric(x)
                    if val < best_val:
                        best_val = val
                        improved = True
                        break
                    else:
                        x[idx] = orig
                if not improved:
                    break
            if not improved:
                break

    return best_x
