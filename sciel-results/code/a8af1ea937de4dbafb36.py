import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Precompute FFT basis for fast convolution
    # We use FFT to compute autoconvolution c = conv(h, h)
    # c_k = sum_i h_i h_{i+k}
    # This is equivalent to inverse FFT of |FFT(h)|^2
    
    # We want to minimize V = 2N * max(c) / (sum h)^2
    # By scaling invariance, we can fix sum(h) = N (so mean is 1).
    # Then V = 2 * max(c).
    # We want to minimize max(c).
    # max(c) >= ||c||_1 / (2N-1) is not quite right, but minimizing L1 norm of c
    # generally pushes down the peaks if the energy is spread out.
    # Actually, min max(c) is a convex problem?
    # max_k sum_i h_i h_{i+k} is convex in h?
    # The function f(h) = max_k (h^T A_k h) is convex if A_k are PSD.
    # A_k corresponds to shift matrices. For k=0, A_0 = I (PSD).
    # For k != 0, A_k is not symmetric, but h^T A_k h is not necessarily convex?
    # Wait, c_k = sum_i h_i h_{i+k}.
    # If we view h as a vector, c is a quadratic form.
    # Minimizing the maximum of a set of convex functions is convex.
    # Is c_k convex in h?
    # c_k(h) = sum_i h_i h_{i+k}.
    # Hessian is symmetric part of shift matrix.
    # For k=0, Hessian is 2I (convex).
    # For k != 0, the quadratic form is not convex in general?
    # Actually, let's look at the spectral representation.
    # c_k = (1/2pi) int H(e^{i w}) H(e^{-i w}) e^{i k w} dw ...
    # It's simpler to use Subgradient on the L1 norm of c, which is a convex surrogate
    # for the L-infinity norm (since L1 <= (2N-1) L-inf, and we want to reduce L-inf).
    # Minimizing L1(c) subject to sum(h)=N is a convex problem.
    # Gradient of L1(c) w.r.t h:
    # c = IFFT(|FFT(h)|^2)
    # Let H = FFT(h). C = H * conj(H) = |H|^2.
    # c = IFFT(C).
    # L1(c) = sum_j |c_j|.
    # d L1 / d c_j = sign(c_j).
    # d c / d h: c = h * h (convolution).
    # In FFT domain, C = H^2 (element-wise? No, C = H * conj(H) for real h? 
    # Wait, autoconvolution of real h:
    # FFT(h * h) = FFT(h) * FFT(h) ? No.
    # FFT(conv(h, g)) = FFT(h) * FFT(g).
    # So FFT(c) = FFT(h) * FFT(h) = H^2.
    # Since h is real, H has conjugate symmetry.
    # C = H^2.
    # c = IFFT(C).
    # We want grad_h L1(c).
    # L1(c) = sum |c_j|.
    # d L1 / d h = (d c / d h)^T sign(c).
    # c = IFFT(H^2).
    # Let's use the chain rule in FFT domain.
    # L1(c) = sum_j |IFFT(C)_j|.
    # This is tricky. Let's use the property that for real signals,
    # the gradient of sum |c_j| w.r.t h can be computed via:
    # grad_h = 2 * IFFT( sign(c) * H ) ? 
    # Let's verify.
    # c = h * h.
    # <g, h> = sum g_i h_i.
    # d L1 / d h_i = sum_j sign(c_j) d c_j / d h_i.
    # c_j = sum_k h_k h_{k-j}.
    # d c_j / d h_i = h_{i-j} + h_{i+j} (with boundary handling for linear conv).
    # This is essentially the cross-correlation of sign(c) with h.
    # In FFT: grad_h = IFFT( FFT(sign(c)) * H )?
    # Let's check:
    # FFT( sign(c) * h )? No.
    # The operation d c / d h is related to convolution.
    # c = h * h.
    # Variation dc = h * dh + dh * h = 2 h * dh (if commutative, which it is).
    # So <sign(c), dc> = <sign(c), 2 h * dh> = <2 (h * sign(c)), dh>.
    # So grad_h = 2 * (h * sign(c)).
    # In FFT: FFT(grad_h) = 2 * FFT(h) * FFT(sign(c)) = 2 * H * S, where S = FFT(sign(c)).
    # So grad_h = IFFT( 2 * H * S ).
    
    def compute_grad_and_value(h):
        # h is 1D array
        # Ensure h is real
        H = np.fft.fft(h)
        C = H * H  # This is FFT of linear convolution? 
        # Wait. np.fft.fft computes circular convolution if not padded.
        # We need linear convolution length 2N-1.
        # Pad h to size 2N-1 (or next power of 2) for FFT.
        # Let's use size 2*N for simplicity (power of 2 is 512 for N=400? 2*400=800, next pow2 1024).
        n_fft = 1024
        h_pad = np.zeros(n_fft)
        h_pad[:N] = h
        H = np.fft.fft(h_pad)
        C = H * H  # This gives FFT of circular conv of padded h with itself.
        # Since h_pad is zero-padded to 2N-1 effectively (1024 > 799),
        # the first 2N-1 coefficients of IFFT(C) are the linear autoconvolution.
        c_full = np.fft.ifft(C).real
        c = c_full[:2*N-1]
        
        # Value
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, None
        val = abs(2.0 * N * np.max(c) / (s * s))
        
        # Gradient of L1(c) w.r.t h
        # We want to minimize L1(c) to push down peaks.
        # grad_h = IFFT( 2 * H * FFT(sign(c)) )
        # But we need to be careful with the mapping.
        # sign(c) is length 2N-1.
        # Pad sign(c) to n_fft.
        sc = np.sign(c)
        sc_pad = np.zeros(n_fft)
        sc_pad[:2*N-1] = sc
        S = np.fft.fft(sc_pad)
        
        # grad_h_pad = IFFT( 2 * H * S )
        grad_h_pad = np.fft.ifft(2.0 * H * S).real
        grad_h = grad_h_pad[:N]
        
        return val, grad_h

    # Initialize h with mean 1
    h = np.ones(N)
    rng = np.random.default_rng(42)
    
    # Add small random noise to break symmetry
    h += rng.normal(0, 0.01, N)
    # Normalize to sum N
    h *= N / np.sum(h)
    
    best_h = h.copy()
    best_val = compute_grad_and_value(h)[0]
    
    # Projected Subgradient Method
    # Minimize L1(c) subject to sum(h) = N
    # Step size: eta_t = c / sqrt(t)
    # Projection onto sum(h)=N: h = h - (sum(h)-N)/N * ones
    
    eta = 1.0
    alpha_decay = 0.995
    
    for i in range(50000):
        if time.time() > deadline:
            break
            
        val, grad = compute_grad_and_value(h)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        # Subgradient step
        # h_new = h - eta * grad
        h_new = h - eta * grad
        
        # Project onto sum(h) = N
        # h_proj = h_new - (sum(h_new) - N) / N * ones
        s_new = np.sum(h_new)
        h_new -= (s_new - N) / N
        
        h = h_new
        
        # Decay step size
        eta *= alpha_decay
        if eta < 1e-6:
            break
            
    # Final polish with Nelder-Mead on a few random restarts if time permits?
    # Or just return best_h
    
    # One more check: try a few random restarts with different initial noise
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            h = np.ones(N)
            h += rng.normal(0, 0.1, N)
            h *= N / np.sum(h)
            
            eta = 1.0
            for i in range(5000):
                if time.time() > deadline - 5:
                    break
                val, grad = compute_grad_and_value(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                h_new = h - eta * grad
                s_new = np.sum(h_new)
                h_new -= (s_new - N) / N
                h = h_new
                eta *= 0.995
                if eta < 1e-6:
                    break

    return best_h

