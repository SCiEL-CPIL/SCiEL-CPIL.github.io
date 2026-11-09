import numpy as np
import time

N = 600
DELTA = 1.0 / N
L = 0.5
_NFFT = 1 << (2 * N - 1).bit_length()

def compute_l2_ratio(h):
    """Compute sum((f*f)^2) * DELTA^2 / (int f)^4. 
    Normalized so that for uniform h, it returns a baseline.
    Actually, let's just return the raw value of sum(conv^2) * DELTA / (sum(h)*DELTA)^2?
    The score is the upper bound on C1.
    C1 is related to max(f*f) / (int f)^2.
    Minimizing L2 of f*f is a proxy.
    Let's define the objective as:
    J = max(conv) / (sum(h)*DELTA)^2  <-- This is what we want to minimize (the score).
    But the gradient of max is non-smooth.
    The knowledge says: "L2 proxy (sum |F|^4) is smooth... enabling stable convergence to a flatter autoconvolution profile".
    So we optimize L2, but we report the Max score.
    
    Objective to optimize: sum(conv^2)
    Score to report: max(conv) / (sum(h)*DELTA)^2
    """
    h = np.maximum(np.asarray(h, dtype=np.float64), 0.0)
    s = np.sum(h) * DELTA
    if s < 1e-15:
        return 1e15, 1e15
    h_norm = h / s
    F = np.fft.rfft(h_norm, _NFFT)
    conv = np.fft.irfft(F * F, _NFFT)[:2*N-1]
    l2_val = np.sum(conv**2) * DELTA
    max_val = np.max(conv) * DELTA
    return l2_val, max_val

def compute_grad_l2(h):
    """Compute gradient of sum(conv^2) w.r.t h_norm."""
    s = np.sum(h) * DELTA
    h_norm = h / s
    F = np.fft.rfft(h_norm, _NFFT)
    # C = conv
    # Grad = 2 * IFFT(|F|^2 * F)
    # Note: rfft/irfft handle the real spectrum.
    # |F|^2 is real. F is complex.
    # The product |F|^2 * F is complex.
    # irfft will take the real part of the inverse FFT of the conjugate symmetric extension.
    # Since F is the rfft of a real signal, F is conjugate symmetric in the full spectrum.
    # |F|^2 is real and symmetric.
    # |F|^2 * F has the symmetry of F.
    # So irfft(|F|^2 * F) gives the real signal C * h.
    
    grad_freq = (np.abs(F)**2) * F
    grad_h = 2.0 * np.fft.irfft(grad_freq, _NFFT)[:N]
    return grad_h * DELTA, s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialize with uniform
    h = np.ones(N)
    
    # Normalize
    s = np.sum(h) * DELTA
    h = h / s
    
    best_h = h.copy()
    _, best_score = compute_l2_ratio(h)
    
    alpha = 1e-4
    tau_anneal = 0.99
    
    for i_iter in range(5000):
        if time.time() - start_time > deadline - 5:
            break
        
        grad_l2, s = compute_grad_l2(h)
        
        # Gradient step
        h_new = h - alpha * grad_l2
        
        # Project to simplex: h >= 0, sum(h)*DELTA = 1
        h_new = np.maximum(h_new, 0)
        s_new = np.sum(h_new) * DELTA
        if s_new > 1e-15:
            h_new = h_new / s_new
        else:
            h_new = h.copy()
            
        # Evaluate score (Max ratio)
        _, score_new = compute_l2_ratio(h_new)
        
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            h = h_new
            # Decay alpha slightly or keep it
            # alpha *= 0.99
        else:
            # If no improvement, reduce alpha
            alpha *= 0.9
            if alpha < 1e-8:
                # Reset alpha and try a random restart or perturbation?
                # For now, just break if alpha is too small
                # Or increase alpha back up?
                # Let's just break to save time if stuck
                pass
                
    h_final = best_h
    s_final = np.sum(h_final) * DELTA
    if s_final > 1e-15:
        h_final = h_final / s_final
        
    return [float(x) for x in h_final]
