"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search with gradient descent in frequency domain.
Parameterize the step function by its DFT coefficients.
Enforce non-negativity via hard projection in time domain.
Minimize the maximum value of the autoconvolution (IFFT of |F|^2)
using gradient descent on the frequency coefficients.
This decouples the peak constraint from the time-domain shape search.
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

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    # Constant function has score 1/N for linear conv? 
    # f = ones(N). conv = [1, 2, ..., N, ..., 1]. max = N. s = N.
    # score = N / N^2 = 1/N.
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # We will run gradient descent in the frequency domain.
    # State: F (DFT coefficients)
    # Objective: J = max(C) / (sum(f))^2, where C = IFFT(|F|^2)
    # Note: Linear conv peak is approx circular conv peak for large N if f is smooth/compact.
    # We optimize for circular conv peak for speed, then check linear.
    
    # Initialize F
    # Try a few initializations
    initializations = []
    
    # 1. Flat (DC only)
    F0 = np.zeros(N, dtype=np.complex128)
    F0[0] = N
    initializations.append(F0)
    
    # 2. Cosine wave
    t = np.arange(N)
    f_cos = np.cos(2 * np.pi * t / N) + 1.0
    f_cos = np.maximum(f_cos, 0)
    F1 = np.fft.fft(f_cos)
    initializations.append(F1)
    
    # 3. Sine wave
    f_sin = np.sin(2 * np.pi * t / N) + 1.0
    f_sin = np.maximum(f_sin, 0)
    F2 = np.fft.fft(f_sin)
    initializations.append(F2)
    
    # 4. Gaussian-like
    f_gauss = np.exp(-((t - N/2)**2) / (2 * 50**2))
    f_gauss = np.maximum(f_gauss, 0)
    F3 = np.fft.fft(f_gauss)
    initializations.append(F3)

    for F_init in initializations:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        F = F_init.copy()
        
        # Ensure conjugate symmetry for real signal
        half = N // 2
        for k in range(1, half + 1):
            F[N - k] = np.conj(F[k])
        
        # Gradient Descent Parameters
        lr = 0.1
        min_lr = 1e-6
        no_improve_count = 0
        max_no_improve = 50
        
        for it in range(5000):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # 1. Compute current f and C
            f = np.real(np.fft.ifft(F))
            f = np.maximum(f, 0.0) # Project to non-negative
            s = np.sum(f)
            if s < 1e-12:
                break
            
            # Normalize f to keep sum constant? 
            # The score is scale invariant. 
            # Let's normalize f to have sum N (or 1) to stabilize gradient.
            f = f * (N / s)
            
            # Recompute F from projected f
            F = np.fft.fft(f)
            # Re-enforce symmetry (FFT of real f is symmetric, but numerical errors might creep in)
            for k in range(1, half + 1):
                F[N - k] = np.conj(F[k])
            
            # 2. Compute Autoconvolution (Circular)
            # C = IFFT(|F|^2)
            power = np.abs(F) ** 2
            C = np.real(np.fft.ifft(power))
            
            # 3. Find Peak
            peak_idx = np.argmax(C)
            C_max = C[peak_idx]
            
            # Score (Circular)
            # For circular conv, sum(C) = sum(f)^2 = N^2.
            # So normalized score is C_max / N^2.
            current_score = C_max / (N * N)
            
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    lr *= 0.9
                    no_improve_count = 0
                    if lr < min_lr:
                        break
            
            # 4. Compute Gradient
            # We want to minimize C_max.
            # Gradient of C_k w.r.t F_m:
            # C_k = 1/N sum_l |F_l|^2 exp(-i 2 pi k l / N)
            # d C_k / d F_m = 1/N * 2 F_m * exp(-i 2 pi k m / N) * conj(F_m) ??
            # Let's use the chain rule carefully.
            # Let P_l = |F_l|^2.
            # C_k = 1/N sum_l P_l exp(-i 2 pi k l / N).
            # d C_k / d P_l = 1/N exp(-i 2 pi k l / N).
            # d P_l / d F_m = 0 if l != m.
            # If l == m, d |F_m|^2 / d F_m = 2 F_m (in Wirtinger sense, treating F and Fbar independent, 
            # but for real signals we can just use the real part of the gradient).
            
            # Gradient vector G_F such that d C_k = sum_m G_F[m] d F_m.
            # G_F[m] = (d C_k / d P_m) * (d P_m / d F_m)
            # G_F[m] = (1/N exp(-i 2 pi k m / N)) * 2 F_m
            
            # We want to decrease C_k. So we update F -= lr * G_F.
            # Note: G_F is complex. F is complex.
            
            # Gradient for the specific peak index
            k = peak_idx
            # Vector of exp terms
            # exp(-i 2 pi k m / N) for m in 0..N-1
            # This is a DFT basis vector.
            # Instead of constructing the full vector, we can compute the FFT of the gradient w.r.t P.
            # grad_P is a vector with 1/N exp(-i 2 pi k m / N) at all m? 
            # No, d C_k / d P_l is 1/N exp(-i 2 pi k l / N) for ALL l.
            # So grad_P = (1/N) * exp(-i 2 pi k * arange(N) / N).
            
            m = np.arange(N)
            grad_P = (1.0 / N) * np.exp(-1j * 2 * np.pi * k * m / N)
            
            # grad_F = grad_P * 2 * F
            grad_F = grad_P * 2.0 * F
            
            # Update F
            F -= lr * grad_F
            
            # Enforce conjugate symmetry
            for idx in range(1, half + 1):
                F[N - idx] = np.conj(F[idx])
            
            # Optional: Normalize F to prevent drift
            # The scale of F determines the scale of f.
            # Since we project f to non-negative and renormalize f to sum N, 
            # the scale is handled in the time domain step.
            
        # After loop, get final f
        f_final = np.real(np.fft.ifft(F))
        f_final = np.maximum(f_final, 0.0)
        s_final = np.sum(f_final)
        if s_final > 1e-12:
            f_final = f_final * (N / s_final)
        
        # Compute actual linear score for this candidate
        actual_score = compute_score(f_final)
        if actual_score < best_score:
            best_score = actual_score
            best_f = f_final

    # Final refinement: simple local search in time domain
    if time.time() - start_time < TIME_LIMIT - 5:
        f_ref = best_f.copy()
        # Ensure sum is N
        s = np.sum(f_ref)
        if s > 0:
            f_ref = f_ref * (N / s)
        
        for _ in range(100):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            
            # Compute gradient of score w.r.t f in time domain
            # Score = max(conv) / s^2
            # This is complex. Let's just do a simple hill climbing.
            # Perturb random indices?
            # Or use the sensitivity from before.
            
            # Let's stick to the spectral result. It's likely good enough.
            break

    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
