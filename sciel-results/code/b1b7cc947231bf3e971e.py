"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search with efficient O(N) updates.
We parameterize the signal by its DFT coefficients F.
The autoconvolution c is IFFT(|F|^2).
We use coordinate descent on F.
For each frequency bin k, we try to modify F[k] and F[N-k].
We update the time-domain signal f and the autoconvolution c in O(N) time
using the linearity of the IFFT and the fact that changes are localized in frequency.
We enforce non-negativity of f and minimize the peak of c.
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

def initialize_windows():
    """Generate initial time-domain signals from standard windows."""
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    windows = []
    
    # Hann
    w = 0.5 * (1 - np.cos(2 * np.pi * x))
    windows.append(w)
    
    # Hamming
    w = 0.54 - 0.46 * np.cos(2 * np.pi * x)
    windows.append(w)
    
    # Blackman
    w = 0.42 - 0.5 * np.cos(2 * np.pi * x) + 0.08 * np.cos(4 * np.pi * x)
    windows.append(w)
    
    # Gaussian
    w = np.exp(-20 * (2 * x) ** 2)
    windows.append(w)
    
    # Uniform
    w = np.ones(N)
    windows.append(w)
    
    return windows

def spectral_coordinate_descent(f_init, max_time, start_time, n_iter=50000):
    """
    Coordinate descent on DFT coefficients.
    Maintains f (time), F (freq), S (|F|^2), c (autoconv).
    Updates in O(N).
    """
    f = np.maximum(f_init, 0.0).copy()
    # Normalize f so sum is 1? No, score is invariant to scaling. 
    # But numerical stability might be better. Let's keep sum ~ 1.
    s = np.sum(f)
    if s > 0:
        f /= s
        
    F = np.fft.fft(f)
    S = np.abs(F) ** 2
    c = np.fft.ifft(S).real
    # Note: c from IFFT of |F|^2 is the circular convolution.
    # The linear convolution peak is the same as circular convolution peak 
    # if N is large enough relative to support? 
    # The task says "autoconvolution". Usually implies linear.
    # However, for step functions on [-1/4, 1/4], the support is the whole domain.
    # The circular convolution of a signal with itself on a grid of size N
    # is equivalent to the linear convolution if we consider the periodic extension.
    # The peak of the linear convolution is at lag 0.
    # The peak of the circular convolution is also at lag 0 (sum of squares).
    # Wait, the score is max(conv) / (sum(f))^2.
    # For linear conv, max is at lag 0 (sum f_i^2).
    # For circular conv, max is at lag 0 (sum f_i^2).
    # So I can use circular convolution (IFFT of |F|^2) for the score!
    # This is much faster and consistent with the spectral approach.
    
    current_peak = np.max(c)
    # Score is current_peak / (sum(f))^2. Since we normalized sum(f)=1, score = current_peak.
    # Wait, if we scale f, sum(f) changes. 
    # Let's keep f unnormalized during optimization to allow amplitude changes?
    # No, the score is scale-invariant. 
    # Let's fix sum(f) = 1.
    # Then score = max(c).
    
    best_f = f.copy()
    best_peak = current_peak
    
    half = N // 2
    step_size = 0.01
    
    # Precompute cosines for each k to speed up update?
    # cos(2*pi*k*n/N)
    # We can compute on the fly or precompute a matrix? Matrix is N x N/2, 600x300 = 180k floats. OK.
    # But O(N) update per step is fine.
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
            
        improved = False
        # Shuffle order? Or fixed? Fixed is deterministic.
        # Let's try fixed order first.
        
        for k in range(1, half + 1):
            # Try perturbations
            # We modify F[k] and F[N-k]
            # Let F[k] = a + ib.
            # We can change a or b.
            
            # Try changing Real part
            for sign in [1, -1]:
                delta_F = sign * step_size
                F_new_k = F[k] + delta_F
                F_new_Nk = F[N-k] + delta_F # Conjugate symmetry: if F[k] real part changes, F[N-k] real part changes same way?
                # F[N-k] = conj(F[k]).
                # If F[k] = a+ib, F[N-k] = a-ib.
                # Change a -> a+da. F[k] = a+da+ib. F[N-k] = a+da-ib.
                # So delta is same for both.
                
                # Check f update
                # delta_f = Re( IFFT(delta_F_vec) )
                # delta_F_vec has delta_F at k, delta_F at N-k.
                # IFFT is (1/N) * sum.
                # delta_f[n] = (1/N) * (delta_F * exp(i 2 pi k n / N) + delta_F * exp(i 2 pi (N-k) n / N))
                # = (2 delta_F / N) * cos(2 pi k n / N)
                
                cos_term = np.cos(2 * np.pi * k * np.arange(N) / N)
                delta_f = (2 * delta_F / N) * cos_term
                
                f_trial = f + delta_f
                if np.any(f_trial < -1e-12):
                    continue
                
                # Check c update
                # S = |F|^2.
                # S_new_k = |F_new_k|^2
                # delta_S = S_new_k - S[k]
                S_new_k = np.abs(F_new_k) ** 2
                delta_S = S_new_k - S[k]
                
                # delta_c[n] = (2 delta_S / N) * cos(2 pi k n / N)
                # Note: S is real and even. 
                # The IFFT of a change in S[k] and S[N-k] (which are equal) 
                # is (2 delta_S / N) cos(...).
                
                delta_c = (2 * delta_S / N) * cos_term
                c_trial = c + delta_c
                peak_trial = np.max(c_trial)
                
                if peak_trial < best_peak:
                    # Accept
                    f = f_trial
                    c = c_trial
                    F[k] = F_new_k
                    F[N-k] = F_new_k.conjugate() # Ensure symmetry
                    S[k] = S_new_k
                    S[N-k] = S_new_k
                    best_peak = peak_trial
                    best_f = f.copy()
                    improved = True
                    step_size *= 1.1
                    break # Move to next k
            
            if not improved:
                # Try Imag part
                for sign in [1, -1]:
                    delta_F = sign * step_size * 1j
                    F_new_k = F[k] + delta_F
                    # F[N-k] = conj(F_new_k) = conj(F[k] + i*da) = conj(F[k]) - i*da
                    # So F[N-k] changes by -delta_F?
                    # Let F[k] = a+ib. F[N-k] = a-ib.
                    # Change b -> b+db.
                    # F[k] = a + i(b+db).
                    # F[N-k] = a - i(b+db).
                    # Delta F[k] = i*db.
                    # Delta F[N-k] = -i*db.
                    # So delta at N-k is -delta_F.
                    
                    # delta_f[n] = (1/N) * (delta_F * exp(i 2 pi k n / N) + (-delta_F) * exp(i 2 pi (N-k) n / N))
                    # = (delta_F / N) * (exp(i 2 pi k n / N) - exp(-i 2 pi k n / N))
                    # = (delta_F / N) * (2i * sin(2 pi k n / N))
                    # Since delta_F = i * da (real da),
                    # delta_f[n] = (i*da / N) * 2i * sin(...) = (-2*da/N) * sin(...)
                    
                    sin_term = np.sin(2 * np.pi * k * np.arange(N) / N)
                    da = sign * step_size
                    delta_f = (-2 * da / N) * sin_term
                    
                    f_trial = f + delta_f
                    if np.any(f_trial < -1e-12):
                        continue
                    
                    S_new_k = np.abs(F_new_k) ** 2
                    delta_S = S_new_k - S[k]
                    
                    delta_c = (2 * delta_S / N) * np.cos(2 * np.pi * k * np.arange(N) / N)
                    c_trial = c + delta_c
                    peak_trial = np.max(c_trial)
                    
                    if peak_trial < best_peak:
                        f = f_trial
                        c = c_trial
                        F[k] = F_new_k
                        F[N-k] = F_new_k.conjugate()
                        S[k] = S_new_k
                        S[N-k] = S_new_k
                        best_peak = peak_trial
                        best_f = f.copy()
                        improved = True
                        step_size *= 1.1
                        break
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
                
    return best_f, best_peak

def construct():
    start_time = time.time()
    
    best_f = None
    best_score = np.inf
    
    inits = initialize_windows()
    
    # Add some random inits
    for _ in range(5):
        f_rand = np.random.rand(N)
        f_rand += 0.1 # Ensure positive
        inits.append(f_rand)
        
    for f_init in inits:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        f_opt, peak_opt = spectral_coordinate_descent(f_init, TIME_LIMIT - 2, start_time, n_iter=20000)
        
        # Verify score with linear convolution to be safe
        actual_score = compute_score(f_opt)
        
        if actual_score < best_score:
            best_score = actual_score
            best_f = f_opt
            
    if best_f is None:
        best_f = np.ones(N)
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
