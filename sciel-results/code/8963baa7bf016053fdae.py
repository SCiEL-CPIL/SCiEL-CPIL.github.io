import numpy as np
import time

N = 600

def compute_sidelobe_score(f):
    """Compute max sidelobe / main lobe."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    c0 = np.sum(f**2)
    if c0 < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    # conv is symmetric, length 2N-1
    # c[0] is at index N-1
    # Sidelobes are conv[:N-1] and conv[N:]
    max_side = np.max(np.concatenate([conv[:N-1], conv[N:]]))
    return max_side / c0

def construct():
    start_time = time.time()
    best_f = None
    best_score = np.inf
    
    # Multiple restarts with random initializations
    for restart in range(20):
        if time.time() - start_time > 140:
            break
            
        rng = np.random.RandomState(restart)
        # Initialize with random positive values
        f = rng.uniform(0.5, 1.5, N)
        # Normalize
        f = f / np.sqrt(np.sum(f**2))
        
        # Optimization loop
        alpha = 0.01
        beta = 10.0 # For soft max
        for i in range(500):
            if time.time() - start_time > 140:
                break
                
            c = np.convolve(f, f, mode='full')
            c0 = c[N-1]
            # Sidelobes
            sides = np.concatenate([c[:N-1], c[N:]])
            
            # Soft max of sides
            # We want to minimize max(sides)
            # Use log-sum-exp
            m = np.max(sides)
            if m < 1e-15:
                break
            # Scale to avoid overflow
            log_sum = np.log(np.sum(np.exp(beta * (sides - m))))
            soft_max_val = m + log_sum / beta
            
            # Gradient of soft max w.r.t sides
            # d(soft_max)/d(side_k) = exp(beta*(side_k - m)) / sum(exp(...))
            exp_terms = np.exp(beta * (sides - m))
            grad_sides = exp_terms / np.sum(exp_terms)
            
            # Map gradient from sides to f
            # sides[k] corresponds to c[k] and c[2N-2-k]
            # c[k] = sum f_i f_{i+k}
            # We need grad_f
            # This is complex to map directly.
            # Alternative: Use FFT based gradient.
            
            # Let's use the FFT method for gradient.
            # Objective: J = sum_{k!=0} exp(beta c_k)
            # dJ/df = sum_{k!=0} beta exp(beta c_k) dc_k/df
            # dc_k/df = shift(f, k) + shift(f, -k)
            
            # Efficient computation:
            # c = ifft(|fft(f)|^2)
            # We want to reduce c_k for large k.
            # Let w_k = exp(beta c_k) for k != 0, w_0 = 0.
            # grad_f = ifft(fft(w) * conj(fft(f))) ? No.
            # c_k = sum f_i f_{i+k}
            # d c_k / d f_j = f_{j+k} + f_{j-k}
            # grad_f_j = sum_{k!=0} w_k (f_{j+k} + f_{j-k})
            # This is a convolution of w with f?
            # sum_k w_k f_{j+k} = (w * f)_j
            # sum_k w_k f_{j-k} = (w_rev * f)_j
            # So grad_f = w * f + w_rev * f
            
            w = np.zeros(2*N-1)
            w[N-1] = 0 # Exclude main lobe
            # w corresponds to lags
            # sides array is unrolled lags.
            # Let's construct w in the full convolution shape.
            # c is length 2N-1.
            # w[c_index] = exp(beta * c[c_index]) if c_index != N-1 else 0
            w_full = np.exp(beta * c)
            w_full[N-1] = 0
            
            # grad_f = convolve(w_full, f) + convolve(w_full[::-1], f)
            # Note: w_full is symmetric? No, c is symmetric, so w_full is symmetric.
            # So w_full[::-1] = w_full.
            # grad_f = 2 * convolve(w_full, f, mode='same')?
            # Let's check.
            # (w * f)_j = sum_k w_k f_{j-k}
            # We need sum_k w_k f_{j+k} + sum_k w_k f_{j-k}
            # Let m = -k. sum_m w_{-m} f_{j-m} + sum_k w_k f_{j-k}
            # Since w is symmetric (w_{-m} = w_m), this is 2 * (w * f)_j.
            
            grad_f = 2.0 * np.convolve(w_full, f, mode='same')
            
            # Update
            f_new = f - alpha * grad_f
            f_new = np.maximum(f_new, 0.0)
            
            # Normalize
            norm = np.sqrt(np.sum(f_new**2))
            if norm < 1e-15:
                break
            f_new = f_new / norm
            
            # Check improvement
            score_new = compute_sidelobe_score(f_new)
            if score_new < best_score:
                best_score = score_new
                best_f = f_new.copy()
            
            # Adaptive alpha
            # If score improved, keep alpha. Else reduce.
            # We need current score to compare.
            # We can approximate or just use a fixed small alpha with many steps.
            # Let's use a fixed alpha and rely on the soft max smoothing.
            f = f_new
            
    if best_f is None:
        # Fallback to constant if optimization failed
        best_f = np.ones(N)
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_sidelobe_score(f)}")
