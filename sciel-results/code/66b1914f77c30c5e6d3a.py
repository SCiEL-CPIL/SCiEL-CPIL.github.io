import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    # Pre-compute FFT size
    n_fft = 1 << (2 * N - 1).bit_length()
    
    # Soft-max parameter for differentiability
    # Too high = too sharp (hard to optimize), too low = too flat (poor approximation)
    alpha = 20.0 

    def compute_max_autocorr_soft(h):
        """
        Compute a smooth approximation of the maximum autocorrelation.
        h: continuous vector
        """
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        acf = acf[:N]
        
        # Soft-max approximation of max(acf)
        # We want to maximize the negative of this to minimize the max autocorr
        m = np.max(acf)
        # To avoid overflow/underflow in exp, subtract max
        # exp(alpha * (acf - m)) / sum(exp(alpha * (acf - m))) * m + ... 
        # Actually, soft_max(x) = (1/alpha) * log(sum(exp(alpha*x)))
        # For stability:
        shifted = alpha * (acf - m)
        exp_shifted = np.exp(shifted)
        log_sum = np.log(np.sum(exp_shifted))
        soft_max_val = m + log_sum / alpha
        
        return soft_max_val

    def objective_and_gradient(x):
        """
        Objective: minimize the soft-max of autocorrelation.
        Gradient: computed via FFT derivatives.
        """
        h = x
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        acf = acf[:N]
        
        # Soft-max value
        m = np.max(acf)
        shifted = alpha * (acf - m)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        val = m + np.log(sum_exp) / alpha
        
        # Gradient of soft_max(acf) w.r.t. h
        # d(acf_k)/dh_j is a linear operator.
        # acf = ifft(|fft(h)|^2)
        # This is a quadratic form. 
        # Instead of deriving the full matrix-vector product analytically which is complex 
        # and error-prone in 130s, we can use a simpler gradient estimation or 
        # a specific property.
        
        # However, L-BFGS-B requires an accurate gradient.
        # Let's derive the gradient of the max autocorrelation term.
        # acf[k] = sum_j h[j] h[j+k] (circular)
        # d(acf[k])/dh[i] = h[i+k] + h[i-k] (with circular indexing)
        # 
        # Gradient of soft_max = sum_k weights[k] * d(acf[k])/dh
        # 
        # We can compute this efficiently using FFT or direct convolution.
        # Since N=400, direct O(N^2) is 160,000 ops, which is trivial for Python/NumPy.
        # But let's try to be efficient.
        
        # d(acf)/dh is a circulant-like matrix but for autocorrelation it's symmetric.
        # grad[i] = sum_k weights[k] * (h[(i+k)%N] + h[(i-k)%N])
        
        # We can compute this using FFT:
        # h[(i+k)%N] is a circular shift.
        # Let W = weights.
        # grad = ifft(fft(W) * (fft(h) + fft(h)[::-1] ... ))? 
        # Actually, circular convolution of W with h gives sum_k W[k] h[i-k].
        # And circular correlation is sum_k W[k] h[i+k].
        
        # conv = ifft(fft(W) * fft(h)) -> sum_k W[k] h[i-k]
        # corr = ifft(fft(W) * conj(fft(h))) -> sum_k W[k] h[i+k] ? No, correlation is different.
        
        # Let's just use the direct sum for robustness and speed (N=400 is small).
        # grad = np.zeros(N)
        # for i in range(N):
        #     grad[i] = np.sum(weights * (np.roll(h, -np.arange(N)) + np.roll(h, np.arange(N))))
        # This is O(N^2) = 160k operations. Very fast.
        
        # Vectorized O(N^2) approach:
        # shifts = np.arange(N)
        # h_plus = np.roll(h, -shifts, axis=0) # h[i+k] for all i,k
        # h_minus = np.roll(h, shifts, axis=0) # h[i-k] for all i,k
        # grad = weights @ (h_plus + h_minus)
        
        # Wait, np.roll(h, -shifts) creates a matrix of size N x N.
        # h_plus[i, k] = h[(i+k)%N]
        # We want sum_k weights[k] * h[(i+k)%N]
        # This is weights @ h_plus.T ?
        # h_plus is N x N. h_plus.T is N x N where [k, i] = h[(i+k)%N]
        # weights @ h_plus.T -> [i] = sum_k weights[k] h_plus.T[k, i] = sum_k weights[k] h[(i+k)%N]
        # Correct.
        
        shifts = np.arange(N)
        h_plus = np.roll(h, -shifts, axis=0)
        h_minus = np.roll(h, shifts, axis=0)
        
        # grad[i] = sum_k weights[k] (h[i+k] + h[i-k])
        grad = (weights @ h_plus.T + weights @ h_minus.T)
        
        # The objective is to MINIMIZE the soft max autocorr.
        # So we return val, grad
        return val, grad

    best_h = None
    best_val = float('inf')
    
    # Multiple random restarts to escape local minima
    rng = np.random.RandomState(123)
    
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize with random continuous values, e.g., uniform [-1, 1]
        # Or start near a random binary sequence
        x0 = rng.uniform(-1, 1, N)
        
        try:
            res = minimize(
                objective_and_gradient, 
                x0, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            if res.success or res.status == 0:
                # Threshold the result to binary
                h_bin = np.sign(res.x)
                # Handle zeros by randomizing or defaulting to 1
                h_bin[h_bin == 0] = 1
                
                # Evaluate true max autocorrelation on the binary sequence
                F = np.fft.fft(h_bin, n=n_fft)
                power = np.abs(F) ** 2
                acf = np.fft.ifft(power).real
                true_val = np.max(acf[:N])
                
                if true_val < best_val:
                    best_val = true_val
                    best_h = h_bin.copy()
                    
        except Exception:
            continue

    # If optimization failed to find anything better than a random guess, 
    # fall back to a simple random binary sequence evaluation
    if best_h is None:
        h_bin = rng.choice([1, -1], N)
        F = np.fft.fft(h_bin, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        best_val = np.max(acf[:N])
        best_h = h_bin

    return best_h
