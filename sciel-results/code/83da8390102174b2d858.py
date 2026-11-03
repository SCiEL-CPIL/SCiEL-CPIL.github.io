"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Base solution: constant sequence
    x_const = np.full(N, 1.0)
    score_const = 1.0 / N  # Theoretical lower bound
    
    # Continuous relaxation optimization
    # We use gradient descent to minimize J(x) = max_k |c_k(x)| / (sum x_i)^2
    # Initialization: random signs to explore landscape
    rng = np.random.default_rng(42)
    x = rng.choice([-1.0, 1.0], size=N)
    
    # Hyperparameters
    lr = 0.01
    steps = 500
    eps = 1e-8
    
    best_x = x.copy()
    best_score = float('inf')
    
    for _ in range(steps):
        # Compute autocorrelation via FFT
        # c_k = sum_i x_i x_{i+k}
        # Using FFT: ifft(fft(x) * conj(fft(x))) gives circular autocorrelation.
        # For linear autocorrelation, we need to zero-pad.
        x_fft = np.fft.fft(x, n=2*N)
        power = x_fft * np.conj(x_fft)
        c = np.fft.ifft(power).real
        # c[0] is sum x_i^2. c[k] for k>0 is sum x_i x_{i+k} (roughly, depends on shift)
        # Standard ifft of power spectrum gives circular correlation.
        # c[0] = sum x_i^2
        # c[k] = sum x_i x_{(i+k) mod N}
        # We want linear autoconvolution.
        # Let's compute linear autocorrelation directly for accuracy or use zero-padded FFT.
        # Zero-padded FFT:
        x_pad = np.zeros(2*N)
        x_pad[:N] = x
        x_pad_fft = np.fft.fft(x_pad, n=2*N)
        power_lin = x_pad_fft * np.conj(x_pad_fft)
        c_lin = np.fft.ifft(power_lin).real
        # c_lin[0] = sum x_i^2
        # c_lin[k] for k=1..N-1 is sum x_i x_{i+k}
        # c_lin[N-k] is sum x_i x_{i-k}
        
        # We care about max absolute value.
        # Usually c_lin is symmetric if x is real? No, c_lin[k] = c_lin[-k].
        # In the array c_lin, indices 0..N-1 are positive lags, N..2N-1 are negative lags (wrapped).
        # Actually, ifft gives c[0..N-1] corresponding to lags 0..N-1.
        # c[0] = sum x_i^2
        # c[k] = sum x_i x_{i+k} for k>0
        # c[N-k] = sum x_i x_{i-k} = c[k]
        
        max_abs_c = np.max(np.abs(c_lin[:N]))
        S = np.sum(x)
        S_sq = S**2 + eps
        
        score = max_abs_c / S_sq
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
        # Gradient computation
        # J = A / B where A = max_k |c_k|, B = S^2
        # Let k_max be the index with max |c_k|.
        k_max = np.argmax(np.abs(c_lin[:N]))
        
        # Gradient of A = |c_{k_max}|
        # c_k = sum_i x_i x_{i+k}
        # d c_k / d x_j = x_{j-k} + x_{j+k} (with bounds)
        
        # Construct gradient of c_k
        grad_c = np.zeros(N)
        if k_max == 0:
            # c_0 = sum x_i^2
            # d c_0 / d x_j = 2 x_j
            grad_c = 2 * x
        else:
            # c_k = sum_{i=0}^{N-1-k} x_i x_{i+k}
            # Term x_j appears in:
            # 1. As x_i where i=j: x_j x_{j+k} (if j+k < N)
            # 2. As x_{i+k} where i+k=j => i=j-k: x_{j-k} x_j (if j-k >= 0)
            
            # Vectorized approach using shifts
            # Shift x by k
            x_shifted_k = np.roll(x, -k_max)
            # Mask valid indices
            mask_pos = np.arange(N) + k_max < N
            mask_neg = np.arange(N) - k_max >= 0
            
            # d c_k / d x_j = (x_{j+k} if j+k<N else 0) + (x_{j-k} if j-k>=0 else 0)
            term1 = np.zeros(N)
            term1[:N-k_max] = x[k_max:]
            
            term2 = np.zeros(N)
            term2[k_max:] = x[:N-k_max]
            
            grad_c = term1 + term2
            
        # Sign of c_k to handle absolute value
        sign_c = np.sign(c_lin[k_max])
        grad_A = sign_c * grad_c
        
        # Gradient of B = S^2
        # d S^2 / d x_j = 2 S
        grad_B = 2 * S * np.ones(N)
        
        # Gradient of J = (grad_A * B - A * grad_B) / B^2
        grad_J = (grad_A * S_sq - max_abs_c * grad_B) / (S_sq**2)
        
        # Update x
        x = x - lr * grad_J
        
        # Optional: Project to signs to stay in "signed step" domain?
        # The relaxation allows reals, but final output should be signs.
        # We can periodically project or just keep reals and project at end.
        # Keeping reals allows smoother optimization.
        
    # Final projection to signs
    x_final = np.sign(best_x)
    # Ensure no zeros
    x_final[x_final == 0] = 1.0
    
    # Evaluate final score
    x_pad = np.zeros(2*N)
    x_pad[:N] = x_final
    x_pad_fft = np.fft.fft(x_pad, n=2*N)
    power_lin = x_pad_fft * np.conj(x_pad_fft)
    c_lin = np.fft.ifft(power_lin).real
    max_abs_c = np.max(np.abs(c_lin[:N]))
    S = np.sum(x_final)
    S_sq = S**2 + 1e-8
    score_final = max_abs_c / S_sq
    
    # Compare with constant sequence
    if score_final < score_const:
        return x_final
    else:
        return x_const
# EVOLVE-BLOCK-END
