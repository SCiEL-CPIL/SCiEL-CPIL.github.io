import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _autocorr(x):
    """Compute linear autocorrelation R(k) for k=0..N-1."""
    n = len(x)
    # Zero-pad to 2n
    x_pad = np.zeros(2*n, dtype=np.complex128)
    x_pad[:n] = x
    X = np.fft.fft(x_pad)
    Y = np.fft.ifft(X * np.conj(X))
    return Y.real[:n]

def _peak_sidelobe(x):
    """Compute max |R(k)| for k=1..N-1."""
    R = _autocorr(x)
    return np.max(np.abs(R[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_value = np.inf
    best_x = np.ones(N)
    
    # Try multiple initializations
    for init_idx in range(20):
        if time.time() > deadline:
            break
        
        # Initialize z (continuous variables)
        if init_idx == 0:
            # Thue-Morse based initialization
            z = np.zeros(N)
            for i in range(N):
                # Count bits
                cnt = 0
                v = i
                while v:
                    cnt += v & 1
                    v >>= 1
                z[i] = 1.0 if cnt % 2 == 0 else -1.0
        elif init_idx == 1:
            # Random
            z = np.random.randn(N) * 0.5
        else:
            # Random with larger scale
            z = np.random.randn(N) * 2.0
        
        # Smooth surrogate objective
        tau = 5.0  # temperature for soft-max
        eps = 1e-6
        
        def objective(z):
            if time.time() > deadline:
                return 1e10
            x = np.tanh(z)
            R = _autocorr(x)
            s = np.sqrt(R[1:]**2 + eps)
            # Soft-max: (1/tau) * log(sum exp(tau * s))
            max_s = np.max(s)
            log_sum = max_s + np.log(np.sum(np.exp(tau * (s - max_s))))
            return log_sum / tau
        
        def gradient(z):
            if time.time() > deadline:
                return np.ones(N)
            x = np.tanh(z)
            sech2 = 1.0 - x**2
            R = _autocorr(x)
            s = np.sqrt(R[1:]**2 + eps)
            # Weights for soft-max
            max_s = np.max(s)
            exp_terms = np.exp(tau * (s - max_s))
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp  # These are the "attention" weights
            
            # ds_k/dz_j = (R(k)/s_k) * dR(k)/dz_j
            # dR(k)/dz_j = sech2[j] * (x[j+k] if j+k<N else 0 + x[j-k] if j-k>=0 else 0)
            
            # For each k in 1..N-1, dR(k)/dz is a vector of length N
            # dR(k)/dz_j = sech2[j] * (x[j+k] * I(j+k<N) + x[j-k] * I(j-k>=0))
            
            # Let's compute this as: for each k, the vector is sech2 * (shifted_x_plus + shifted_x_minus)
            # where shifted_x_plus[j] = x[j+k] if j+k<N else 0
            # and shifted_x_minus[j] = x[j-k] if j-k>=0 else 0
            
            # We can compute the full gradient as:
            # grad[j] = sum_k weights[k] * (R[k]/s[k]) * sech2[j] * (x[j+k]*I(j+k<N) + x[j-k]*I(j-k>=0))
            
            # Let's compute this efficiently.
            # For a fixed j, the sum over k is:
            # sech2[j] * sum_k weights[k] * (R[k]/s[k]) * (x[j+k]*I(j+k<N) + x[j-k]*I(j-k>=0))
            
            # Let w[k] = weights[k] * R[k] / s[k] for k=1..N-1
            # Then grad[j] = sech2[j] * (sum_{k=1}^{N-1-j} w[k]*x[j+k] + sum_{k=1}^{j} w[k]*x[j-k])
            
            w = weights * R[1:] / s  # length N-1
            
            # For each j, we need:
            # sum_{k=1}^{N-1-j} w[k]*x[j+k] + sum_{k=1}^{j} w[k]*x[j-k]
            
            # This is a convolution-like operation.
            # Let's compute it directly for N=400 (O(N^2) = 160k ops)
            
            grad = np.zeros(N)
            for j in range(N):
                # First sum: k from 1 to N-1-j
                if N-1-j >= 1:
                    s1 = np.dot(w[:N-1-j], x[j+1:j+N-j])
                else:
                    s1 = 0.0
                # Second sum: k from 1 to j
                if j >= 1:
                    s2 = np.dot(w[:j], x[j-1::-1])
                else:
                    s2 = 0.0
                grad[j] = sech2[j] * (s1 + s2)
            
            return grad
        
        # Run L-BFGS-B
        try:
            result = minimize(objective, z, jac=gradient, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            z_opt = result.x
        except Exception:
            z_opt = z
        
        # Round to ±1
        x_rounded = np.sign(z_opt)
        x_rounded[x_rounded == 0] = 1.0
        
        # Local search polish
        for ls_iter in range(10):
            if time.time() > deadline:
                break
            improved = False
            for j in range(N):
                if time.time() > deadline:
                    break
                x_try = x_rounded.copy()
                x_try[j] *= -1
                val = _peak_sidelobe(x_try)
                if val < best_value:
                    best_value = val
                    best_x = x_try
                    x_rounded = x_try
                    improved = True
            if not improved:
                break
        
        # Also evaluate the rounded solution
        val = _peak_sidelobe(x_rounded)
        if val < best_value:
            best_value = val
            best_x = x_rounded.copy()
    
    return best_x
