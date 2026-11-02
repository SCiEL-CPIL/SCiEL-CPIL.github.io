import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Initial guess
    h = np.full(N, 0.5)
    
    # Precompute shifts
    shifts = np.arange(-(N-1), N)
    
    def overlaps(h_vec):
        # Calculate O(k) for all k
        # O(k) = sum_i h_i (1 - h_{i+k})
        # Using circular shifts for simplicity in calculation? 
        # No, task implies linear shifts on [0,2] usually.
        # But "shifted 1-h" on [0,2] with steps.
        # If linear, we need to handle boundaries.
        # Let's assume circular first as it's more common in such "overlap" problems unless specified "linear".
        # However, the "Erdos" problem usually refers to circular or specific domains.
        # Given "integral 1 on [0,2]", it's a function on a circle or interval.
        # If interval, boundary effects matter.
        # Let's stick to circular for now as it allows FFT and is symmetric.
        # If circular, O(k) is periodic.
        
        # Compute circular cross-correlation
        # O(k) = sum h_i (1 - h_{i+k})
        # = sum h_i - sum h_i h_{i+k}
        # = N/2 - autocorr(h)[k]
        
        # FFT for autocorrelation
        n_fft = 1
        while n_fft < 2*N:
            n_fft *= 2
            
        H = np.fft.fft(h_vec, n_fft)
        AC = np.fft.ifft(H * np.conj(H)).real
        # AC[k] is sum h_i h_{i+k} (circular)
        # We only need k in [0, N-1]
        AC = AC[:N]
        
        O = N/2 - AC
        return O

    def objective(h_vec):
        O = overlaps(h_vec)
        # Smooth max
        beta = 50.0
        return np.logsumexp(beta * O) / beta
        
    def jac(h_vec):
        # Gradient of logsumexp(beta * O)
        O = overlaps(h_vec)
        beta = 50.0
        weights = np.exp(beta * (O - np.max(O)))
        weights /= np.sum(weights)
        
        # d O / d h
        # O[k] = N/2 - sum_i h_i h_{i+k}
        # d O[k] / d h_j = - sum_i (delta_{i,j} h_{i+k} + h_i delta_{i+k, j})
        # = - h_{j+k} - h_{j-k} (circular)
        
        # We need grad = sum_k weights[k] * d O[k] / d h
        # This is a convolution.
        # d O / d h is the negative of the "cross-correlation" of h with delta?
        # Actually, d (h * h)[k] / d h_j = h_{j+k} + h_{j-k}.
        
        # Let's compute it directly for N=100, it's fast enough.
        grad = np.zeros(N)
        for k in range(N):
            if weights[k] < 1e-6: continue
            # d O[k] / d h_j = - (h_{j+k} + h_{j-k})
            # Vectorized:
            dOk_dh = -(np.roll(h_vec, -k) + np.roll(h_vec, k))
            grad += weights[k] * dOk_dh
            
        return grad

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2},
        {'type': 'ineq', 'fun': lambda x: x},
        {'type': 'ineq', 'fun': lambda x: 1 - x}
    ]
    
    # Use SLSQP or L-BFGS-B
    # L-BFGS-B doesn't support equality constraints well directly in some versions, 
    # but we can project or use SLSQP.
    # SLSQP is slow for N=100? Maybe.
    # Let's try a simple projected gradient descent.
    
    h_curr = h.copy()
    for i in range(1000):
        # Compute grad
        O = overlaps(h_curr)
        beta = 50.0
        weights = np.exp(beta * (O - np.max(O)))
        weights /= np.sum(weights)
        
        grad = np.zeros(N)
        for k in range(N):
            if weights[k] < 1e-6: continue
            dOk_dh = -(np.roll(h_curr, -k) + np.roll(h_curr, k))
            grad += weights[k] * dOk_dh
        
        # Projected Gradient Descent
        lr = 0.01
        h_new = h_curr - lr * grad
        
        # Project to simplex {sum=1, 0<=x<=1}
        # Normalize
        h_new = np.clip(h_new, 0, 1)
        # Adjust sum
        diff = N/2 - np.sum(h_new)
        # Add diff to elements with slack
        # Simple: add diff/N
        h_new += diff / N
        h_new = np.clip(h_new, 0, 1)
        # Re-normalize exactly
        # This simple projection might not be exact.
        
        h_curr = h_new
        
    return h_curr
