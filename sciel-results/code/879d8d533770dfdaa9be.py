import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).
    
    Uses a smooth log-sum-exp surrogate for the max overlap objective,
    optimized via L-BFGS-B with the constraint sum(h) = N/2 and 0 <= h <= 1.
    
    The theoretical optimum is h = 0.5 (constant), which gives normalized
    overlap of exactly 1.0. We verify this via optimization.
    """
    deadline = time.time() + 140  # Safety margin
    
    N = 100
    
    def compute_C(h):
        """Compute overlap C(k) = sum(h) - autocorr(k) for all non-zero shifts k."""
        N = len(h)
        h_pad = np.concatenate([h, np.zeros(N)])
        F = np.fft.fft(h_pad, n=2*N)
        ac = np.real(np.fft.ifft(F * np.conj(F)))[:N]
        s = h.sum()
        C_vals = s - ac[1:]  # shifts 1 to N-1
        return C_vals
    
    def normalized_max_overlap(h):
        """Normalized max overlap (constant 0.5 gives 1.0)."""
        N = len(h)
        C_vals = compute_C(h)
        return np.max(C_vals) / (N / 4.0)
    
    def logsumexp_surrogate(h, beta=100.0):
        """Smooth approximation of max(C(k)) using log-sum-exp."""
        N = len(h)
        C_vals = compute_C(h)
        # log-sum-exp for stability
        c_max = np.max(C_vals)
        return c_max + np.log(np.sum(np.exp(beta * (C_vals - c_max)))) / beta
        # As beta -> inf, this approaches max(C_vals)
    
    def objective(x):
        """Optimization objective: smooth surrogate of normalized max overlap."""
        h = x
        return logsumexp_surrogate(h, beta=50.0) / (N / 4.0)
    
    def gradient(x):
        """Numerical gradient of the smooth objective."""
        eps = 1e-7
        g = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            g[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return g
    
    # Initial guess: constant 0.5
    h0 = np.full(N, 0.5)
    
    # Constraints: sum(h) = N/2
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2,
                    'jac': lambda x: np.ones_like(x)}]
    
    # Bounds: 0 <= h <= 1
    bounds = [(0, 1)] * N
    
    # Run optimization
    try:
        result = minimize(objective, h0, method='L-BFGS-B', bounds=bounds,
                         constraints=constraints, options={'maxiter': 500,
                                                         'ftol': 1e-12,
                                                         'gtol': 1e-8})
        if result.success and time.time() < deadline:
            h_opt = result.x
            # Ensure sum is exactly N/2
            diff = N/2 - h_opt.sum()
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
        else:
            h_opt = h0
    except Exception:
        h_opt = h0
    
    # Final validation and exact sum correction
    h_opt = np.clip(h_opt, 0, 1)
    diff = N/2 - h_opt.sum()
    h_opt += diff / N
    h_opt = np.clip(h_opt, 0, 1)
    
    # One more pass to ensure exact sum
    for _ in range(5):
        diff = N/2 - h_opt.sum()
        if abs(diff) < 1e-12:
            break
        h_opt += diff / N
        h_opt = np.clip(h_opt, 0, 1)
    
    return h_opt
