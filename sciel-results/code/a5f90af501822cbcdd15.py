"""Erdos minimum overlap: find equal-width step heights h in [0,1], sum N/2,
minimizing the maximum overlap of h with shifted (1-h)."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _overlap_max(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts k=0..N-1.
    O_k = sum(h) - autocorr(k) = N/2 - A_k where A_k = sum h_i * h_{i+k mod N}
    We want to minimize max_k O_k.
    """
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H))) * N
    O = N/2.0 - A
    return float(np.max(O))

def _objective(params):
    """Objective function for scipy optimizer: maximize min autocorrelation."""
    h = params
    h = np.clip(h, 0.0, 1.0)
    # Fix sum to N/2
    current_sum = h.sum()
    if current_sum > 0:
        h = h * (N/2.0) / current_sum
        h = np.clip(h, 0.0, 1.0)
    # Re-fix sum
    diff = N/2.0 - h.sum()
    if abs(diff) > 1e-10:
        # Distribute adjustment
        mask = (h > 0.01) & (h < 0.99)
        if mask.sum() > 0:
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
    # Compute negative min autocorrelation (we want to maximize min A_k)
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H))) * N
    return -np.min(A)

def construct():
    """
    Returns a sequence of step heights h in [0, 1] with sum N/2.
    
    Strategy: Use scipy optimization to directly maximize the minimum circular
    autocorrelation, which minimizes the maximum overlap.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Baseline: constant 0.5
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try multiple random initializations with scipy optimization
    np.random.seed(42)
    
    n_attempts = 20
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if attempt == 0:
            # Start from constant
            h0 = np.full(N, 0.5)
        elif attempt == 1:
            # Small random perturbation
            h0 = np.full(N, 0.5) + 0.1 * np.random.randn(N)
            h0 = np.clip(h0, 0.0, 1.0)
        elif attempt == 2:
            # Sinusoidal initialization
            m = np.random.randint(1, N//2 + 1)
            A = np.random.uniform(0.1, 0.4)
            phi = np.random.uniform(0, 2*np.pi)
            i = np.arange(N)
            h0 = 0.5 + A * np.cos(2*np.pi*m*i/N + phi)
            h0 = np.clip(h0, 0.0, 1.0)
        elif attempt == 3:
            # Two-level initialization
            h0 = np.random.choice([0.2, 0.8], size=N)
            # Adjust to have sum N/2
            h0 = h0 * (N/2.0) / h0.sum()
            h0 = np.clip(h0, 0.0, 1.0)
        elif attempt == 4:
            # Gradient-like pattern
            i = np.arange(N)
            h0 = 0.5 + 0.3 * np.sin(2*np.pi*3*i/N) + 0.2 * np.cos(2*np.pi*7*i/N)
            h0 = np.clip(h0, 0.0, 1.0)
        else:
            # Random with structure
            freqs = np.random.choice(range(1, N//2+1), size=3, replace=False)
            amps = np.random.uniform(0.1, 0.3, size=3)
            phis = np.random.uniform(0, 2*np.pi, size=3)
            i = np.arange(N)
            h0 = 0.5 * np.ones(N)
            for f, a, p in zip(freqs, amps, phis):
                h0 += a * np.cos(2*np.pi*f*i/N + p)
            h0 = np.clip(h0, 0.0, 1.0)
        
        # Use SLSQP with bounds
        bounds = [(0.0, 1.0)] * N
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
        
        try:
            result = minimize(
                _objective,
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            h_opt = np.clip(result.x, 0.0, 1.0)
            
            # Ensure exact sum
            diff = N/2.0 - h_opt.sum()
            if abs(diff) > 1e-8:
                mask = (h_opt > 0.01) & (h_opt < 0.99)
                if mask.sum() > 0:
                    h_opt[mask] += diff / mask.sum()
                    h_opt = np.clip(h_opt, 0.0, 1.0)
            
            score = _overlap_max(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            continue
    
    # Verify
    best_h = np.clip(best_h, 0.0, 1.0)
    final_diff = N/2.0 - best_h.sum()
    if abs(final_diff) > 1e-8:
        mask = (best_h > 0.01) & (best_h < 0.99)
        if mask.sum() > 0:
            best_h[mask] += final_diff / mask.sum()
            best_h = np.clip(best_h, 0.0, 1.0)
    
    assert abs(best_h.sum() - N/2.0) < 1e-6, f"Sum is {best_h.sum()}, expected {N/2.0}"
    assert np.all(best_h >= 0.0 - 1e-10) and np.all(best_h <= 1.0 + 1e-10)
    
    return best_h

# EVOLVE-BLOCK-START
# (kept for compatibility)
# EVOLVE-BLOCK-END
