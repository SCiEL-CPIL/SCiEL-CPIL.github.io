"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Objective: minimize max_k C(k) where C(k) = sum_i h_i * (1 - h_{i+k mod N})
    # = N/2 - A(k), so minimize max C = N/2 - min A(k)
    # Equivalently maximize min_k A(k) where A(k) = sum_i h_i * h_{i+k mod N}
    # Use smooth surrogate: -softmin(A) + penalty for constraints
    
    def compute_autocorr(h):
        # A(k) = sum_i h[i] * h[(i+k) % N]
        # Use FFT for efficiency
        H = np.fft.fft(h)
        power = np.abs(H)**2
        A = np.fft.ifft(power).real  # circular autocorrelation
        return A
    
    def objective_and_grad(params):
        h = params  # will be projected
        A = compute_autocorr(h)
        # We want to maximize min A(k), use -softmin
        beta = 20.0
        # softmin = -logsumexp(-beta*A)/beta
        max_neg = np.max(-beta * A)
        softmin = -(max_neg + np.log(np.sum(np.exp(-beta * A - max_neg)))) / beta
        # We want to minimize -softmin + constraint penalties
        # Constraint: sum h = N/2, h in [0,1]
        # Use penalty method
        sum_h = np.sum(h)
        sum_pen = (sum_h - N/2)**2 * 100.0
        # Box penalty
        below = np.clip(-h, 0, None)**2 * 100.0
        above = np.clip(h - 1, 0, None)**2 * 100.0
        box_pen = np.sum(below) + np.sum(above)
        
        val = -softmin + sum_pen + box_pen
        return val
    
    def objective(params):
        return objective_and_grad(params)
    
    # Initial guess: 0.5 + small perturbation
    np.random.seed(42)
    h0 = np.full(N, 0.5)
    h0 += 0.1 * np.random.randn(N)
    h0 = np.clip(h0, 0, 1)
    # Normalize to sum N/2
    h0 = h0 / np.sum(h0) * (N/2)
    h0 = np.clip(h0, 0, 1)
    
    best_h = h0.copy()
    best_val = objective(h0)
    
    # Try multiple initializations and optimization
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            h_cur = h0.copy()
        else:
            # Different initializations
            if r % 2 == 1:
                # Cosine-based
                freq = np.random.uniform(1, N//4)
                phase = np.random.uniform(0, 2*np.pi)
                h_cur = 0.5 + 0.3 * np.cos(2*np.pi*freq*np.arange(N)/N + phase)
            else:
                # Random
                h_cur = 0.5 + 0.2 * np.random.randn(N)
            h_cur = np.clip(h_cur, 0, 1)
            h_cur = h_cur / np.sum(h_cur) * (N/2)
            h_cur = np.clip(h_cur, 0, 1)
        
        # BFGS optimization
        try:
            res = minimize(objective, h_cur, method='L-BFGS-B', 
                          bounds=[(0,1)]*N,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if time.time() > deadline:
                break
            if res.fun < best_val:
                best_val = res.fun
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Final projection: ensure sum is exactly N/2 and h in [0,1]
    # Use iterative water-filling style projection
    h = best_h.copy()
    for _ in range(100):
        if time.time() > deadline:
            break
        # Clip to [0,1]
        h = np.clip(h, 0, 1)
        # Adjust to make sum = N/2
        diff = N/2 - np.sum(h)
        if abs(diff) < 1e-12:
            break
        # Distribute adjustment to non-saturated elements
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        if np.sum(mask) == 0:
            # All saturated, just clip
            h = np.clip(h, 0, 1)
            break
        h[mask] += diff / np.sum(mask)
    
    # Final exact sum correction
    h = np.clip(h, 0, 1)
    diff = N/2 - np.sum(h)
    if abs(diff) > 1e-15:
        # Find elements not at boundary
        mask = (h > 1e-12) & (h < 1 - 1e-12)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    
    return h
# EVOLVE-BLOCK-END
