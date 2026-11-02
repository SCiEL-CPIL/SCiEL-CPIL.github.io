"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _max_overlap(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    The overlap for shift k is sum_i h[i] * (1 - h[(i+k) % N]).
    We want to minimize the maximum over all shifts k.
    """
    N = len(h)
    # For each shift k, compute overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
    # = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    # = N/2 - sum_i h[i]*h[(i+k)%N]  (since sum h = N/2)
    # So minimizing max overlap is equivalent to minimizing max (N/2 - autocorr[k])
    # which is equivalent to maximizing min autocorr[k].
    
    # Compute circular autocorrelation
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    # Use FFT for efficiency
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] corresponds to circular autocorrelation at lag k
    # But note: numpy's ifft gives unnormalized, so we need to be careful
    # Actually, np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(x))) gives N * circular_autocorr
    # Let me verify: for x, the circular autocorrelation at lag k is sum_i x[i]*x[(i+k)%N]
    # The FFT method: ifft(|FFT(x)|^2) gives sum_i x[i] * x[(i+k)%N] * N? No.
    # Actually, the inverse FFT of |X|^2 gives the circular autocorrelation directly (unnormalized).
    
    # Let me just compute it directly for correctness with N=100 (fast enough)
    overlaps = []
    for k in range(N):
        shifted = np.roll(h, k)
        overlap = np.sum(h * (1 - shifted))
        overlaps.append(overlap)
    return max(overlaps)

def _score(h):
    """
    Score is the upper bound on C5, which is the max overlap.
    Lower is better.
    """
    return _max_overlap(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial guess: uniform 0.5
    h0 = np.full(N, 0.5)
    
    # Try a structured initial guess: alternating high/low
    # This might give a better starting point
    h1 = np.zeros(N)
    h1[0::2] = 1.0
    h1[1::2] = 0.0
    # But sum must be N/2 = 50, and this has 50 ones, so sum=50. Good.
    
    best_h = h0.copy()
    best_score = _score(h0)
    
    # Try a few different initial structures
    candidates = [h0, h1]
    
    # Add a sinusoidal candidate
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    h2 = 0.5 + 0.5 * np.cos(x)
    # Ensure sum is N/2
    h2 = h2 / np.sum(h2) * (N / 2)
    h2 = np.clip(h2, 0, 1)
    # Re-normalize
    if np.sum(h2) > 0:
        h2 = h2 / np.sum(h2) * (N / 2)
    candidates.append(h2)
    
    # Add another sinusoidal with different frequency
    h3 = 0.5 + 0.5 * np.cos(2*x)
    h3 = h3 / np.sum(h3) * (N / 2)
    h3 = np.clip(h3, 0, 1)
    if np.sum(h3) > 0:
        h3 = h3 / np.sum(h3) * (N / 2)
    candidates.append(h3)
    
    # Evaluate all candidates
    for h in candidates:
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Use scipy.optimize to minimize the max overlap
    # We use a smooth approximation of the max: log-sum-exp
    def objective(h):
        # Ensure constraints
        h = np.clip(h, 0, 1)
        # Compute all overlaps
        N_local = len(h)
        overlaps = []
        for k in range(N_local):
            shifted = np.roll(h, k)
            overlap = np.sum(h * (1 - shifted))
            overlaps.append(overlap)
        overlaps = np.array(overlaps)
        # Smooth max using log-sum-exp
        max_val = np.max(overlaps)
        if max_val < 1e-10:
            return 0.0
        # Use a temperature to smooth
        temp = 0.01
        smooth_max = max_val + temp * np.log(np.sum(np.exp((overlaps - max_val) / temp)))
        return smooth_max
    
    # Try optimization from the best candidate
    for h_init in [best_h, h1, h2, h3]:
        if time.time() > deadline - 5:
            break
        
        # Use SLSQP with bounds and equality constraint
        bounds = [(0, 1)] * N
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
        
        try:
            res = minimize(objective, h_init, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            if res.success or res.fun < best_score:
                h_opt = np.clip(res.x, 0, 1)
                # Re-normalize to satisfy constraint exactly
                if np.sum(h_opt) > 0:
                    h_opt = h_opt / np.sum(h_opt) * (N / 2)
                    h_opt = np.clip(h_opt, 0, 1)
                s = _score(h_opt)
                if s < best_score:
                    best_score = s
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final check: ensure the result is valid
    h_final = np.clip(best_h, 0, 1)
    # Ensure sum is exactly N/2
    current_sum = np.sum(h_final)
    if current_sum > 0:
        h_final = h_final / current_sum * (N / 2)
        h_final = np.clip(h_final, 0, 1)
    
    return h_final
# EVOLVE-BLOCK-END
