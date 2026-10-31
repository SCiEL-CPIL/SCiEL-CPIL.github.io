import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    def compute_overlaps(h):
        """Compute overlaps: sum_i h[i] * (1 - h[(i+t)%N]) for all t"""
        # autocorr[t] = sum_i h[i] * h[(i+t)%N]
        # Use FFT for efficiency
        h_ext = np.concatenate([h, h])
        # Circular autocorrelation via FFT
        Nfft = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, Nfft)
        autocorr = np.fft.irfft(H * np.conj(H), Nfft)[:N]
        # overlap[t] = N/2 - autocorr[t]
        return N / 2.0 - autocorr

    def objective(h):
        h_clipped = np.clip(h, 0, 1)
        overlaps = compute_overlaps(h_clipped)
        # Smooth max approximation with adaptive beta
        beta = 100.0
        m = np.max(overlaps)
        # Use log-sum-exp for numerical stability
        smooth_max = m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta
        
        # Penalty for sum constraint
        sum_penalty = 500.0 * (np.sum(h_clipped) - N / 2.0) ** 2
        
        return smooth_max + sum_penalty

    def objective_grad(h):
        h_clipped = np.clip(h, 0, 1)
        overlaps = compute_overlaps(h_clipped)
        beta = 100.0
        m = np.max(overlaps)
        weights = np.exp(beta * (overlaps - m))
        weights /= np.sum(weights)
        
        # Gradient of smooth max w.r.t. h
        # overlap[t] = N/2 - sum_i h[i]*h[(i+t)%N]
        # d(overlap[t])/dh[j] = -h[(j+t)%N] - h[(j-t)%N] (for circular)
        # Actually: d/dh[j] of sum_i h[i]*h[(i+t)%N] = h[(j+t)%N] + h[(j-t)%N] if j != (i+t)%N... 
        # Let me compute gradient numerically for safety, or derive properly
        # For now, use finite difference approximation
        grad = np.zeros(N)
        eps = 1e-7
        for j in range(N):
            h_plus = h_clipped.copy()
            h_plus[j] += eps
            h_plus = np.clip(h_plus, 0, 1)
            ov_plus = compute_overlaps(h_plus)
            smooth_plus = m + np.log(np.sum(np.exp(beta * (ov_plus - m)))) / beta
            grad[j] = (smooth_plus - (m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta)) / eps
        
        # Gradient of sum penalty
        grad += 2 * 500.0 * (np.sum(h_clipped) - N / 2.0) * (np.clip(h, 0, 1) == h_clipped).astype(float)
        
        return grad

    # Initial guess: alternating pattern
    x0 = np.full(N, 0.5)
    # Try a few different initializations
    best_h = None
    best_score = np.inf
    
    initializations = [
        np.full(N, 0.5),
        np.where(np.arange(N) % 2 == 0, 0.6, 0.4),
        np.clip(0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N), 0, 1),
        np.clip(0.5 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N), 0, 1),
        np.clip(0.5 + 0.2 * np.sin(4 * np.pi * np.arange(N) / N), 0, 1),
    ]
    
    for x0_init in initializations:
        x0_init = np.clip(x0_init, 0, 1)
        try:
            res = minimize(objective, x0_init, method='L-BFGS-B', bounds=[(0.0, 1.0)] * N,
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            h_candidate = np.clip(res.x, 0, 1)
            # Enforce sum constraint
            s = np.sum(h_candidate)
            if s > 0:
                h_candidate = h_candidate * (N / 2.0) / s
            h_candidate = np.clip(h_candidate, 0, 1)
            
            # Fine-tune sum
            diff = N / 2.0 - np.sum(h_candidate)
            idx = np.argsort(h_candidate)[::-1]
            for i in idx:
                if abs(diff) < 1e-10:
                    break
                if diff > 0:
                    add = min(diff, 1.0 - h_candidate[i])
                    h_candidate[i] += add
                    diff -= add
                else:
                    sub = min(-diff, h_candidate[i])
                    h_candidate[i] -= sub
                    diff += sub
            
            score = np.max(compute_overlaps(h_candidate))
            if score < best_score:
                best_score = score
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    return best_h
