"""Optimized program for third autocorrelation inequality."""
import numpy as np
import time

N = 400

def _compute_value(h):
    """Compute the objective value for a given h."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))

def construct():
    """Construct step heights to minimize the autocorrelation inequality constant."""
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # Should be 2.0
    
    # Strategy: Spectral Parameterization + Simulated Annealing
    # h[n] = C + sum_{k=1}^K (A_k cos(2*pi*k*n/N) + B_k sin(2*pi*k*n/N))
    # We optimize C, A_k, B_k.
    
    # Choose number of harmonics. Low K gives smooth signals.
    K = 15
    dim = 1 + 2 * K
    
    # Precompute frequency vectors for fast reconstruction
    n_idx = np.arange(N)
    freqs = np.arange(1, K + 1)
    # Shape: (K, N)
    cos_terms = np.cos(2 * np.pi * np.outer(freqs, n_idx) / N)
    sin_terms = np.sin(2 * np.pi * np.outer(freqs, n_idx) / N)
    
    def reconstruct(theta):
        """Reconstruct h from spectral coefficients."""
        C = theta[0]
        A = theta[1:K+1]
        B = theta[K+1:]
        # h = C + cos_terms @ A + sin_terms @ B
        h = C + cos_terms.dot(A) + sin_terms.dot(B)
        return h

    def evaluate_theta(theta):
        h = reconstruct(theta)
        return _compute_value(h)
    
    # Initialize theta
    # Start with ones (C=1, others 0)
    theta_best = np.zeros(dim)
    theta_best[0] = 1.0
    
    # We might want to allow the signal to shift or scale?
    # The objective is scale invariant: val(c*h) = val(h).
    # So we can fix the scale, e.g., sum(h) = 1, or just let SA find the magnitude.
    # Since it's scale invariant, the magnitude of theta doesn't matter for the value,
    # but it matters for the "step" size in SA.
    # It's better to normalize h after reconstruction to a fixed sum, e.g., sum=1,
    # to keep the search space bounded and consistent.
    
    def evaluate_theta_normalized(theta):
        h = reconstruct(theta)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.inf
        h_norm = h / s
        return _compute_value(h_norm)
    
    # Simulated Annealing
    # Initial temperature
    T_start = 1.0
    T_end = 1e-4
    # Number of steps
    max_steps = 200000
    
    # Current state
    theta_cur = theta_best.copy()
    val_cur = evaluate_theta_normalized(theta_cur)
    
    # Track best
    best_val_sa = val_cur
    best_theta_sa = theta_cur.copy()
    
    # Adaptive step size? Or fixed?
    # Let's use a fixed step size that decays with T.
    # Perturbation: add Gaussian noise scaled by T.
    
    # We need to be careful with the "scale invariance".
    # If we normalize h to sum=1, then C is effectively fixed relative to A,B.
    # Actually, if h is normalized, the absolute value of C, A, B doesn't matter, only their ratios.
    # So we can just perturb the ratios.
    # Let's perturb the raw theta, reconstruct, normalize, evaluate.
    
    # To make SA work well, we should ensure the perturbations are meaningful.
    # Let's try to keep the "energy" of theta somewhat bounded.
    
    for i in range(max_steps):
        if time.time() > deadline:
            break
        
        # Cooling schedule: geometric
        T = T_start * (T_end / T_start) ** (i / max_steps)
        
        # Generate candidate
        # Perturb theta
        # Scale of perturbation: proportional to T
        # We want to explore different shapes.
        # Let's perturb A and B with std = T * 0.1
        # And C with std = T * 0.1
        noise = np.random.normal(0, T * 0.1, dim)
        theta_new = theta_cur + noise
        
        val_new = evaluate_theta_normalized(theta_new)
        
        if not np.isfinite(val_new):
            # Reject if invalid
            continue
            
        delta_val = val_new - val_cur
        
        if delta_val < 0 or np.random.rand() < np.exp(-delta_val / T):
            # Accept
            theta_cur = theta_new
            val_cur = val_new
            
            if val_new < best_val_sa:
                best_val_sa = val_new
                best_theta_sa = theta_new.copy()
                # Update global best
                h_best_candidate = reconstruct(best_theta_sa)
                s_best = np.sum(h_best_candidate)
                if abs(s_best) > 1e-10:
                    h_best_candidate = h_best_candidate / s_best
                    if val_new < best_val:
                        best_val = val_new
                        best_h = h_best_candidate.copy()
        
        # Occasionally, if stuck, do a larger jump (restart-ish)
        # But standard SA should handle it.
        
    # Final check and return
    # Ensure best_h is the one corresponding to best_val
    # The loop updates best_h when best_val_sa improves.
    # But what if the very last steps didn't improve best_val_sa but we have a better h from before?
    # The logic above updates best_h only if val_new < best_val.
    # This is correct.
    
    return best_h
