"""Optimized program for first autocorrelation inequality upper bound."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 140s budget, leaving 10s safety margin
    
    # Initial: try several starting shapes and pick the best
    def compute_score(f):
        f = np.maximum(f, 0)
        l1 = np.sum(f)
        if l1 < 1e-15:
            return np.inf
        l2sq = np.sum(f**2)
        return l2sq / (l1**2)
    
    # Candidate initial shapes
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/N
    candidates = []
    
    # Box (seed)
    candidates.append(np.ones(N))
    
    # Gaussian-like peak
    for sigma in [0.05, 0.08, 0.1, 0.12, 0.15]:
        g = np.exp(-0.5 * (x/sigma)**2)
        g = g / np.sum(g) * N  # normalize to similar L1
        candidates.append(g)
    
    # Cosine
    c = 0.5 * (1 + np.cos(2*np.pi*x))
    candidates.append(c)
    
    # Triangle
    t = np.maximum(0, 1 - np.abs(x)/0.125)
    candidates.append(t)
    
    # Find best initial
    best_f = None
    best_score = np.inf
    for cand in candidates:
        s = compute_score(cand)
        if s < best_score:
            best_score = s
            best_f = cand.copy()
    
    # Spectral domain coordinate descent to minimize sum(f^2) subject to sum(f)=const, f>=0
    # Transform to frequency domain
    f = best_f.astype(float)
    l1_target = np.sum(f)
    
    # Normalize: we can scale, so fix L1 = 1 for optimization
    f = f / l1_target
    l1_target = 1.0
    
    # DFT
    F = np.fft.fft(f)
    
    # Coordinate descent in spectral domain
    # For each frequency bin, adjust magnitude/phase to reduce sum(f^2)
    # sum(f^2) = sum(|F|^2) / N (by Parseval)
    # So minimize sum(|F|^2) subject to F[0] = 1 (DC component fixed by L1=1) and f >= 0
    
    n_iter_max = 500
    for it in range(n_iter_max):
        if time.time() > deadline - 5:
            break
        # For each frequency bin k (k != 0), we can adjust F[k]
        # The constraint is f >= 0 in time domain
        # Simple approach: for each k, try reducing |F[k]| and check if f stays nonnegative
        improved = False
        for k in range(1, N//2 + 1):
            if time.time() > deadline - 5:
                break
            # Try reducing magnitude of F[k] (and F[N-k] for conjugate symmetry)
            for reduction in [0.1, 0.05, 0.01]:
                F_old_k = F[k].copy()
                F_old_Nk = F[N-k].copy()
                
                # Reduce magnitude
                scale = 1.0 - reduction
                F[k] = F_old_k * scale
                F[N-k] = F_old_Nk * scale
                
                f_new = np.fft.ifft(F).real
                if np.min(f_new) >= -1e-10:
                    # Accept if it reduces the objective
                    obj_new = np.sum(f_new**2)
                    obj_old = np.sum(np.fft.ifft(np.concatenate([[F[0]], [F_old_k], F[2:N-k], [F_old_Nk], [F[N-1]]])).real**2)
                    if obj_new < obj_old:
                        improved = True
                        # Keep the change
                        continue
                    else:
                        # Revert
                        F[k] = F_old_k
                        F[N-k] = F_old_Nk
                        break
                else:
                    # Revert
                    F[k] = F_old_k
                    F[N-k] = F_old_Nk
                    break
        
        if not improved:
            break
    
    f = np.fft.ifft(F).real
    f = np.maximum(f, 0)
    
    # Time-domain polishing: gradient descent on sum(f^2) with projection
    if time.time() < deadline - 10:
        f = f / np.sum(f)  # renormalize
        lr = 0.01
        for it in range(200):
            if time.time() > deadline - 10:
                break
            # Gradient of sum(f^2) is 2f
            grad = 2 * f
            # Projected gradient descent: minimize sum(f^2) s.t. sum(f)=1, f>=0
            f_new = f - lr * grad
            # Project onto {f >= 0, sum(f) = 1}
            f_new = np.maximum(f_new, 0)
            s = np.sum(f_new)
            if s < 1e-15:
                f = np.ones(N) / N
            else:
                f_new = f_new / s
            # Check improvement
            if np.sum(f_new**2) < np.sum(f**2):
                f = f_new
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break
    
    # Final: ensure nonnegativity and return as list of heights
    f = np.maximum(f, 0)
    
    # Scale: the actual value doesn't matter for the ratio, but let's keep reasonable scale
    # The task asks for step heights; the ratio is scale-invariant
    # Return f directly (the evaluator will compute the normalized max)
    return list(f)
# EVOLVE-BLOCK-END
