"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar
import time

def _compute_score(x):
    """Compute max|autoconv| / (sum(x))^2"""
    S = x.sum()
    if abs(S) < 1e-10:
        return 1e10
    # Autoconvolution via FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    X = np.fft.rfft(x, n=n_fft)
    C = np.fft.irfft(X * X, n=n_fft)[:2 * N - 1]
    max_abs = np.max(np.abs(C))
    return max_abs / (S * S)

def _zadoff_chu_real(m, N):
    """Real part of Zadoff-Chu sequence, signed to ±1"""
    n = np.arange(N)
    # Zadoff-Chu: exp(j * pi * m * n * (n+1) / N)
    phase = np.pi * m * n * (n + 1) / N
    x = np.cos(phase)
    # Sign to get ±1
    x = np.sign(x)
    x[x == 0] = 1  # handle zeros
    return x

def _local_search(x, max_iter=5000, deadline=None):
    """Local search: flip bits to reduce max|autoconv|/S^2"""
    best = x.copy()
    best_score = _compute_score(best)
    
    # Precompute autoconv
    n_fft = 1 << (2 * N - 1).bit_length()
    
    def get_score_from_C(C, S):
        if abs(S) < 1e-10:
            return 1e10
        return np.max(np.abs(C)) / (S * S)
    
    # Maintain C and S incrementally
    S = best.sum()
    C = np.fft.irfft(np.fft.rfft(best, n=n_fft) ** 2, n=n_fft)[:2 * N - 1]
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        # Try flipping each position
        for i in range(N):
            # Effect of flipping x[i] on S: delta_S = -2*x[i]
            delta_S = -2 * best[i]
            new_S = S + delta_S
            
            if abs(new_S) < 1e-10:
                continue
            
            # Effect on autoconvolution: C[k] changes by -2*x[i]*x[i+k] for each k
            # C_new[k] = C[k] - 2*x[i]*x[i+k] (for k where both in range)
            # This is complex; let's just compute the new C for this flip
            # Actually, let's compute the change more carefully
            
            # C[k] = sum_j x[j]*x[j+k]
            # Flipping x[i] affects terms where j=i (x[i]*x[i+k]) and j=i-k (x[i-k]*x[i])
            # So delta_C[k] = -2*x[i]*x[i+k] - 2*x[i-k]*x[i] (with bounds)
            
            # For efficiency, compute new C
            new_x = best.copy()
            new_x[i] = -new_x[i]
            new_C = np.fft.irfft(np.fft.rfft(new_x, n=n_fft) ** 2, n=n_fft)[:2 * N - 1]
            new_score = get_score_from_C(new_C, new_S)
            
            if new_score < best_score * 0.999:  # require meaningful improvement
                best[i] = -best[i]
                S = new_S
                C = new_C
                best_score = new_score
                improved = True
                break  # restart search after a flip
        
        if not improved:
            break
    
    return best, best_score

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    # Try multiple Zadoff-Chu parameters
    best_x = None
    best_score = 1e10
    
    # Try various chirp rates m
    for m in range(1, N):
        if np.gcd(m, N) != 1:
            continue
        
        x = _zadoff_chu_real(m, N)
        score = _compute_score(x)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        if time.time() > deadline - 30:
            break
    
    # Also try negated version (should be same score, but just in case)
    if best_x is not None:
        score_neg = _compute_score(-best_x)
        if score_neg < best_score:
            best_score = score_neg
            best_x = -best_x
    
    # Local search polish
    if best_x is not None and time.time() < deadline - 5:
        polished_x, polished_score = _local_search(best_x, max_iter=5000, deadline=deadline - 5)
        if polished_score < best_score:
            best_x = polished_x
            best_score = polished_score
    
    # Final check: compare with all-ones baseline
    ones_score = 1.0 / N
    if best_score > ones_score:
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
