"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _project_to_simplex(h, target_sum):
    """Project h to satisfy sum(h) = target_sum and 0<=h<=1."""
    h = np.clip(h, 0, 1)
    diff = target_sum - np.sum(h)
    for _ in range(20):
        if abs(diff) < 1e-12:
            break
        mask = (h > 1e-8) & (h < 1 - 1e-8)
        if np.sum(mask) == 0:
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0, 1)
        diff = target_sum - np.sum(h)
    # Final exact adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-10:
        h += diff / len(h)
        h = np.clip(h, 0, 1)
    return h

def _fourier_to_h(coeffs, n):
    """Convert Fourier coefficients (DC removed) to h values.
    coeffs: array of shape (n,) where coeffs[0] is DC (ignored), coeffs[1:n] are complex.
    Returns h in [0,1] with sum = n/2.
    """
    # Build h from Fourier: h[j] = 0.5 + sum_{k=1}^{n-1} Re(c_k * exp(2*pi*i*k*j/n))
    # Use only first few harmonics for stability
    h = np.full(n, 0.5)
    for k in range(1, n):
        if k >= len(coeffs) or np.abs(coeffs[k]) < 1e-15:
            continue
        c = coeffs[k]
        h += np.real(c * np.exp(2j * np.pi * k * np.arange(n) / n))
    h = np.clip(h, 0, 1)
    # Project to exact sum
    h = _project_to_simplex(h, n / 2.0)
    return h

def _h_to_fourier(h, n):
    """Convert h to Fourier coefficients (DC removed)."""
    h_centered = h - np.mean(h)
    coeffs = np.zeros(n, dtype=complex)
    for k in range(1, n):
        coeffs[k] = np.sum(h_centered * np.exp(-2j * np.pi * k * np.arange(n) / n)) / n
    return coeffs

def _max_overlap_subgrad(h):
    """Compute subgradient of max_k R_k w.r.t. h.
    Returns (max_val, subgradient) where subgradient points in direction to decrease max overlap.
    """
    n = len(h)
    corrs = _compute_overlaps(h)
    max_val = np.max(corrs)
    
    # Find which k achieves max (or near-max for smoothing)
    threshold = max_val - 1e-6
    active = corrs >= threshold
    
    subgrad = np.zeros(n)
    for k in range(1, n):
        if not active[k-1]:
            continue
        w = (2.0 / n) * corrs[k-1] / max_val if max_val > 1e-15 else (2.0/n)
        subgrad[:n-k] += w * h[k:]
        subgrad[k:] += w * h[:n-k]
    
    # Normalize subgradient
    norm = np.linalg.norm(subgrad)
    if norm > 1e-15:
        subgrad /= norm
    
    return max_val, subgrad

def construct():
    n = N
    target_sum = n / 2.0
    
    best_h = np.full(n, 0.5)
    best_score = float('inf')
    
    def _score(h):
        corrs = _compute_overlaps(h)
        return 1.0 - np.min(corrs)
    
    best_score = _score(best_h)
    
    np.random.seed(42)
    
    # Strategy: Use cyclical gradient descent in Fourier space
    # Optimize a small number of Fourier coefficients to find patterns that minimize max overlap
    
    # Start with various initial patterns in Fourier space
    best_coeffs = np.zeros(n, dtype=complex)
    
    # Try different numbers of active harmonics
    for n_harm in [1, 2, 3, 4, 5, 6, 8, 10]:
        if _time_remaining() < 5.0:
            break
        
        # Random initial Fourier coefficients
        for trial in range(15):
            if _time_remaining() < 2.0:
                break
            
            # Initialize coefficients for harmonics 1..n_harm
            coeffs = np.zeros(n, dtype=complex)
            for k in range(1, n_harm+1):
                # Random amplitude and phase
                amp = np.random.uniform(0.1, 0.4)
                phase = np.random.uniform(0, 2*np.pi)
                coeffs[k] = amp * np.exp(1j * phase)
            
            # Also try structured initializations
            if trial == 0:
                # Single dominant harmonic
                coeffs = np.zeros(n, dtype=complex)
                coeffs[1] = 0.3
            elif trial == 1:
                # Two harmonics
                coeffs = np.zeros(n, dtype=complex)
                coeffs[1] = 0.25
                coeffs[2] = 0.15
            elif trial == 2:
                # Alternating
                coeffs = np.zeros(n, dtype=complex)
                coeffs[1] = 0.2
                coeffs[3] = 0.2
            elif trial == 3:
                # Cosine-like (real coefficients)
                coeffs = np.zeros(n, dtype=complex)
                for k in range(1, min(n_harm, 5)+1):
                    coeffs[k] = 0.3 / k
            
            h = _fourier_to_h(coeffs, n)
            score = _score(h)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
                best_coeffs = coeffs.copy()
            
            # Gradient descent in Fourier coefficient space
            current_coeffs = coeffs.copy()
            current_h = h.copy()
            current_score = score
            
            lr = 0.01
            for step in range(500):
                if _time_remaining() < 1.5:
                    break
                
                # Compute max overlap and subgradient in h-space
                max_val, subgrad_h = _max_overlap_subgrad(current_h)
                
                # Project subgradient to Fourier space
                # dR/dcoeffs_k = sum_j (dR/dh_j) * conj(exp(2*pi*i*k*j/n)) / n
                subgrad_coeffs = np.zeros(n, dtype=complex)
                for k in range(1, n_harm+1):
                    subgrad_coeffs[k] = np.sum(subgrad_h * np.exp(-2j * np.pi * k * np.arange(n) / n)) / n
                
                # Update coefficients
                current_coeffs[1:n_harm+1] -= lr * subgrad_coeffs[1:n_harm+1]
                
                # Clip coefficients to reasonable range
                current_coeffs[1:n_harm+1] = np.clip(np.abs(current_coeffs[1:n_harm+1]), 0, 0.5) * \
                    np.exp(1j * np.angle(current_coeffs[1:n_harm+1]))
                
                # Convert to h
                new_h = _fourier_to_h(current_coeffs, n)
                new_score = _score(new_h)
                
                if new_score < current_score:
                    current_h = new_h
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_h = new_h.copy()
                        best_coeffs = current_coeffs.copy()
                    lr = min(lr * 1.05, 0.05)
                else:
                    lr *= 0.9
                    if lr < 1e-5:
                        break
            
            # Polish with direct h-space optimization
            if _time_remaining() > 3.0:
                h_polish = current_h.copy()
                for step in range(200):
                    if _time_remaining() < 1.0:
                        break
                    
                    max_val, subgrad_h = _max_overlap_subgrad(h_polish)
                    
                    # Simple projected gradient step
                    h_new = h_polish - 0.005 * subgrad_h
                    h_new = _project_to_simplex(h_new, target_sum)
                    
                    new_score = _score(h_new)
                    if new_score < _score(h_polish):
                        h_polish = h_new
                        if new_score < best_score:
                            best_score = new_score
                            best_h = h_polish.copy()
    
    # Final: try bang-bang patterns with local search
    if _time_remaining() > 10.0:
        for trial in range(30):
            if _time_remaining() < 3.0:
                break
            
            # Random bang-bang
            h_bb = np.random.choice([0, 1], size=n).astype(float)
            h_bb = _project_to_simplex(h_bb, target_sum)
            
            score = _score(h_bb)
            if score < best_score:
                best_score = score
                best_h = h_bb.copy()
            
            # Local search: flip individual bits
            for step in range(100):
                if _time_remaining() < 1.0:
                    break
                
                i = np.random.randint(n)
                h_trial = h_bb.copy()
                h_trial[i] = 1.0 - h_trial[i]
                h_trial = _project_to_simplex(h_trial, target_sum)
                
                new_score = _score(h_trial)
                if new_score < score:
                    h_bb = h_trial
                    score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_h = h_bb.copy()
    
    return best_h
# EVOLVE-BLOCK-END
