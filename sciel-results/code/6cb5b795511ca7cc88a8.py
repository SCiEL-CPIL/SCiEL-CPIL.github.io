"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_max_abs(f):
    """Compute max absolute value of autoconvolution (excluding zero-lag self term)."""
    n = len(f)
    # Linear convolution via FFT
    size = 2 * n
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.fft(f_pad, size)
    conv = np.fft.ifft(F * F).real
    # conv[0] is the zero-lag (self) term: sum(f^2)
    # We want max |conv[k]| for k != 0
    if len(conv) > 1:
        return np.max(np.abs(conv[1:]))
    return 0.0


def _compute_score(f):
    """The score is the upper bound on C3 (lower is better)."""
    return _autoconvolution_max_abs(f)


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Strategy: sparse combinatorial design
    # Select a small subset K of intervals to have non-zero values.
    # For each active interval, assign +1 or -1.
    # Minimize the autoconvolution peak.
    
    # Start with a moderate sparsity level
    best_f = np.zeros(N)
    best_score = np.inf
    
    # Try different sparsity levels
    sparsity_levels = [20, 30, 40, 50, 60, 80, 100]
    
    for K in sparsity_levels:
        if time.time() > deadline - 30:
            break
        
        # Random initialization: pick K random indices
        np.random.seed(42)
        active_indices = np.random.choice(N, size=K, replace=False)
        signs = np.random.choice([-1, 1], size=K)
        
        f = np.zeros(N)
        f[active_indices] = signs
        
        current_score = _compute_score(f)
        
        # Local search: flip individual signs
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            for i in range(K):
                if time.time() > deadline - 30:
                    break
                idx = active_indices[i]
                old_sign = f[idx]
                f[idx] = -old_sign
                new_score = _compute_score(f)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    f[idx] = old_sign
        
        # Also try swapping active indices
        if time.time() < deadline - 30:
            for _ in range(200):
                if time.time() > deadline - 30:
                    break
                # Pick a random active index to remove
                i_remove = np.random.randint(K)
                # Pick a random inactive index to add
                inactive = np.setdiff1d(np.arange(N), active_indices)
                if len(inactive) == 0:
                    break
                j_add = inactive[np.random.randint(len(inactive))]
                
                idx_remove = active_indices[i_remove]
                f[idx_remove] = 0
                f[j_add] = 1  # Start with +1
                
                active_indices = np.delete(active_indices, i_remove)
                active_indices = np.append(active_indices, j_add)
                
                new_score = _compute_score(f)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    # Refine sign of the new element
                    f[j_add] = -1
                    alt_score = _compute_score(f)
                    if alt_score < current_score:
                        current_score = alt_score
                    else:
                        f[j_add] = 1
                else:
                    # Revert
                    f[j_add] = 0
                    f[idx_remove] = 1
                    active_indices = np.delete(active_indices, len(active_indices) - 1)
                    active_indices = np.insert(active_indices, i_remove, idx_remove)
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
    
    # Final polish: try all sign flips one more time on best solution
    if time.time() < deadline - 10:
        active = np.nonzero(best_f)[0]
        for idx in active:
            if time.time() > deadline - 10:
                break
            old_val = best_f[idx]
            best_f[idx] = -old_val
            s = _compute_score(best_f)
            if s < best_score - 1e-12:
                best_score = s
            else:
                best_f[idx] = old_val
    
    return best_f
# EVOLVE-BLOCK-END
