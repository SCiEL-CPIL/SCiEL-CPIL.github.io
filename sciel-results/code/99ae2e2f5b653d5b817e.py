"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_max_abs(f):
    """Compute the max absolute value of the autoconvolution (excluding zero-lag self term)."""
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


def _quasi_periodic_init(n, n_harmonics=5, seed=42):
    """
    Initialize using a quasi-periodic structure based on real parts of complex exponentials.
    h[i] = sign(sum_{k=1}^{K} A_k * cos(2*pi*f_k*i + phi_k))
    Using irrational frequencies (like sqrt(2), sqrt(3), etc.) to create non-repeating patterns
    that inherently have lower autocorrelation peaks.
    """
    rng = np.random.default_rng(seed)
    
    # Use irrational frequencies scaled to [0, 1]
    # e.g., sqrt(2)/10, sqrt(3)/10, etc.
    freqs = []
    for k in range(1, n_harmonics + 1):
        # Use different irrationals for each harmonic
        if k % 3 == 1:
            base = np.sqrt(2)
        elif k % 3 == 2:
            base = np.sqrt(3)
        else:
            base = np.sqrt(5)
        # Scale to a reasonable frequency range
        freqs.append(base / (10 + k))
    
    freqs = np.array(freqs)
    
    # Random phases
    phases = rng.uniform(0, 2 * np.pi, size=n_harmonics)
    
    # Random amplitudes (positive)
    amplitudes = rng.uniform(0.5, 1.5, size=n_harmonics)
    
    i = np.arange(n)
    signal = np.zeros(n)
    for k in range(n_harmonics):
        signal += amplitudes[k] * np.cos(2 * np.pi * freqs[k] * i + phases[k])
    
    # Convert to signs: +1 if signal > 0, else -1
    h = np.sign(signal)
    # Ensure no zeros (replace 0 with 1)
    h[h == 0] = 1.0
    
    return h


def _local_search_sign_flips(h, max_time=20.0, rng=None):
    """
    Local search: try flipping individual signs to reduce the autoconvolution peak.
    Uses a greedy approach with some randomization to escape local minima.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(h)
    h = h.copy()
    current_score = _autoconvolution_max_abs(h)
    best_h = h.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Multi-pass local search
    for _ in range(10):
        if time.time() > deadline:
            break
        
        improved = False
        # Try each position once
        for i in range(n):
            if time.time() > deadline:
                break
            
            old_val = h[i]
            h[i] = -old_val
            new_score = _autoconvolution_max_abs(h)
            
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                h[i] = old_val
        
        if not improved:
            break
    
    return best_h, best_score


def _sparse_local_search(f, active_indices, max_time=20.0, rng=None):
    """
    Local search for sparse patterns: flip signs and swap active indices.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(f)
    f = f.copy()
    active_indices = active_indices.copy()
    current_score = _autoconvolution_max_abs(f)
    best_f = f.copy()
    best_score = current_score
    best_active = active_indices.copy()
    
    start_time = time.time()
    deadline = start_time + max_time
    
    K = len(active_indices)
    
    # Phase 1: Flip signs
    for _ in range(5):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(K):
            if time.time() > deadline:
                break
            
            idx = active_indices[i]
            old_val = f[idx]
            f[idx] = -old_val
            new_score = _autoconvolution_max_abs(f)
            
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                    best_active = active_indices.copy()
            else:
                f[idx] = old_val
        
        if not improved:
            break
    
    # Phase 2: Swap active indices
    if time.time() < deadline - 5:
        inactive = np.setdiff1d(np.arange(n), active_indices)
        if len(inactive) > 0:
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                # Pick a random active index to remove
                i_remove = rng.integers(0, K)
                # Pick a random inactive index to add
                j_add = inactive[rng.integers(0, len(inactive))]
                
                idx_remove = active_indices[i_remove]
                old_val_remove = f[idx_remove]
                
                f[idx_remove] = 0
                f[j_add] = 1.0
                
                active_indices[i_remove] = j_add
                inactive = np.setdiff1d(np.arange(n), active_indices)
                
                new_score = _autoconvolution_max_abs(f)
                
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_f = f.copy()
                        best_active = active_indices.copy()
                else:
                    # Revert
                    f[j_add] = 0
                    f[idx_remove] = old_val_remove
                    active_indices[i_remove] = idx_remove
                    inactive = np.setdiff1d(np.arange(n), active_indices)
    
    return best_f, best_active, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_f = None
    best_score = np.inf
    
    # Strategy 1: Quasi-periodic initialization with local search
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        
        h = _quasi_periodic_init(N, n_harmonics=5, seed=42 + trial)
        h, score = _local_search_sign_flips(h, max_time=15.0, rng=np.random.default_rng(100 + trial))
        
        if score < best_score:
            best_score = score
            best_f = h.copy()
    
    # Strategy 2: Sparse combinatorial design
    sparsity_levels = [30, 50, 80, 120]
    
    for K in sparsity_levels:
        if time.time() > deadline - 20:
            break
        
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.default_rng(200 + K + trial)
            active_indices = rng.choice(N, size=K, replace=False)
            signs = rng.choice([-1.0, 1.0], size=K)
            
            f = np.zeros(N)
            f[active_indices] = signs
            
            current_score = _autoconvolution_max_abs(f)
            
            f, active_indices, current_score = _sparse_local_search(
                f, active_indices, max_time=10.0, rng=np.random.default_rng(300 + K + trial)
            )
            
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
    
    # Final polish on the best solution found
    if best_f is not None and time.time() < deadline - 5:
        # Check if it's sparse or dense
        nonzero_count = np.count_nonzero(best_f)
        
        if nonzero_count < N * 0.5:
            # Sparse: do more sign flips
            active = np.nonzero(best_f)[0]
            current_score = best_score
            for idx in active:
                if time.time() > deadline - 2:
                    break
                old_val = best_f[idx]
                best_f[idx] = -old_val
                s = _autoconvolution_max_abs(best_f)
                if s < current_score - 1e-12:
                    current_score = s
                else:
                    best_f[idx] = old_val
            if current_score < best_score:
                best_score = current_score
        else:
            # Dense: do more sign flips
            current_score = best_score
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                old_val = best_f[i]
                best_f[i] = -old_val
                s = _autoconvolution_max_abs(best_f)
                if s < current_score - 1e-12:
                    current_score = s
                else:
                    best_f[i] = old_val
            if current_score < best_score:
                best_score = current_score
    
    if best_f is None:
        # Fallback: all ones
        best_f = np.ones(N)
    
    return best_f
# EVOLVE-BLOCK-END
