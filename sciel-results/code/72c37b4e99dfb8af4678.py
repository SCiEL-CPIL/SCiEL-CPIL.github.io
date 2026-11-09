"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr(h):
    """Compute autocorrelation r[k] = sum_i h[i]*h[i+k] for k=0..N-1."""
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    return r[:N]


def _score(h):
    """Compute the upper bound on C3 (peak sidelobe metric)."""
    r = _autocorr(h)
    # Peak sidelobe: max |r[k]| for k = 1..N-1
    return np.max(np.abs(r[1:N]))


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_h = None
    best_score = np.inf
    
    # Start with random initialization
    np.random.seed(42)
    
    # Phase 1: Random restarts to find good initial candidates
    n_restarts = 0
    while time.time() < deadline - 30.0:
        n_restarts += 1
        h = np.random.choice([-1, 1], N).astype(np.float64)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 2: Local search (flip individual bits) from best candidate
    if best_h is not None and time.time() < deadline - 5.0:
        current_h = best_h.copy()
        current_score = best_score
        n_no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline - 5.0 and n_no_improve < max_no_improve:
            # Try flipping a random position
            idx = np.random.randint(N)
            old_val = current_h[idx]
            current_h[idx] = -old_val
            
            new_score = _score(current_h)
            if new_score < current_score:
                current_score = new_score
                n_no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
            else:
                # Revert
                current_h[idx] = old_val
                n_no_improve += 1
        
        # Phase 3: Try multiple restarts of local search from different random starts
        while time.time() < deadline - 5.0:
            h = np.random.choice([-1, 1], N).astype(np.float64)
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
            
            # Local search from this point
            current_h = h.copy()
            current_score = s
            n_no_improve = 0
            max_no_improve = 200
            
            while time.time() < deadline - 5.0 and n_no_improve < max_no_improve:
                idx = np.random.randint(N)
                old_val = current_h[idx]
                current_h[idx] = -old_val
                
                new_score = _score(current_h)
                if new_score < current_score:
                    current_score = new_score
                    n_no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()
                else:
                    current_h[idx] = old_val
                    n_no_improve += 1
    
    if best_h is None:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
