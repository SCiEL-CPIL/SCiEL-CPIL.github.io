"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv(x):
    """Compute linear autocorrelation via FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    fx = np.fft.rfft(x, n_fft)
    ac = np.fft.irfft(fx * np.conj(fx), n_fft)[:N]
    return ac

def _score_candidate(x):
    """Compute the objective: max |autoconv| / (sum x)^2 for a linear autocorrelation."""
    S = x.sum()
    if abs(S) < 1e-12:
        return np.inf
    ac = _autoconv(x)
    max_abs = np.max(np.abs(ac))
    return max_abs / (S * S)

def _greedy_refine(x, time_budget):
    """Greedy coordinate descent: flip individual signs if it improves the score."""
    current_score = _score_candidate(x)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() > time_budget:
                break
            x[i] *= -1
            s_new = _score_candidate(x)
            if s_new < current_score - 1e-14:
                current_score = s_new
                improved = True
            else:
                x[i] *= -1  # revert
        if time.time() > time_budget:
            break
    return x, current_score

def _pairwise_refine(x, time_budget):
    """Try flipping pairs of signs to escape local minima."""
    current_score = _score_candidate(x)
    improved = True
    while improved:
        improved = False
        for i in range(0, N, 2):
            if time.time() > time_budget:
                break
            for j in range(i + 1, min(i + 10, N)):
                if time.time() > time_budget:
                    break
                x[i] *= -1
                x[j] *= -1
                s_new = _score_candidate(x)
                if s_new < current_score - 1e-14:
                    current_score = s_new
                    improved = True
                else:
                    x[i] *= -1
                    x[j] *= -1
        if time.time() > time_budget:
            break
    return x, current_score

def _random_pattern(rng):
    """Generate a random sign pattern."""
    return rng.choice([-1.0, 1.0], size=N)

def _barker_like_pattern(rng):
    """Generate a Barker-like pattern (balanced, low autocorrelation)."""
    # Start with random, then try to balance
    x = rng.choice([-1.0, 1.0], size=N)
    # Flip signs to make sum close to 0
    S = x.sum()
    if S > 0:
        n_flip = int(S / 2)
        idx = rng.choice(N, size=n_flip, replace=False)
        x[idx] *= -1
    else:
        n_flip = int(-S / 2)
        idx = rng.choice(N, size=n_flip, replace=False)
        x[idx] *= -1
    return x

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Start with all ones
    x = np.ones(N)
    best_score = _score_candidate(x)
    best_x = x.copy()
    
    # Phase 1: Multiple random restarts with greedy refinement
    n_restarts = 0
    while time.time() < deadline - 10.0:
        n_restarts += 1
        
        # Try different initial patterns
        if n_restarts % 3 == 1:
            x_try = _random_pattern(rng)
        elif n_restarts % 3 == 2:
            x_try = _barker_like_pattern(rng)
        else:
            x_try = _random_pattern(rng)
        
        s_try = _score_candidate(x_try)
        
        # Greedy refine
        x_ref, s_ref = _greedy_refine(x_try.copy(), deadline - 5.0)
        
        if s_ref < best_score - 1e-14:
            best_score = s_ref
            best_x = x_ref.copy()
            print(f"Restart {n_restarts}: improved to {best_score:.6f}")
        
        # Pairwise refine on best
        if time.time() < deadline - 5.0:
            x_ref, s_ref = _pairwise_refine(best_x.copy(), deadline - 5.0)
            if s_ref < best_score - 1e-14:
                best_score = s_ref
                best_x = x_ref.copy()
                print(f"Pairwise {n_restarts}: improved to {best_score:.6f}")
    
    # Final refinement pass
    if time.time() < deadline - 2.0:
        x_ref, s_ref = _greedy_refine(best_x.copy(), deadline - 2.0)
        if s_ref < best_score - 1e-14:
            best_score = s_ref
            best_x = x_ref.copy()
    
    return best_x
# EVOLVE-BLOCK-END
