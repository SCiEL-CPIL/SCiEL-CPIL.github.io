"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_c3_prime(x):
    """Compute the maximum absolute autoconvolution divided by squared integral."""
    # Autoconvolution (linear)
    ac = np.convolve(x, x, mode='full')
    # Integral (sum)
    integral = np.sum(x)
    sq_integral = integral * integral
    if abs(sq_integral) < 1e-12:
        return float('inf')
    # The autoconvolution at lag 0 is sum(x^2), which is not necessarily the max
    # We want max |ac[k]| / integral^2
    max_abs_ac = np.max(np.abs(ac))
    return max_abs_ac / abs(sq_integral)


def _flat_spectrum_design(n, num_iters=50, rng=None):
    """
    Design a signed step function with flat spectrum (constant magnitude DFT).
    This minimizes off-zero-lag autoconvolution coefficients.
    
    We use a phase-only optimization in frequency domain, then project back
    to time domain and threshold to get signed values.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Target: |X[k]| = constant for all k
    # We'll optimize phases to make the time-domain signal have small autoconvolution
    # except at lag 0.
    
    # Start with random phases
    phases = rng.uniform(0, 2 * np.pi, size=n)
    best_x = None
    best_score = float('inf')
    
    # Target magnitude: we want the DFT to have constant magnitude
    # For a real signal, X[k] = X[n-k]* (conjugate symmetry)
    # Let's work with the full DFT and enforce conjugate symmetry
    
    for it in range(num_iters):
        # Create complex spectrum with target magnitude and current phases
        # For real signal: X[0] is real, X[n/2] is real if n even
        # X[k] = A * exp(i * phi[k]) for k=1..n-1, with phi[n-k] = -phi[k]
        
        # Use a simpler approach: generate a random sequence, compute its DFT,
        # replace magnitudes with constant, keep phases, inverse DFT, threshold
        
        # Actually, let's use iterative projection:
        # 1. Start with random signed sequence
        # 2. Compute DFT
        # 3. Replace magnitudes with constant (keeping phases)
        # 4. Inverse DFT
        # 5. Threshold to get signed sequence
        # 6. Repeat
        
        if it == 0:
            x = rng.choice([-1.0, 1.0], size=n)
        else:
            # Use previous iteration's x to compute DFT
            X = np.fft.fft(x)
            # Replace magnitudes with constant
            # Target magnitude: we want sum |X[k]|^2 = n * sum x^2 = n * n (since x is ±1)
            # So |X[k]|^2 = n^2 / n = n, so |X[k]| = sqrt(n)
            target_mag = np.sqrt(n)
            X_new = target_mag * X / np.where(np.abs(X) < 1e-12, 1e-12, np.abs(X))
            # Inverse DFT
            x_complex = np.fft.ifft(X_new)
            # Take real part
            x_real = np.real(x_complex)
            # Threshold to ±1
            x = np.sign(x_real)
            # Handle zeros
            x[x == 0] = 1.0
        
        # Compute score
        score = _compute_c3_prime(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
    
    return best_x, best_score


def _greedy_improve(x, max_iters=100):
    """
    Greedy local search: flip individual bits to reduce max |autoconv| / integral^2.
    """
    n = len(x)
    best_x = x.copy()
    best_score = _compute_c3_prime(best_x)
    
    for it in range(max_iters):
        improved = False
        # Try flipping each bit
        for i in range(n):
            x_trial = best_x.copy()
            x_trial[i] *= -1  # Flip sign
            score = _compute_c3_prime(x_trial)
            if score < best_score:
                best_score = score
                best_x = x_trial
                improved = True
                break  # Restart scan from beginning
        
        if not improved:
            break
    
    return best_x, best_score


def _random_search(n, num_trials=20, rng=None):
    """
    Random search with greedy improvement.
    """
    if rng is None:
        rng = np.random.default_rng(123)
    
    best_x = None
    best_score = float('inf')
    
    for t in range(num_trials):
        # Random signed sequence
        x = rng.choice([-1.0, 1.0], size=n)
        # Greedy improvement
        x_imp, score = _greedy_improve(x, max_iters=50)
        if score < best_score:
            best_score = score
            best_x = x_imp
    
    return best_x, best_score


def construct():
    """
    Construct a signed step function minimizing max |autoconv| / integral^2.
    
    Returns:
        numpy array of shape (N,) with values in {-1, 1}
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    best_x = None
    best_score = float('inf')
    
    # Strategy 1: Flat spectrum design
    try:
        x_flat, score_flat = _flat_spectrum_design(N, num_iters=30, rng=np.random.default_rng(42))
        if score_flat < best_score:
            best_score = score_flat
            best_x = x_flat.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_x is not None:
            return best_x
        return np.ones(N)
    
    # Strategy 2: Random search with greedy improvement
    try:
        x_rand, score_rand = _random_search(N, num_trials=10, rng=np.random.default_rng(123))
        if score_rand < best_score:
            best_score = score_rand
            best_x = x_rand.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_x is not None:
            return best_x
        return np.ones(N)
    
    # Strategy 3: More iterations of flat spectrum with different seeds
    for seed in range(5):
        if time.time() > deadline:
            break
        try:
            x_fs, score_fs = _flat_spectrum_design(N, num_iters=50, rng=np.random.default_rng(seed + 1000))
            if score_fs < best_score:
                best_score = score_fs
                best_x = x_fs.copy()
        except Exception:
            pass
    
    if best_x is not None:
        return best_x
    return np.ones(N)
# EVOLVE-BLOCK-END
