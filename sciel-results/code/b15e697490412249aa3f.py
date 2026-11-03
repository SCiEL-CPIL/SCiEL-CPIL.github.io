"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv(x):
    """Compute autoconvolution (full) of x."""
    n = len(x)
    # Use FFT for efficient convolution
    size = 2 * n - 1
    size = 1 << (size - 1).bit_length()  # next power of 2
    F = np.fft.rfft(x, size)
    ac = np.fft.irfft(F * F.conj(), size)[:size]
    return ac


def _score(x):
    """Score: max |autoconv| / (sum x)^2. Lower is better."""
    ac = _autoconv(x)
    max_abs = np.max(np.abs(ac))
    s = np.sum(x)
    if s == 0:
        return float('inf')
    return max_abs / (s * s)


def _local_improve(x, max_iter=200, deadline=None):
    """Greedy local improvement: flip signs to reduce score."""
    x = x.copy()
    n = len(x)
    current_score = _score(x)
    
    # Precompute: for each position i, flipping x[i] changes:
    #   autoconv[k] for k in range(n-1-i, n+i) roughly
    #   sum changes by -2*x[i]
    # We'll just recompute for simplicity but use vectorized approach.
    
    best_x = x.copy()
    best_score = current_score
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try flipping each position
        for i in range(n):
            x[i] = -x[i]
            s = _score(x)
            if s < best_score:
                best_score = s
                best_x = x.copy()
                improved = True
            else:
                x[i] = -x[i]  # revert
        
        if not improved:
            break
    
    return best_x, best_score


def _gaussian_elimination_seed(rng, n):
    """Generate a random ±1 sequence with some structure."""
    return rng.choice([-1.0, 1.0], size=n)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_score = float('inf')
    
    # Strategy 1: Random restarts with local improvement
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate random ±1 sequence
        x = rng.choice([-1.0, 1.0], size=N)
        
        # Local improvement
        x_imp, s_imp = _local_improve(x, max_iter=100, deadline=deadline)
        if s_imp < best_score:
            best_score = s_imp
            best_x = x_imp.copy()
    
    # Strategy 2: Try structured sequences (e.g., based on quadratic residues)
    if time.time() < deadline:
        # Quadratic residue sequence
        x_qr = np.ones(N)
        for i in range(N):
            if i > 0 and (i * i) % (N - 1) != 0:
                # Check if i is a quadratic residue mod (N-1)
                is_residue = False
                for j in range(1, (N - 1) // 2 + 1):
                    if (j * j) % (N - 1) == i % (N - 1):
                        is_residue = True
                        break
                x_qr[i] = 1.0 if is_residue else -1.0
        
        x_qr_imp, s_qr = _local_improve(x_qr, max_iter=200, deadline=deadline)
        if s_qr < best_score:
            best_score = s_qr
            best_x = x_qr_imp.copy()
    
    # Strategy 3: Gradient-based on continuous relaxation, then round
    if time.time() < deadline and best_x is not None:
        # Start from best discrete solution, relax to continuous
        x_cont = best_x.astype(float).copy()
        lr = 0.01
        for it in range(500):
            if time.time() > deadline:
                break
            
            # Compute gradient of score w.r.t. x
            ac = _autoconv(x_cont)
            s = np.sum(x_cont)
            
            # d(max|ac|)/dx is complex; use subgradient at max abs position
            max_idx = np.argmax(np.abs(ac))
            sign_ac = np.sign(ac[max_idx]) if ac[max_idx] != 0 else 1.0
            
            # Gradient of ac[k] w.r.t. x[i]: ac[k] = sum_j x[j]*x[j+k-n+1] (circular-ish)
            # For full autoconv: ac[k] = sum_{j} x[j] * x[j + k - (n-1)] for valid j
            # This is complicated; use numerical gradient for a few steps
            pass
        
        # Just do a few rounds of sign flipping from the relaxed solution
        x_rounded = np.sign(x_cont)
        x_rounded[x_rounded == 0] = 1.0
        x_rounded_imp, s_rounded = _local_improve(x_rounded, max_iter=200, deadline=deadline)
        if s_rounded < best_score:
            best_score = s_rounded
            best_x = x_rounded_imp.copy()
    
    # Final: ensure we return something valid
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
