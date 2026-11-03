"""Rewritten program: signed step heights minimizing max |autoconv| / (sum)^2."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _autoconvolve(x):
    """Compute autoconvolution of x (length N) via FFT. Returns length 2N-1."""
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    ac = np.fft.irfft(f * f, n=2*n)
    return ac[:2*n-1]

def _objective(x):
    """Compute max |autoconv| / (sum x)^2. If sum is 0, return inf."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconvolve(x)
    return np.max(np.abs(ac)) / (s * s)

def _incremental_ac_delta(x, i, old_val, new_val):
    """
    Compute the change in autoconvolution when x[i] changes from old_val to new_val.
    Returns a vector of length 2N-1 with the deltas.
    """
    N = len(x)
    delta = new_val - old_val
    # The autoconvolution at lag k is sum_j x[j]*x[j+k]
    # Changing x[i] affects:
    #   terms where i is the first index: x[i]*x[i+k] for k=0..N-1-i
    #   terms where i is the second index: x[i-k]*x[i] for k=0..i
    # So the delta at lag k is:
    #   if k <= N-1-i: delta * x[i+k]
    #   if k <= i:     delta * x[i-k]
    # Note: for k=0, both apply: delta*x[i] + delta*x[i] = 2*delta*x[i] if we count both,
    # but actually the term x[i]*x[i] is counted once. Let's be careful.
    
    # ac[k] = sum_{j=0}^{N-1-k} x[j] * x[j+k]
    # When we change x[i] by delta:
    #   ac[k] changes by:
    #     delta * x[i+k] if 0 <= i+k < N and i <= N-1-k (i.e., i+k <= N-1)
    #     delta * x[i-k] if 0 <= i-k < N and i-k >= 0 and k <= i
    #     For k=0: the term x[i]*x[i] changes by delta*x[i] + x[i]*delta = 2*delta*x[i]
    #     But wait, ac[0] = sum_j x[j]^2. Changing x[i] changes it by (x[i]+delta)^2 - x[i]^2 = 2*delta*x[i] + delta^2.
    #     So we need to include delta^2 for k=0.
    
    ac_delta = np.zeros(2*N-1)
    # First index i contributes to lags k where i+k < N, i.e., k = 0..N-1-i
    for k in range(0, N - i):
        ac_delta[k] += delta * x[i + k]
    # Second index i contributes to lags k where i-k >= 0, i.e., k = 0..i
    for k in range(0, i + 1):
        ac_delta[k] += delta * x[i - k]
    # For k=0, we have double-counted the cross term but missed the delta^2 term.
    # Actually: ac[0] = sum x[j]^2. Change = (x[i]+d)^2 - x[i]^2 = 2*d*x[i] + d^2.
    # Our loop added: d*x[i] (from first) + d*x[i] (from second) = 2*d*x[i]. Missing d^2.
    ac_delta[0] += delta * delta
    
    return ac_delta

def construct():
    """
    Construct signed step heights (length N, values in {-1, +1}) minimizing
    max |autoconvolution| / (sum x)^2.
    
    Uses local search with incremental autoconvolution updates.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Initialize with random +/- 1
    rng = np.random.default_rng(42)
    x = rng.choice([-1.0, 1.0], size=N)
    
    # Compute initial autoconvolution
    ac = _autoconvolve(x)
    best_score = _objective(x)
    best_x = x.copy()
    
    # Local search: try flipping each element
    # We'll do multiple passes, trying to improve
    max_passes = 500
    for p in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        # Try each position
        for i in range(N):
            if time.time() > deadline:
                break
            
            old_val = x[i]
            new_val = -old_val  # Flip sign
            
            # Compute delta in autoconvolution
            ac_delta = _incremental_ac_delta(x, i, old_val, new_val)
            
            # New sum
            new_sum = np.sum(x) + (new_val - old_val)
            if abs(new_sum) < 1e-12:
                continue
            
            # New max abs autoconv
            new_ac = ac + ac_delta
            new_max_ac = np.max(np.abs(new_ac))
            new_score = new_max_ac / (new_sum * new_sum)
            
            if new_score < best_score - 1e-12:
                # Accept the flip
                x[i] = new_val
                ac = new_ac
                best_score = new_score
                best_x = x.copy()
                improved = True
        
        if not improved:
            # No improvement in this pass; try a different strategy
            # Random restart with perturbation
            if time.time() > deadline - 10:
                break
            
            # Perturb: flip a random subset
            n_flips = rng.integers(5, 30)
            flip_indices = rng.choice(N, size=n_flips, replace=False)
            for i in flip_indices:
                x[i] = -x[i]
            ac = _autoconvolve(x)
            cur_score = _objective(x)
            if cur_score < best_score:
                best_score = cur_score
                best_x = x.copy()
    
    # Final polish: try single flips one more time on best_x
    if time.time() < deadline - 5:
        x = best_x.copy()
        ac = _autoconvolve(x)
        cur_score = _objective(x)
        for _ in range(50):  # Limited polish passes
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                old_val = x[i]
                new_val = -old_val
                ac_delta = _incremental_ac_delta(x, i, old_val, new_val)
                new_sum = np.sum(x) + (new_val - old_val)
                if abs(new_sum) < 1e-12:
                    continue
                new_ac = ac + ac_delta
                new_max_ac = np.max(np.abs(new_ac))
                new_score = new_max_ac / (new_sum * new_sum)
                if new_score < cur_score - 1e-12:
                    x[i] = new_val
                    ac = new_ac
                    cur_score = new_score
                    improved = True
            if not improved:
                break
        if cur_score < best_score:
            best_score = cur_score
            best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
