import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _compute_acf_fft(h):
    """Compute autocorrelation using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return acf

def _update_acf_flip(acf, h, i):
    """
    Update autocorrelation vector after flipping h[i].
    acf[tau] = sum_j h[j] h[j+tau]
    Flipping h[i] means h[i] becomes -h[i].
    Terms involving h[i] change sign.
    In acf[tau], h[i] appears as h[i]h[i+tau] (if i+tau < N) and h[i-tau]h[i] (if i-tau >= 0).
    Actually, acf[tau] = sum_{j=0}^{N-1-tau} h[j] h[j+tau].
    If we flip h[i]:
    New term for j=i: -h[i] h[i+tau] (if i+tau < N)
    New term for j+tau=i => j=i-tau: h[i-tau] (-h[i]) (if i-tau >= 0)
    
    Old contribution from these positions:
    If i < N-tau: h[i] h[i+tau]
    If i-tau >= 0: h[i-tau] h[i]
    
    Change in acf[tau]:
    delta = -2 * h[i] * ( (h[i+tau] if i+tau < N else 0) + (h[i-tau] if i-tau >= 0 else 0) )
    Note: h[i] in the formula is the OLD value before flip.
    """
    old_h_i = h[i]
    # We need to update acf in place
    # h is already flipped in the caller? No, let's say caller flips h[i] before calling this, 
    # so h[i] is the NEW value.
    # Let's stick to: caller flips h[i]. This function updates acf.
    # New h[i] = -Old h[i].
    # Let old_val = h[i] (which is now the new value, so old was -h[i]).
    # This is confusing. Let's do it explicitly.
    
    # Let's assume h[i] has ALREADY been flipped to the new value.
    # We want to compute the difference.
    # Old acf[tau] = ... + old_h_i * neighbor + ...
    # New acf[tau] = ... + new_h_i * neighbor + ...
    # new_h_i = -old_h_i
    # Diff = new_h_i * neighbor - old_h_i * neighbor = -2 * old_h_i * neighbor
    # Since h[i] is now new_h_i, old_h_i = -h[i].
    # Diff = -2 * (-h[i]) * neighbor = 2 * h[i] * neighbor
    
    # Neighbors for lag tau:
    # 1. j=i, term h[i]h[i+tau]. Valid if i+tau < N. Neighbor is h[i+tau].
    # 2. j+tau=i, term h[i-tau]h[i]. Valid if i-tau >= 0. Neighbor is h[i-tau].
    
    # We iterate over all tau from 1 to N-1.
    # This is O(N).
    for tau in range(1, N):
        delta = 0.0
        # Contribution from j=i
        if i + tau < N:
            delta += h[i + tau]
        # Contribution from j=i-tau
        if i - tau >= 0:
            delta += h[i - tau]
        
        # The change in acf[tau] is 2 * h[i] * delta
        # Wait, h[i] is the NEW value.
        # Old term: old_h_i * neighbor. New term: new_h_i * neighbor.
        # new_h_i = -old_h_i.
        # New - Old = -old_h_i * neighbor - old_h_i * neighbor = -2 * old_h_i * neighbor.
        # Since h[i] (new) = -old_h_i, then old_h_i = -h[i].
        # New - Old = -2 * (-h[i]) * neighbor = 2 * h[i] * neighbor.
        
        acf[tau] += 2.0 * h[i] * delta

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    np.random.seed(42)
    
    best_h = np.random.choice([-1, 1], size=N).astype(np.float64)
    best_acf = _compute_acf_fft(best_h)
    best_val = np.max(np.abs(best_acf[1:]))
    
    # Simulated Annealing
    T = 10.0  # Initial temperature
    T_min = 0.1
    cooling = 0.9999
    
    current_h = best_h.copy()
    current_acf = best_acf.copy()
    current_val = best_val
    
    n_iter = 0
    
    while time.time() < deadline:
        # Pick random index
        i = np.random.randint(N)
        
        # Flip
        current_h[i] = -current_h[i]
        
        # Update ACF
        _update_acf_flip(current_acf, current_h, i)
        
        # Evaluate
        new_val = np.max(np.abs(current_acf[1:]))
        
        # Metropolis Criterion
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = current_h.copy()
                best_acf = current_acf.copy()
        else:
            # Revert
            current_h[i] = -current_h[i]
            _update_acf_flip(current_acf, current_h, i)
            # current_val remains unchanged
        
        n_iter += 1
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: restart or just stay low
            # If stuck, we might want to reheat slightly?
            # For now, keep it simple.
            
        # Check time periodically to avoid overhead of time.time() in tight loop?
        # time.time() is fast enough.
        
    return best_h
# EVOLVE-BLOCK-END
