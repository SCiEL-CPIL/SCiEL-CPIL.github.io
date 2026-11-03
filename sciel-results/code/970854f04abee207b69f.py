"""Search for a low-autocorrelation signed step function (400 equal intervals).

Combines simulated annealing with block-flip moves and fast O(N) in-place
autocorrelation updates for the peak sidelobe objective.
"""
import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i] h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))

def _fast_flip_update(h, ac, i):
    """
    Flip h[i] and update ac in place.
    Returns the new peak sidelobe value.
    """
    val = h[i]
    
    # ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k]
    # Flipping h[i] affects terms where j=i (if i+k<N) or j+k=i i.e. j=i-k (if i-k>=0)
    # delta_ac[k] = -2*val*(h[i+k]*1_{i+k<N} + h[i-k]*1_{i-k>=0})
    
    # For k where i+k < N: delta = -2*val*h[i+k]
    # k ranges from 1 to N-1-i
    if i + 1 < N:
        ac[1:N-i] -= 2.0 * val * h[i+1:N]
    # For k where i-k >= 0: delta = -2*val*h[i-k]
    # k ranges from 1 to i
    if i > 0:
        ac[1:i+1] -= 2.0 * val * h[i-1::-1]
        
    # Flip h[i]
    h[i] = -val
    
    # Calculate new peak sidelobe
    return float(np.max(np.abs(ac[1:])))


def _compute_acf(h):
    """Compute aperiodic autocorrelation via FFT, return ac[0..N-1]."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _apply_block_flip(h, ac, i, L):
    """
    Flip h[i:i+L] and update ac in place. Returns the change in h (for undo).
    ac is updated to reflect the new h.
    """
    # Delta for flipping a block: for each position j in [i, i+L-1], h[j] -> -h[j]
    # The autocorrelation ac[k] = sum_j h[j] * h[j+k] (with h[j+k]=0 if out of range)
    # Flipping h[j] for j in block changes ac by:
    #   For each j in block: -2*h[j]*(h[j+k] + h[j-k])  [but careful with boundaries]
    #
    # More precisely, the change in ac[k] when flipping positions in block B:
    #   delta_ac[k] = -2 * sum_{j in B} h[j] * (h[j+k] * 1_{j+k in [0,N-1]} + h[j-k] * 1_{j-k in [0,N-1] and j != j+k})
    # Wait, let me think more carefully.
    #
    # ac[k] = sum_{j=0}^{N-1-k} h[j] * h[j+k]
    # After flipping h[j] for j in B: h'[j] = -h[j] for j in B, h'[j] = h[j] otherwise
    # ac'[k] = sum_{j=0}^{N-1-k} h'[j] * h'[j+k]
    #
    # The change: ac'[k] - ac[k] = sum_j (h'[j]*h'[j+k] - h[j]*h[j+k])
    # For a given k, the terms where both j and j+k are in B: (-h[j])*(-h[j+k]) = h[j]*h[j+k] (no change)
    # Where j in B, j+k not in B: (-h[j])*h[j+k] = -h[j]*h[j+k], change = -2*h[j]*h[j+k]
    # Where j not in B, j+k in B: h[j]*(-h[j+k]) = -h[j]*h[j+k], change = -2*h[j]*h[j+k]
    # Where neither in B: no change
    #
    # So delta_ac[k] = -2 * [sum_{j in B, j+k not in B, 0<=j+k<N} h[j]*h[j+k] + sum_{j not in B, j+k in B, 0<=j<N-k} h[j]*h[j+k]]
    #
    # This is complex to vectorize for a general block. For N=400, let's just recompute
    # ac via FFT if the block is large, or use a simpler approach.
    #
    # Actually, for a single flip (L=1), the delta is simple:
    # delta_ac[k] = -2*h[i]*(h[i+k]*1_{i+k<N} + h[i-k]*1_{i-k>=0, i-k!=i})
    #
    # For a block, let's just recompute ac. The FFT is fast enough (~0.1ms for N=400).
    # The benefit of in-place update is mainly for single flips. For blocks, FFT recompute is fine.
    
    # Flip the block
    h[i:i+L] *= -1.0
    # Recompute ac
    ac[:] = _compute_acf(h)


def _undo_block_flip(h, i, L):
    """Undo a block flip."""
    h[i:i+L] *= -1.0


def _soft_peak(ac, beta=50.0):
    """Soft approximation of max|ac[1:]| using log-sum-exp."""
    x = np.abs(ac[1:])
    m = np.max(x)
    if m < 1e-12:
        return 0.0
    # log-sum-exp: m + log(sum(exp(x - m)))
    s = np.sum(np.exp((x - m) * beta))
    return float(m + np.log(s) / beta)

def _anneal(seed, time_budget):
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    cur_soft = _soft_peak(ac)
    best_h = h.copy()
    best = cur
    T = 5.0
    T_min = 1e-8
    start = time.time()
    n_steps = 0
    
    while time.time() - start < time_budget:
        # Use single flips with fast update
        i = int(rng.integers(0, N))
        
        # Apply flip
        new_val = _fast_flip_update(h, ac, i)
        new_soft = _soft_peak(ac)
        d = new_soft - cur_soft
        
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new_val
            cur_soft = new_soft
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            # Undo
            _fast_flip_update(h, ac, i)
            cur_soft = _soft_peak(ac)
            
        n_steps += 1
        # Adaptive cooling: faster early, slower later
        frac = min(1.0, n_steps / 100000.0)
        T = max(T_min, 5.0 * (1.0 - frac) + T_min)
        
    return best_h, best


def _greedy_polish(h, time_budget, start):
    """Greedy single-flip polish: try each position, keep the best flip."""
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        best_delta = 0.0
        
        for i in range(N):
            # Compute delta in ac without modifying, find new peak
            val = h[i]
            # Save relevant portions
            # For efficiency, compute the change and check if peak decreases
            old_ac = ac.copy()  # This is expensive; let's use a different approach
            
            # Actually, let's just do the flip/undo but track the max more efficiently
            _fast_flip_update(h, ac, i)
            c = float(np.max(np.abs(ac[1:])))
            
            if c < best_c:
                best_c = c
                best_i = i
            # Undo i
            _fast_flip_update(h, ac, i)
            
        if best_i < 0:
            break
            
        # Apply the best flip
        _fast_flip_update(h, ac, best_i)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
            
    return best_h, best


def construct():
    try:
        hard_deadline = 145.0
        start = time.time()
        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        # Long initial anneal from a random +-1 start.
        budget = min(50.0, hard_deadline - (time.time() - start) - 10.0)
        if budget > 1.0:
            h, c = _anneal(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Multiple shorter restarts for diversity.
        restart_id = 1
        while time.time() - start < hard_deadline - 20.0:
            sub = min(12.0, hard_deadline - (time.time() - start) - 20.0)
            if sub < 1.0:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1

        # Final greedy single-flip polish with whatever time remains.
        remaining = hard_deadline - (time.time() - start)
        if remaining > 3.0:
            h, c = _greedy_polish(global_best_h, remaining - 5.0, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Safety: ensure we return before hard deadline
        if time.time() - start > hard_deadline - 3.0:
            return global_best_h

        return global_best_h
    except Exception:
        return np.ones(N)
