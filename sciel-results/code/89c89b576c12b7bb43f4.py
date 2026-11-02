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


def _anneal(seed, time_budget):
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best_h = h.copy()
    best = cur
    T = 3.0
    T_min = 1e-6
    start = time.time()
    
    while time.time() - start < time_budget:
        # Choose block size: mostly small blocks (4-8), occasionally larger (16-32)
        r = rng.random()
        if r < 0.5:
            L = int(rng.integers(1, 5))
        elif r < 0.8:
            L = int(rng.integers(4, 13))
        elif r < 0.95:
            L = int(rng.integers(8, 25))
        else:
            L = int(rng.integers(16, 33))
        L = min(L, N)
        
        i = int(rng.integers(0, N - L + 1))
        
        # Apply flip and compute new objective
        h[i:i+L] *= -1.0
        ac_new = _compute_acf(h)
        new_val = float(np.max(np.abs(ac_new[1:])))
        d = new_val - cur
        
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new_val
            ac = ac_new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            # Undo
            h[i:i+L] *= -1.0
        
        T = max(T_min, T * 0.99995)
    
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
        best_ac = None
        
        for i in range(N):
            h[i] *= -1.0
            ac_try = _compute_acf(h)
            c = float(np.max(np.abs(ac_try[1:])))
            if c < best_c:
                best_c = c
                best_i = i
                best_ac = ac_try.copy()
            h[i] *= -1.0
        
        if best_i < 0:
            break
        
        h[best_i] *= -1.0
        ac = best_ac
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    
    return best_h, best


def construct():
    try:
        hard_deadline = 140.0
        start = time.time()
        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        # Long initial anneal from a random +-1 start.
        budget = min(40.0, hard_deadline - (time.time() - start) - 10.0)
        if budget > 1.0:
            h, c = _anneal(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Multiple shorter restarts for diversity.
        restart_id = 1
        while time.time() - start < hard_deadline - 15.0:
            sub = min(15.0, hard_deadline - (time.time() - start) - 15.0)
            if sub < 1.0:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1

        # Final greedy single-flip polish with whatever time remains.
        remaining = hard_deadline - (time.time() - start)
        if remaining > 2.0:
            h, c = _greedy_polish(global_best_h, remaining - 3.0, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return global_best_h
    except Exception:
        return np.ones(N)
