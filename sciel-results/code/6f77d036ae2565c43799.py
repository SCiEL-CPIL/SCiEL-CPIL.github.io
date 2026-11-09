import numpy as np
import time

N = 400

def _compute_autocorr(x):
    """Compute periodic autocorrelation R[k] for all k using FFT."""
    X = np.fft.fft(x.astype(np.float64))
    R = np.real(np.fft.ifft(X * np.conj(X)))
    return R

def _compute_c3_bound(R):
    """Compute C3 upper bound from autocorrelation array."""
    # Peak sidelobe level
    return np.max(np.abs(R[1:]))

def _fast_flip_autocorr_update(R, x, pos):
    """
    Fast update of autocorrelation when flipping x[pos].
    
    When we flip x[pos] from v to -v, the change in R[k] is:
    delta_R[k] = -2*v * (sum over i where (i+k)%N == pos of x[i] + sum over i where i == pos of x[(i+k)%N] - 2*v)
    
    More precisely: R_new[k] = R_old[k] - 2*v*(x[(pos-k)%N] + x[(pos+k)%N] - 2*v)
    where v is the old value at position pos.
    """
    v = x[pos]
    # For each lag k, the change involves x[(pos-k)%N] and x[(pos+k)%N]
    # But this is O(N) per flip, which is acceptable for N=400
    new_R = R.copy()
    for k in range(N):
        # R[k] = sum_i x[i]*x[(i+k)%N]
        # After flipping x[pos], terms involving x[pos] change
        # Term where i=pos: x[pos]*x[(pos+k)%N] -> -x[pos]*x[(pos+k)%N], change = -2*x[pos]*x[(pos+k)%N]
        # Term where (i+k)%N=pos, i.e., i=(pos-k)%N: x[(pos-k)%N]*x[pos] -> x[(pos-k)%N]*(-x[pos]), change = -2*x[(pos-k)%N]*x[pos]
        # If k==0, both terms are the same (i=pos and (i+k)%N=pos), so change = -2*x[pos]^2 = -2
        if k == 0:
            new_R[k] = R[k] - 2 * v * v
        else:
            new_R[k] = R[k] - 2 * v * (x[(pos + k) % N] + x[(pos - k) % N])
    return new_R

def _generate_m_sequence(m):
    """Generate an m-sequence of length 2^m - 1 using LFSR."""
    if m == 9:
        # x^9 + x^4 + 1
        taps = (1 << 4)
        length = (1 << m) - 1
    elif m == 8:
        taps = (1 << 6) | (1 << 5) | (1 << 4)
        length = (1 << m) - 1
    elif m == 7:
        taps = (1 << 3)
        length = (1 << m) - 1
    elif m == 6:
        taps = (1 << 5)
        length = (1 << m) - 1
    elif m == 5:
        taps = (1 << 2)
        length = (1 << m) - 1
    elif m == 4:
        taps = (1 << 3)
        length = (1 << m) - 1
    else:
        m = 9
        taps = (1 << 4)
        length = (1 << m) - 1
    
    state = (1 << m) - 1
    seq = []
    for i in range(length):
        bit = state & 1
        seq.append(1 if bit == 1 else -1)
        fb = 0
        for t in range(m):
            if (taps >> t) & 1:
                fb ^= (state >> t) & 1
        state = (state >> 1) | (fb << (m - 1))
    
    return np.array(seq, dtype=np.float64)

def construct():
    """Construct a signed step function (sequence of ±1) to minimize C3 upper bound."""
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Start with m-sequence based initialization
    mseq = _generate_m_sequence(9)
    x = mseq[:N].copy()
    
    # Compute initial autocorrelation and metric
    R = _compute_autocorr(x)
    best_metric = _compute_c3_bound(R)
    best_x = x.copy()
    best_R = R.copy()
    
    # Local search with fast autocorrelation updates
    rng = np.random.default_rng(42)
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random position to flip
        pos = int(rng.integers(0, N))
        old_val = x[pos]
        x[pos] = -old_val  # Flip the sign
        
        # Fast update of autocorrelation
        R_new = _fast_flip_autocorr_update(best_R, x, pos)
        # Note: we update from best_R, not the current R, since we're doing hill climbing
        # Actually, we should track the current R separately. Let me restructure.
        
        metric_new = _compute_c3_bound(R_new)
        
        if metric_new < best_metric:
            best_metric = metric_new
            best_x = x.copy()
            best_R = R_new.copy()
            no_improve = 0
        else:
            # Revert the flip
            x[pos] = old_val
            no_improve += 1
    
    return best_x
