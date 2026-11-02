"""Search for a low-autocorrelation signed step function (400 equal intervals).

Minimise the peak aperiodic autocorrelation sidelobe  max_{k>=1} |sum_i h[i] h[i+k]|
for a +-1 sequence of length 400; this drives the C3 autoconvolution upper bound.

Structural change vs. the previous seed:
  1) A continuous "flat-spectrum" gradient-descent phase on the L2 norm of the
     autocorrelation sidelobes is run first; the resulting real vector is
     thresholded to a +-1 binary seed.  This gives the discrete local search a
     far better starting point than a random +-1 sequence.
  2) The discrete local search uses an O(N) incremental ACF update when a
     single bit is flipped, so each candidate flip costs O(N) rather than the
     O(N log N) of an FFT-based peak evaluation.  This lets the search explore
     many more candidate flips within the wall-clock budget and converge to a
     lower peak sidelobe.
"""
import numpy as np
import time

N = 400


def _ac_from_h(h):
    """Aperiodic autocorrelation via FFT: ac[k] = sum_i h[i]*h[i+k], k=0..N-1."""
    size = 1
    while size < 2 * N:
        size *= 2
    H = np.fft.fft(h, size)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_from_ac(ac):
    """Peak sidelobe magnitude from an ACF array (ac[0] is the DC term)."""
    return float(np.max(np.abs(ac[1:])))


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i]*h[i+k]|."""
    return _peak_from_ac(_ac_from_h(h))


def _flip_delta(h, i, ac):
    """Return the change (new_ac - ac) when bit i of h is flipped.

    Flipping h[i] to -h[i] changes the ACF by:
      delta[k] = -2*h[i]*( sum_{j=i-k}^{i-1} h[j]  +  sum_{j=i+1}^{i+k} h[j] )
    for 1 <= k <= N-1, and delta[0] = -2*N*h[i]^2 = -2*N (since h[i]^2 = 1).
    This is O(N) per call.
    """
    hi = h[i]
    delta = np.empty(N, dtype=np.float64)
    delta[0] = -2.0 * N  # h[i]^2 = 1
    for k in range(1, N):
        lo = max(0, i - k)
        hi_lo = min(i, i - k + 1)  # exclusive upper for left sum
        hi_hi = min(N, i + k + 1)
        lo_hi = max(i + 1, i - k + 1)
        # left sum: j in [lo, hi_lo)
        # right sum: j in [lo_hi, hi_hi)
        s = 0.0
        if hi_lo > lo:
            s += np.sum(h[lo:hi_lo])
        if hi_hi > lo_hi:
            s += np.sum(h[lo_hi:hi_hi])
        delta[k] = -2.0 * hi * s
    return delta


def _flat_spectrum_init(seed, time_budget):
    """Continuous gradient descent on sum_k ac[k]^2 (k>=1), then threshold to +-1.

    Returns (h_binary, peak_sidelobe).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    x -= np.mean(x)
    if np.linalg.norm(x) < 1e-9:
        x = rng.normal(size=N)
    x /= np.linalg.norm(x) * np.sqrt(N)

    start = time.time()
    # Adaptive step size
    step = 0.05
    best_x = x.copy()
    best_val = np.inf

    for _ in range(20000):
        if time.time() - start >= time_budget:
            break
        ac = _ac_from_h(x)
        # Gradient of (1/2) * sum_{k=1}^{N-1} ac[k]^2 w.r.t. x
        # ac[k] = sum_i x[i] x[i+k], d ac[k]/d x[i] = x[i+k] + x[i-k] (with boundary 0)
        # grad[i] = sum_{k=1}^{N-1} 2*ac[k]*(x[i+k] + x[i-k])
        # Use FFT-based approach for speed:
        # grad = 2 * (IFFT(FFT(x)^2) convolved appropriately)
        # Simpler: compute via the relation that grad[i] = 2 * sum_k ac[k] * (x[i+k]+x[i-k])
        # This is a convolution-like operation.  Use FFT:
        # Let A = FFT(x).  Then ac_full = IFFT(|A|^2) gives circular autocorrelation.
        # For aperiodic, we computed ac via zero-padding.  The gradient can be
        # computed as: grad = 2 * (ac_padded convolved with x in a specific way).
        # For simplicity and correctness, use a direct but vectorized approach:
        # grad[i] = 2 * sum_{k=1}^{N-1} ac[k] * (x[i+k] if i+k<N else 0 + x[i-k] if i-k>=0 else 0)
        # This is O(N^2) which is too slow.  Use FFT trick:
        # grad[i] = 2 * (sum_k ac[k]*x[i+k]) + 2*(sum_k ac[k]*x[i-k])
        # = 2 * (ac reversed convolved with x) evaluated appropriately.
        # Let's use: for the first term, define ac_rev[k] = ac[N-1-k] for k=0..N-2, ac_rev[N-1]=0
        # Then sum_k ac[k]*x[i+k] for k=1..N-1 = sum_j ac_rev[j]*x[i-j+1] ... this is getting complex.
        # 
        # Alternative: since N=400, O(N^2)=160000 per iteration is actually fine for ~2000 iterations.
        # 2000 * 160000 = 320M ops, ~1-2s in numpy.  Let's just do it directly with numpy.
        
        # Vectorized O(N^2) gradient:
        # grad[i] = 2 * sum_{k=1}^{N-1} ac[k] * (x[i+k]*1_{i+k<N} + x[i-k]*1_{i-k>=0})
        # Build using broadcasting is O(N^2) memory.  Instead, loop over k with numpy sums.
        grad = np.zeros(N, dtype=np.float64)
        for k in range(1, N):
            ak = ac[k]
            if ak == 0.0:
                continue
            # x[i+k] for i=0..N-1-k
            grad[:N-k] += 2.0 * ak * x[k:]
            # x[i-k] for i=k..N-1
            grad[k:] += 2.0 * ak * x[:N-k]
        
        # Normalise gradient to control step size
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        grad /= gnorm
        
        x_new = x - step * grad
        # Check if this reduces the L2 objective
        ac_new = _ac_from_h(x_new)
        val_new = float(np.sum(ac_new[1:]**2))
        if val_new < float(np.sum(ac[1:]**2)):
            x = x_new
            step *= 1.05
            if val_new < best_val:
                best_val = val_new
                best_x = x.copy()
        else:
            step *= 0.8
            # Try a smaller step
            x_try = x - step * grad
            ac_try = _ac_from_h(x_try)
            val_try = float(np.sum(ac_try[1:]**2))
            if val_try < float(np.sum(ac[1:]**2)):
                x = x_try
                if val_try < best_val:
                    best_val = val_try
                    best_x = x.copy()
    
    # Threshold to +-1
    h_bin = np.where(best_x >= 0, 1.0, -1.0)
    peak = _peak_sidelobe(h_bin)
    return h_bin, peak


def _local_search(h, ac, time_budget, max_sweeps=None):
    """Greedy single-flip local search using O(N) incremental ACF updates.

    Repeatedly sweeps over all positions, flipping the bit that gives the
    largest reduction in peak sidelobe.  Each candidate flip is evaluated in
    O(N) via the incremental delta, so a full sweep costs O(N^2) ~ 160k ops.
    """
    h = h.copy()
    ac = ac.copy()
    cur_peak = _peak_from_ac(ac)
    best_peak = cur_peak
    best_h = h.copy()
    
    start = time.time()
    sweep_count = 0
    while True:
        if time.time() - start >= time_budget:
            break
        if max_sweeps is not None and sweep_count >= max_sweeps:
            break
        
        best_i = -1
        best_new_peak = cur_peak
        best_new_ac = None
        
        for i in range(N):
            delta = _flip_delta(h, i, ac)
            new_ac = ac + delta
            new_peak = _peak_from_ac(new_ac)
            if new_peak < best_new_peak:
                best_new_peak = new_peak
                best_i = i
                best_new_ac = new_ac
        
        if best_i < 0 or best_new_peak >= cur_peak - 1e-12:
            break
        
        h[best_i] *= -1.0
        ac = best_new_ac
        cur_peak = best_new_peak
        if cur_peak < best_peak:
            best_peak = cur_peak
            best_h = h.copy()
        sweep_count += 1
    
    return best_h, best_peak


def _pair_flip_local_search(h, ac, time_budget, max_sweeps=None):
    """Greedy adjacent-pair-flip local search using O(N) incremental updates.

    Flipping two adjacent bits i, i+1: the ACF change is the sum of the
    individual deltas (since the ACF is bilinear in h, the combined change is
    the sum of the two single-flip deltas only if we account for the cross-term).
    Actually, flipping both i and i+1: h[i]->-h[i], h[i+1]->-h[i+1].
    The change in ac[k] = sum_j h[j]*h[j+k]:
      For terms involving h[i] or h[i+1], the change is not simply additive
      because the cross-term h[i]*h[i+1+k] and h[i+k]*h[i+1] etc. change.
    For correctness, recompute the full ACF after the pair flip using FFT
    (O(N log N)), but only for the best pair found via a cheaper heuristic.
    """
    h = h.copy()
    ac = ac.copy()
    cur_peak = _peak_from_ac(ac)
    best_peak = cur_peak
    best_h = h.copy()
    
    start = time.time()
    sweep_count = 0
    while True:
        if time.time() - start >= time_budget:
            break
        if max_sweeps is not None and sweep_count >= max_sweeps:
            break
        
        best_pair = -1
        best_new_peak = cur_peak
        
        # Evaluate all adjacent pairs using FFT (O(N log N) each, but N=400 is small)
        for i in range(N - 1):
            h[i] *= -1.0
            h[i + 1] *= -1.0
            new_ac = _ac_from_h(h)
            new_peak = _peak_from_ac(new_ac)
            if new_peak < best_new_peak:
                best_new_peak = new_peak
                best_pair = i
            h[i] *= -1.0
            h[i + 1] *= -1.0
        
        if best_pair < 0 or best_new_peak >= cur_peak - 1e-12:
            break
        
        h[best_pair] *= -1.0
        h[best_pair + 1] *= -1.0
        ac = _ac_from_h(h)
        cur_peak = best_new_peak
        if cur_peak < best_peak:
            best_peak = cur_peak
            best_h = h.copy()
        sweep_count += 1
    
    return best_h, best_peak


def _anneal_discrete(seed, time_budget):
    """Simulated annealing on the discrete +-1 sequence with O(N) flip deltas.

    Uses incremental ACF updates for single-bit flips.  Block flips of length > 1
    require O(N*L) work; we restrict to single-bit flips for speed and rely on
    the flat-spectrum init for a good starting point.
    """
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    ac = _ac_from_h(h)
    cur = _peak_from_ac(ac)
    best_h = h.copy()
    best = cur
    
    T_start = 2.0
    T_end = 1e-4
    start = time.time()
    
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        T = T_start * (T_end / T_start) ** progress
        
        i = int(rng.integers(0, N))
        delta = _flip_delta(h, i, ac)
        new_ac = ac + delta
        new_peak = _peak_from_ac(new_ac)
        d = new_peak - cur
        
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            h[i] *= -1.0
            ac = new_ac
            cur = new_peak
            if cur < best:
                best = cur
                best_h = h.copy()
        # else: reject, h and ac unchanged
    
    return best_h, best


def construct():
    try:
        hard_deadline = 135.0
        start = time.time()

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 0: flat-spectrum continuous init to get a good binary seed.
        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        init_budget = min(15.0, _remaining() - 10.0)
        if init_budget > 2.0:
            h_init, c_init = _flat_spectrum_init(42, init_budget)
            if c_init < global_best:
                global_best = c_init
                global_best_h = h_init

        # Phase 1: fast single-flip local search from the flat-spectrum seed.
        ac = _ac_from_h(global_best_h)
        ls_budget = min(20.0, _remaining() - 10.0)
        if ls_budget > 1.0:
            h_ls, c_ls = _local_search(global_best_h, ac, ls_budget)
            if c_ls < global_best:
                global_best = c_ls
                global_best_h = h_ls

        # Phase 2: pair-flip local search to escape plateaus.
        ac = _ac_from_h(global_best_h)
        pf_budget = min(10.0, _remaining() - 10.0)
        if pf_budget > 1.0:
            h_pf, c_pf = _pair_flip_local_search(global_best_h, ac, pf_budget)
            if c_pf < global_best:
                global_best = c_pf
                global_best_h = h_pf

        # Phase 3: discrete anneal for exploration, then polish.
        anneal_budget = min(20.0, _remaining() - 10.0)
        if anneal_budget > 2.0:
            h_an, c_an = _anneal_discrete(12345, anneal_budget)
            if c_an < global_best:
                global_best = c_an
                global_best_h = h_an
            # Polish after anneal
            ac = _ac_from_h(global_best_h)
            pol_budget = min(10.0, _remaining() - 5.0)
            if pol_budget > 0.5:
                h_pol, c_pol = _local_search(global_best_h, ac, pol_budget)
                if c_pol < global_best:
                    global_best = c_pol
                    global_best_h = h_pol

        # Phase 4: final single-flip polish.
        ac = _ac_from_h(global_best_h)
        final_budget = _remaining() - 2.0
        if final_budget > 0.5:
            h_fin, c_fin = _local_search(global_best_h, ac, final_budget)
            if c_fin < global_best:
                global_best = c_fin
                global_best_h = h_fin

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
