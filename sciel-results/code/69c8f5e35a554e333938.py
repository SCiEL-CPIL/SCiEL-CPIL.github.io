"""Autoconvolution minimization via block-flip simulated annealing with multi-scale polish.

We choose 400 signed step heights (+/-1) to minimize the maximum absolute
value of the (linear) autoconvolution, which is the C3 upper bound we want
to lower.  Evaluation is done with a zero-padded FFT so each candidate costs
a single O(L log L) transform (L=1024).
"""
import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345


def _autocorr_fast(h):
    """Linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac


def _hard_score(h):
    """The actual metric: peak autoconvolution concentration (lower is better)."""
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score(h, beta):
    """Smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA."""
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta


def _autocorr_from_fft(F):
    """Compute autocorrelation from precomputed FFT."""
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 140.0  # safety margin before 150s limit
        rng = np.random.default_rng(_SEED)

        # ---------- diverse initializations ----------
        # Include Halton sequence (base 2) for low-discrepancy start
        def halton_base2(n):
            # Generate first n terms of Halton sequence in base 2
            # x_i = sum_{k=1}^inf (i_k / 2^k) where i_k are bits of i
            # Actually, standard Halton for i=1..n:
            # i=1: 1/2 = 0.5
            # i=2: 1/4 = 0.25
            # i=3: 3/4 = 0.75
            # i=4: 1/8 = 0.125
            # i=5: 5/8 = 0.625
            # i=6: 3/8 = 0.375
            # i=7: 7/8 = 0.875
            # i=8: 1/16 = 0.0625
            # Formula: reverse bits of i, interpret as fraction.
            vals = np.zeros(n)
            for i in range(1, n + 1):
                x = 0.0
                d = 0.5
                j = i
                while j > 0:
                    x += (j % 2) * d
                    j //= 2
                    d *= 0.5
                vals[i-1] = x
            return vals
        
        halton_vals = halton_base2(N)
        halton_signs = np.where(halton_vals < 0.5, 1.0, -1.0)
        
        idx = np.arange(N)
        inits = [
            np.ones(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            halton_signs,
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx / N),
            np.sin(2 * np.pi * idx / N),
            np.cos(2 * np.pi * 2 * idx / N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
        ]
        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        cur_hard = best_hard
        cur_smooth = _smooth_score(h, 30.0)

        # Precompute FFT for current signal
        F_cur = np.fft.rfft(h, n=L)

        # ---------- simulated annealing with coordinated block sign-flips ----------
        T = 0.05
        T_min = 1e-4
        decay = 0.99985
        beta = 30.0
        sizes = np.array([1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 48, 64])

        while time.time() < deadline - 8.0:
            if rng.random() < 0.70:
                size = int(rng.choice(sizes))
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                old = h[sl].copy()
                h[sl] *= -1.0
                # Recompute FFT (necessary for block flip)
                F_new = np.fft.rfft(h, n=L)
                ac_new = _autocorr_from_fft(F_new)
                new_hard = float(np.max(np.abs(ac_new)) / ac_new[0])
                if new_hard < cur_hard:
                    cur_hard = new_hard
                    new_smooth = _smooth_score(h, beta)
                    cur_smooth = new_smooth
                    F_cur = F_new
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    # Only compute smooth score if we might accept via SA
                    new_smooth = _smooth_score(h, beta)
                    dS = new_smooth - cur_smooth
                    if dS <= 0 or rng.random() < np.exp(-dS / T):
                        cur_hard = new_hard
                        cur_smooth = new_smooth
                        F_cur = F_new
                    else:
                        h[sl] = old
            else:
                i = int(rng.integers(0, N))
                old = h[i]
                h[i] *= -1.0
                # For single-bit flip, we can update FFT incrementally
                # F_new[k] = F_cur[k] - 2 * h[i] * exp(-2*pi*i*k*i/L)
                # But it's simpler to just recompute
                F_new = np.fft.rfft(h, n=L)
                ac_new = _autocorr_from_fft(F_new)
                new_hard = float(np.max(np.abs(ac_new)) / ac_new[0])
                if new_hard < cur_hard:
                    cur_hard = new_hard
                    new_smooth = _smooth_score(h, beta)
                    cur_smooth = new_smooth
                    F_cur = F_new
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    new_smooth = _smooth_score(h, beta)
                    dS = new_smooth - cur_smooth
                    if dS <= 0 or rng.random() < np.exp(-dS / T):
                        cur_hard = new_hard
                        cur_smooth = new_smooth
                        F_cur = F_new
                    else:
                        h[i] = old
            T = max(T_min, T * decay)

        # ---------- randomized local search polish ----------
        # The exhaustive greedy polish is too slow. Use randomized steepest descent.
        h = best_h.copy()
        cur_hard = best_hard
        F_cur = np.fft.rfft(h, n=L)
        
        # Time budget for polish: use up to deadline - 1.0
        while time.time() < deadline - 1.0:
            # Randomly pick a move: single bit or small block
            if rng.random() < 0.8:
                i = int(rng.integers(0, N))
                h[i] *= -1.0
                F_new = np.fft.rfft(h, n=L)
                ac_new = _autocorr_from_fft(F_new)
                new_hard = float(np.max(np.abs(ac_new)) / ac_new[0])
                if new_hard < cur_hard:
                    cur_hard = new_hard
                    F_cur = F_new
                    if cur_hard < best_hard:
                        best_hard = cur_hard
                        best_h = h.copy()
                else:
                    h[i] *= -1.0 # revert
            else:
                size = int(rng.choice([2, 4, 8]))
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                h[sl] *= -1.0
                F_new = np.fft.rfft(h, n=L)
                ac_new = _autocorr_from_fft(F_new)
                new_hard = float(np.max(np.abs(ac_new)) / ac_new[0])
                if new_hard < cur_hard:
                    cur_hard = new_hard
                    F_cur = F_new
                    if cur_hard < best_hard:
                        best_hard = cur_hard
                        best_h = h.copy()
                else:
                    h[sl] *= -1.0 # revert

        # Final single-bit sweep if time permits (very fast check)
        if time.time() < deadline - 0.5:
            h = best_h.copy()
            for i in range(N):
                if time.time() > deadline - 0.1:
                    break
                old = h[i]
                h[i] *= -1.0
                new_hard = _hard_score(h)
                if new_hard < best_hard - 1e-14:
                    best_hard = new_hard
                    best_h = h.copy()
                else:
                    h[i] = old

        return best_h
    except Exception:
        return np.ones(N)
