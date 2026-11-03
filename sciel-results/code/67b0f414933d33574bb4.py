"""Autoconvolution minimization (circular metric) for the C3 upper bound.

We choose 400 signed step heights (+/-1) on the circle [-1/4,1/4] to minimize
the peak of the *circular* autoconvolution (the C3 upper bound).  The seed
program measured the linear (zero-padded) autocorrelation; here we explicitly
select the circular metric, which is the actual autoconvolution of the step
function on the circle and the quantity that bounds C3.

Pipeline (same family as the seed):
  1. Continuous gradient descent on a smooth LSE-peak / L2 off-DC surrogate of
     the normalized *circular* autocorrelation, plus CAZAC-style inits ->
     threshold to {-1,1}.
  2. Discrete simulated annealing with incremental-FFT single-bit and block
     flips.
  3. Greedy multi-scale local polish + high-precision single-bit polish.
Each FFT-based score costs O(L log L); single-bit flips use an O(L) update.
"""
import time
import numpy as np

N = 400
L = N  # circular autoconvolution -> FFT length equals the number of intervals
_SEED = 12345
_PI2I_OVER_L = -2j * np.pi / L
# Precompute the per-index phase vectors for incremental FFT updates:
# PHASE[i] = exp(-2*pi*i*k*i/L) for k = 0..L/2  (shape L//2+1)
_PHASE = np.exp(_PI2I_OVER_L * np.outer(np.arange(N), np.arange(L // 2 + 1)))  # (N, L//2+1)


def _autocorr_fast(h):
    """Circular one-sided autocorrelation ac[k] = sum_i h[i] h[(i+k) mod N]."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)
    return ac


def _hard_score(h):
    """Peak autoconvolution concentration (lower is better)."""
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _autocorr_from_fft(F):
    return np.fft.irfft(F * np.conj(F), n=L)


def _lse_peak_score(h, beta):
    """Smooth LSE surrogate of the peak of |ac[1:]|/ac[0]. Returns (score, ac, F)."""
    F = np.fft.rfft(h, n=L)
    ac = _autocorr_from_fft(F)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9, ac, F
    y = np.abs(ac[1:]) / dc
    m = float(np.max(y))
    z = beta * (y - m)
    s = m + float(np.log(np.sum(np.exp(z)))) / beta
    return s, ac, F


def _l2_offdc_score_and_grad(h):
    """Minimize sum_{k>=1} ac[k]^2 (circular off-DC energy). Returns (score, grad)."""
    F = np.fft.rfft(h, n=L)
    ac = _autocorr_from_fft(F)
    ac_ext = ac.copy()
    ac_ext[0] = 0.0
    Ac = np.fft.rfft(ac_ext, n=L)
    Hc = np.fft.rfft(h, n=L)
    # circular cross-correlation and convolution via FFT
    conv1 = np.fft.irfft(Ac * np.conj(Hc), n=L)
    conv2 = np.fft.irfft(Ac * Hc, n=L)
    grad = 2.0 * (conv1 + conv2)
    score = float(np.sum(ac[1:] ** 2))
    return score, grad


def _lse_peak_gradient(h, beta):
    """Gradient of the LSE-peak surrogate w.r.t. h (circular). Returns (grad, score)."""
    F = np.fft.rfft(h, n=L)
    ac = _autocorr_from_fft(F)
    dc = ac[0]
    if dc <= 1e-30:
        return np.zeros(N), 1e9
    y = np.abs(ac[1:]) / dc
    m = float(np.max(y))
    z = beta * (y - m)
    e = np.exp(z)
    w = e / float(np.sum(e))
    w_ext = np.zeros(L)
    w_ext[1:] = w / dc
    signs = np.zeros(L)
    signs[1:] = np.sign(ac[1:])
    w_signed = w_ext * signs
    Ws = np.fft.rfft(w_signed, n=L)
    Hc = np.fft.rfft(h, n=L)
    conv1 = np.fft.irfft(Ws * np.conj(Hc), n=L)
    conv2 = np.fft.irfft(Ws * Hc, n=L)
    grad = conv1 + conv2
    score = m + float(np.log(np.sum(e))) / beta
    return grad, score


def _fft_update_single_flip(F, old_val, i):
    """New FFT after flipping h[i] -> -h[i]. O(L) using precomputed phase."""
    return F - 2.0 * old_val * _PHASE[i]


def _fft_update_block_flip(F, old, start, size):
    """New FFT after flipping h[start:start+size] -> -h[...]."""
    idx = np.arange(start, start + size)
    delta = 2.0 * (old[:, None] * _PHASE[idx]).sum(axis=0)
    return F - delta


def _zadoff_chu_init(rng):
    """CAZAC-style Zadoff-Chu phase sequence, thresholded to {-1,1}."""
    # 400 = 16*25; use root 1 with length 400 (not prime but gives low periodic ac)
    n = np.arange(N)
    # quadratic chirp: h[i] = exp(-j*pi*i*(i+ (N%2?1:0))/N)
    h = np.exp(-1j * np.pi * n * (n + (N % 2)) / N)
    h_bin = np.sign(h.real)
    h_bin[h_bin == 0] = 1.0
    # also try a few chirp roots
    best = h_bin
    best_s = _hard_score(h_bin)
    for root in range(1, 6):
        hc = np.exp(-1j * np.pi * root * n * (n + (N % 2)) / N)
        hb = np.sign(hc.real)
        hb[hb == 0] = 1.0
        s = _hard_score(hb)
        if s < best_s:
            best_s = s
            best = hb
    return best


def _continuous_seed(rng, deadline):
    """Continuous gradient descent on circular surrogates, threshold to {-1,1}."""
    # Start from a Zadoff-Chu-like continuous signal (already good for periodic ac)
    n = np.arange(N)
    h = np.exp(-1j * np.pi * n * (n + (N % 2)) / N).real
    h = np.clip(h, -1.0, 1.0)
    best_h = h.copy()
    best_score = float(np.sum(_autocorr_fast(h)[1:] ** 2))

    # Phase A: minimize L2 off-DC energy (flat periodic spectrum)
    lr = 0.05
    for step in range(2500):
        if time.time() > deadline - 80.0:
            break
        score, grad = _l2_offdc_score_and_grad(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        gnorm = float(np.linalg.norm(grad))
        if gnorm < 1e-12:
            break
        h -= lr * grad / (gnorm + 1e-12) * 0.5
        h = np.clip(h, -2.0, 2.0)
        lr = max(lr * 0.9995, 0.005)

    h_bin = np.sign(best_h)
    h_bin[h_bin == 0] = 1.0

    # Phase B: short LSE-peak refinement
    if time.time() < deadline - 60.0:
        h = best_h.copy()
        beta = 50.0
        lr = 0.1
        for step in range(600):
            if time.time() > deadline - 60.0:
                break
            grad, score = _lse_peak_gradient(h, beta)
            gnorm = float(np.linalg.norm(grad))
            if gnorm < 1e-12:
                break
            h -= lr * grad / (gnorm + 1e-12) * 0.1
            h = np.clip(h, -2.0, 2.0)
            beta = min(beta * 1.005, 300.0)
            lr = max(lr * 0.999, 0.01)
        h_bin2 = np.sign(h)
        h_bin2[h_bin2 == 0] = 1.0
        if _hard_score(h_bin2) < _hard_score(h_bin):
            h_bin = h_bin2
    return h_bin


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 140.0
        rng = np.random.default_rng(_SEED)
        idx = np.arange(N)

        # ---------- Phase 1: continuous gradient-descent seed ----------
        h_cont_seed = _continuous_seed(rng, deadline)

        # ---------- Phase 2: diverse initializations ----------
        inits = [
            h_cont_seed,
            _zadoff_chu_init(rng),
            np.ones(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N - 1)),
            np.cos(2 * np.pi * idx ** 2 / (5 * N)),
            np.cos(2 * np.pi * idx ** 2 / (7 * N)),
            np.cos(2 * np.pi * idx / N),
            np.sin(2 * np.pi * idx / N),
            np.cos(2 * np.pi * 2 * idx / N),
            np.sin(2 * np.pi * 2 * idx / N),
            np.cos(2 * np.pi * 3 * idx / N),
            np.sin(2 * np.pi * 3 * idx / N),
            np.cos(2 * np.pi * 5 * idx / N),
            np.sin(2 * np.pi * 5 * idx / N),
            np.sign(np.cos(2 * np.pi * idx * np.sqrt(2) / N)),
            np.sign(np.cos(2 * np.pi * idx * np.sqrt(3) / N)),
            np.sign(np.sin(2 * np.pi * idx * 8 / N)),
            np.sign(np.sin(2 * np.pi * idx * 16 / N)),
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
        F_cur = np.fft.rfft(h, n=L)

        # ---------- Phase 3: discrete SA with incremental FFT ----------
        T = 0.08
        T_min = 1e-4
        decay = 0.9999
        beta = 50.0
        sizes = np.array([2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128])

        def _smooth_from_ac(ac):
            dc = ac[0]
            if dc <= 1e-30:
                return 1e9
            y = np.abs(ac[1:]) / dc
            m = float(np.max(y))
            z = beta * (y - m)
            return m + float(np.log(np.sum(np.exp(z)))) / beta

        ac_cur = _autocorr_from_fft(F_cur)
        cur_smooth = _smooth_from_ac(ac_cur)
        step_count = 0

        while time.time() < deadline - 30.0:
            step_count += 1
            if rng.random() < 0.35:
                # Block flip with incremental FFT
                size = int(rng.choice(sizes))
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                old = h[sl].copy()
                h[sl] *= -1.0
                F_new = _fft_update_block_flip(F_cur, old, start, size)
                ac_new = _autocorr_from_fft(F_new)
                dc = ac_new[0]
                new_hard = 1e9 if dc <= 1e-30 else float(np.max(np.abs(ac_new)) / dc)
                if new_hard < cur_hard:
                    cur_hard = new_hard
                    F_cur = F_new
                    ac_cur = ac_new
                    cur_smooth = _smooth_from_ac(ac_new)
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    new_smooth = _smooth_from_ac(ac_new)
                    dS = new_smooth - cur_smooth
                    if dS <= 0 or rng.random() < np.exp(-dS / T):
                        cur_hard = new_hard
                        F_cur = F_new
                        ac_cur = ac_new
                        cur_smooth = new_smooth
                    else:
                        h[sl] = old
            else:
                # Single-bit flip (incremental FFT, O(L))
                i = int(rng.integers(0, N))
                old_val = h[i]
                h[i] *= -1.0
                F_new = _fft_update_single_flip(F_cur, old_val, i)
                ac_new = _autocorr_from_fft(F_new)
                dc = ac_new[0]
                new_hard = 1e9 if dc <= 1e-30 else float(np.max(np.abs(ac_new)) / dc)
                if new_hard < cur_hard:
                    cur_hard = new_hard
                    F_cur = F_new
                    ac_cur = ac_new
                    cur_smooth = _smooth_from_ac(ac_new)
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    new_smooth = _smooth_from_ac(ac_new)
                    dS = new_smooth - cur_smooth
                    if dS <= 0 or rng.random() < np.exp(-dS / T):
                        cur_hard = new_hard
                        F_cur = F_new
                        ac_cur = ac_new
                        cur_smooth = new_smooth
                    else:
                        h[i] = old_val
            T = max(T_min, T * decay)
            # Periodic restart from best if stuck
            if step_count % 5000 == 0 and time.time() < deadline - 35.0:
                if cur_hard > best_hard * 1.05:
                    h = best_h.copy()
                    F_cur = np.fft.rfft(h, n=L)
                    ac_cur = _autocorr_from_fft(F_cur)
                    cur_hard = best_hard
                    cur_smooth = _smooth_from_ac(ac_cur)

        # ---------- Phase 4: greedy multi-scale local polish ----------
        h = best_h.copy()
        F_cur = np.fft.rfft(h, n=L)
        sizes_polish = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
        for _pass in range(40):
            if time.time() > deadline - 8.0:
                break
            any_improve = False
            for size in sizes_polish:
                for start in range(0, N - size + 1):
                    if time.time() > deadline - 8.0:
                        return best_h
                    sl = slice(start, start + size)
                    old = h[sl].copy()
                    h[sl] *= -1.0
                    F_new = _fft_update_block_flip(F_cur, old, start, size)
                    ac_new = _autocorr_from_fft(F_new)
                    dc = ac_new[0]
                    new_hard = 1e9 if dc <= 1e-30 else float(np.max(np.abs(ac_new)) / dc)
                    if new_hard < best_hard - 1e-12:
                        best_hard = new_hard
                        best_h = h.copy()
                        F_cur = F_new
                        any_improve = True
                    else:
                        h[sl] = old
            if not any_improve:
                break

        # ---------- Phase 5: high-precision single-bit polish ----------
        h = best_h.copy()
        F_cur = np.fft.rfft(h, n=L)
        for _pass in range(120):
            if time.time() > deadline - 3.0:
                break
            any_improve = False
            for i in range(N):
                if time.time() > deadline - 3.0:
                    return best_h
                old_val = h[i]
                h[i] *= -1.0
                F_new = _fft_update_single_flip(F_cur, old_val, i)
                ac_new = _autocorr_from_fft(F_new)
                dc = ac_new[0]
                new_hard = 1e9 if dc <= 1e-30 else float(np.max(np.abs(ac_new)) / dc)
                if new_hard < best_hard - 1e-14:
                    best_hard = new_hard
                    best_h = h.copy()
                    F_cur = F_new
                    any_improve = True
                else:
                    h[i] = old_val
            if not any_improve:
                break

        return best_h
    except Exception:
        return np.ones(N)
