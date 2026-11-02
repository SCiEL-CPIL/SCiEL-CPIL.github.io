"""Autoconvolution minimization via continuous relaxation seeding and block-flip simulated annealing.

We choose 400 signed step heights (+/-1) to minimize the maximum absolute
value of the (linear) autoconvolution, which is the C3 upper bound we want
to lower. 

The strategy is:
1. Initialize using a continuous relaxation: minimize the L2 norm of the 
   non-DC components of the FFT (equivalent to minimizing total sidelobe 
   energy) using gradient descent. Threshold the result to get a high-quality 
   binary seed.
2. Refine using Simulated Annealing with block sign-flips, using the peak 
   sidelobe ratio as the primary metric and a smooth log-sum-exp surrogate 
   for exploration.
3. Polish with greedy local search.
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
    if m < 1e-30:
        return 0.0
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta


def _continuous_seed(rng):
    """
    Generate a high-quality binary seed by minimizing the total sidelobe energy
    of a continuous signal via gradient descent, then thresholding.
    """
    # Start with a random continuous signal
    x = rng.standard_normal(N)
    
    # Gradient descent to minimize sum_{f != 0} |X_f|^2
    # This is equivalent to maximizing |X_0|^2 for fixed energy.
    # Gradient of sum |X_f|^2 w.r.t x is 2 * Real(IFFT(FFT(x) * conj(FFT(x))))
    # We want to minimize the non-DC part.
    
    # Mask for DC component (index 0 in rfft)
    mask = np.ones(L // 2 + 1)
    mask[0] = 0.0
    
    lr = 0.01
    for step in range(500):
        F = np.fft.rfft(x, n=L)
        # Power spectrum
        P = F * np.conj(F)
        # Zero out DC power
        P_masked = P * mask
        # Gradient in time domain
        # d/dx sum |X_f|^2 = 2 * Real(IFFT(2 * X * conj(X) * mask)) / L ?
        # Let's use the standard relation: 
        # autocorr = IFFT(FFT(x) * conj(FFT(x)))
        # We want to minimize sum_{k=1}^{N-1} ac[k]^2.
        # This is equivalent to minimizing sum_{f != 0} |X_f|^4 ? No.
        # Parseval: sum ac[k]^2 = (1/L) sum |X_f|^4.
        # So we want to minimize sum |X_f|^4 for f != 0.
        # Gradient of |X_f|^4 is 4 |X_f|^2 X_f.
        # So gradient of sum_{f != 0} |X_f|^4 is IFFT(4 * |F|^2 * F * mask).
        
        grad_freq = 4.0 * (F * np.conj(F) * F) * mask
        grad_time = np.fft.irfft(grad_freq, n=L)[:N]
        
        x -= lr * grad_time
        
        # Normalize x to keep it bounded
        norm = np.linalg.norm(x)
        if norm > 0:
            x /= norm
    
    # Threshold to binary
    h = np.sign(x)
    # Ensure no zeros (sign(0) is 0)
    h[h == 0] = 1.0
    return h


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 140.0  # safety margin before 150s limit
        rng = np.random.default_rng(_SEED)

        # ---------- Initialization ----------
        # 1. Generate continuous relaxation seed
        h_cont = _continuous_seed(rng)
        
        # 2. Also prepare standard inits just in case
        idx = np.arange(N)
        inits = [
            h_cont,
            np.ones(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
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

        # ---------- Simulated Annealing ----------
        T = 0.02
        T_min = 1e-4
        decay = 0.9997
        beta = 30.0
        sizes = np.array([1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 48, 64])

        while time.time() < deadline - 8.0:
            if rng.random() < 0.70:
                size = int(rng.choice(sizes))
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                old = h[sl].copy()
                h[sl] *= -1.0
                new_hard = _hard_score(h)
                if new_hard < cur_hard:
                    cur_hard = new_hard
                    new_smooth = _smooth_score(h, beta)
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    new_smooth = _smooth_score(h, beta)
                    dS = new_smooth - cur_smooth
                    if dS <= 0 or rng.random() < np.exp(-dS / T):
                        cur_hard = new_hard
                        cur_smooth = new_smooth
                    else:
                        h[sl] = old
            else:
                i = int(rng.integers(0, N))
                old = h[i]
                h[i] *= -1.0
                new_hard = _hard_score(h)
                if new_hard < cur_hard:
                    cur_hard = new_hard
                    new_smooth = _smooth_score(h, beta)
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    new_smooth = _smooth_score(h, beta)
                    dS = new_smooth - cur_smooth
                    if dS <= 0 or rng.random() < np.exp(-dS / T):
                        cur_hard = new_hard
                        cur_smooth = new_smooth
                    else:
                        h[i] = old
            T = max(T_min, T * decay)

        # ---------- Greedy multi-scale local polish ----------
        h = best_h.copy()
        sizes_polish = [1, 2, 4, 8, 16, 32]
        for _pass in range(60):
            if time.time() > deadline - 3.0:
                break
            any_improve = False
            for size in sizes_polish:
                for start in range(0, N - size + 1):
                    if time.time() > deadline - 3.0:
                        return best_h
                    sl = slice(start, start + size)
                    old = h[sl].copy()
                    h[sl] *= -1.0
                    new_hard = _hard_score(h)
                    if new_hard < best_hard - 1e-12:
                        best_hard = new_hard
                        best_h = h.copy()
                        any_improve = True
                    else:
                        h[sl] = old
            if not any_improve:
                break

        # ---------- Final high-precision single-bit polish ----------
        for _pass in range(10):
            if time.time() > deadline - 1.0:
                break
            any_improve = False
            for i in range(N):
                if time.time() > deadline - 1.0:
                    return best_h
                old = h[i]
                h[i] *= -1.0
                new_hard = _hard_score(h)
                if new_hard < best_hard - 1e-14:
                    best_hard = new_hard
                    best_h = h.copy()
                    any_improve = True
                else:
                    h[i] = old
            if not any_improve:
                break

        return best_h
    except Exception:
        return np.ones(N)
