"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
d = 1.0 / (2 * N)  # interval width

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin below 150s limit

    def remaining():
        return deadline - time.time()

    def _autoconv_peak(h):
        # max of circular autoconvolution (continuous autoconvolution peak)
        H = np.fft.fft(h)
        ac = np.fft.ifft(H * np.conj(H)).real
        return np.max(ac)

    def score(h):
        h = np.asarray(h, dtype=float)
        sum_sq = float(np.sum(h * h))
        sum_h = float(np.sum(h))
        if sum_h < 1e-15 or sum_sq < 1e-30:
            return 0.0
        peak = _autoconv_peak(h)
        if peak < 1e-30:
            return 0.0
        return sum_sq / (d * sum_h * peak)

    def try_candidate(h, best_h, best_score):
        h = np.asarray(h, dtype=float)
        h = np.maximum(0.0, h)
        if float(np.sum(h)) < 1e-15:
            return h, best_score
        s = score(h)
        if s > best_score:
            return h.copy(), s
        return best_h, best_score

    # Single spike baseline: ratio = 1/d
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = 1.0 / d

    # ---------- Phase 1: exhaustive flat-top / plateau enumeration ----------
    # Flat-top on a contiguous block of k cells has ratio 1/(d*sqrt(k));
    # wider blocks dominate. Enumerate all contiguous blocks and a few
    # plateau variants (core + partial edge cells) to be safe.
    for i in range(N):
        if remaining() < 40.0:
            break
        for j in range(i, N):
            h = np.zeros(N)
            h[i:j + 1] = 1.0
            best_h, best_score = try_candidate(h, best_h, best_score)
            # plateau with half-height edge cells (wider effective support)
            h2 = np.zeros(N)
            h2[i:j + 1] = 1.0
            if i > 0:
                h2[i - 1] = 0.5
            if j < N - 1:
                h2[j + 1] = 0.5
            best_h, best_score = try_candidate(h2, best_h, best_score)

    # ---------- Phase 2: parametric shapes (Gaussian, Lorentzian, power, box) ----------
    t = np.arange(N, dtype=float)
    centers = np.linspace(0.0, N - 1.0, 25)
    widths = np.logspace(-1.0, 2.0, 35)
    for center in centers:
        if remaining() < 30.0:
            break
        for width in widths:
            if remaining() < 30.0:
                break
            # Gaussian
            h = np.exp(-0.5 * ((t - center) / width) ** 2)
            best_h, best_score = try_candidate(h, best_h, best_score)
            # Lorentzian
            h = 1.0 / (1.0 + ((t - center) / width) ** 2)
            best_h, best_score = try_candidate(h, best_h, best_score)
            # Power laws
            for p in (1.0, 1.5, 2.0, 3.0, 5.0):
                h = (1.0 + np.abs(t - center) / width) ** (-p)
                best_h, best_score = try_candidate(h, best_h, best_score)
            # Box
            h = np.where(np.abs(t - center) <= width, 1.0, 0.0)
            best_h, best_score = try_candidate(h, best_h, best_score)
            # Triangular
            h = np.maximum(0.0, 1.0 - np.abs(t - center) / width)
            best_h, best_score = try_candidate(h, best_h, best_score)

    # ---------- Phase 3: random sparse / concentrated shapes ----------
    if remaining() > 15.0:
        rng = np.random.RandomState(42)
        for _ in range(300):
            if remaining() < 15.0:
                break
            n_active = rng.choice([1, 2, 3, 4, 5, 8, 12])
            indices = rng.choice(N, size=n_active, replace=False)
            h = np.zeros(N)
            for idx in indices:
                h[idx] = rng.lognormal(0.0, 1.0)
            best_h, best_score = try_candidate(h, best_h, best_score)

    # ---------- Phase 4: smooth gradient polish (soft-max surrogate + log-barrier) ----------
    # Maximize  sum_sq / (d * sum_h)  subject to  softmax(ac) <= 1
    # via log-barrier; ac is smooth in h (quadratic), softmax is smooth.
    if remaining() > 12.0:
        def objective(x, tau):
            h = np.maximum(0.0, x)
            s2 = float(np.sum(h * h))
            s1 = float(np.sum(h))
            if s1 < 1e-12:
                return 1e12
            H = np.fft.fft(h)
            ac = np.fft.ifft(H * np.conj(H)).real
            m = np.max(ac)
            if m <= 0.0:
                m = 1e-12
            base = ac / m
            sm = m * np.log(np.sum(np.exp((base - 1.0) / tau)))  # log-sum-exp
            if sm > 1.0:
                return 1e12 + (sm - 1.0) ** 2
            return -(s2 / (d * s1))

        for tau in (1e-3, 5e-3, 1e-2, 3e-2):
            if remaining() < 8.0:
                break
            h0 = best_h.copy()
            if np.sum(h0) < 1e-15:
                continue
            # normalize so peak ~ 1 to keep barrier well-scaled
            pk = _autoconv_peak(h0)
            if pk > 1e-15:
                h0 = h0 * (1.0 / pk)
            try:
                res = minimize(
                    lambda x: objective(x, tau),
                    h0, method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12,
                             'maxls': 40},
                )
                hx = np.maximum(0.0, res.x)
                best_h, best_score = try_candidate(hx, best_h, best_score)
            except Exception:
                pass

    # ---------- Phase 5: high-precision coordinate-descent finish ----------
    if remaining() > 3.0:
        h = best_h.copy()
        for _ in range(120):
            if remaining() < 3.0:
                break
            improved = False
            for i in range(N):
                old = h[i]
                if old <= 0.0:
                    # try turning on
                    for val in (0.05, 0.2, 0.5, 1.0, 2.0):
                        h[i] = val
                        s = score(h)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_h = h.copy()
                            improved = True
                            break
                        h[i] = 0.0
                else:
                    for mult in (0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 0.1):
                        h[i] = max(0.0, old * mult)
                        s = score(h)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_h = h.copy()
                            improved = True
                            break
                        h[i] = old
                h[i] = old
            if not improved:
                break
        best_h = h.copy()

    # Final: ensure nonnegativity and validity
    best_h = np.maximum(0.0, best_h)
    if float(np.sum(best_h)) < 1e-15:
        best_h = np.zeros(N)
        best_h[N // 2] = 1.0

    return list(best_h)

# EVOLVE-BLOCK-END
